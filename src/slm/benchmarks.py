"""Multiple-choice benchmarks scored by log-likelihood.

Each benchmark is fetched at a pinned revision and converted to one JSONL format:
``{"id", "context", "choices": [...], "label"}``. A model's answer is the choice with the highest
log-probability given the context (``acc``), or the highest log-probability per byte of the choice
(``acc_norm``), following the conventions of lm-evaluation-harness. Prompt templates follow that
harness so that numbers are comparable in kind, though not guaranteed identical.

Belebele is parallel across languages, which gives a like-for-like English/French comparison.
Benchmark files are evaluation data: every one used in a reported result must also be listed in the
decontamination step of the training-data pipelines (docs/DATA_POLICY.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from slm.model import SLM
from slm.runlog import sha256_file, write_json
from slm.tokenizer import EOS

BENCHMARKS: dict[str, dict[str, Any]] = {
    "arc_easy": {"repo": "allenai/ai2_arc", "revision": "210d026faf9955653af8916fad021475a3f00453",
                 "file": "ARC-Easy/test-00000-of-00001.parquet", "license": "CC-BY-SA-4.0"},
    "hellaswag": {"repo": "Rowan/hellaswag", "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                  "file": "data/validation-00000-of-00001.parquet", "license": "MIT (declared in the original repository)"},
    "belebele_en": {"repo": "facebook/belebele", "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
                    "file": "data/eng_Latn.jsonl", "license": "CC-BY-SA-4.0"},
    "belebele_fr": {"repo": "facebook/belebele", "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
                    "file": "data/fra_Latn.jsonl", "license": "CC-BY-SA-4.0"},
}


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def convert(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for i, r in enumerate(rows):
        if name == "arc_easy":
            labels = list(r["choices"]["label"])
            items.append({"id": r["id"], "context": f"Question: {r['question']}\nAnswer:",
                          "choices": [f" {c}" for c in r["choices"]["text"]], "label": labels.index(r["answerKey"])})
        elif name == "hellaswag":
            ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
            items.append({"id": str(r.get("ind", i)), "context": _hellaswag_clean(r["activity_label"] + ": " + ctx),
                          "choices": [" " + _hellaswag_clean(e) for e in r["endings"]], "label": int(r["label"])})
        elif name.startswith("belebele"):
            items.append({"id": f"{r['link']}#{r['question_number']}",
                          "context": f"P: {r['flores_passage']}\nQ: {r['question']}\nA:",
                          "choices": [f" {r[f'mc_answer{k}']}" for k in range(1, 5)],
                          "label": int(r["correct_answer_num"]) - 1})
        else:
            raise ValueError(f"unknown benchmark {name!r}")
    return items


def fetch(name: str, out_dir: str | Path) -> Path:
    from huggingface_hub import hf_hub_download

    spec = BENCHMARKS[name]
    local = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"])
    if local.endswith(".parquet"):
        import pyarrow.parquet as pq

        rows = pq.read_table(local).to_pylist()
    else:
        with open(local, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    items = convert(name, rows)
    out = Path(out_dir) / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    write_json(out.with_suffix(".manifest.json"), {**spec, "name": name, "items": len(items),
                                                   "source_sha256": sha256_file(local), "sha256": sha256_file(out)})
    return out


def load_items(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return items[:limit] if limit else items


@torch.no_grad()
def choice_logprobs(model: SLM, tok, context: str, choices: list[str], device: torch.device) -> list[float]:
    """Sum of log-probabilities of each choice's tokens after the context (left-truncated to fit)."""
    # Choices start with a space and the pre-tokenizer splits before spaces, so encoding the context
    # and the choice separately gives the same tokens as encoding their concatenation.
    ctx = [tok.token_to_id(EOS)] + tok.encode(context).ids
    scores = []
    for choice in choices:
        cont = tok.encode(choice).ids
        ids = (ctx + cont)[-(model.cfg.max_seq_len + 1) :]
        x = torch.tensor(ids[:-1], device=device)[None]
        y = torch.tensor(ids[1:], device=device)[None]
        logp = model(x)[0].float().log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)
        scores.append(logp[0, -len(cont) :].sum().item())
    return scores


@torch.no_grad()
def batched_choice_logprobs(model: SLM, tok, items: list[dict[str, Any]], device: torch.device) -> list[list[float]]:
    """Same scores as ``choice_logprobs`` for several items in one forward pass. Sequences are
    right-padded; with causal attention padding cannot affect the scored positions."""
    eos = tok.token_to_id(EOS)
    seqs, spans = [], []
    for it in items:
        ctx = [eos] + tok.encode(it["context"]).ids
        for choice in it["choices"]:
            cont = tok.encode(choice).ids
            ids = (ctx + cont)[-(model.cfg.max_seq_len + 1) :]
            seqs.append(ids)
            spans.append(min(len(cont), len(ids) - 1))
    width = max(len(s) for s in seqs) - 1
    x = torch.full((len(seqs), width), eos, dtype=torch.long)
    y = torch.full((len(seqs), width), eos, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s) - 1] = torch.tensor(s[:-1])
        y[i, : len(s) - 1] = torch.tensor(s[1:])
    x, y = x.to(device), y.to(device)
    logp = model(x)[0].float().log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)
    flat = [logp[i, len(s) - 1 - n : len(s) - 1].sum().item() for i, (s, n) in enumerate(zip(seqs, spans))]
    out, k = [], 0
    for it in items:
        out.append(flat[k : k + len(it["choices"])])
        k += len(it["choices"])
    return out


@torch.no_grad()
def multiple_choice(model: SLM, tok, items: list[dict[str, Any]], device: torch.device, batch_items: int = 16) -> dict[str, Any]:
    model.eval()
    correct = correct_norm = 0
    all_scores: list[list[float]] = []
    for i in range(0, len(items), batch_items):
        all_scores += batched_choice_logprobs(model, tok, items[i : i + batch_items], device)
    for it, scores in zip(items, all_scores):
        norm = [s / max(1, len(c.encode("utf-8"))) for s, c in zip(scores, it["choices"])]
        correct += int(max(range(len(scores)), key=scores.__getitem__) == it["label"])
        correct_norm += int(max(range(len(norm)), key=norm.__getitem__) == it["label"])
    n = len(items)
    chance = sum(1 / len(it["choices"]) for it in items) / max(1, n)
    return {"items": n, "acc": correct / n, "acc_norm": correct_norm / n, "chance": chance,
            "acc_stderr": (correct / n * (1 - correct / n) / n) ** 0.5}
