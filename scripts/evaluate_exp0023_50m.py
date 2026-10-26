"""Focused EXP-0023 evaluation: FSC held-out BPB plus the six pinned MC tasks."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm import tokenizer as slm_tokenizer  # noqa: E402
from slm.benchmarks import load_items, multiple_choice  # noqa: E402
from slm.evaluation import bits_per_byte, read_texts  # noqa: E402
from slm.runlog import sha256_file, write_json  # noqa: E402
from slm.train import load_model  # noqa: E402
from slm.tokenizer import EOS  # noqa: E402


TEXT_SETS = {
    "fsc_en_validation": "data/processed/fsc-pilot-v1-token-balanced/en-val.jsonl",
    "fsc_fr_validation": "data/processed/fsc-pilot-v1-token-balanced/fr-val.jsonl",
}
BENCHMARKS = [
    "data/benchmarks/arc_easy.jsonl", "data/benchmarks/arc_challenge.jsonl",
    "data/benchmarks/hellaswag.jsonl", "data/benchmarks/belebele_en.jsonl",
    "data/benchmarks/belebele_fr.jsonl", "data/benchmarks/winogrande.jsonl",
]


@torch.no_grad()
def score_documents_batched(model, tok, texts: list[str], device, batch_size: int = 8) -> list[dict]:
    """Per-document NLL with padded batches; right padding cannot affect causal-prefix scores."""
    eos = tok.token_to_id(EOS)
    max_len = model.cfg.max_seq_len
    chunks: list[tuple[int, list[int]]] = []
    byte_counts = [len(text.encode("utf-8")) for text in texts]
    for doc_index, text in enumerate(texts):
        ids = [eos] + tok.encode(text).ids
        chunks.extend((doc_index, ids[start:start + max_len + 1])
                      for start in range(0, len(ids) - 1, max_len))
    nlls = [0.0] * len(texts)
    token_counts = [0] * len(texts)
    model.eval()
    for start in range(0, len(chunks), batch_size):
        rows = chunks[start:start + batch_size]
        width = max(len(ids) - 1 for _, ids in rows)
        x = torch.full((len(rows), width), eos, dtype=torch.long)
        y = torch.full((len(rows), width), eos, dtype=torch.long)
        lengths = []
        for i, (_, ids) in enumerate(rows):
            n = len(ids) - 1
            x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long)
            y[i, :n] = torch.tensor(ids[1:], dtype=torch.long)
            lengths.append(n)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        token_nll = -logits.float().log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)
        for i, ((doc_index, _), n) in enumerate(zip(rows, lengths)):
            nlls[doc_index] += float(token_nll[i, :n].sum().item())
            token_counts[doc_index] += n
    return [
        {"index": i, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
         "nll_per_token": nlls[i] / max(1, token_counts[i]),
         "bytes": byte_counts[i], "tokens": token_counts[i], "nll": nlls[i]}
        for i, text in enumerate(texts)
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/fsc-pilot-v1-train-only"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-progress", action="store_true",
                        help="reuse completed datasets/benchmarks from the adjacent progress JSON")
    parser.add_argument("--verify-batching", action="store_true",
                        help="compare batched and reference BPB on two FSC documents, then exit")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    model = load_model(checkpoint, device).eval()
    tok = slm_tokenizer.load(args.tokenizer)
    if args.verify_batching:
        probe = read_texts(ROOT / TEXT_SETS["fsc_en_validation"])[:2]
        batched = score_documents_batched(model, tok, probe, device)
        reference = [bits_per_byte(model, tok, [text], device) for text in probe]
        max_delta = max(abs(row["nll"] / (row["bytes"] * math.log(2)) - ref["bits_per_byte"])
                        for row, ref in zip(batched, reference))
        print(json.dumps({"batch_documents": len(probe), "max_abs_bpb_delta": max_delta,
                          "within_1e-5": max_delta <= 1e-5}, indent=2))
        if max_delta > 1e-5:
            raise SystemExit("Batched BPB differs from the reference scorer by more than 1e-5")
        return
    progress_path = args.out.with_name(args.out.stem + "-progress.json")
    progress = {"checkpoint_sha256": sha256_file(checkpoint), "text_sets": {}, "benchmarks": {}}
    if args.resume_progress and progress_path.is_file():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("checkpoint_sha256") != progress["checkpoint_sha256"]:
            raise ValueError("Progress file checkpoint hash does not match current checkpoint")
        progress = saved

    text_results = progress["text_sets"]
    for name, rel_path in TEXT_SETS.items():
        if name in text_results:
            print(f"Reusing saved text set {name}", flush=True)
            continue
        path = ROOT / rel_path
        texts = read_texts(path)
        rows = score_documents_batched(model, tok, texts, device)
        total_bytes = sum(r["bytes"] for r in rows)
        total_tokens = sum(r["tokens"] for r in rows)
        total_nll = sum(r["nll"] for r in rows)
        for row in rows:
            row["bits_per_byte"] = row.pop("nll") / (row["bytes"] * math.log(2))
        text_results[name] = {
            "bits_per_byte": total_nll / (total_bytes * math.log(2)),
            "nll_per_token": total_nll / total_tokens,
            "tokens": total_tokens,
            "bytes": total_bytes,
            "per_document": rows,
            "path": rel_path,
            "sha256": sha256_file(path),
        }
        write_json(progress_path, progress)
        print(f"Completed held-out set {name}: {text_results[name]['bits_per_byte']:.5f} BPB", flush=True)

    benchmark_results = progress["benchmarks"]
    for rel_path in BENCHMARKS:
        path = ROOT / rel_path
        if path.stem in benchmark_results:
            print(f"Reusing saved benchmark {path.stem}", flush=True)
            continue
        benchmark_results[path.stem] = {
            **multiple_choice(model, tok, load_items(path), device, batch_items=8),
            "path": rel_path,
            "sha256": sha256_file(path),
        }
        write_json(progress_path, progress)
        print(json.dumps({"benchmark": path.stem,
                          "acc": benchmark_results[path.stem]["acc"]}), flush=True)

    result = {
        "kind": "slm.exp0023_50m_fsc_and_mc_evaluation",
        "experiment_id": "EXP-0023",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_sha256": sha256_file(args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer),
        "params": model.num_params(),
        "device": str(device),
        "evaluator": "SLM-Tavenel held-out bits-per-byte and local multiple-choice log-likelihood evaluator",
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "text_sets": text_results,
        "benchmarks": benchmark_results,
        "evaluation_mode": "fsc_primary_text_and_six_mc",
        "limitations": [
            "single-seed exploratory continuation; no training-seed uncertainty estimate",
            "existing multiple-choice items are repeatedly inspected and are not confirmatory",
            "paired intervals are conditional on these checkpoints and sampled documents/items",
            "FSC source rights and privacy review remain unresolved; weights stay local",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    progress_path.unlink(missing_ok=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
