"""Score the sealed v4 text split and analyze the predeclared paired-seed effect."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from exp0010_integrity import (validate_code_root, validate_hash,
                              TOKENIZER_SHA256, SPLIT_MANIFEST_SHA256)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22"
DATA = ROOT / "data/processed/fsc-cc0-open-weights-v4/splits"
TOKENIZER_PATH = ROOT / "data/tokenizers/fsc-cc0-open-weights-v4/tokenizer.json"
SEEDS = list(range(421101, 421111))
BOOTSTRAP_SEED = 20261022
BOOTSTRAP_DRAWS = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.inference_mode()
def score_document(model, tokenizer, text: str, device: torch.device, max_len: int) -> tuple[float, int]:
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None or max_len <= 0 or not text:
        raise ValueError("Scoring requires EOS, nonempty text and a positive context length")
    ids = [eos] + tokenizer.encode(text).ids
    nll = 0.0
    for start in range(0, len(ids) - 1, max_len):
        chunk = torch.tensor(ids[start:start + max_len + 1], device=device)[None]
        logits, _ = model(chunk[:, :-1])
        log_probs = logits.float().log_softmax(-1)
        nll -= log_probs.gather(-1, chunk[:, 1:, None]).sum().item()
    return nll / np.log(2.0), len(text.encode("utf-8"))


def bpb_for(scores: dict, arm: str, seed: int, language: str, docs: list[str]) -> float:
    bits = sum(scores[(arm, seed, language, doc)][0] for doc in docs)
    nbytes = sum(scores[(arm, seed, language, doc)][1] for doc in docs)
    return bits / nbytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True,
                        help="same clean detached worktree used for training")
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    validate_code_root(code_root)
    validate_hash(TOKENIZER_PATH, TOKENIZER_SHA256)
    validate_hash(DATA / "split_manifest.json", SPLIT_MANIFEST_SHA256)
    split_manifest = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))
    for row in split_manifest["splits"]:
        if row["split"] == "test":
            validate_hash(DATA / f"{row['language']}-test.jsonl", row["sha256"])
    sys.path.insert(0, str(code_root / "src"))
    from slm.config import ModelConfig  # noqa: E402
    from slm.model import SLM  # noqa: E402

    if not torch.cuda.is_available():
        raise SystemExit("The locked evaluation expects the recorded CUDA environment")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    docs = {
        "en": read_records(DATA / "en-test.jsonl"),
        "fr": read_records(DATA / "fr-test.jsonl"),
    }
    doc_hashes = {
        lang: [hashlib.sha256(row["id"].encode("utf-8")).hexdigest() for row in rows]
        for lang, rows in docs.items()
    }
    if len(set(doc_hashes["en"])) != len(doc_hashes["en"]) or len(set(doc_hashes["fr"])) != len(doc_hashes["fr"]):
        raise ValueError("Duplicate document IDs in a sealed language split")

    scores: dict[tuple, tuple[float, int]] = {}
    per_doc = []
    for seed in SEEDS:
        for arm in ("gated", "ungated"):
            ckpt = RUNS / f"seed-{seed}/{arm}/checkpoints/step_00000488.pt"
            pair = json.loads((RUNS / f"seed-{seed}/pair-record.json").read_text(encoding="utf-8"))
            validate_hash(ckpt, pair["arms"][arm]["checkpoint_sha256"])
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if state["step"] != 488 or state["train_config"]["seed"] != seed:
                raise ValueError(f"Checkpoint does not match the predeclared step and seed: {ckpt}")
            model = SLM(ModelConfig(**state["model_config"]))
            model.load_state_dict(state["model"])
            model = model.to(device).eval()
            for lang, rows in docs.items():
                for row, doc_hash in zip(rows, doc_hashes[lang]):
                    bits, nbytes = score_document(model, tokenizer, row["text"], device, model.cfg.max_seq_len)
                    scores[(arm, seed, lang, doc_hash)] = (bits, nbytes)
                    per_doc.append({"arm": arm, "seed": seed, "language": lang,
                                    "document_id_sha256": doc_hash, "bits": bits, "bytes": nbytes,
                                    "bits_per_byte": bits / nbytes})
            del model, state
            torch.cuda.empty_cache()

    seed_rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for arm in ("gated", "ungated"):
            lang_bpbs = {lang: bpb_for(scores, arm, seed, lang, doc_hashes[lang]) for lang in ("en", "fr")}
            row[arm] = {**lang_bpbs, "equal_language_macro": float(np.mean(list(lang_bpbs.values())))}
        row["effect_ungated_minus_gated"] = {
            lang: row["ungated"][lang] - row["gated"][lang] for lang in ("en", "fr")
        }
        row["effect_ungated_minus_gated"]["equal_language_macro"] = (
            row["ungated"]["equal_language_macro"] - row["gated"]["equal_language_macro"]
        )
        seed_rows.append(row)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    seed_array = np.array(SEEDS)
    for b in range(BOOTSTRAP_DRAWS):
        sampled_seeds = rng.choice(seed_array, size=len(seed_array), replace=True)
        sampled_docs = {
            lang: rng.choice(np.array(doc_hashes[lang]), size=len(doc_hashes[lang]), replace=True).tolist()
            for lang in ("en", "fr")
        }
        effects = []
        for seed in sampled_seeds:
            lang_effects = []
            for lang in ("en", "fr"):
                gated = bpb_for(scores, "gated", int(seed), lang, sampled_docs[lang])
                ungated = bpb_for(scores, "ungated", int(seed), lang, sampled_docs[lang])
                lang_effects.append(ungated - gated)
            effects.append(float(np.mean(lang_effects)))
        bootstrap[b] = float(np.mean(effects))

    point = float(np.mean([r["effect_ungated_minus_gated"]["equal_language_macro"] for r in seed_rows]))
    interval = [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]
    threshold = 0.01
    if point >= threshold and interval[0] > 0:
        decision = "supports_predeclared_practical_gate_benefit_on_this_corpus_and_budget"
    elif interval[1] < threshold:
        decision = "does_not_support_predeclared_practical_gate_benefit"
    else:
        decision = "inconclusive_for_predeclared_practical_gate_benefit"

    report = {
        "schema": "slm.exp0010_paired_seed_bpb.v1",
        "status": "complete",
        "protocol": "experiments/EXP-0010-1m-paired-replication-2026-10-22.md",
        "protocol_sha256": sha256(ROOT / "experiments/EXP-0010-1m-paired-replication-2026-10-22.md"),
        "protocol_amendment": "experiments/EXP-0010-amendment-2026-10-22.md",
        "protocol_amendment_sha256": sha256(ROOT / "experiments/EXP-0010-amendment-2026-10-22.md"),
        "code_commit": "02396acd15ef9bcdc453bcd53aade40819e2f69f",
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "split_manifest_sha256": sha256(ROOT / "data/processed/fsc-cc0-open-weights-v4/splits/split_manifest.json"),
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "test_scored_for_first_time_in_this_protocol": not (RUNS / "sealed-test-bpb-results.json").exists(),
        "test_documents": {lang: len(rows) for lang, rows in docs.items()},
        "seeds": SEEDS,
        "primary_metric": "equal-language macro-average document-disjoint test BPB",
        "effect_definition": "ungated BPB minus gated BPB; positive favors the gate",
        "practical_effect_threshold_bpb": threshold,
        "estimate_bpb": point,
        "hierarchical_bootstrap_95_percent_interval_bpb": interval,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "resampling_units": ["paired training seeds", "test documents within language"]},
        "decision": decision,
        "seed_results": seed_rows,
        "per_document_scores": per_doc,
        "limitations": [
            "one small science-text corpus and one-pass token budget",
            "ten paired training seeds do not establish general model quality",
            "source-rights and manual privacy review remain unsigned",
            "no causal explanation of model behavior is tested",
        ],
    }
    out = RUNS / "sealed-test-bpb-results.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("status", "estimate_bpb", "hierarchical_bootstrap_95_percent_interval_bpb", "decision", "test_documents")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
