"""Evaluate an EXP-0016 25M checkpoint on matched text sets and six existing tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
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

TEXT_SETS = {
    "fsc_en_validation": "data/processed/fsc-pilot-v1-token-balanced/en-val.jsonl",
    "fsc_fr_validation": "data/processed/fsc-pilot-v1-token-balanced/fr-val.jsonl",
    "wiki_en_bpb_holdout": "data/processed/exp0016-wiki/wiki_bpb_en.jsonl",
    "wiki_fr_bpb_holdout": "data/processed/exp0016-wiki/wiki_bpb_fr.jsonl",
    "mot_2022_2023_dev": "data/processed/exp0012-mot-v1.11/temporal/dev.jsonl",
    "mot_2024_plus_test": "data/processed/exp0012-mot-v1.11/temporal/test.jsonl",
}
BENCHMARKS = [
    "data/benchmarks/arc_easy.jsonl", "data/benchmarks/arc_challenge.jsonl",
    "data/benchmarks/hellaswag.jsonl", "data/benchmarks/belebele_en.jsonl",
    "data/benchmarks/belebele_fr.jsonl", "data/benchmarks/winogrande.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/fsc-pilot-v1-train-only"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true",
                        help="score held-out text only; use for the matched EXP-0014 control checkpoint")
    parser.add_argument("--primary-text-only", action="store_true",
                        help="score FSC EN/FR and wiki EN/FR only, for the paired primary BPB analysis")
    parser.add_argument("--experiment-id", default="EXP-0016",
                        help="experiment identifier recorded in the output manifest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    model = load_model(checkpoint, device).eval()
    tok = slm_tokenizer.load(args.tokenizer)

    text_results = {}
    text_sets = list(TEXT_SETS.items())
    if args.primary_text_only:
        primary_sets = {"fsc_en_validation", "fsc_fr_validation",
                        "wiki_en_bpb_holdout", "wiki_fr_bpb_holdout"}
        text_sets = [(name, path) for name, path in text_sets if name in primary_sets]
    for name, rel_path in text_sets:
        path = ROOT / rel_path
        texts = read_texts(path)
        per_document = []
        for index, text in enumerate(texts):
            row = bits_per_byte(model, tok, [text], device)
            per_document.append({
                "index": index,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "bits_per_byte": row["bits_per_byte"],
                "nll_per_token": row["nll_per_token"],
                "bytes": row["bytes"],
                "tokens": row["tokens"],
            })
        total_bytes = sum(row["bytes"] for row in per_document)
        total_tokens = sum(row["tokens"] for row in per_document)
        total_nll = sum(row["nll_per_token"] * row["tokens"] for row in per_document)
        text_results[name] = {"bits_per_byte": sum(row["bits_per_byte"] * row["bytes"]
                                                    for row in per_document) / max(1, total_bytes),
                              "nll_per_token": total_nll / max(1, total_tokens),
                              "tokens": total_tokens, "bytes": total_bytes,
                              "per_document": per_document,
                              "path": rel_path, "sha256": sha256_file(path)}
    benchmark_results = {}
    if not (args.text_only or args.primary_text_only):
        for rel_path in BENCHMARKS:
            path = ROOT / rel_path
            benchmark_results[path.stem] = {**multiple_choice(model, tok, load_items(path), device, batch_items=8),
                                            "path": rel_path, "sha256": sha256_file(path)}
    result = {
        "kind": f"slm.{args.experiment_id.lower().replace('-', '_')}_evaluation",
        "experiment_id": args.experiment_id,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_sha256": sha256_file(args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer),
        "params": model.num_params(), "device": str(device),
        "evaluator": "SLM-Tavenel bits-per-byte and local multiple-choice log-likelihood evaluator",
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "text_sets": text_results, "benchmarks": benchmark_results,
        "evaluation_mode": "text_only_primary" if args.primary_text_only else
                           "text_only" if args.text_only else "full",
        "limitations": [
            "single-seed exploratory continuation; no training-seed uncertainty estimate",
            "existing multiple-choice items are repeatedly inspected and are not confirmatory",
            "exact 13-gram decontamination does not detect paraphrase or semantic overlap",
            "Wikipedia source-rights and privacy review remain incomplete; weights are local only",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
