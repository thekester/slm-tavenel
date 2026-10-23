"""Evaluate a pre-registered EXP-0012 arm on shared held-out text and benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from slm import tokenizer as slm_tokenizer
from slm.benchmarks import load_items, multiple_choice
from slm.evaluation import bits_per_byte, read_texts
from slm.runlog import sha256_file, write_json
from slm.train import load_model


TEXT_SETS = {
    "fsc_en_validation": "data/processed/fsc-pilot-v1-token-balanced/en-val.jsonl",
    "fsc_fr_validation": "data/processed/fsc-pilot-v1-token-balanced/fr-val.jsonl",
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    model = load_model(checkpoint, device).eval()
    tok = slm_tokenizer.load(args.tokenizer)
    text_results = {}
    for name, path in TEXT_SETS.items():
        text_results[name] = {
            **bits_per_byte(model, tok, read_texts(path), device),
            "path": path,
            "sha256": sha256_file(path),
        }
    benchmark_results = {}
    for path in BENCHMARKS:
        benchmark_results[Path(path).stem] = {
            **multiple_choice(model, tok, load_items(path), device, batch_items=8),
            "path": path,
            "sha256": sha256_file(path),
        }
    result = {
        "kind": "slm.exp0012-heldout-evaluation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_sha256": sha256_file(args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer),
        "params": model.num_params(),
        "evaluator": "SLM-Tavenel bits-per-byte and local multiple-choice log-likelihood evaluator",
        "device": str(device),
        "text_sets": text_results,
        "benchmarks": benchmark_results,
        "interpretation": {
            "mot_2024_plus_test": "temporal held-out test; report once after both pre-registered arms finish",
            "single_seed": "paired pilot only; does not estimate training-seed variability",
            "multiple_choice": "exploratory, post-hoc secondary outcomes; do not claim broad capability from small differences",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
