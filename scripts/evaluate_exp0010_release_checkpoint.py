"""Run the fixed local benchmark harness on the preselected EXP-0010 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slm import tokenizer as slm_tokenizer
from slm.benchmarks import load_items, multiple_choice
from slm.runlog import sha256_file, write_json
from slm.train import load_model


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22"
CHECKPOINT = RUNS / "seed-421101/gated/checkpoints/step_00000488.pt"
TOKENIZER = ROOT / "data/tokenizers/fsc-cc0-open-weights-v4/tokenizer.json"
TASKS = (
    "arc_easy", "arc_challenge", "hellaswag", "belebele_en", "belebele_fr", "winogrande",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=RUNS / "selected-checkpoint-mc.json")
    args = parser.parse_args()

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    device = torch.device(args.device)
    model = load_model(CHECKPOINT, device)
    tok = slm_tokenizer.load(TOKENIZER)
    benchmark_records = []
    results = {}
    for task in TASKS:
        path = ROOT / "data/benchmarks" / f"{task}.jsonl"
        items = load_items(path)
        results[task] = multiple_choice(model, tok, items, device, batch_items=8)
        benchmark_records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "items": len(items),
            "split": "validation" if task in {"hellaswag", "winogrande"} else "test",
        })

    report = {
        "kind": "slm.exp0010-selected-checkpoint-multiple-choice",
        "status": "exploratory_posthoc",
        "model": "SLM-Tavenel 1M EXP-0010 gated seed 421101",
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "tokenizer": TOKENIZER.relative_to(ROOT).as_posix(),
        "tokenizer_sha256": sha256_file(TOKENIZER),
        "parameters": model.num_params(),
        "max_seq_len": model.cfg.max_seq_len,
        "evaluation_harness": "SLM-Tavenel local log-likelihood evaluator; batch size 8",
        "benchmark_files": benchmark_records,
        "results": results,
        "interpretation": (
            "Post-training descriptive scores. They are not the EXP-0010 primary outcome, "
            "do not estimate training-seed variability, and do not establish superiority. "
            "External models have different training corpora, budgets and possible benchmark exposure."
        ),
    }
    write_json(args.out, report)
    print(json.dumps({"status": "complete", "out": str(args.out),
                      "parameters": report["parameters"],
                      "scores": {k: v["acc"] for k, v in results.items()}}, indent=2))
    del model, tok
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
