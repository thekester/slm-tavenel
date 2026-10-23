"""Analyze paired EXP-0014 checkpoint evaluations with item-level bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ["arc_easy", "arc_challenge", "hellaswag", "belebele_en", "belebele_fr", "winogrande"]


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_parts(record: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    if "benchmarks" in record and isinstance(record["benchmarks"], dict):
        b = record["benchmarks"]
        return ({k: v["sha256"] for k, v in b.items()}, b)
    b = record["results"]
    entries = record.get("benchmarks", record.get("benchmark_files", []))
    hashes = {Path(x["path"]).stem: x["sha256"] for x in entries}
    return hashes, b


def compare(base: dict[str, Any], treatment: dict[str, Any], rng: np.random.Generator,
            draws: int) -> dict[str, Any]:
    base_hashes, a_tasks = benchmark_parts(base)
    treatment_hashes, b_tasks = benchmark_parts(treatment)
    if base_hashes != treatment_hashes:
        raise ValueError("Paired comparison requires identical benchmark files and hashes")
    rows = {}
    for task in TASKS:
        a = a_tasks[task]["per_item"]
        b = b_tasks[task]["per_item"]
        if [x["id"] for x in a] != [x["id"] for x in b]:
            raise ValueError(f"{task}: item IDs/order differ")
        if [x["label"] for x in a] != [x["label"] for x in b]:
            raise ValueError(f"{task}: labels differ")
        wins = sum(bool(y["correct"]) and not bool(x["correct"]) for x, y in zip(a, b))
        losses = sum(bool(x["correct"]) and not bool(y["correct"]) for x, y in zip(a, b))
        n = len(a)
        counts = rng.multinomial(n, [wins / n, losses / n, (n - wins - losses) / n], size=draws)
        deltas = (counts[:, 0] - counts[:, 1]) / n
        acc_a = sum(bool(x["correct"]) for x in a) / n
        acc_b = sum(bool(x["correct"]) for x in b) / n
        rows[task] = {
            "items": n,
            "baseline_acc": acc_a,
            "treatment_acc": acc_b,
            "delta_treatment_minus_baseline": acc_b - acc_a,
            "paired_item_bootstrap_95ci": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            "treatment_only_correct": wins,
            "baseline_only_correct": losses,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="original EXP-0009 grown-25M item-level benchmark record")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--draws", type=int, default=20_000)
    args = parser.parse_args()

    control, mixture, reference = read(args.control), read(args.mixture), read(args.reference)
    rng = np.random.default_rng(args.seed)
    out = {
        "kind": "slm.exp0014_25m_paired_analysis",
        "seed": args.seed,
        "draws": args.draws,
        "interval_scope": "paired item-sampling uncertainty conditional on checkpoints; no training-seed uncertainty",
        "comparisons": {
            "mixture_vs_fsc_control": compare(control, mixture, rng, args.draws),
            "fsc_control_vs_original_exp0009_25m": compare(reference, control, rng, args.draws),
            "mixture_vs_original_exp0009_25m": compare(reference, mixture, rng, args.draws),
        },
        "heldout_bpb": {
            "control": {k: v["bits_per_byte"] for k, v in control["text_sets"].items()},
            "mixture": {k: v["bits_per_byte"] for k, v in mixture["text_sets"].items()},
        },
        "source_evaluation_hashes": {
            "control_checkpoint": control["checkpoint_sha256"],
            "mixture_checkpoint": mixture["checkpoint_sha256"],
            "control_tokenizer": control["tokenizer_sha256"],
            "mixture_tokenizer": mixture["tokenizer_sha256"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
