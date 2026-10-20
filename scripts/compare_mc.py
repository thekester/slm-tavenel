"""Paired item-level comparison for two outputs from `slm eval mc`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(baseline_path: str, treatment_path: str, seed: int, draws: int) -> dict:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    baseline, treatment = read(baseline_path), read(treatment_path)
    rng = np.random.default_rng(seed)
    report = {
        "baseline": baseline_path,
        "treatment": treatment_path,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "benchmarks": {},
    }
    for result in (baseline, treatment):
        fingerprints = result.get("benchmarks")
        if not fingerprints or any(not entry.get("path") or not entry.get("sha256") for entry in fingerprints):
            raise ValueError("benchmark paths/hashes are missing; rerun evaluation")
    if baseline["benchmarks"] != treatment["benchmarks"]:
        raise ValueError("benchmark paths/hashes differ; paired comparison requires identical files")
    for name, base in baseline["results"].items():
        if name not in treatment["results"]:
            raise ValueError(f"treatment is missing benchmark {name}")
        a = base.get("per_item")
        b = treatment["results"][name].get("per_item")
        if a is None or b is None:
            raise ValueError(f"{name}: per-item predictions are missing; rerun evaluation with the item-level evaluator")
        if not a or not b:
            raise ValueError(f"{name}: no items to compare")
        if len(a) != len(b) or [x["id"] for x in a] != [x["id"] for x in b]:
            raise ValueError(f"{name}: item IDs or ordering differ")
        if [x["label"] for x in a] != [x["label"] for x in b]:
            raise ValueError(f"{name}: labels differ")
        wins = sum(bool(y["correct"]) and not bool(x["correct"]) for x, y in zip(a, b))
        losses = sum(bool(x["correct"]) and not bool(y["correct"]) for x, y in zip(a, b))
        ties = len(a) - wins - losses
        n = len(a)
        base_acc = sum(bool(x["correct"]) for x in a) / n
        treatment_acc = sum(bool(x["correct"]) for x in b) / n
        # The paired per-item difference is +1, -1, or 0. Resampling its three
        # observed categories gives a paired bootstrap CI without storing N×draws.
        sampled = rng.multinomial(n, [wins / n, losses / n, ties / n], size=draws)
        deltas = (sampled[:, 0] - sampled[:, 1]) / n
        report["benchmarks"][name] = {
            "items": n,
            "baseline_acc": base_acc,
            "treatment_acc": treatment_acc,
            "delta_treatment_minus_baseline": treatment_acc - base_acc,
            "paired_bootstrap_95ci": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            "treatment_only_correct": wins,
            "baseline_only_correct": losses,
            "both_same_correctness": ties,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("treatment")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.treatment, args.seed, args.draws)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
