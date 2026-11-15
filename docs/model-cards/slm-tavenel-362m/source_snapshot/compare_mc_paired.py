#!/usr/bin/env python3
"""Paired item-level comparison of two multiple-choice benchmark files (same evaluator, same items).

For each task: accuracy of both models, the difference (treatment - control) with a 95% paired
bootstrap interval over items, and the exact two-sided McNemar test on discordant items. The six-task
mean difference gets a bootstrap interval by resampling items independently within every task.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired(control: dict, treatment: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    # Items are paired by position: some datasets (HellaSwag) repeat ids, so ids only check alignment.
    a, b = control["per_item"], treatment["per_item"]
    if [item["id"] for item in a] != [item["id"] for item in b]:
        raise SystemExit("the two files do not score the same items in the same order")
    return np.array([item[key] for item in a], dtype=float), np.array([item[key] for item in b], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--labels", default="control,treatment")
    parser.add_argument("--key", default="correct", help="per-item field: correct or correct_norm")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    control_label, treatment_label = args.labels.split(",")
    control = json.loads(args.control.read_text(encoding="utf-8"))["results"]
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))["results"]
    rng = np.random.default_rng(args.seed)
    tasks, mean_draws = {}, []
    for task in sorted(control):
        a, b = paired(control[task], treatment[task], args.key)
        diff = b - a
        draws = diff[rng.integers(0, len(diff), size=(args.draws, len(diff)))].mean(1)
        mean_draws.append(draws)
        tasks[task] = {"items": len(diff), control_label: float(a.mean()), treatment_label: float(b.mean()),
                       "difference": float(diff.mean()),
                       "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                       "treatment_only_correct": int((diff > 0).sum()), "control_only_correct": int((diff < 0).sum()),
                       "mcnemar_p": mcnemar_p(int((diff > 0).sum()), int((diff < 0).sum()))}
        print(task, tasks[task], flush=True)
    mean = np.mean(mean_draws, axis=0)
    summary = {control_label: float(np.mean([t[control_label] for t in tasks.values()])),
               treatment_label: float(np.mean([t[treatment_label] for t in tasks.values()])),
               "difference": float(np.mean([t["difference"] for t in tasks.values()])),
               "ci95": [float(np.quantile(mean, 0.025)), float(np.quantile(mean, 0.975))]}
    print("mean", summary)
    result = {"kind": "slm.mc_paired_comparison", "control": str(args.control), "treatment": str(args.treatment),
              "labels": [control_label, treatment_label], "key": args.key, "draws": args.draws, "seed": args.seed,
              "tasks": tasks, "task_mean": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
