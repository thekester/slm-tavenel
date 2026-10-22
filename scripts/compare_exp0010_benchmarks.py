"""Build descriptive same-harness comparisons for the selected EXP-0010 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22"
REF_ROOT = ROOT / "runs/reference"
ADDITIONAL_ROOT = ROOT / "runs/fsc-additional"
REFERENCE_NAMES = ("tiny1m", "pythia-14m", "pythia-31m", "smollm2-135m")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def combine_reference(name: str) -> dict:
    first_path = REF_ROOT / name / "fsc-pilot-v1-mc.json"
    second_path = ADDITIONAL_ROOT / name / "fsc-pilot-v2-mc.json"
    first = read(first_path) if first_path.is_file() else {}
    second = read(second_path) if second_path.is_file() else {}
    records = {Path(item["path"]).stem: item
               for item in first.get("benchmarks", []) + second.get("benchmarks", [])}
    return {
        "model": name,
        "parameters": first.get("model_params", second.get("model", {}).get("model_params")),
        "benchmarks": records,
        "results": {**first.get("results", {}), **second.get("results", {})},
    }


def combine_internal(name: str, display_name: str) -> dict:
    first = read(ROOT / f"runs/fsc-scaling/{name}-gated-s1234/analysis/fsc-pilot-v1-mc.json")
    second = read(ADDITIONAL_ROOT / f"slm-tavenel-{name}/fsc-pilot-v2-mc.json")
    records = {Path(item["path"]).stem: item for item in first["benchmarks"] + second["benchmarks"]}
    return {
        "model": display_name,
        "parameters": first["model_params"],
        "benchmarks": records,
        "results": {**first["results"], **second["results"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUNS / "benchmark-comparisons.json")
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    candidate = read(RUNS / "selected-checkpoint-mc.json")
    references = [combine_internal("10m", "SLM-Tavenel 10M FSC pilot")]
    references.extend(combine_reference(name) for name in REFERENCE_NAMES)
    tasks = list(candidate["results"])
    rng = np.random.default_rng(args.seed)
    reference_table = []
    comparisons = []

    candidate_files = {Path(item["path"]).stem: item for item in candidate["benchmark_files"]}
    for reference in references:
        row = {"model": reference["model"], "parameters": reference["parameters"], "scores": {}}
        for task in tasks:
            record = reference["benchmarks"].get(task)
            if record is None:
                continue
            if record["sha256"] != candidate_files[task]["sha256"]:
                raise ValueError(f"{reference['model']} has no identical pinned benchmark input for {task}")
            base = reference["results"][task]["per_item"]
            treatment = candidate["results"][task]["per_item"]
            if ([x["id"] for x in base] != [x["id"] for x in treatment]
                    or [x["label"] for x in base] != [x["label"] for x in treatment]):
                raise ValueError(f"{reference['model']} item order/labels differ for {task}")
            base_acc = float(np.mean([bool(x["correct"]) for x in base]))
            cand_acc = float(np.mean([bool(x["correct"]) for x in treatment]))
            delta_per_item = np.array([int(bool(y["correct"])) - int(bool(x["correct"]))
                                       for x, y in zip(base, treatment)], dtype=np.int8)
            n = len(delta_per_item)
            probs = np.bincount(delta_per_item + 1, minlength=3) / n
            counts = rng.multinomial(n, probs, size=args.draws)
            deltas = (counts[:, 2] - counts[:, 0]) / n
            interval = [float(x) for x in np.quantile(deltas, [0.025, 0.975])]
            row["scores"][task] = {"items": n, "acc": base_acc}
            comparisons.append({
                "reference": reference["model"], "task": task, "items": n,
                "reference_accuracy": base_acc, "slm_tavenel_accuracy": cand_acc,
                "paired_delta_tavenel_minus_reference": cand_acc - base_acc,
                "paired_item_bootstrap_95ci": interval,
                "tavenel_only_correct": int((delta_per_item == 1).sum()),
                "reference_only_correct": int((delta_per_item == -1).sum()),
                "both_same_correctness": int((delta_per_item == 0).sum()),
                "multiple_comparison_adjustment": "none",
            })
        reference_table.append(row)

    payload = {
        "kind": "slm.exp0010-descriptive-benchmark-comparisons",
        "status": "exploratory_posthoc",
        "candidate": {
            "model": candidate["model"], "parameters": candidate["parameters"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "scores": {task: {"items": values["items"], "acc": values["acc"]}
                       for task, values in candidate["results"].items()},
        },
        "references": reference_table,
        "paired_item_comparisons": comparisons,
        "bootstrap": {"method": "paired resampling of item correctness differences",
                      "draws": args.draws, "seed": args.seed,
                      "interval": "percentile 95%", "familywise_correction": "none"},
        "interpretation": (
            "Scores use identical pinned evaluation files and local scoring code. They are conditional "
            "on these checkpoints and do not estimate training-seed variability. External models have "
            "different corpora, budgets, tokenizers and possible benchmark exposure; differences are descriptive."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "references": len(references),
                      "tasks": tasks, "comparisons": len(comparisons)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
