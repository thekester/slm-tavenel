"""Summarize the paired EXP-0012 10M and EXP-0013 1M mixture pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("arc_easy", "arc_challenge", "hellaswag", "belebele_en", "belebele_fr", "winogrande")
EXP12 = ROOT / "runs"
EXP13 = ROOT / "runs"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exp12_arm(name: str) -> dict:
    folder = EXP12 / f"EXP-0012-10m-fsc-{name}-py311-torch210-2026-10-22"
    full = read(folder / "analysis/heldout-evaluation.json")
    missing = read(folder / "analysis/mc-missing-two.json")
    benchmarks = dict(full["benchmarks"])
    overlap = set(benchmarks) & set(missing["results"])
    if overlap:
        raise ValueError(f"EXP-0012 duplicate benchmark records: {sorted(overlap)}")
    metadata = {Path(item["path"]).stem: item for item in missing["benchmarks"]}
    for task, result in missing["results"].items():
        source = metadata[task]
        benchmarks[task] = {**result, "path": source["path"], "sha256": source["sha256"]}
    return {"evaluation": full, "benchmarks": benchmarks, "missing_eval": missing}


def exp13_arm(name: str) -> dict:
    full = read(EXP13 / f"EXP-0013-1m-fsc-{name}-2026-10-22/analysis/heldout-evaluation.json")
    return {"evaluation": full, "benchmarks": full["benchmarks"]}


def relative_to_repo(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path


def compare_pair(study: str, seed: int, control: dict, treatment: dict,
                 rng: np.random.Generator, draws: int) -> dict:
    base, mix = control["benchmarks"], treatment["benchmarks"]
    if set(base) != set(TASKS) or set(mix) != set(TASKS):
        raise ValueError(f"{study}: expected all six pinned tasks in both arms")
    benchmark_results = {}
    for task in TASKS:
        a, b = base[task], mix[task]
        if a.get("sha256") != b.get("sha256") or a.get("path") != b.get("path"):
            raise ValueError(f"{study}/{task}: evaluation inputs differ")
        a_items, b_items = a["per_item"], b["per_item"]
        if ([x["id"] for x in a_items] != [x["id"] for x in b_items]
                or [x["label"] for x in a_items] != [x["label"] for x in b_items]):
            raise ValueError(f"{study}/{task}: item IDs, order, or labels differ")
        changes = np.array([int(bool(y["correct"])) - int(bool(x["correct"]))
                            for x, y in zip(a_items, b_items)], dtype=np.int8)
        n = len(changes)
        win = int(np.count_nonzero(changes == 1))
        loss = int(np.count_nonzero(changes == -1))
        tie = int(n - win - loss)
        sampled = rng.multinomial(n, [win / n, loss / n, tie / n], size=draws)
        ci = np.quantile((sampled[:, 0] - sampled[:, 1]) / n, [0.025, 0.975])
        benchmark_results[task] = {
            "items": n,
            "control_accuracy": float(np.mean([bool(x["correct"]) for x in a_items])),
            "mixture_accuracy": float(np.mean([bool(x["correct"]) for x in b_items])),
            "delta_accuracy_mixture_minus_control": float(changes.mean()),
            "paired_item_bootstrap_95ci": [float(ci[0]), float(ci[1])],
            "mixture_only_correct": win,
            "control_only_correct": loss,
            "same_correctness": tie,
            "benchmark_sha256": a["sha256"],
        }

    base_text = control["evaluation"].get("text_sets", control["evaluation"].get("text_scores", {}))
    mix_text = treatment["evaluation"].get("text_sets", treatment["evaluation"].get("text_scores", {}))
    text_results = {}
    for name in sorted(set(base_text) & set(mix_text)):
        a = float(base_text[name]["bits_per_byte"])
        b = float(mix_text[name]["bits_per_byte"])
        text_results[name] = {
            "control_bpb": a,
            "mixture_bpb": b,
            "delta_bpb_mixture_minus_control": b - a,
            "input_path": base_text[name].get("path"),
            "input_sha256": base_text[name].get("sha256"),
            "interpretation": "negative favors the mixture on this text set",
        }

    return {
        "study": study,
        "status": "single_paired_seed_pilot",
        "paired_training_seed": seed,
        "control_checkpoint": relative_to_repo(control["evaluation"]["checkpoint"]),
        "control_checkpoint_sha256": control["evaluation"].get("checkpoint_sha256"),
        "mixture_checkpoint": relative_to_repo(treatment["evaluation"]["checkpoint"]),
        "mixture_checkpoint_sha256": treatment["evaluation"].get("checkpoint_sha256"),
        "parameters": control["evaluation"].get("params"),
        "text_outcomes": text_results,
        "benchmark_outcomes": benchmark_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/EXP-0012-0013-mixture-pilot-results.json")
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    studies = [
        compare_pair("EXP-0012 10M", 1234, exp12_arm("control"), exp12_arm("mot-mixture"), rng, args.draws),
        compare_pair("EXP-0013 1M", 421021, exp13_arm("control"), exp13_arm("mot-mixture"), rng, args.draws),
    ]
    external = read(ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22/benchmark-comparisons.json")
    external_scores = read(ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22/selected-checkpoint-mc.json")
    external_hashes = {Path(item["path"]).stem: item["sha256"]
                       for item in external_scores["benchmark_files"]}
    for study in studies:
        for task, outcome in study["benchmark_outcomes"].items():
            if external_hashes.get(task) != outcome["benchmark_sha256"]:
                raise ValueError(f"{study['study']}/{task}: benchmark differs from EXP-0010 reference file")
    payload = {
        "kind": "slm.exp0012-exp0013-mixture-pilot-comparison",
        "status": "exploratory_single_paired_seed_per_size",
        "bootstrap": {
            "method": "paired resampling of item correctness differences",
            "draws": args.draws,
            "seed": args.seed,
            "interval": "percentile 95%; item-sampling uncertainty only",
            "familywise_correction": "none",
        },
        "limitations": [
            "Each size has one matched seed pair; training-seed variability is not estimated.",
            "Benchmark intervals condition on these checkpoints and benchmark items; they do not measure run-to-run variation.",
            "Benchmark outcomes are secondary and exploratory; no checkpoint was selected from them.",
            "The MOT-containing checkpoints are not cleared for public weight release.",
            "The FSC test was already evaluated in EXP-0010 and is not a new untouched confirmatory test.",
        ],
        "studies": studies,
        "same_harness_external_reference": external,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
