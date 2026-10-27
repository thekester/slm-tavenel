"""Paired document bootstrap for the EXP-0030 FSC/HPLT development results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(rows_a: list[dict], rows_b: list[dict], draws: int, seed: int) -> dict:
    a = {r["text_sha256"]: r for r in rows_a}
    b = {r["text_sha256"]: r for r in rows_b}
    keys = sorted(a.keys() & b.keys())
    if len(keys) != len(a) or len(keys) != len(b):
        raise ValueError("Control and treatment must score the exact same development documents")
    aa, bb = [a[k] for k in keys], [b[k] for k in keys]
    def bpb(rows: list[dict]) -> float:
        return sum(r["nll"] for r in rows) / (sum(r["bytes"] for r in rows) * np.log(2))
    delta = bpb(bb) - bpb(aa)
    rng = np.random.default_rng(seed)
    values = []
    # Batch draws to bound memory even if the held-out sample grows later.
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        idx = rng.integers(0, len(keys), size=(count, len(keys)))
        nll_a = np.fromiter((r["nll"] for r in aa), dtype=np.float64)[idx].sum(axis=1)
        nll_b = np.fromiter((r["nll"] for r in bb), dtype=np.float64)[idx].sum(axis=1)
        bytes_a = np.fromiter((r["bytes"] for r in aa), dtype=np.float64)[idx].sum(axis=1)
        bytes_b = np.fromiter((r["bytes"] for r in bb), dtype=np.float64)[idx].sum(axis=1)
        values.extend((nll_b / (bytes_b * np.log(2)) - nll_a / (bytes_a * np.log(2))).tolist())
    return {
        "documents": len(keys),
        "control_bpb": bpb(aa),
        "treatment_bpb": bpb(bb),
        "treatment_minus_control_bpb": delta,
        "paired_document_bootstrap_95ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def paired_mc(control_path: Path, treatment_path: Path, seed: int) -> dict:
    control, treatment = load(control_path), load(treatment_path)
    hashes_a = {Path(x["path"]).name: x["sha256"] for x in control["benchmarks"]}
    hashes_b = {Path(x["path"]).name: x["sha256"] for x in treatment["benchmarks"]}
    if hashes_a != hashes_b:
        raise ValueError("Benchmark files differ between the two checkpoints")
    output = {}
    rng = np.random.default_rng(seed)
    for task in sorted(control["results"]):
        a = control["results"][task]["per_item"]
        b = treatment["results"][task]["per_item"]
        if [(r["id"], r["label"]) for r in a] != [(r["id"], r["label"]) for r in b]:
            raise ValueError(f"Benchmark item order/labels differ for {task}")
        wins = sum(bool(y["correct"]) and not bool(x["correct"]) for x, y in zip(a, b))
        losses = sum(bool(x["correct"]) and not bool(y["correct"]) for x, y in zip(a, b))
        ties = len(a) - wins - losses
        draws = rng.multinomial(len(a), [wins / len(a), losses / len(a), ties / len(a)], size=20000)
        distribution = (draws[:, 0] - draws[:, 1]) / len(a)
        output[task] = {
            "items": len(a),
            "control_accuracy": sum(bool(r["correct"]) for r in a) / len(a),
            "treatment_accuracy": sum(bool(r["correct"]) for r in b) / len(a),
            "treatment_minus_control": (wins - losses) / len(a),
            "paired_item_bootstrap_95ci": [float(x) for x in np.quantile(distribution, [0.025, 0.975])],
            "treatment_only_correct": wins,
            "control_only_correct": losses,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-mc", type=Path)
    parser.add_argument("--treatment-mc", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", default="slm.exp0030_25m_paired_data_diversity_analysis")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    control, treatment = load(args.control), load(args.treatment)
    if control["tokenizer_sha256"] != treatment["tokenizer_sha256"]:
        raise ValueError("Tokenizers differ; paired comparison is invalid")
    results = {}
    for dataset in control["results"]:
        results[dataset] = {}
        for lang in ("en", "fr"):
            rows_a = [r for r in control["results"][dataset]["per_document"] if r["language"] == lang]
            rows_b = [r for r in treatment["results"][dataset]["per_document"] if r["language"] == lang]
            results[dataset][lang] = paired(rows_a, rows_b, args.draws, args.seed + (0 if lang == "en" else 1))
    benchmark_results = None
    if args.control_mc and args.treatment_mc:
        benchmark_results = paired_mc(args.control_mc, args.treatment_mc, args.seed + 20)
    elif args.control_mc or args.treatment_mc:
        raise ValueError("Provide both --control-mc and --treatment-mc")
    output = {
        "kind": args.kind,
        "control_checkpoint_sha256": control["checkpoint_sha256"],
        "treatment_checkpoint_sha256": treatment["checkpoint_sha256"],
        "tokenizer_sha256": control["tokenizer_sha256"],
        "delta_convention": "treatment minus control BPB; negative favors treatment, positive favors control",
        "results": results,
        "secondary_benchmark_results": benchmark_results,
        "limitations": [
            "one seed per arm; document intervals do not measure seed variability",
            "development-set results are for recipe selection, not final confirmation",
            "source-rights and privacy reviews for the training corpora remain incomplete",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
