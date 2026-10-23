"""Analyze EXP-0016 against the matched EXP-0014 FSC-only continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_exp0014_25m_mixture import compare  # noqa: E402


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_document_bpb(control: dict[str, Any], treatment: dict[str, Any],
                        rng: np.random.Generator, draws: int) -> dict[str, Any]:
    results = {}
    shared_sets = sorted(set(control["text_sets"]) & set(treatment["text_sets"]))
    if not shared_sets:
        raise ValueError("Control and treatment have no text sets in common")
    for name in shared_sets:
        a = control["text_sets"][name]["per_document"]
        b = treatment["text_sets"][name]["per_document"]
        if [x["text_sha256"] for x in a] != [x["text_sha256"] for x in b]:
            raise ValueError(f"{name}: document hashes/order differ")
        if [x["bytes"] for x in a] != [x["bytes"] for x in b]:
            raise ValueError(f"{name}: byte counts differ")
        byte_counts = np.array([x["bytes"] for x in a], dtype=np.float64)
        delta_bits = np.array([(y["bits_per_byte"] - x["bits_per_byte"]) * x["bytes"]
                               for x, y in zip(a, b)], dtype=np.float64)
        n = len(a)
        indices = rng.integers(0, n, size=(draws, n))
        deltas = delta_bits[indices].sum(axis=1) / byte_counts[indices].sum(axis=1)
        point = float(delta_bits.sum() / byte_counts.sum())
        results[name] = {
            "documents": n,
            "control_bpb": control["text_sets"][name]["bits_per_byte"],
            "treatment_bpb": treatment["text_sets"][name]["bits_per_byte"],
            "delta_treatment_minus_control_bpb": point,
            "paired_document_bootstrap_95ci": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            "interval_scope": "paired document-sampling uncertainty conditional on the two checkpoints; excludes training-seed uncertainty",
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-text", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-benchmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-0016",
                        help="experiment identifier recorded in the output analysis")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--draws", type=int, default=20_000)
    args = parser.parse_args()

    control_text = read(args.control_text)
    treatment = read(args.treatment)
    control_benchmarks = read(args.control_benchmarks)
    rng = np.random.default_rng(args.seed)
    aggregate_text_deltas = {}
    for name, baseline in control_benchmarks["text_sets"].items():
        if name in treatment["text_sets"]:
            estimate = treatment["text_sets"][name]["bits_per_byte"]
            aggregate_text_deltas[name] = {
                "control_bpb": baseline["bits_per_byte"],
                "treatment_bpb": estimate,
                "delta_treatment_minus_control_bpb": estimate - baseline["bits_per_byte"],
                "uncertainty": "point estimates only; control per-document scores were not retained",
            }
    result = {
        "kind": f"slm.{args.experiment_id.lower().replace('-', '_')}_analysis",
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "draws": args.draws,
        "text_bpb_vs_fsc_control": paired_document_bpb(control_text, treatment, rng, args.draws),
        "aggregate_text_bpb_vs_exp0014_control": aggregate_text_deltas,
        "benchmarks_vs_fsc_control": compare(control_benchmarks, treatment, rng, args.draws),
        "source_hashes": {
            "control_checkpoint": control_text["checkpoint_sha256"],
            "treatment_checkpoint": treatment["checkpoint_sha256"],
            "control_tokenizer": control_text["tokenizer_sha256"],
            "treatment_tokenizer": treatment["tokenizer_sha256"],
            "control_text_evaluation": sha256(args.control_text),
            "treatment_evaluation": sha256(args.treatment),
            "control_benchmark_evaluation": sha256(args.control_benchmarks),
        },
        "limitations": [
            "single matched continuation pair; intervals do not estimate training-seed variability",
            "multiple-choice results are post-hoc and repeatedly inspected",
            "exact n-gram decontamination does not detect paraphrase or semantic overlap",
            "source-rights and privacy reviews remain unresolved",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
