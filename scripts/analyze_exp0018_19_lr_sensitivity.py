"""Analyze the 2x2 FSC/wiki-by-learning-rate BPB sensitivity grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    bpb = np.asarray([row["bits_per_byte"] for row in rows], dtype=np.float64)
    byte_counts = np.asarray([row["bytes"] for row in rows], dtype=np.float64)
    text_hashes = [row["text_sha256"] for row in rows]
    return bpb, byte_counts, text_hashes


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights) / weights.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-fsc", type=Path, required=True)
    parser.add_argument("--high-wiki", type=Path, required=True)
    parser.add_argument("--low-fsc", type=Path, required=True)
    parser.add_argument("--low-wiki", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--draws", type=int, default=20_000)
    args = parser.parse_args()

    paths = {"high_fsc": args.high_fsc, "high_wiki": args.high_wiki,
             "low_fsc": args.low_fsc, "low_wiki": args.low_wiki}
    runs = {name: read(path) for name, path in paths.items()}
    common_sets = sorted(set.intersection(*(set(run["text_sets"]) for run in runs.values())))
    if not common_sets:
        raise ValueError("The four runs have no held-out text sets in common")
    rng = np.random.default_rng(args.seed)
    results = {}
    for name in common_sets:
        vectors = {arm: arrays(run["text_sets"][name]["per_document"])
                   for arm, run in runs.items()}
        reference_hashes = vectors["high_fsc"][2]
        reference_bytes = vectors["high_fsc"][1]
        for arm, (values, byte_counts, hashes) in vectors.items():
            if hashes != reference_hashes or not np.array_equal(byte_counts, reference_bytes):
                raise ValueError(f"{name}: document identities, order or byte counts differ in {arm}")

        hf, hw, lf, lw = (vectors[k][0] for k in ("high_fsc", "high_wiki", "low_fsc", "low_wiki"))
        n = len(reference_hashes)
        draws = rng.integers(0, n, size=(args.draws, n))
        sampled_weights = reference_bytes[draws]
        denom = sampled_weights.sum(axis=1)

        def boot_mean(v: np.ndarray) -> np.ndarray:
            return (v[draws] * sampled_weights).sum(axis=1) / denom

        high_fsc_b = boot_mean(hf)
        high_wiki_b = boot_mean(hw)
        low_fsc_b = boot_mean(lf)
        low_wiki_b = boot_mean(lw)
        effect_high = high_wiki_b - high_fsc_b
        effect_low = low_wiki_b - low_fsc_b
        interaction = effect_low - effect_high

        point_high_effect = weighted_mean(hw, reference_bytes) - weighted_mean(hf, reference_bytes)
        point_low_effect = weighted_mean(lw, reference_bytes) - weighted_mean(lf, reference_bytes)
        point_interaction = point_low_effect - point_high_effect
        results[name] = {
            "documents": n,
            "fsc_bpb": {
                "high_lr_3e-4": weighted_mean(hf, reference_bytes),
                "low_lr_1_5e-4": weighted_mean(lf, reference_bytes),
                "low_minus_high": weighted_mean(lf, reference_bytes) - weighted_mean(hf, reference_bytes),
                "low_minus_high_paired_document_95ci": [float(v) for v in np.quantile(low_fsc_b - high_fsc_b, [0.025, 0.975])],
            },
            "wiki_bpb": {
                "high_lr_3e-4": weighted_mean(hw, reference_bytes),
                "low_lr_1_5e-4": weighted_mean(lw, reference_bytes),
                "low_minus_high": weighted_mean(lw, reference_bytes) - weighted_mean(hw, reference_bytes),
                "low_minus_high_paired_document_95ci": [float(v) for v in np.quantile(low_wiki_b - high_wiki_b, [0.025, 0.975])],
            },
            "wiki_minus_fsc_recipe_effect": {
                "high_lr_3e-4": point_high_effect,
                "high_lr_3e-4_paired_document_95ci": [float(v) for v in np.quantile(effect_high, [0.025, 0.975])],
                "low_lr_1_5e-4": point_low_effect,
                "low_lr_1_5e-4_paired_document_95ci": [float(v) for v in np.quantile(effect_low, [0.025, 0.975])],
            },
            "recipe_by_learning_rate_interaction_low_minus_high": point_interaction,
            "interaction_paired_document_95ci": [float(v) for v in np.quantile(interaction, [0.025, 0.975])],
            "uncertainty_scope": "paired held-out-document bootstrap conditional on these four checkpoints; excludes training-seed uncertainty",
        }

    output = {
        "kind": "slm.exp0018_0019_lr_sensitivity_analysis",
        "learning_rates": {"high": 3e-4, "low": 1.5e-4},
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.draws,
        "runs": {name: {"evaluation_sha256": sha256(path),
                        "checkpoint_sha256": run["checkpoint_sha256"],
                        "tokenizer_sha256": run["tokenizer_sha256"]}
                 for (name, path), run in zip(paths.items(), runs.values())},
        "text_sets": results,
        "limitations": [
            "one training seed per cell",
            "the high-LR cells were completed before the low-LR follow-up and the follow-up choice used observed results",
            "the 10% wiki sample is nested with the earlier 20% wiki sample",
            "BPB intervals quantify document sampling only, not training-seed variability",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
