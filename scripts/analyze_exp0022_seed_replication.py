"""Compare the two matched 10%-wiki/FSC seed pairs using paired document BPB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pair(control: dict[str, Any], wiki: dict[str, Any], rng: np.random.Generator,
         draws: int) -> dict[str, Any]:
    if control["tokenizer_sha256"] != wiki["tokenizer_sha256"]:
        raise ValueError("Tokenizer hashes differ")
    results = {}
    for name in sorted(set(control["text_sets"]) & set(wiki["text_sets"])):
        a, b = control["text_sets"][name]["per_document"], wiki["text_sets"][name]["per_document"]
        if [x["text_sha256"] for x in a] != [x["text_sha256"] for x in b]:
            raise ValueError(f"{name}: held-out documents differ")
        if [x["bytes"] for x in a] != [x["bytes"] for x in b]:
            raise ValueError(f"{name}: byte counts differ")
        sizes = np.array([x["bytes"] for x in a], dtype=np.float64)
        delta_bits = np.array([(y["bits_per_byte"] - x["bits_per_byte"]) * x["bytes"]
                               for x, y in zip(a, b)], dtype=np.float64)
        ix = rng.integers(0, len(a), size=(draws, len(a)))
        samples = delta_bits[ix].sum(axis=1) / sizes[ix].sum(axis=1)
        results[name] = {
            "documents": len(a),
            "control_bpb": control["text_sets"][name]["bits_per_byte"],
            "wiki10_bpb": wiki["text_sets"][name]["bits_per_byte"],
            "wiki10_minus_control_bpb": float(delta_bits.sum() / sizes.sum()),
            "paired_document_bootstrap_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1-control", type=Path, required=True)
    ap.add_argument("--seed1-wiki", type=Path, required=True)
    ap.add_argument("--seed2-control", type=Path, required=True)
    ap.add_argument("--seed2-wiki", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--draws", type=int, default=20_000)
    args = ap.parse_args()
    paths = [args.seed1_control, args.seed1_wiki, args.seed2_control, args.seed2_wiki]
    data = [read(p) for p in paths]
    rng = np.random.default_rng(args.seed)
    pairs = {
        "seed_20261023": pair(data[0], data[1], rng, args.draws),
        "seed_20261024": pair(data[2], data[3], rng, args.draws),
    }
    sets = sorted(set(pairs["seed_20261023"]) & set(pairs["seed_20261024"]))
    averaged = {}
    for name in sets:
        d1 = pairs["seed_20261023"][name]["wiki10_minus_control_bpb"]
        d2 = pairs["seed_20261024"][name]["wiki10_minus_control_bpb"]
        averaged[name] = {
            "mean_of_two_seed_effects_bpb": (d1 + d2) / 2,
            "seed_effects_bpb": [d1, d2],
            "direction_replicated": (d1 <= 0 and d2 <= 0) or (d1 >= 0 and d2 >= 0),
            "uncertainty": "no between-seed CI reported; only two matched seeds",
        }
    result = {
        "kind": "slm.exp0022_wiki10_seed_replication_analysis",
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.draws,
        "pairs": pairs,
        "mean_effects_across_two_seeds": averaged,
        "input_evaluation_sha256": {str(p): sha256(p) for p in paths},
        "limitations": [
            "only two training seeds; seed variation remains poorly estimated",
            "paired document intervals are conditional on each pair of checkpoints",
            "FSC and Wikipedia source rights/privacy review remains unresolved",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
