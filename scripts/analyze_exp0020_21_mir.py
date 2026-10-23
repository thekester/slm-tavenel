"""Paired document-bootstrap analysis for the EXP-0020/0021 MIR pilot."""

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


def paired(control: dict[str, Any], mir: dict[str, Any], seed: int, draws: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result = {}
    if control["tokenizer_sha256"] != mir["tokenizer_sha256"]:
        raise ValueError("Tokenizer hashes differ")
    for name in sorted(set(control["text_sets"]) & set(mir["text_sets"])):
        a = control["text_sets"][name]["per_document"]
        b = mir["text_sets"][name]["per_document"]
        if [x["text_sha256"] for x in a] != [x["text_sha256"] for x in b]:
            raise ValueError(f"{name}: held-out document ids/order differ")
        if [x["bytes"] for x in a] != [x["bytes"] for x in b]:
            raise ValueError(f"{name}: held-out byte counts differ")
        sizes = np.array([x["bytes"] for x in a], dtype=np.float64)
        delta = np.array([(y["bits_per_byte"] - x["bits_per_byte"]) * x["bytes"]
                          for x, y in zip(a, b)], dtype=np.float64)
        ix = rng.integers(0, len(a), size=(draws, len(a)))
        draws_bpb = delta[ix].sum(axis=1) / sizes[ix].sum(axis=1)
        result[name] = {
            "documents": len(a),
            "control_bpb": control["text_sets"][name]["bits_per_byte"],
            "mir_bpb": mir["text_sets"][name]["bits_per_byte"],
            "mir_minus_control_bpb": float(delta.sum() / sizes.sum()),
            "paired_document_bootstrap_95ci": [float(x) for x in np.quantile(draws_bpb, [0.025, 0.975])],
            "uncertainty_scope": "paired held-out-document bootstrap conditional on these checkpoints; excludes training-seed uncertainty",
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--mir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--draws", type=int, default=20_000)
    args = ap.parse_args()
    control, mir = read(args.control), read(args.mir)
    result = {
        "kind": "slm.exp0020_0021_mir_analysis",
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.draws,
        "control_checkpoint_sha256": control["checkpoint_sha256"],
        "mir_checkpoint_sha256": mir["checkpoint_sha256"],
        "control_evaluation_sha256": sha256(args.control),
        "mir_evaluation_sha256": sha256(args.mir),
        "tokenizer_sha256": control["tokenizer_sha256"],
        "text_sets": paired(control, mir, args.seed, args.draws),
        "limitations": [
            "one matched training seed; bootstrap intervals do not estimate seed variability",
            "MIR consumes an additional forward pass, so equal token exposure is not equal compute",
            "the motivating paper starts at 72M parameters, above this 25M pilot",
            "FSC rights and privacy reviews remain unresolved; the study is local only",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
