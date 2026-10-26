"""Paired document/item bootstrap for the EXP-0023 grown-50M treatment vs 25M control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bpb(rows: list[dict]) -> float:
    nll = sum(float(r["nll_per_token"]) * int(r["tokens"]) for r in rows)
    nbytes = sum(int(r["bytes"]) for r in rows)
    return nll / (nbytes * math.log(2))


def bootstrap_document_delta(control: list[dict], treatment: list[dict], rng: np.random.Generator,
                             draws: int) -> dict:
    if len(control) != len(treatment):
        raise ValueError("Held-out document counts do not match")
    for a, b in zip(control, treatment):
        if a["text_sha256"] != b["text_sha256"] or int(a["bytes"]) != int(b["bytes"]):
            raise ValueError("Held-out documents are not paired identically")
    ctrl_nll = np.asarray([r["nll_per_token"] * r["tokens"] for r in control], dtype=np.float64)
    trt_nll = np.asarray([r["nll_per_token"] * r["tokens"] for r in treatment], dtype=np.float64)
    sizes = np.asarray([r["bytes"] for r in control], dtype=np.float64)
    delta = np.empty(draws, dtype=np.float64)
    n = len(control)
    chunk = max(1, min(256, 2_000_000 // n))
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        ix = rng.integers(0, n, size=(count, n), dtype=np.int32)
        delta[start:start + count] = ((ctrl_nll[ix].sum(1) - trt_nll[ix].sum(1)) /
                                      (sizes[ix].sum(1) * math.log(2)))
    return {
        "documents": n,
        "control_bpb": bpb(control),
        "treatment_bpb": bpb(treatment),
        "control_minus_50m_bpb": bpb(control) - bpb(treatment),
        "paired_document_bootstrap_95ci": np.quantile(delta, [0.025, 0.975]).tolist(),
    }


def bootstrap_item_delta(control: list[dict], treatment: list[dict], rng: np.random.Generator,
                         draws: int) -> dict:
    if len(control) != len(treatment):
        raise ValueError("Benchmark item counts do not match")
    for a, b in zip(control, treatment):
        if (a.get("id"), a.get("label")) != (b.get("id"), b.get("label")):
            raise ValueError("Benchmark items are not paired identically")
    result = {"items": len(control)}
    for field in ("correct", "correct_norm"):
        c = np.asarray([r[field] for r in control], dtype=np.float64)
        t = np.asarray([r[field] for r in treatment], dtype=np.float64)
        diffs = t - c
        n = len(c)
        boot = np.empty(draws, dtype=np.float64)
        chunk = max(1, min(256, 2_000_000 // n))
        for start in range(0, draws, chunk):
            count = min(chunk, draws - start)
            ix = rng.integers(0, n, size=(count, n), dtype=np.int32)
            boot[start:start + count] = diffs[ix].mean(1)
        result[field] = {
            "control_accuracy": float(c.mean()),
            "50m_accuracy": float(t.mean()),
            "50m_minus_control": float(diffs.mean()),
            "paired_item_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--control-mc", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    control_mc = json.loads(args.control_mc.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    control_benchmarks = control_mc.get("results", control_mc.get("benchmarks", {}))
    treatment_benchmarks = treatment.get("benchmarks", treatment.get("results", {}))

    text = {}
    for key in ("fsc_en_validation", "fsc_fr_validation"):
        text[key] = bootstrap_document_delta(control["text_sets"][key]["per_document"],
                                             treatment["text_sets"][key]["per_document"], rng, args.draws)
    benchmarks = {}
    for key, trt in treatment_benchmarks.items():
        if key not in control_benchmarks:
            raise ValueError(f"Control missing benchmark: {key}")
        benchmarks[key] = bootstrap_item_delta(control_benchmarks[key]["per_item"],
                                               trt["per_item"], rng, args.draws)
    result = {
        "kind": "slm.exp0023_50m_vs_25m_paired_analysis",
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.draws,
        "control_checkpoint": control["checkpoint"],
        "treatment_checkpoint": treatment["checkpoint"],
        "fsc_validation_bpb": text,
        "multiple_choice": benchmarks,
        "uncertainty_scope": "document/item resampling conditional on one matched seed and these checkpoints; no training-seed variance estimate",
        "input_sha256": {str(args.control): sha256(args.control), str(args.control_mc): sha256(args.control_mc),
                          str(args.treatment): sha256(args.treatment)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "fsc_validation_bpb": text,
                      "multiple_choice": {k: {"acc": v["correct"]["50m_minus_control"],
                                               "ci": v["correct"]["paired_item_bootstrap_95ci"]}
                                          for k, v in benchmarks.items()}}, indent=2))


if __name__ == "__main__":
    main()
