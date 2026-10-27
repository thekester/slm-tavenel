"""Paired item/document analysis for the equal-budget EXP-0029 corpus comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paired_bootstrap(control: np.ndarray, treatment: np.ndarray, rng: np.random.Generator,
                     draws: int) -> list[float]:
    if control.shape != treatment.shape:
        raise ValueError("Paired inputs must have the same shape")
    n = len(control)
    boot = np.empty(draws, dtype=np.float64)
    chunk = max(1, min(256, 2_000_000 // n))
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        ix = rng.integers(0, n, size=(count, n), dtype=np.int32)
        boot[start:start + count] = (treatment[ix] - control[ix]).mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def item_result(control: list[dict], treatment: list[dict], rng: np.random.Generator,
                draws: int) -> dict:
    if len(control) != len(treatment):
        raise ValueError("Benchmark item counts do not match")
    for c, t in zip(control, treatment):
        if (c.get("id"), c.get("label")) != (t.get("id"), t.get("label")):
            raise ValueError("Benchmark items are not identically ordered")
    result = {"items": len(control)}
    for field in ("correct", "correct_norm"):
        c = np.asarray([row[field] for row in control], dtype=np.float64)
        t = np.asarray([row[field] for row in treatment], dtype=np.float64)
        result[field] = {
            "old_corpus_accuracy": float(c.mean()),
            "hplt_rich_accuracy": float(t.mean()),
            "hplt_rich_minus_old_corpus": float((t - c).mean()),
            "paired_item_bootstrap_95ci": paired_bootstrap(c, t, rng, draws),
        }
    return result


def document_result(control: list[dict], treatment: list[dict], rng: np.random.Generator,
                    draws: int) -> dict:
    if len(control) != len(treatment):
        raise ValueError("Held-out document counts do not match")
    for c, t in zip(control, treatment):
        if (c.get("id"), c.get("text_sha256"), c.get("bytes")) != (
                t.get("id"), t.get("text_sha256"), t.get("bytes")):
            raise ValueError("Held-out documents are not paired identically")
    nll_c = np.asarray([float(row["nll"]) for row in control], dtype=np.float64)
    nll_t = np.asarray([float(row["nll"]) for row in treatment], dtype=np.float64)
    sizes = np.asarray([int(row["bytes"]) for row in control], dtype=np.float64)
    ix = np.arange(len(control))
    denom = sizes.sum() * math.log(2)
    c_bpb = float(nll_c.sum() / denom)
    t_bpb = float(nll_t.sum() / denom)
    boot = np.empty(draws, dtype=np.float64)
    chunk = max(1, min(256, 2_000_000 // len(ix)))
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        sample = rng.integers(0, len(ix), size=(count, len(ix)), dtype=np.int32)
        boot[start:start + count] = ((nll_c[sample] - nll_t[sample]).sum(axis=1) /
                                     (sizes[sample].sum(axis=1) * math.log(2)))
    return {
        "documents": len(control),
        "old_corpus_bpb": c_bpb,
        "hplt_rich_bpb": t_bpb,
        "old_corpus_minus_hplt_rich_bpb": c_bpb - t_bpb,
        "paired_document_bootstrap_95ci_for_old_minus_hplt": np.quantile(boot, [0.025, 0.975]).tolist(),
        "positive_delta_favors_hplt_rich": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-mc", type=Path, required=True)
    parser.add_argument("--hplt-mc", type=Path, required=True)
    parser.add_argument("--control-test", type=Path, required=True)
    parser.add_argument("--hplt-test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    ctrl_mc = json.loads(args.control_mc.read_text(encoding="utf-8"))
    hplt_mc = json.loads(args.hplt_mc.read_text(encoding="utf-8"))
    ctrl_test = json.loads(args.control_test.read_text(encoding="utf-8"))
    hplt_test = json.loads(args.hplt_test.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    if ctrl_mc.get("benchmarks") != hplt_mc.get("benchmarks"):
        raise ValueError("Benchmark file manifests differ")
    if ctrl_test.get("test_sha256") != hplt_test.get("test_sha256"):
        raise ValueError("Corpus test split hashes differ")

    control_results = ctrl_mc.get("results", ctrl_mc.get("benchmarks", {}))
    hplt_results = hplt_mc.get("results", hplt_mc.get("benchmarks", {}))
    benchmarks = {}
    for name, ctrl in control_results.items():
        if name not in hplt_results:
            raise ValueError(f"HPLT result missing benchmark {name}")
        benchmarks[name] = item_result(ctrl["per_item"], hplt_results[name]["per_item"], rng, args.draws)

    corpus = document_result(ctrl_test["per_document"], hplt_test["per_document"], rng, args.draws)
    result = {
        "kind": "slm.exp0029_equal_budget_corpus_comparison",
        "control_checkpoint": ctrl_mc.get("checkpoint"),
        "hplt_checkpoint": hplt_mc.get("checkpoint"),
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.draws,
        "hplt_test_sha256": ctrl_test.get("test_sha256"),
        "held_out_hplt_test": corpus,
        "multiple_choice": benchmarks,
        "uncertainty_scope": "paired item/document resampling conditional on one matched training seed and these checkpoints; no training-seed variance estimate",
        "input_sha256": {str(path): sha256(path) for path in
                         (args.control_mc, args.hplt_mc, args.control_test, args.hplt_test)},
        "limitations": [
            "single-seed exploratory comparison; old-corpus control uses the prior FSC pilot scaling corpus",
            "HPLT-rich treatment is a mixture containing prior candidate data, not HPLT-only",
            "HPLT source-page rights and human privacy review remain unresolved",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "held_out_hplt_test": corpus,
                      "accuracy_delta": {name: values["correct"]["hplt_rich_minus_old_corpus"]
                                         for name, values in benchmarks.items()}}, indent=2))


if __name__ == "__main__":
    main()
