"""Paired item/document analysis for a token-matched scale comparison (EXP-0025, EXP-0026).

The smaller arm is a same-parent continuation control; the larger arm is a function-preserving growth
of that parent trained on the same data order, seed and token budget. Positive deltas favour the
larger model. Bootstrap intervals resample benchmark items or held-out documents only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_exp0029_data_comparison import paired_bootstrap, sha256


def item_result(smaller: list[dict], larger: list[dict], rng: np.random.Generator, draws: int) -> dict:
    if len(smaller) != len(larger):
        raise ValueError("Benchmark item counts do not match")
    for s, l in zip(smaller, larger):
        if (s.get("id"), s.get("label")) != (l.get("id"), l.get("label")):
            raise ValueError("Benchmark items are not identically ordered")
    result = {"items": len(smaller)}
    for field in ("correct", "correct_norm"):
        s = np.asarray([row[field] for row in smaller], dtype=np.float64)
        l = np.asarray([row[field] for row in larger], dtype=np.float64)
        result[field] = {
            "smaller_accuracy": float(s.mean()),
            "larger_accuracy": float(l.mean()),
            "larger_minus_smaller": float((l - s).mean()),
            "paired_item_bootstrap_95ci": paired_bootstrap(s, l, rng, draws),
        }
    return result


def document_result(smaller: list[dict], larger: list[dict], rng: np.random.Generator, draws: int,
                    group: str | None = None) -> dict:
    if len(smaller) != len(larger):
        raise ValueError("Held-out document counts do not match")
    for s, l in zip(smaller, larger):
        if (s.get("id"), s.get("text_sha256"), s.get("bytes")) != (l.get("id"), l.get("text_sha256"), l.get("bytes")):
            raise ValueError("Held-out documents are not paired identically")
    keep = [i for i, row in enumerate(smaller) if group is None or row.get("language") == group]
    nll_s = np.asarray([float(smaller[i]["nll"]) for i in keep], dtype=np.float64)
    nll_l = np.asarray([float(larger[i]["nll"]) for i in keep], dtype=np.float64)
    sizes = np.asarray([int(smaller[i]["bytes"]) for i in keep], dtype=np.float64)
    denom = sizes.sum() * math.log(2)
    boot = np.empty(draws, dtype=np.float64)
    chunk = max(1, min(256, 2_000_000 // len(keep)))
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        sample = rng.integers(0, len(keep), size=(count, len(keep)), dtype=np.int32)
        boot[start:start + count] = ((nll_s[sample] - nll_l[sample]).sum(axis=1) /
                                     (sizes[sample].sum(axis=1) * math.log(2)))
    s_bpb, l_bpb = float(nll_s.sum() / denom), float(nll_l.sum() / denom)
    return {
        "documents": len(keep),
        "smaller_bpb": s_bpb,
        "larger_bpb": l_bpb,
        "smaller_minus_larger_bpb": s_bpb - l_bpb,
        "paired_document_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def relabel(obj, control: str, treatment: str):
    """Rename smaller/larger in keys, for comparisons other than scale (e.g. ce vs distill)."""
    if isinstance(obj, dict):
        return {k.replace("smaller", control).replace("larger", treatment): relabel(v, control, treatment)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [relabel(v, control, treatment) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--smaller-mc", type=Path, required=True)
    parser.add_argument("--larger-mc", type=Path, required=True)
    parser.add_argument("--smaller-test", type=Path, required=True)
    parser.add_argument("--larger-test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--labels", default="smaller,larger",
                        help="output names for the control and treatment arms (inputs keep --smaller/--larger)")
    parser.add_argument("--limitations", nargs="*", default=None)
    parser.add_argument("--test-name", default="hplt_test", help="name of the held-out corpus split in the output")
    args = parser.parse_args()
    control_label, treatment_label = args.labels.split(",")
    small_mc = json.loads(args.smaller_mc.read_text(encoding="utf-8"))
    large_mc = json.loads(args.larger_mc.read_text(encoding="utf-8"))
    small_test = json.loads(args.smaller_test.read_text(encoding="utf-8"))
    large_test = json.loads(args.larger_test.read_text(encoding="utf-8"))
    if small_mc.get("benchmarks") != large_mc.get("benchmarks"):
        raise ValueError("Benchmark file manifests differ")
    if small_test.get("test_sha256") != large_test.get("test_sha256"):
        raise ValueError("Corpus test split hashes differ")
    rng = np.random.default_rng(args.seed)

    benchmarks = {name: item_result(res["per_item"], large_mc["results"][name]["per_item"], rng, args.draws)
                  for name, res in small_mc["results"].items()}
    docs_s, docs_l = small_test["per_document"], large_test["per_document"]
    corpus = {"all": document_result(docs_s, docs_l, rng, args.draws)}
    for language in sorted({row.get("language") for row in docs_s if row.get("language")}):
        corpus[f"language:{language}"] = document_result(docs_s, docs_l, rng, args.draws, language)
    result = {
        "kind": "slm.scale_comparison",
        "experiment_id": args.experiment_id,
        "smaller_checkpoint": small_mc.get("checkpoint"),
        "larger_checkpoint": large_mc.get("checkpoint"),
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.draws,
        f"{args.test_name}_sha256": small_test.get("test_sha256"),
        f"held_out_{args.test_name}": corpus,
        "multiple_choice": benchmarks,
        "positive_delta_favors_larger": True,
        "uncertainty_scope": "paired item/document resampling conditional on one matched training seed and these checkpoints; no training-seed variance estimate",
        "input_sha256": {str(p): sha256(p) for p in (args.smaller_mc, args.larger_mc, args.smaller_test, args.larger_test)},
        "limitations": args.limitations if args.limitations is not None else [
            "single-seed, token-matched (not compute-matched) comparison",
            "both arms continue on the HPLT-rich candidate mixture; its test split is in-domain",
            "HPLT source-page rights and human privacy review remain unresolved",
        ],
    }
    result = relabel(result, control_label, treatment_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(relabel({"out": str(args.out), f"held_out_{args.test_name}": corpus,
                              "accuracy_delta": {n: v["correct"]["larger_minus_smaller"] for n, v in benchmarks.items()}},
                             control_label, treatment_label), indent=2))


if __name__ == "__main__":
    main()
