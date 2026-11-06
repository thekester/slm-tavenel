#!/usr/bin/env python3
"""EXP-0042 (preregistered in experiments/EXP-0042-lfm3-compiled-computer-2026-11-06.md): LFM3-VL-2B alone
(N), with a host Python tool (P) and with the compiled machine (C) on 700 seeded arithmetic items.

Resumable: answers are appended per arm to runs/EXP-0042/<arm>.jsonl and finished items are skipped.
``--score`` only analyses existing answers. Run with PYTHONPATH=data/envs/tf5 (transformers>=5.21).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "runs/EXP-0042"
SEED = 20261106
CELLS = [("+", 6, 6), ("+", 10, 10), ("+", 15, 15), ("*", 3, 3), ("*", 5, 4), ("*", 6, 5), ("*", 8, 8)]
ARMS = {"N": "none", "P": "python", "C": "compiled"}


def make_items(per_cell: int = 100) -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for op, da, db in CELLS:
        for k in range(per_cell):
            a, b = rng.randrange(10 ** (da - 1), 10 ** da), rng.randrange(10 ** (db - 1), 10 ** db)
            items.append({"id": f"{'add' if op == '+' else 'mul'}{da}x{db}-{k:03d}", "cell": f"{op}{da}x{db}",
                          "a": a, "b": b, "op": op, "question": f"What is {a} {op} {b}?",
                          "answer": a + b if op == "+" else a * b})
    return items


def final_integer(text: str) -> int | None:
    """Return the last integer, accepting common thousands separators."""
    text = text.replace("{,}", "").replace(",", "").replace("\u202f", "").replace("\u2009", "").replace("\u00a0", "")
    found = re.findall(r"\d+", text)
    return int(found[-1]) if found else None

def score(items: list[dict], out: Path = OUT) -> dict:
    by_id = {it["id"]: it for it in items}
    rows = {}
    for arm in ARMS:
        path = out / f"{arm}.jsonl"
        rows[arm] = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else {}
    correct = {arm: {i: final_integer(r["final"]) == by_id[i]["answer"] for i, r in rs.items()} for arm, rs in rows.items()}
    cells = [f"{op}{da}x{db}" for op, da, db in CELLS]
    result = {"items": len(items), "answered": {a: len(r) for a, r in rows.items()}, "by_cell": {}, "arms": {}}
    for arm, rs in rows.items():
        calls = [c for r in rs.values() for c in r["calls"]]
        well_formed = [c for c in calls if "result" in c]
        result["arms"][arm] = {
            "accuracy": float(np.mean(list(correct[arm].values()))) if rs else None,
            "tool_call_rate": len([r for r in rs.values() if r["calls"]]) / len(rs) if rs else None,
            "argument_copy_accuracy": float(np.mean([
                int(c["arguments"].get("a", -1)) == by_id[i]["a"] and int(c["arguments"].get("b", -1)) == by_id[i]["b"]
                for i, r in rs.items() for c in r["calls"] if "result" in c])) if well_formed else None,
            "tool_result_exact": float(np.mean([
                c["result"] == (int(c["arguments"]["a"]) + int(c["arguments"]["b"]) if c["function"] == "add"
                                else int(c["arguments"]["a"]) * int(c["arguments"]["b"])) for c in well_formed])) if well_formed else None,
            "mean_trace_tokens_per_call": float(np.mean([c["trace_tokens"] for c in well_formed if "trace_tokens" in c]))
            if any("trace_tokens" in c for c in well_formed) else None,
            "mean_seconds_per_item": float(np.mean([r["seconds"] for r in rs.values()])) if rs else None,
        }
    for cell in cells:
        ids = [it["id"] for it in items if it["cell"] == cell]
        result["by_cell"][cell] = {arm: float(np.mean([correct[arm][i] for i in ids if i in correct[arm]]))
                                   for arm in rows if any(i in correct[arm] for i in ids)}
    mul = [it["id"] for it in items if it["op"] == "*"]
    if all(i in correct["C"] and i in correct["N"] for i in mul):
        diff = np.array([correct["C"][i] - correct["N"][i] for i in mul], dtype=float)
        draws = diff[np.random.default_rng(SEED).integers(0, len(diff), size=(10000, len(diff)))].mean(1)
        result["criterion_2_C_minus_N_multiplication"] = {
            "difference": float(diff.mean()), "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "met": bool(diff.mean() >= 0.10 and np.quantile(draws, 0.025) > 0)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="N,P,C")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="items per arm (0 = all), for smoke tests")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT, help="smoke tests must use another directory")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    items = make_items()
    (out / "items.json").write_text(json.dumps({"seed": SEED, "items": items}, indent=1) + "\n", encoding="utf-8")
    if not args.score:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from lfm3_compiled_computer import MODEL, answer
        from slm.computer import CompiledMachine

        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True,
                                                            dtype=torch.bfloat16).to(args.device).eval()
        machine = CompiledMachine()
        for arm in args.arms.split(","):
            path = out / f"{arm}.jsonl"
            done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
            todo = [it for it in items if it["id"] not in done][: args.limit or None]
            with path.open("a", encoding="utf-8") as stream:
                for it in todo:
                    start = time.time()
                    record = answer(model, processor, it["question"], ARMS[arm], machine)
                    record.update(id=it["id"], seconds=round(time.time() - start, 2))
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
            print(f"arm {arm}: {len(todo)} new answers", flush=True)
    summary = score(items, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
