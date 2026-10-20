"""Evaluate completed local runs and pinned SLM references on the added FSC tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slm.benchmarks import fetch, load_items, multiple_choice
from slm.hf_adapter import load as load_hf
from slm.runlog import sha256_file, write_json
from slm.train import load_model


TASKS = ("arc_challenge", "winogrande")
REFERENCES = {
    "tiny1m": {"model": "shibatch/tiny1m", "revision": "fdacad8ff9dc27bd72c77d3684075a494be1ba11",
               "subfolder": "hf", "max_seq_len": 256},
    "pythia-14m": {"model": "EleutherAI/pythia-14m", "revision": "d31f659f077f9ea500c4d7faa999893d7f28cbdf",
                   "max_seq_len": 1024},
    "pythia-31m": {"model": "EleutherAI/pythia-31m", "revision": "e556ace21b489575e94e9d50b6dad2fcc7419679",
                   "max_seq_len": 2048},
    "smollm2-135m": {"model": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                     "max_seq_len": 1024},
}
LOCAL = {
    "slm-tavenel-1m": Path("runs/fsc-scaling/1m-gated-s1234/checkpoints/step_00015000.pt"),
    "slm-tavenel-10m": Path("runs/fsc-scaling/10m-gated-s1234/checkpoints/step_00015000.pt"),
}
TOKENIZER = Path("data/tokenizer/fsc-pilot-v1-train-only")


def validate_audit(path: Path) -> dict:
    audit = json.loads(path.read_text(encoding="utf-8"))
    audited = {Path(entry["path"]).stem: entry["sha256"] for entry in audit["benchmarks"]}
    for task in TASKS:
        if audited.get(task) != sha256_file(f"data/benchmarks/{task}.jsonl"):
            raise ValueError(f"{task}: benchmark differs from the contamination audit; rerun the audit")
    return {"path": str(path), "sha256": sha256_file(path)}


def evaluate(name: str, device: torch.device, out_root: Path,
             audit_path: Path = Path("runs/fsc-additional/contamination-audit.json")) -> Path:
    audit_record = validate_audit(audit_path)
    if name in LOCAL:
        checkpoint = LOCAL[name]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = load_model(checkpoint, device)
        from slm.tokenizer import load as load_tokenizer

        tok = load_tokenizer(TOKENIZER)
        model_record = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                        "tokenizer_sha256": sha256_file(TOKENIZER / "tokenizer.json"),
                        "model_params": model.num_params(), "max_seq_len": model.cfg.max_seq_len}
    else:
        spec = REFERENCES[name]
        model, tok = load_hf(spec["model"], device, revision=spec["revision"],
                             max_seq_len=spec["max_seq_len"], subfolder=spec.get("subfolder"))
        model_record = {"hf_model": spec["model"], "hf_revision": spec["revision"],
                        "hf_subfolder": spec.get("subfolder"), "model_params": model.num_params(),
                        "max_seq_len": model.cfg.max_seq_len}

    results = {}
    for task in TASKS:
        path = Path("data/benchmarks") / f"{task}.jsonl"
        items = load_items(path)
        results[task] = multiple_choice(model, tok, items, device, batch_items=8)
    out = out_root / name / "fsc-pilot-v2-mc.json"
    write_json(out, {
        "kind": "slm.additional-benchmark-evaluation",
        "contamination_audit": audit_record,
        "evaluation_harness": "SLM-Tavenel local log-likelihood evaluator; batch size 8",
        "model": model_record,
        "benchmarks": [{"path": f"data/benchmarks/{task}.jsonl",
                        "sha256": sha256_file(f"data/benchmarks/{task}.jsonl"),
                        "split": "test" if task == "arc_challenge" else "validation"}
                       for task in TASKS],
        "results": results,
        "contamination_status": "post-hoc exact-overlap audit recorded separately; external pretraining exposure unknown",
    })
    del model, tok
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[*LOCAL, *REFERENCES],
                        choices=[*LOCAL, *REFERENCES])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", default="runs/fsc-additional")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-only", action="store_true", help="Prepare benchmark files before the overlap audit")
    mode.add_argument("--skip-fetch", action="store_true", help="Use the benchmark files already audited")
    parser.add_argument("--audit", default="runs/fsc-additional/contamination-audit.json")
    args = parser.parse_args()
    out_root = Path(args.out_root)
    if not args.skip_fetch:
        for task in TASKS:
            fetch(task, "data/benchmarks")
    if args.fetch_only:
        return 0
    for name in args.models:
        path = evaluate(name, torch.device(args.device), out_root, Path(args.audit))
        print(json.dumps({"model": name, "output": str(path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
