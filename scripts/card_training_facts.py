#!/usr/bin/env python3
"""Training-compute facts for the published model cards, measured from the published training logs.

For each Hugging Face repository (read-only download at its current revision), reads the training
metrics log and run manifest and reports: optimizer steps, token positions, GPU compute time estimated
as the sum over log intervals of (tokens processed / logged tokens per second; segments recomputed after
a resume count again, since they used the GPU again), peak allocated GPU
memory if logged, and the hardware/software recorded in the manifest. Energy is NOT measured: no power
log exists for these runs, so none is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPOS = {
    "slm-tavenel-1m": ("training_record/selected-metrics.jsonl", "training_record/selected-run-manifest.json"),
    "slm-tavenel-10m": ("training_record/metrics.jsonl", "training_record/run_manifest.json"),
    "slm-tavenel-25m": ("training_record/metrics.jsonl", "training_record/run_manifest.json"),
    "slm-tavenel-25m-hplt-10pct": ("training_record/treatment/metrics.jsonl", "training_record/treatment/run-manifest-summary.json"),
    "slm-tavenel-124m": ("training_record/metrics.jsonl", "training_record/run_manifest.json"),
}


def find(obj, keys: tuple[str, ...]):
    """First value of any of ``keys`` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and not isinstance(obj[key], (dict, list)):
                return obj[key]
        for value in obj.values():
            found = find(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find(value, keys)
            if found is not None:
                return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    api, facts = HfApi(), {}
    for repo, (metrics_name, manifest_name) in REPOS.items():
        sha = api.model_info(f"athenista/{repo}").sha
        metrics_path = hf_hub_download(f"athenista/{repo}", metrics_name, revision=sha, local_dir=args.cache / repo)
        manifest_path = hf_hub_download(f"athenista/{repo}", manifest_name, revision=sha, local_dir=args.cache / repo)
        rows = [json.loads(line) for line in Path(metrics_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [r for r in rows if "tokens" in r and r.get("tokens_per_s")]
        seconds, previous = 0.0, 0
        for r in rows:
            seconds += max(r["tokens"] - previous, 0) / r["tokens_per_s"]
            previous = r["tokens"]
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        facts[repo] = {
            "revision": sha,
            "steps": rows[-1].get("step") if rows else None,
            "token_positions": rows[-1]["tokens"] if rows else None,
            "gpu_compute_hours_from_logged_throughput": round(seconds / 3600, 2),
            "median_tokens_per_s": sorted(r["tokens_per_s"] for r in rows)[len(rows) // 2] if rows else None,
            "peak_allocated_gpu_gb": max((r["max_mem_gb"] for r in rows if "max_mem_gb" in r), default=None),
            "gpu": find(manifest, ("gpu", "gpu_name", "device_name", "cuda_device")),
            "torch": find(manifest, ("torch", "torch_version", "pytorch")),
            "python": find(manifest, ("python", "python_version")),
            "dtype": find(manifest, ("dtype",)),
            "metrics_file": metrics_name,
            "energy": "not measured (no power log recorded)",
        }
        print(repo, facts[repo], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
