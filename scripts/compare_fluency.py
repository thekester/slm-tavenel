"""Compare frozen local HF snapshots; write every completion and a blind review file.

Requires the project's HF extra; SentencePiece is also needed for tiny1m.
No download, training, remote judge or model upload occurs in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from slm.fluency import generate_sample
from slm.runlog import code_revision, environment_info, sha256_file, write_json


def load_snapshot(spec, trust_custom_code):
    folder = Path(spec["path"])
    if not folder.is_dir() or not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("Each snapshot needs an existing local directory and full Hub revision")
    # Local hashes bind the actual inputs; the revision is declared by the downloader.
    files = {str(p.relative_to(folder)): sha256_file(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix in {".json", ".py", ".model", ".safetensors"}}
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Only safetensors model snapshots are supported")
    if "tokenizer.json" not in files and "tokenizer.model" in files:
        tokenizer = LlamaTokenizer(vocab_file=str(folder / "tokenizer.model"), legacy=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True,
                                                   trust_remote_code=trust_custom_code)
    model = AutoModelForCausalLM.from_pretrained(
        folder, local_files_only=True, trust_remote_code=trust_custom_code,
        use_safetensors=True, torch_dtype=torch.float32,
    ).eval()
    return model, tokenizer, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True,
                        help='JSON list of {label, repo_id, revision, path}')
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[421201, 421202])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.8])
    parser.add_argument("--max-new-bytes", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--trust-custom-code", action="store_true")
    args = parser.parse_args()
    specs = json.loads(args.snapshots.read_text(encoding="utf-8"))
    prompts = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(specs) < 2 or len({s["label"] for s in specs}) != len(specs):
        raise ValueError("Supply at least two uniquely labelled snapshots")
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("Prompts must be nonempty with unique ids")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.temperatures)) != len(args.temperatures):
        raise ValueError("Duplicate seeds or temperatures would duplicate diagnostic evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest = {
        "kind": "slm.fluency.diagnostic.v1", "status": "running",
        "scope": "Exploratory generation diagnostics; no automated coherence score or superiority claim",
        "code": code_revision(), "environment": {**environment_info(), "transformers": transformers.__version__},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "prompts_sha256": sha256_file(args.prompts), "snapshots_spec_sha256": sha256_file(args.snapshots),
        "diagnostic_code_sha256": {str(p): sha256_file(p) for p in
                                   [Path(__file__), Path(__file__).parents[1] / "src/slm/fluency.py"]},
        "models": [], "energy_j": None,
    }
    write_json(args.output / "manifest.json", manifest)
    records = []
    with (args.output / "generations.jsonl").open("w", encoding="utf-8") as stream:
        for spec in specs:
            model, tok, hashes = load_snapshot(spec, args.trust_custom_code)
            model.to(args.device)
            manifest["models"].append({**spec, "files_sha256": hashes,
                                       "parameters": sum(p.numel() for p in model.parameters())})
            write_json(args.output / "manifest.json", manifest)
            for prompt in prompts:
                for temp in args.temperatures:
                    # One deterministic sample is enough; duplicate greedy seeds add no evidence.
                    for seed in args.seeds[:1] if temp == 0 else args.seeds:
                        record = {"model": spec["label"], "id": prompt["id"],
                                  "language": prompt["language"], "domain": prompt["domain"],
                                  **generate_sample(model, tok, prompt["prompt"],
                                      max_new_bytes=args.max_new_bytes, max_new_tokens=args.max_new_tokens,
                                      temperature=temp, top_p=args.top_p, seed=seed)}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        records.append(record)
            del model
    # Keep the model key separate; give blind_review.csv alone to human reviewers.
    grouped = {}
    for rec in records:
        key = (rec["id"], rec["temperature"], rec["seed"])
        grouped.setdefault(key, []).append(rec)
    review_key = []
    with (args.output / "blind_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "variant", "language", "prompt", "completion",
                                                   "grammar_0_to_4", "coherence_0_to_4", "relevance_0_to_4", "notes"])
        writer.writeheader()
        for key, group in grouped.items():
            group_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            random.Random(group_id).shuffle(group)
            for i, rec in enumerate(group):
                variant = str(i + 1)
                writer.writerow({"group": group_id, "variant": variant, "language": rec["language"],
                                 "prompt": rec["prompt"], "completion": rec["completion"]})
                review_key.append({"group": group_id, "variant": variant, "model": rec["model"],
                                   "id": rec["id"], "temperature": rec["temperature"], "seed": rec["seed"]})
    write_json(args.output / "review_key.json", review_key)
    manifest.update(status="completed", generations=len(records),
                    generations_sha256=sha256_file(args.output / "generations.jsonl"))
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output), "generations": len(records)}, indent=2))


if __name__ == "__main__":
    main()
