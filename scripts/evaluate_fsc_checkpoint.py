"""Evaluate a local checkpoint on the frozen FSC validation set and MC benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slm import tokenizer as slm_tokenizer
from slm.benchmarks import load_items, multiple_choice
from slm.evaluation import bits_per_byte, read_texts
from slm.runlog import sha256_file, write_json
from slm.train import load_model
from fsc_evaluation_cache import complete, input_signature


BENCHMARKS = [
    "data/benchmarks/arc_easy.jsonl", "data/benchmarks/hellaswag.jsonl",
    "data/benchmarks/belebele_en.jsonl", "data/benchmarks/belebele_fr.jsonl",
]
VAL_FILES = {"en": "data/processed/fsc-pilot-v1-token-balanced/en-val.jsonl",
             "fr": "data/processed/fsc-pilot-v1-token-balanced/fr-val.jsonl"}


def evaluate(checkpoint: Path, tokenizer_path: Path, output_dir: Path, device: str) -> dict:
    mc_path = output_dir / "fsc-pilot-v1-mc.json"
    bpb_paths = {lang: output_dir / f"fsc-pilot-v1-bpb-{lang}.json" for lang in VAL_FILES}
    tokenizer_file = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    signature = input_signature({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                                 "tokenizer_sha256": sha256_file(tokenizer_file)}, BENCHMARKS, VAL_FILES)
    if complete([mc_path, *bpb_paths.values()], signature):
        return {"status": "already_complete", "mc": str(mc_path)}
    torch_device = torch.device(device)
    model = load_model(checkpoint, torch_device)
    tok = slm_tokenizer.load(tokenizer_path)
    results = {}
    for path in BENCHMARKS:
        results[Path(path).stem] = multiple_choice(model, tok, load_items(path), torch_device, batch_items=8)
    mc = {
        "evaluation_inputs": signature,
        "checkpoint": str(checkpoint), "hf_model": None, "hf_revision": None,
        "model_params": model.num_params(), "max_seq_len": model.cfg.max_seq_len,
        "evaluation_harness": "SLM-Tavenel local evaluator; batch size 8",
        "benchmarks": [{"path": path, "sha256": sha256_file(path)} for path in BENCHMARKS],
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(mc_path, mc)
    bpb = {}
    for language, path in VAL_FILES.items():
        result = bits_per_byte(model, tok, read_texts(path), torch_device)
        bpb[language] = {**result, "data": path, "data_sha256": sha256_file(path)}
        write_json(bpb_paths[language], {"evaluation_inputs": signature,
                                       "checkpoint": str(checkpoint), "model_params": model.num_params(), **bpb[language]})
    del model, tok
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "complete", "checkpoint": str(checkpoint), "mc": str(mc_path), "bpb": bpb}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer/fsc-pilot-v1-train-only")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(Path(args.checkpoint), Path(args.tokenizer), Path(args.out_dir), args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
