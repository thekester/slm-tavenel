#!/usr/bin/env python3
"""Check the KV-cache inference code (src/slm/hf_model.py) on the published checkpoints, without uploading.

For each repository: download the weights, tokenizer and configuration at the pinned revision into a
local staging folder, add the new ``hf_model.py`` / ``hf_config.py``, then compare with the published
code at the same revision:
  * logits on a fixed 128-token prompt (new code, cache off) against the published code;
  * greedy generation of 64 tokens with the cache against the same generation without the cache
    (token-identical required);
  * latency and tokens/s with and without the cache (benchmark protocol of benchmark_card_inference.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "slm-tavenel-1m": "33ddccd588a693efeaceae9ae239fa0fbd0c94b6",
    "slm-tavenel-10m": "9818b706a2a2467efb22848967a5842871f7422c",
    "slm-tavenel-25m": "a34c08d1bd4b91a244f170ae26096ff059afc37a",
    "slm-tavenel-25m-hplt-10pct": "d377cfb8cae4a6018ea44d9133da5f82dce6b4d3",
    "slm-tavenel-124m": "a4ccf8189c738eab28ab5d78990974e98930b9a7",
}
PROMPT = ("La recherche scientifique repose sur des observations reproductibles. "
          "Scientific research relies on reproducible observations and careful measurement. ") * 20
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json"]


def timed_generate(model, ids, use_cache: bool, device: str, repeats: int = 5) -> tuple[torch.Tensor, float]:
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    out = None
    times = []
    for i in range(repeats + 1):
        sync()
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=64,
                                 min_new_tokens=64, do_sample=False, use_cache=use_cache, pad_token_id=0)
        sync()
        if i:  # first run is warm-up
            times.append(time.perf_counter() - t)
    return out, statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=ROOT / "docs/model-cards/revised-2026-11-06/_kv-cache-release")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/model-cards/revised-2026-11-06/kv-cache-verification.json")
    args = parser.parse_args()
    results = []
    for repo, revision in REVISIONS.items():
        local = args.stage / repo
        snapshot_download(f"athenista/{repo}", revision=revision, allow_patterns=FILES, local_dir=local)
        for name in ("hf_model.py", "hf_config.py"):
            shutil.copyfile(ROOT / "src/slm" / name, local / name)
        tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][:, :128]
        old = AutoModelForCausalLM.from_pretrained(f"athenista/{repo}", revision=revision, trust_remote_code=True).eval()
        new = AutoModelForCausalLM.from_pretrained(local, trust_remote_code=True).eval()
        with torch.no_grad():
            max_logit_diff = (old(ids).logits - new(ids, use_cache=False).logits).abs().max().item()
        row = {"repo": f"athenista/{repo}", "revision": revision, "max_abs_logit_diff_vs_published_code": max_logit_diff}
        for device in ("cpu", "cuda"):
            model = new.to(device)
            x = ids.to(device)
            cached, t_cached = timed_generate(model, x, True, device)
            plain, t_plain = timed_generate(model, x, False, device)
            row[device] = {"identical_tokens_with_and_without_cache": bool(torch.equal(cached, plain)),
                           "no_cache_s": round(t_plain, 3), "cache_s": round(t_cached, 3),
                           "no_cache_tokens_per_s": round(64 / t_plain, 1), "cache_tokens_per_s": round(64 / t_cached, 1),
                           "speedup": round(t_plain / t_cached, 2)}
            new = model.to("cpu")
        results.append(row)
        print(json.dumps(row), flush=True)
    import transformers

    args.out.write_text(json.dumps({"kind": "slm.kv_cache_verification", "torch": torch.__version__,
                                    "transformers": transformers.__version__, "prompt_tokens": 128,
                                    "new_tokens": 64, "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
