#!/usr/bin/env python3
"""Local inference measurements for the model cards (one common protocol for every published model).

Protocol: the published Hugging Face checkpoint at a pinned revision, loaded with Transformers and
``trust_remote_code=True`` exactly as the card shows; batch size 1; a fixed 128-token prompt (French and
English science text, repeated and truncated); greedy decoding of exactly 64 new tokens; one warm-up run,
then the median of 5 timed runs. Reported: load time, prefill latency (one forward pass over the
prompt, a proxy for time to first token), total generation latency, generated tokens per second,
peak process RAM (Windows peak working set, whole process including Python and PyTorch) and peak
allocated VRAM (``torch.cuda.max_memory_allocated``). Each configuration runs in a fresh subprocess so
memory peaks do not leak between models. The SLM-Tavenel implementations have no KV cache, so each new
token recomputes the whole prefix; latency therefore grows with prompt plus output length.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPOS = ["slm-tavenel-1m", "slm-tavenel-10m", "slm-tavenel-25m", "slm-tavenel-25m-hplt-10pct", "slm-tavenel-124m"]
CONFIGS = [("cpu", "float32"), ("cuda", "float32"), ("cuda", "bfloat16")]
PROMPT = ("La recherche scientifique repose sur des observations reproductibles. "
          "Scientific research relies on reproducible observations and careful measurement. ") * 20
PROMPT_TOKENS, NEW_TOKENS, REPEATS = 128, 64, 5


def measure(repo: str, revision: str, device: str, dtype: str) -> dict:
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(max(1, (psutil.cpu_count(logical=False) or 4)))
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(f"athenista/{repo}", revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(f"athenista/{repo}", revision=revision, trust_remote_code=True,
                                                 torch_dtype=getattr(torch, dtype)).to(device).eval()
    load_seconds = time.perf_counter() - start
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][:, :PROMPT_TOKENS].to(device)
    if ids.shape[1] < PROMPT_TOKENS:
        raise RuntimeError(f"prompt is only {ids.shape[1]} tokens for {repo}")
    mask = torch.ones_like(ids)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def prefill() -> float:
        sync()
        t = time.perf_counter()
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        sync()
        return time.perf_counter() - t

    def generate() -> float:
        sync()
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=NEW_TOKENS,
                                 min_new_tokens=NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        sync()
        assert out.shape[1] == PROMPT_TOKENS + NEW_TOKENS
        return time.perf_counter() - t

    prefill(), generate()  # warm-up
    prefills = [prefill() for _ in range(REPEATS)]
    generations = [generate() for _ in range(REPEATS)]
    total = statistics.median(generations)
    return {
        "repo": f"athenista/{repo}", "revision": revision, "device": device, "dtype": dtype,
        "parameters": sum(p.numel() for p in model.parameters()),
        "load_seconds": round(load_seconds, 2),
        "prefill_ms_median": round(1000 * statistics.median(prefills), 2),
        "generate_64_tokens_s_median": round(total, 3),
        "generated_tokens_per_s": round(NEW_TOKENS / total, 1),
        "peak_process_ram_gb": round(psutil.Process().memory_info().peak_wset / 1e9, 3),
        "peak_allocated_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--one", nargs=4, metavar=("REPO", "REVISION", "DEVICE", "DTYPE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print("RESULT " + json.dumps(measure(*args.one)), flush=True)
        return
    import torch
    from huggingface_hub import HfApi

    api = HfApi()
    results = []
    for repo in REPOS:
        revision = api.model_info(f"athenista/{repo}").sha
        for device, dtype in CONFIGS:
            proc = subprocess.run([sys.executable, __file__, "--one", repo, revision, device, dtype],
                                  capture_output=True, text=True)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            row = json.loads(line[7:]) if line else {"repo": f"athenista/{repo}", "revision": revision, "device": device,
                                                     "dtype": dtype, "error": proc.stderr.strip().splitlines()[-1:]}
            results.append(row)
            print(json.dumps(row), flush=True)
    import psutil

    env = {"cpu": platform.processor(), "physical_cores": psutil.cpu_count(logical=False),
           "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "os": platform.platform(), "python": platform.python_version()}
    protocol = {"prompt_tokens": PROMPT_TOKENS, "new_tokens": NEW_TOKENS, "batch_size": 1, "decoding": "greedy",
                "warmup_runs": 1, "timed_runs": REPEATS, "statistic": "median", "kv_cache": "not implemented"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"kind": "slm.card_inference_benchmark", "environment": env,
                                    "protocol": protocol, "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
