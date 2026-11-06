#!/usr/bin/env python3
"""Reduced-precision variants of the published SLM-Tavenel checkpoints, measured against float32.

Variants (all from the published weights, nothing retrained):
  * ``bf16``: weights cast to bfloat16 and saved as safetensors (half the file size), run on the GPU;
  * ``int8-dynamic``: PyTorch dynamic quantization of every ``nn.Linear`` (int8 weights, activations
    quantized on the fly; CPU only), applied at load time (no file: PyTorch stores it only as a pickle).
Quality is measured on one fixed text sample, the same for every model: the first 100 English and 100
French documents of the SLM-Tavenel corpus-v2 held-out test split (FineWeb-Edu / FineWeb2-HQ; truncated
to each model's context). Reported: bits per byte and its change against float32 on the same device
family, next-token top-1 agreement with float32, file size, and greedy generation speed (128-token
prompt, 64 new tokens, KV cache on, median of 3). The sample is in-domain for the 124M and out-of-domain
for the French-Science-Commons models; only within-model changes are meaningful.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REPOS = ["slm-tavenel-1m", "slm-tavenel-10m", "slm-tavenel-25m", "slm-tavenel-25m-hplt-10pct", "slm-tavenel-124m"]
PROMPT = ("La recherche scientifique repose sur des observations reproductibles. "
          "Scientific research relies on reproducible observations and careful measurement. ") * 20


def sample_texts(per_language: int = 100) -> list[str]:
    texts, counts = [], {"en": 0, "fr": 0}
    with gzip.open(ROOT / "data/processed/slm-tavenel-edu-corpus-v2/test.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            doc = json.loads(line)
            if counts.get(doc["language"], per_language) < per_language:
                counts[doc["language"]] += 1
                texts.append(doc["text"])
            if all(v >= per_language for v in counts.values()):
                break
    return texts


@torch.no_grad()
def score(model, tokenizer, texts: list[str], device: str, reference: list[torch.Tensor] | None):
    """Bits per byte over the sample and top-1 agreement with reference argmax predictions."""
    nll_bits, n_bytes, agree, total, argmaxes = 0.0, 0, 0, 0, []
    limit = model.config.max_seq_len
    for i, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt")["input_ids"][:, :limit].to(device)
        if ids.shape[1] < 2:
            argmaxes.append(torch.empty(0, dtype=torch.long))
            continue
        logits = model(ids, use_cache=False).logits[0, :-1].float()
        targets = ids[0, 1:]
        nll_bits += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item() / math.log(2)
        n_bytes += len(tokenizer.decode(ids[0, 1:]).encode("utf-8"))
        top = logits.argmax(-1).cpu()
        argmaxes.append(top)
        if reference is not None:
            agree += int((top == reference[i]).sum())
            total += top.numel()
    return nll_bits / n_bytes, (agree / total if total else None), argmaxes


def speed(model, tokenizer, device: str) -> float:
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][:, :128].to(device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    times = []
    for i in range(4):
        sync()
        t = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, min_new_tokens=64,
                           do_sample=False, use_cache=True, pad_token_id=0)
        sync()
        if i:
            times.append(time.perf_counter() - t)
    return round(64 / statistics.median(times), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, default=ROOT / "docs/model-cards/revised-2026-11-06/_kv-cache-release",
                        help="local snapshots with the KV-cache code (from verify_kv_cache_release.py)")
    parser.add_argument("--stage", type=Path, default=ROOT / "docs/model-cards/revised-2026-11-06")
    args = parser.parse_args()
    torch.manual_seed(0)
    texts = sample_texts()
    results = []
    for repo in REPOS:
        local = args.code / repo
        tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        fp32 = AutoModelForCausalLM.from_pretrained(local, trust_remote_code=True, dtype=torch.float32).eval()
        bpb32, _, ref = score(fp32, tokenizer, texts, "cpu", None)
        rows = [{"variant": "float32", "device": "cpu", "bpb": round(bpb32, 4), "delta_bpb": 0.0, "top1_agreement": 1.0,
                 "file_mb": round((local / "model.safetensors").stat().st_size / 1e6, 1), "tokens_per_s": speed(fp32, tokenizer, "cpu")}]
        # bfloat16 file, saved beside the card for a possible later upload.
        bf16_dir = args.stage / repo / "variants/bf16"
        bf16 = fp32.to(torch.bfloat16)
        bf16.save_pretrained(bf16_dir, safe_serialization=True)
        for name in ("hf_model.py", "hf_config.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "generation_config.json"):
            if (local / name).exists():
                shutil.copyfile(local / name, bf16_dir / name)
        bf16 = bf16.to("cuda")
        bpb16, agree16, _ = score(bf16, tokenizer, texts, "cuda", ref)
        rows.append({"variant": "bfloat16", "device": "cuda", "bpb": round(bpb16, 4), "delta_bpb": round(bpb16 - bpb32, 4),
                     "top1_agreement": round(agree16, 4),
                     "file_mb": round((bf16_dir / "model.safetensors").stat().st_size / 1e6, 1),
                     "tokens_per_s": speed(bf16, tokenizer, "cuda")})
        del bf16
        torch.cuda.empty_cache()
        int8 = torch.ao.quantization.quantize_dynamic(
            AutoModelForCausalLM.from_pretrained(local, trust_remote_code=True, dtype=torch.float32).eval(),
            {torch.nn.Linear}, dtype=torch.qint8)
        bpb8, agree8, _ = score(int8, tokenizer, texts, "cpu", ref)
        size = sum(p.numel() * p.element_size() for p in int8.parameters())  # unquantized params (embeddings, norms)
        for module in int8.modules():
            if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
                size += module.weight().numel()  # one byte per int8 weight (per-tensor scales negligible)
        rows.append({"variant": "int8-dynamic", "device": "cpu", "bpb": round(bpb8, 4), "delta_bpb": round(bpb8 - bpb32, 4),
                     "top1_agreement": round(agree8, 4), "file_mb": round(size / 1e6, 1),
                     "file_note": "in-memory size estimate; not saved as a file",
                     "tokens_per_s": speed(int8, tokenizer, "cpu")})
        result = {"repo": f"athenista/{repo}", "rows": rows}
        results.append(result)
        print(json.dumps(result), flush=True)
    import transformers

    out = {"kind": "slm.quantization_measurements", "torch": torch.__version__, "transformers": transformers.__version__,
           "sample": "first 100 EN + 100 FR documents of slm-tavenel-edu-corpus-v2 test split, truncated to context",
           "documents": len(texts), "results": results}
    (args.stage / "quantization-measurements.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
