#!/usr/bin/env python3
"""Fill the generated sections of the revised model cards (docs/model-cards/revised-2026-11-06/<repo>/)
and stage the files a publication would add. Nothing is uploaded: the folders are a local staging area.

Generated content:
  * ``FAMILY_BENCHMARKS_PLACEHOLDER`` -> ``_shared-family-benchmarks.md``;
  * the block between ``<!-- generated:inference -->`` markers -> local inference tables built from
    ``inference-benchmark.json`` (published code), ``kv-cache-verification.json`` (new KV-cache code) and,
    when present, ``quantization-measurements.json``;
  * staged files: observability figures, the JSON measurements, the KV-cache ``hf_model.py`` /
    ``hf_config.py`` and ``config.json`` with ``use_cache: true``.
The script is idempotent: the marked block is rebuilt on every run.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "docs/model-cards/revised-2026-11-06"
RELEASE = STAGE / "_kv-cache-release"


def load(name: str) -> dict | None:
    path = STAGE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def inference_section(repo: str, bench: dict, kv: dict, quant: dict | None) -> str:
    env, proto = bench["environment"], bench["protocol"]
    rows = [r for r in bench["results"] if r["repo"].endswith("/" + repo)]
    kv_row = next(r for r in kv["results"] if r["repo"].endswith("/" + repo))
    out = [
        f"Measured on {env['gpu']} (8 GB) and an Intel Core i7-12650H (10 cores, {env['ram_gb']:.0f} GB RAM), "
        f"Windows 11, Python {env['python']}, PyTorch {rows[0]['torch']}, Transformers {rows[0]['transformers']}. "
        f"Protocol: batch size 1, {proto['prompt_tokens']}-token prompt, greedy decoding of exactly "
        f"{proto['new_tokens']} new tokens, one warm-up run, median of timed runs "
        "(`scripts/benchmark_card_inference.py`, `scripts/verify_kv_cache_release.py`; raw results in "
        "`inference-benchmark.json` and `kv-cache-verification.json`).",
        "",
        "**Generation speed, with and without the KV cache** (generated tokens per second):",
        "",
        "| Device | Without cache | With KV cache | Speed-up | Same tokens with and without cache |",
        "|---|---:|---:|---:|---|",
    ]
    for device, label in (("cpu", "CPU, float32"), ("cuda", "GPU, float32")):
        d = kv_row[device]
        out.append(f"| {label} | {d['no_cache_tokens_per_s']:.1f} | {d['cache_tokens_per_s']:.1f} | ×{d['speedup']:.2f} | "
                   f"{'yes' if d['identical_tokens_with_and_without_cache'] else 'NO'} |")
    out += [
        "",
        f"The KV-cache code gives the same logits as the code first published with these weights (maximum "
        f"absolute difference {kv_row['max_abs_logit_diff_vs_published_code']:.1g}) and the same greedy tokens with "
        "and without the cache. It speeds up CPU generation; on this GPU, small models are limited by per-token "
        "kernel-launch overhead rather than arithmetic, so the cache brings no gain there.",
        "",
        "**Latency and memory** (code without cache, as first published):",
        "",
        "| Device | Weights dtype | Prefill, 128 tokens (≈ time to first token) | 64 new tokens, total | Peak process RAM | Peak allocated VRAM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        vram = f"{r['peak_allocated_vram_gb'] * 1000:.0f} MB" if r["peak_allocated_vram_gb"] is not None else "—"
        out.append(f"| {'CPU' if r['device'] == 'cpu' else 'GPU'} | {r['dtype']} | {r['prefill_ms_median']:.1f} ms | "
                   f"{r['generate_64_tokens_s_median']:.2f} s | {r['peak_process_ram_gb']:.2f} GB | {vram} |")
    out += [
        "",
        "Peak process RAM includes the Python and PyTorch runtime (about 0.8 GB on CPU, 1.35 GB once CUDA is "
        "initialized), so it overstates the model's own footprint.",
    ]
    if quant is not None:
        q = next(r for r in quant["results"] if r["repo"].endswith("/" + repo))
        out += [
            "",
            "**Reduced-precision variants** (`scripts/quantize_and_measure.py`, `quantization-measurements.json`). "
            "Quality on one fixed text sample (first 100 English + 100 French documents of the SLM-Tavenel corpus-v2 "
            "held-out test split; only the change against float32 is meaningful):",
            "",
            "| Variant | Device | Weights size | BPB on the sample | Change vs float32 | Next-token top-1 agreement with float32 | Tokens/s (KV cache) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for v in q["rows"]:
            size = f"{v['file_mb']:.1f} MB" + (" (in memory)" if v.get("file_note") else "")
            out.append(f"| {v['variant']} | {v['device'].upper() if v['device'] == 'cpu' else 'GPU'} | {size} | {v['bpb']:.4f} | "
                       f"{v['delta_bpb']:+.4f} | {v['top1_agreement']:.1%} | {v['tokens_per_s']:.1f} |")
        out += [
            "",
            "`bfloat16`: weights cast to bfloat16 (`variants/bf16/`, half the file size). `int8-dynamic`: PyTorch "
            "dynamic quantization of every linear layer, applied at load time on CPU "
            "(`torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`); it is not "
            "stored as a file. No 4-bit, GGUF or bitsandbytes variant has been produced or measured.",
        ]
    out.append("")
    out.append("Single machine, single session: treat the timings as indicative.")
    return "\n".join(out)


def main() -> None:
    bench, kv, quant = load("inference-benchmark.json"), load("kv-cache-verification.json"), load("quantization-measurements.json")
    family = (STAGE / "_shared-family-benchmarks.md").read_text(encoding="utf-8").strip()
    for card in sorted(STAGE.glob("slm-tavenel-*/README.md")):
        repo = card.parent.name
        text = card.read_text(encoding="utf-8").replace("FAMILY_BENCHMARKS_PLACEHOLDER", family)
        text = re.sub(r"<!-- generated:inference -->.*?<!-- /generated:inference -->",
                      lambda _: "<!-- generated:inference -->\n" + inference_section(repo, bench, kv, quant)
                      + "\n<!-- /generated:inference -->", text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")
        for ext in ("png", "svg"):
            figure = STAGE / "figures" / f"{repo}-observability.{ext}"
            if figure.exists():
                shutil.copy2(figure, card.parent / figure.name)
        for name in ("inference-benchmark.json", "kv-cache-verification.json", "quantization-measurements.json"):
            if (STAGE / name).exists():
                shutil.copy2(STAGE / name, card.parent / name)
        for name in ("hf_model.py", "hf_config.py"):
            shutil.copy2(ROOT / "src/slm" / name, card.parent / name)
        config = json.loads((RELEASE / repo / "config.json").read_text(encoding="utf-8"))
        config["use_cache"] = True
        (card.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print("built", repo)


if __name__ == "__main__":
    main()
