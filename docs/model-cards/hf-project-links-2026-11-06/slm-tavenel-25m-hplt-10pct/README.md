---
language:
- en
- fr
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: athenista/slm-tavenel-25m
datasets:
- PleIAs/French-Science-Commons
- ashtok897/european-hplt-v1
tags:
- slm-tavenel
- experimental
- research-artifact
- base-model
- hplt-domain-adaptation
model-index:
- name: slm-tavenel-25m-hplt-10pct
  results:
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: ARC-Easy, type: allenai/ai2_arc, config: ARC-Easy, split: test, revision: 210d026faf9955653af8916fad021475a3f00453}
    metrics: [{type: accuracy, value: 0.3194, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0032, url: "https://huggingface.co/athenista/slm-tavenel-25m-hplt-10pct/blob/main/training_record/EXP-0032-results.md"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: ARC-Challenge, type: allenai/ai2_arc, config: ARC-Challenge, split: test, revision: 210d026faf9955653af8916fad021475a3f00453}
    metrics: [{type: accuracy, value: 0.1869, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0032, url: "https://huggingface.co/athenista/slm-tavenel-25m-hplt-10pct/blob/main/training_record/EXP-0032-results.md"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: HellaSwag, type: Rowan/hellaswag, split: validation, revision: 218ec52e09a7e7462a5400043bb9a69a41d06b76}
    metrics: [{type: accuracy, value: 0.2736, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0032, url: "https://huggingface.co/athenista/slm-tavenel-25m-hplt-10pct/blob/main/training_record/EXP-0032-results.md"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: Belebele (English), type: facebook/belebele, config: eng_Latn, split: test, revision: 7899cdfa4e1e0d733fd77c848e2c273cb1d32be2}
    metrics: [{type: accuracy, value: 0.2789, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0032, url: "https://huggingface.co/athenista/slm-tavenel-25m-hplt-10pct/blob/main/training_record/EXP-0032-results.md"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: Belebele (French), type: facebook/belebele, config: fra_Latn, split: test, revision: 7899cdfa4e1e0d733fd77c848e2c273cb1d32be2}
    metrics: [{type: accuracy, value: 0.3078, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0032, url: "https://huggingface.co/athenista/slm-tavenel-25m-hplt-10pct/blob/main/training_record/EXP-0032-results.md"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: "WinoGrande (XL, validation)", type: allenai/winogrande, config: winogrande_xl, split: validation, revision: 546d5aa0e42fc0fa9ad2c13260322ec969e083ee}
    metrics: [{type: accuracy, value: 0.5170, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0032, url: "https://huggingface.co/athenista/slm-tavenel-25m-hplt-10pct/blob/main/training_record/EXP-0032-results.md"}
---

# SLM-Tavenel 25M HPLT 10% continuation

> **Project links:** [SLM-Tavenel overview](https://github.com/thekester/slm-tavenel#slm-tavenel) ? [published model catalog](https://github.com/thekester/slm-tavenel#published-models) ? [research and reproducibility archive](https://github.com/thekester/slm-tavenel#research-and-reproducibility). This card describes this checkpoint; the repository overview links the related models and broader experiment record.

**Experimental English/French base model (25.7M parameters), adapted to an HPLT-derived web mixture.**
It continues the published SLM-Tavenel 25M for 1,500 steps on a frozen 90% French-Science-Commons /
10% HPLT-derived token mixture. A separate variant: it does not replace the 25M checkpoint. Not
instruction tuned; not a chat assistant. Source-rights review (FSC lineage) and page-level rights and
privacy review (HPLT-derived data) remain unresolved; MIT applies to this project's code and weights
and does not relicense either upstream dataset.

| At a glance | |
|---|---|
| Parameters | 25,723,648 |
| Architecture | Decoder-only transformer, 20 layers, width 256, 4 query / 2 key-value heads, head-wise attention gate (same as the parent) |
| Tokenizer | Project byte-level BPE, 8,192 tokens (inherited unchanged) |
| Context window | 1,024 tokens |
| Training precision | bfloat16 mixed precision |
| Published weights | `model.safetensors`, float32 (SHA-256 `fb7c7783…`); no quantized version is published |
| Parent | [SLM-Tavenel 25M](https://huggingface.co/athenista/slm-tavenel-25m) (final checkpoint) |
| Continuation | 1,500 steps, 49,152,000 token positions, AdamW, seed 20261028 |
| HPLT share | ~10% of continuation tokens, English and French balanced within the candidate subset |
| Compute | One RTX 4060 Laptop GPU (8 GB), ~0.8 GPU-hours for this continuation; energy not measured |
| License | MIT (weights, code); upstream datasets keep their own terms |
| Hub revision described | `d377cfb8cae4a6018ea44d9133da5f82dce6b4d3` |

## Contents

[Main result](#main-result) · [Benchmarks](#benchmarks) · [Local inference](#local-inference) ·
[Use](#use) · [Architecture](#architecture) · [Training](#training) ·
[Traceability and explainability](#traceability-and-explainability) ·
[Intended use and safety](#intended-use-out-of-scope-use-and-safety) · [Data, licenses and privacy](#data-licenses-and-privacy) ·
[Limitations](#limitations) · [License and citation](#license-and-citation) · [Model family](#the-slm-tavenel-family)

## Main result

EXP-0032 was a paired, same-seed continuation from the same 25M checkpoint: the treatment used the
frozen 90/10 FSC/HPLT stream, the control continued on FSC only; same tokenizer, model, initialization,
optimizer, seed and token budget. The HPLT test split was held out and scored once, after the recipe was
frozen.

| Evaluation | Language | FSC-only control | HPLT 10% (this model) | Treatment − control | 95% paired document interval |
|---|---|---:|---:|---:|---:|
| Held-out HPLT test BPB | English | 1.54603 | **1.47279** | −0.07323 | [−0.07596, −0.07037] |
| Held-out HPLT test BPB | French | 1.35556 | **1.29059** | −0.06497 | [−0.06960, −0.05989] |
| FSC validation BPB | English | 1.10299 | 1.10421 | +0.00122 | [+0.00100, +0.00143] |
| FSC validation BPB | French | 1.02416 | 1.02493 | +0.00077 | [+0.00055, +0.00098] |

Lower BPB is better. Intervals resample documents within language (13,870 English and 4,712 French HPLT
test documents); they do not measure training-seed variation. The result supports an HPLT-domain
adaptation effect for this pair, not a general capability gain.

[![Paired HPLT development and held-out test BPB differences](hplt-bpb-deltas.png)](hplt-bpb-deltas.svg)

## Benchmarks

**Protocol.** Zero-shot multiple choice scored by log-likelihood with the SLM-Tavenel evaluator on pinned
items. Paired treatment-minus-control accuracy, 95% item bootstrap — every interval includes zero:

| Task | Items | FSC-only control | HPLT 10% | Difference | Paired 95% item interval |
|---|---:|---:|---:|---:|---:|
| ARC-Easy | 2,376 | 0.3161 | 0.3194 | +0.0034 | [−0.0088, +0.0152] |
| ARC-Challenge | 1,172 | 0.1877 | 0.1869 | −0.0009 | [−0.0154, +0.0145] |
| HellaSwag | 10,042 | 0.2757 | 0.2736 | −0.0022 | [−0.0054, +0.0010] |
| Belebele English | 900 | 0.2811 | 0.2789 | −0.0022 | [−0.0144, +0.0111] |
| Belebele French | 900 | 0.3156 | 0.3078 | −0.0078 | [−0.0211, +0.0056] |
| WinoGrande | 1,267 | 0.5185 | 0.5170 | −0.0016 | [−0.0197, +0.0166] |

Family context (same evaluator and items):

| Model | Parameters | ARC-Easy | ARC-Challenge | HellaSwag | Belebele EN | Belebele FR | WinoGrande | Six-task mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLM-Tavenel 1M | 1.18M | 0.280 | 0.196 | 0.259 | 0.264 | 0.266 | 0.493 | 0.293 |
| SLM-Tavenel 10M | 9.98M | 0.306 | 0.184 | 0.275 | 0.296 | 0.319 | 0.511 | 0.315 |
| SLM-Tavenel 25M | 25.7M | 0.319 | 0.180 | 0.273 | 0.283 | 0.309 | 0.519 | 0.314 |
| SLM-Tavenel 25M HPLT 10% | 25.7M | 0.319 | 0.187 | 0.274 | 0.279 | 0.308 | 0.517 | 0.314 |
| SLM-Tavenel 124M | 124M | 0.484 | 0.206 | 0.277 | 0.316 | 0.311 | 0.519 | 0.352 |
| [TinyStories tiny1m](https://huggingface.co/shibatch/tiny1m) | 0.94M | 0.256 | 0.189 | 0.260 | 0.274 | 0.290 | 0.518 | 0.298 |
| [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) | 14.1M | 0.330 | 0.161 | 0.262 | 0.256 | 0.309 | 0.520 | 0.306 |
| [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m) | 70M | 0.317 | 0.194 | 0.268 | 0.282 | 0.306 | 0.487 | 0.309 |
| [GPT-2 Small](https://huggingface.co/openai-community/gpt2) | 124M | 0.447 | 0.185 | 0.292 | 0.310 | 0.318 | 0.517 | 0.345 |
| [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m) | 160M | 0.386 | 0.187 | 0.285 | 0.288 | 0.309 | 0.493 | 0.324 |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 135M | 0.643 | 0.278 | 0.356 | 0.358 | 0.314 | 0.530 | 0.413 |
| Items | | 2,376 | 1,172 | 10,042 | 900 | 900 | 1,267 | |
| 95% binomial half-width (typical) | | ±0.019 | ±0.022 | ±0.009 | ±0.030 | ±0.030 | ±0.028 | |
| Chance | | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.50 | |

Raw zero-shot accuracy, all rows scored by the SLM-Tavenel log-likelihood evaluator on the same pinned
items (ARC `allenai/ai2_arc@210d026f`, HellaSwag `Rowan/hellaswag@218ec52e` validation, Belebele
`facebook/belebele@7899cdfa`, WinoGrande XL `allenai/winogrande@546d5aa0` validation). Half-widths are
item-sampling intervals for one checkpoint; they do not include training-seed variation, and paired
intervals (in each card) are the right tool for comparing two models on the same items. The models
differ in data, tokenizer, training budget and possible benchmark exposure: this table describes
checkpoints, it is not a ranking of recipes.

## Local inference

Measured on NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) and an Intel Core i7-12650H (10 cores, 68 GB RAM), Windows 11, Python 3.11.9, PyTorch 2.10.0+cu130, Transformers 4.57.6, loading this repository exactly as in [Use](#use). Protocol: batch size 1, 128-token prompt, greedy decoding of exactly 64 new tokens, one warm-up run, median of 5 timed runs (`scripts/benchmark_card_inference.py`; raw results in `inference-benchmark.json`).

| Device | Weights dtype | Prefill, 128 tokens (≈ time to first token) | 64 new tokens, total | Generated tokens/s | Peak process RAM | Peak allocated VRAM |
|---|---|---:|---:|---:|---:|---:|
| CPU | float32 | 45.2 ms | 3.52 s | 18.2 | 0.94 GB | — |
| GPU | float32 | 30.3 ms | 1.87 s | 34.2 | 1.36 GB | 119 MB |
| GPU | bfloat16 | 31.7 ms | 2.19 s | 29.2 | 1.55 GB | 64 MB |

The implementation has **no KV cache**: every new token re-runs the whole prefix, so latency grows with prompt plus output length and these figures are far from what an optimized decoder would reach. Peak process RAM includes the Python and PyTorch runtime (about 0.8 GB on CPU, 1.35 GB once CUDA is initialized), so it overstates the model's own footprint; the float32 weights file itself is about 4 bytes per parameter. bfloat16 is not faster here because the run time is dominated by per-token overhead, not arithmetic. No quantized (int8/int4) version has been produced or measured. Single machine, single session: treat the numbers as indicative.

## Use

```bash
pip install -r requirements.txt   # torch>=2.3, transformers==4.57.6, tokenizers==0.22.2, safetensors==0.8.0
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "athenista/slm-tavenel-25m-hplt-10pct"
revision = "d377cfb8cae4a6018ea44d9133da5f82dce6b4d3"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)
inputs = tokenizer("La recherche scientifique", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=48, do_sample=False)[0]))
```

Tested here with PyTorch 2.10.0 and Transformers 4.57.6. `trust_remote_code=True` is required (custom
architecture; `hf_model.py`, `hf_config.py`; read it before running). No KV cache is implemented.

## Architecture

[![SLM-Tavenel transformer architecture](architecture.png)](architecture.svg)

Identical to the parent 25M: twenty pre-RMSNorm decoder blocks with grouped-query causal attention, QK
normalization, RoPE, a head-wise sigmoid gate and SwiGLU, with residual connections.

## Training

| Setting | Value |
|---|---|
| Initialization | Final checkpoint of SLM-Tavenel 25M (EXP-0009) |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Schedule | Warmup–stable–decay: 50 warmup steps, peak 3e-4, final 20% decay to 3e-5 |
| Batch | 4 sequences × 1,024 tokens × 8 accumulation steps = 32,768 token positions per step |
| Steps / tokens | 1,500 steps, 49,152,000 token positions |
| Precision | bfloat16 autocast |
| Hardware | 1 × NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), peak allocated 2.9 GB |
| Software | not recorded in the published run summary (the project environment at the time was Python 3.11.9 / PyTorch 2.10.0+cu130) |
| Compute | ~0.8 GPU-hours, summed from the logged throughput |
| Energy / emissions | **Not measured** (no power log was recorded); not estimated here |

## Traceability and explainability

**Traceability.** `training_record/` includes the EXP-0032 report, training configuration, sanitized
run-manifest summaries for treatment and control, step logs, the mixture manifest, aggregate paired BPB
results and checkpoint provenance; `provenance.json` gives SHA-256 hashes. Source text, token shards,
benchmark prompts and per-example predictions are not included.

**Explainability.** No observability log (attention, gate or residual diagnostics) was recorded for
this 1,500-step continuation; see the parent 25M card for those diagnostics. Not recorded rather than
reconstructed.

## Intended use, out-of-scope use and safety

- **Intended**: research on domain adaptation by data mixture, small-model training and provenance.
- **Out of scope**: chat or assistant use, factual advice, decisions about people, any consequential use.
- **Safety evaluation**: **none was performed** (no toxicity, bias, privacy-leakage or red-team
  evaluation). Web-derived continuation data may raise the chance of reproducing web content.

## Data, licenses and privacy

90% of continuation token positions came from the FSC scaling stream and ~10% from a deterministic
English/French HPLT-derived candidate stream. HPLT source:
[ashtok897/european-hplt-v1](https://huggingface.co/datasets/ashtok897/european-hplt-v1/tree/51edb798468aa71b1000e950d3ab43bb435f72cd),
revision `51edb798468aa71b1000e950d3ab43bb435f72cd`; its card declares CC0-1.0 inherited from HPLT v3.
HPLT describes the collection as web crawls from the Internet Archive and Common Crawl and does not
establish rights over every original page; the declared label is provenance, not evidence of each
page's rights. The parent FSC lineage
([PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons),
`french_open_science`, `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`) has source-declared CC0-1.0,
CC-BY-4.0 and Licence-Ouverte-2.0 records; its remaining audit discrepancies, depositor-authority
questions and unsigned human privacy review are described in the parent's card. No independent rights
or privacy sign-off is represented by this release.

## Limitations

- One fresh seed pair; no estimate of training-seed variation.
- The HPLT test split was used once for the predeclared comparison and should not guide further tuning.
- No observability log for this continuation.
- No KV cache; no quantized weights.

## License and citation

Weights and code: MIT; upstream datasets keep their own terms. Cite as **SLM-Tavenel 25M HPLT 10%
continuation, seed 20261028**, Hub revision `d377cfb8cae4a6018ea44d9133da5f82dce6b4d3`, with the
checkpoint SHA-256 from `provenance.json`, and link the [EXP-0032 report](training_record/EXP-0032-results.md)
as exploratory HPLT-domain adaptation evidence. No paper or DOI is assigned.

## The SLM-Tavenel family

| Model | Parameters | Data | Relation |
|---|---:|---|---|
| [slm-tavenel-1m](https://huggingface.co/athenista/slm-tavenel-1m) | 1.18M | French-Science-Commons (CC0 subset) | ten-seed gate study |
| [slm-tavenel-10m](https://huggingface.co/athenista/slm-tavenel-10m) | 9.98M | French-Science-Commons | ancestor of the 25M |
| [slm-tavenel-25m](https://huggingface.co/athenista/slm-tavenel-25m) | 25.7M | French-Science-Commons | parent of this model |
| **slm-tavenel-25m-hplt-10pct** | 25.7M | + 10% HPLT | continuation of 25M (EXP-0032) |
| [slm-tavenel-124m](https://huggingface.co/athenista/slm-tavenel-124m) | 124M | FineWeb-Edu + FineWeb2-HQ | trained from scratch, new tokenizer |
