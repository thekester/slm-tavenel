---
language:
- en
- fr
license: mit
library_name: transformers
pipeline_tag: text-generation
datasets:
- PleIAs/French-Science-Commons
tags:
- slm-tavenel
- experimental
- research-artifact
- base-model
- open-weights
model-index:
- name: slm-tavenel-10m
  results:
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: ARC-Easy, type: allenai/ai2_arc, config: ARC-Easy, split: test, revision: 210d026faf9955653af8916fad021475a3f00453}
    metrics: [{type: accuracy, value: 0.306, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0009 benchmark record, url: "https://huggingface.co/athenista/slm-tavenel-10m/tree/main/training_record"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: ARC-Challenge, type: allenai/ai2_arc, config: ARC-Challenge, split: test, revision: 210d026faf9955653af8916fad021475a3f00453}
    metrics: [{type: accuracy, value: 0.184, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0009 benchmark record, url: "https://huggingface.co/athenista/slm-tavenel-10m/tree/main/training_record"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: HellaSwag, type: Rowan/hellaswag, split: validation, revision: 218ec52e09a7e7462a5400043bb9a69a41d06b76}
    metrics: [{type: accuracy, value: 0.275, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0009 benchmark record, url: "https://huggingface.co/athenista/slm-tavenel-10m/tree/main/training_record"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: Belebele (English), type: facebook/belebele, config: eng_Latn, split: test, revision: 7899cdfa4e1e0d733fd77c848e2c273cb1d32be2}
    metrics: [{type: accuracy, value: 0.296, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0009 benchmark record, url: "https://huggingface.co/athenista/slm-tavenel-10m/tree/main/training_record"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: Belebele (French), type: facebook/belebele, config: fra_Latn, split: test, revision: 7899cdfa4e1e0d733fd77c848e2c273cb1d32be2}
    metrics: [{type: accuracy, value: 0.319, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0009 benchmark record, url: "https://huggingface.co/athenista/slm-tavenel-10m/tree/main/training_record"}
  - task: {type: multiple-choice, name: "Multiple choice (0-shot, log-likelihood)"}
    dataset: {name: "WinoGrande (XL, validation)", type: allenai/winogrande, config: winogrande_xl, split: validation, revision: 546d5aa0e42fc0fa9ad2c13260322ec969e083ee}
    metrics: [{type: accuracy, value: 0.511, name: accuracy (SLM-Tavenel evaluator)}]
    source: {name: SLM-Tavenel EXP-0009 benchmark record, url: "https://huggingface.co/athenista/slm-tavenel-10m/tree/main/training_record"}
---

# SLM-Tavenel 10M

> **Project links:** [SLM-Tavenel overview](https://github.com/thekester/slm-tavenel#slm-tavenel) · [published model catalog](https://github.com/thekester/slm-tavenel#published-models) · [research and reproducibility archive](https://github.com/thekester/slm-tavenel#research-and-reproducibility). This card describes this checkpoint; the repository overview links the related models and broader experiment record.

**Experimental English/French base model (9.98M parameters), trained from scratch** for 15,000 steps on
a bilingual open-science text pilot (French-Science-Commons). Not instruction tuned; not a chat
assistant. Weights and code are offered under MIT; the source corpus mixes CC0, CC-BY-4.0 and
Licence-Ouverte-2.0 records, and the unresolved rights and privacy checks are disclosed below. This is
not an OSI certification or a rights clearance.

| At a glance | |
|---|---|
| Parameters | 9,978,368 |
| Architecture | Decoder-only transformer, 10 layers, width 256, 4 query / 2 key-value heads, SwiGLU 768, head-wise attention gate |
| Tokenizer | Project byte-level BPE, 8,192 tokens |
| Context window | 1,024 tokens |
| Training precision | bfloat16 mixed precision |
| Published weights | `model.safetensors`, float32; bfloat16 copy staged in ariants/bf16/; CPU int8 recipe in Local inference |
| Training data | `PleIAs/French-Science-Commons` (`french_open_science`), 31,417 retained records, 491.52M token positions |
| Optimizer / seed | AdamW; seed 1234 |
| Compute | One RTX 4060 Laptop GPU (8 GB), ~2.5 GPU-hours; energy not measured |
| Final train / validation loss | 2.29647 / 2.33488 |
| License | MIT (weights, code, Data Information); source records keep their own terms |
| Hub revision described | `9818b706a2a2467efb22848967a5842871f7422c` (weights identical at later revisions: SHA-256 `ef6967c4…`) |

## Contents

[Main result](#main-result) · [Benchmarks](#benchmarks) · [Local inference](#local-inference) ·
[Use](#use) · [Architecture](#architecture) · [Training](#training) ·
[Traceability and explainability](#traceability-and-explainability) ·
[Intended use and safety](#intended-use-out-of-scope-use-and-safety) · [Data, licenses and privacy](#data-licenses-and-privacy) ·
[Limitations](#limitations) · [License and citation](#license-and-citation) · [Model family](#the-slm-tavenel-family)

## Main result

[![Training and validation loss across the completed 10M run](training-curve.png)](training-curve.svg)

A complete 15,000-step run at 10M parameters (validation loss 2.335). On the six multiple-choice tasks
the model is close to chance, as expected at this size and budget; its role in the project is the
trained ancestor of the grown 25M model and a small-scale baseline.

## Benchmarks

**Protocol.** Zero-shot multiple choice scored by log-likelihood with the SLM-Tavenel evaluator on pinned
items (lm-evaluation-harness prompt templates for ARC, HellaSwag and Belebele; cloze scoring for
WinoGrande). One checkpoint, one training seed.

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

Against the grown 25M model on the same items, all six paired 95% item-bootstrap intervals include zero
(see the 25M card). The SLM-Tavenel 10M's FSC validation BPB is 1.1755 (English) and 1.0897 (French).

## Local inference

<!-- generated:inference -->
Measured on NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) and an Intel Core i7-12650H (10 cores, 68 GB RAM), Windows 11, Python 3.11.9, PyTorch 2.10.0+cu130, Transformers 4.57.6. Protocol: batch size 1, 128-token prompt, greedy decoding of exactly 64 new tokens, one warm-up run, median of timed runs (`scripts/benchmark_card_inference.py`, `scripts/verify_kv_cache_release.py`; raw results in `inference-benchmark.json` and `kv-cache-verification.json`).

**Generation speed, with and without the KV cache** (generated tokens per second):

| Device | Without cache | With KV cache | Speed-up | Same tokens with and without cache |
|---|---:|---:|---:|---|
| CPU, float32 | 43.8 | 128.3 | ×2.93 | yes |
| GPU, float32 | 61.9 | 61.3 | ×0.99 | yes |

Cached generation gives the same logits as a full-prefix, no-cache reference (maximum absolute difference 0) and the same greedy tokens with and without the cache. It speeds up CPU generation; on this GPU, small models are limited by per-token kernel-launch overhead rather than arithmetic, so the cache brings no gain there.

**Latency and memory** (code without cache, as first published):

| Device | Weights dtype | Prefill, 128 tokens (≈ time to first token) | 64 new tokens, total | Peak process RAM | Peak allocated VRAM |
|---|---|---:|---:|---:|---:|
| CPU | float32 | 16.7 ms | 1.48 s | 0.87 GB | — |
| GPU | float32 | 16.1 ms | 1.06 s | 1.36 GB | 56 MB |
| GPU | bfloat16 | 17.6 ms | 1.20 s | 1.50 GB | 32 MB |

Peak process RAM includes the Python and PyTorch runtime (about 0.8 GB on CPU, 1.35 GB once CUDA is initialized), so it overstates the model's own footprint.

**Reduced-precision variants** (`scripts/quantize_and_measure.py`, `quantization-measurements.json`). Quality on one fixed text sample (first 100 English + 100 French documents of the SLM-Tavenel corpus-v2 held-out test split; only the change against float32 is meaningful):

| Variant | Device | Weights size | BPB on the sample | Change vs float32 | Next-token top-1 agreement with float32 | Tokens/s (KV cache) |
|---|---|---:|---:|---:|---:|---:|
| float32 | CPU | 39.9 MB | 1.4634 | +0.0000 | 100.0% | 103.6 |
| bfloat16 | GPU | 20.0 MB | 1.4636 | +0.0002 | 98.8% | 53.8 |
| int8-dynamic | CPU | 18.4 MB (in memory) | 1.4872 | +0.0238 | 85.7% | 58.3 |

`bfloat16`: weights cast to bfloat16 (`variants/bf16/`, half the file size). `int8-dynamic`: PyTorch dynamic quantization of every linear layer, applied at load time on CPU (`torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`); it is not stored as a file. No 4-bit, GGUF or bitsandbytes variant has been produced or measured.

Single machine, single session: treat the timings as indicative.
<!-- /generated:inference -->

## Use

```bash
pip install -r requirements.txt   # versions recorded with this package
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "athenista/slm-tavenel-10m"
revision = "9818b706a2a2467efb22848967a5842871f7422c"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)
inputs = tokenizer("La recherche scientifique", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=48, do_sample=False)[0]))
```

`trust_remote_code=True` is required (custom architecture; code in `hf_model.py` and `hf_config.py`,
read it before running). The revision pinned above already contains the KV-cache implementation used for the measurements in Local inference. The cache was checked against uncached generation for this checkpoint; see the recorded logits and token-equivalence results.

## Architecture

[![SLM-Tavenel decoder block and the 1M, 10M and grown 25M model dimensions](architecture.png)](architecture.svg)

Ten pre-RMSNorm decoder blocks with grouped-query causal attention, QK normalization, RoPE, a head-wise
sigmoid gate on attention outputs and a SwiGLU feed-forward sublayer with residual connections. The 25M
model was grown from this trained checkpoint by expanding depth and feed-forward width.

## Training

| Setting | Value |
|---|---|
| Initialization | From scratch |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Schedule | Warmup–stable–decay: 500 warmup steps, peak 3e-3, final 20% decay to 3e-4 |
| Batch | 16 sequences × 1,024 tokens × 2 accumulation steps = 32,768 token positions per step |
| Steps / tokens | 15,000 steps, 491,520,000 token positions |
| Precision | bfloat16 autocast |
| Hardware | 1 × NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), peak allocated 3.7 GB |
| Software | Python 3.11.9, PyTorch 2.10.0+cu130 |
| Compute | ~2.5 GPU-hours, summed from the logged throughput |
| Energy / emissions | **Not measured** for this run (no power log was recorded); not estimated here. Training runs from 2026-11-06 on log GPU board energy through NVML (src/slm/energy.py) |

## Traceability and explainability

**Traceability.** `training_record/` holds the checkpoint provenance, run manifest, train/eval/
observability logs, source registry, processing manifest and decisions, tokenizer and token-shard
manifests, a record-level source manifest (one row per retained document: title, declared authors,
identifiers, DOI, declared license, hashes), the benchmark score record, model code and configurations.
`provenance.json` lists the SHA-256 of every file. The recorded training worktree was dirty and the
uncommitted diff was not captured, so byte-exact reproduction of the training code is not claimed.

**Explainability (observability, not causal explanation).** `training_record/observability.jsonl`
records, every 500 steps on a fixed probe batch, per-layer attention entropy, attention-sink mass,
head-wise gate statistics, residual-stream norms, weight norms and update ratios. Last record:

[![Per-layer internal diagnostics](slm-tavenel-10m-observability.png)](slm-tavenel-10m-observability.svg)

Attention-sink mass stays low in every layer; gates are mostly open except in a few layers; the
residual norm grows sharply in the last layers. These are probe-batch measurements, not explanations of
particular outputs, and were not validated by interventions.

## Intended use, out-of-scope use and safety

- **Intended**: research and teaching on small-model training, bilingual text modeling, reproducibility
  and training observability.
- **Out of scope**: chat or assistant use, factual or scientific advice, decisions about people, any
  consequential use.
<!-- generated:safety -->
- **Safety evaluation**: a basic screening only (`scripts/safety_eval_basic.py`, results in `safety-eval-basic.json`), not a safety certification. Toxicity: 500 prompts sampled from the "challenging" subset of RealToxicityPrompts (English; Gehman et al., 2020), one sampled continuation of 32 tokens each (top-p 0.9), scored by `unitary/toxic-bert`. Personal data: 20 English/French prompts inviting contact details × 10 samples, scanned with the project's PII detector (email, phone, IPv4, IBAN). The prompts themselves score 0.77 mean toxicity.

| Model | Mean toxicity [95% CI] | Continuations with toxicity > 0.5 [95% CI] | PII-shaped outputs | Redaction placeholders (`<EMAIL>`…) |
|---|---:|---:|---:|---:|
| **slm-tavenel-10m** | 0.069 [0.050, 0.089] | 6.2% [4.2%, 8.4%] | 0.0% | 2.0% |
| GPT-2 Small (same harness, reference) | 0.417 [0.378, 0.455] | 42.2% [38.0%, 46.6%] | 21.0% | 0.0% |

Lower toxicity than GPT-2 is not evidence of safety: small or domain-specific models often produce less fluent continuations, which classifiers score as less toxic. The PII probe detects strings shaped like personal data; it does not check whether they belong to real people. Not covered: French toxicity (no French-capable classifier with a compatible license was found), bias, memorization of training text, jailbreaks. No instruction, preference or safety tuning was applied.
<!-- /generated:safety -->

## Data, licenses and privacy

Training data came from the `french_open_science` subset of
[PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), pinned to
`bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. The processing pipeline started with 43,011 records and
retained 31,417 after quality, PII-pattern, exact-duplicate and near-duplicate processing. The retained
source metadata reports 4,528 records labeled CC0-1.0, 26,368 CC-BY-4.0 and 521 Licence-Ouverte-2.0.
The record-level metadata-only manifest lists titles, declared authors, identifiers, DOI fields, source
IDs, declared labels and hashes; all 31,417 rows have non-empty title and author fields and 13,200 have
no DOI (left blank). These are supplied metadata, not independent verification of authorship or rights
ownership. Source texts are not included.

CC-BY-4.0 and Licence-Ouverte-2.0 require attribution: see `ATTRIBUTION.md` and
`training_record/source-record-manifest.jsonl`. The metadata label does not establish that each
submitter had authority to apply it; no blanket rights clearance is claimed. Pattern-based redaction and
screening were limited, not a complete privacy audit.

**HAL file-license cross-check (22 October 2026).** After training, the 31,417 HAL identifiers were
queried against HAL's `licence_s` and `fileLicenses_s` fields
([HAL notice](https://www.ccsd.cnrs.fr/en/2026/02/no-hal-deposit-without-licence-the-essential-information-you-need-to-know/),
[API documentation](https://api.archives-ouvertes.fr/docs/search)): 31,262 matched, 72 explicit
mismatches, 3 conflicting declarations, 24 missing file-license metadata and 56 not found. These 155
remain unresolved. Results and script: `training_record/hal-file-license-audit-2026-10-22.json`,
`training_record/scripts/audit_fsc_mixed_hal_files.py`. This post-training metadata check is not a legal
clearance.

## Limitations

- One seed and one completed run; no training-seed uncertainty estimate.
- Benchmark accuracy is near chance on several tasks and does not establish general model quality.
- Dirty training worktree; exact uncommitted diff unavailable.
- 155 of 31,417 source identifiers have unresolved license-label discrepancies or missing records.
- Quantized variants are limited to bfloat16 weights and CPU int8 dynamic quantization (see Local inference); no 4-bit or GGUF version.

## License and citation

Weights, project code and included Data Information: MIT. Source records remain under their declared
licenses. Cite as **SLM-Tavenel EXP-0009 10M, seed 1234**, Hub revision
`9818b706a2a2467efb22848967a5842871f7422c`; the experiment report and protocol are in
`training_record/EXP-0009-fsc-scaling-and-references.md`. No paper or DOI is assigned.

## The SLM-Tavenel family

| Model | Parameters | Data | Relation |
|---|---:|---|---|
| [slm-tavenel-1m](https://huggingface.co/athenista/slm-tavenel-1m) | 1.18M | French-Science-Commons (CC0 subset) | ten-seed gate study |
| **slm-tavenel-10m** | 9.98M | French-Science-Commons | trained from scratch |
| [slm-tavenel-25m](https://huggingface.co/athenista/slm-tavenel-25m) | 25.7M | French-Science-Commons | grown from this model |
| [slm-tavenel-25m-hplt-10pct](https://huggingface.co/athenista/slm-tavenel-25m-hplt-10pct) | 25.7M | + 10% HPLT | continuation of 25M |
| [slm-tavenel-124m](https://huggingface.co/athenista/slm-tavenel-124m) | 124M | FineWeb-Edu + FineWeb2-HQ | trained from scratch, new tokenizer |
