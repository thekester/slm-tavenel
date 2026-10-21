---
language:
- en
- fr
pipeline_tag: text-generation
library_name: transformers
datasets:
- PleIAs/French-Science-Commons
tags:
- research-artifact
- experimental
- gated-attention
---

# SLM-Tavenel 1M (research candidate)

> **Release status: not approved for public upload or open-weight redistribution.** This card is
> prepared for review. The training corpus is not independently rights-cleared, the run used one
> seed and a dirty code tree. A local fixed-input CPU check found exact tokenizer and logit parity
> with the native checkpoint; it is a smoke check, not a broad independent conformance review. No
> model-weight license is assigned. The MIT license in the source repository
> covers code, not these weights.

## Summary

SLM-Tavenel 1M is a 1,181,920-parameter decoder-only Transformer trained from scratch on a
balanced English/French science-text pilot. It uses RMSNorm, RoPE, grouped-query attention,
per-head QK normalization, SwiGLU, tied token embeddings, and a sigmoid head-wise gate applied to
each attention head's output.

This checkpoint is a small research artifact for studying training and attention behavior. Its
benchmarks are exploratory. It is not validated as a general-purpose assistant, and no claim of
superiority over external models is supported.

## Intended use and limitations

- Research and controlled experiments on compact bilingual language models.
- No instruction tuning or safety tuning is claimed; raw generations may be inaccurate, biased,
  incomplete, or unsuitable for consequential use.
- The training mix is a small science-text pilot, not a representative sample of English and French.
- The observed evaluation results were inspected after training and must not be described as a
  sealed confirmatory evaluation.
- The one training seed does not estimate run-to-run variability.

## Training provenance

| Field | Value |
|---|---|
| Parameters | 1,181,920 |
| Initialization | From scratch |
| Training seed | 1234 |
| Steps | 15,000 |
| Training token budget | 491.52 million |
| Optimizer / schedule | AdamW / warmup-stable-decay |
| Sequence length | 1,024 |
| Precision | bfloat16 training |
| Corpus | French-Science-Commons, `french_open_science`, pinned revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156` |
| Languages | English and French, token-balanced in the prepared pilot |
| Run code revision | `be00a59fe4a8b76a6b82b5ee8a9cd5ddbda748a8`; working tree was dirty |

The FSC source registry remains `candidate`; the recorded per-record license labels, provenance,
privacy review, and attribution handling have not all been independently cleared. See the repository
[FSC pilot review](https://github.com/thekester/slm-tavenel/blob/main/docs/reviews/FSC-PILOT-2026-10-20.md)
and [EXP-0009](https://github.com/thekester/slm-tavenel/blob/main/experiments/EXP-0009-fsc-scaling-and-references.md).
The generated Hub bundle's `provenance.json` includes hashes for the checkpoint, tokenizer, run
manifest, scalar training logs, observability log, and evaluation files; it contains no training
text or token shards.

## Evaluation (exploratory)

| Evaluation | Score | Qualification |
|---|---:|---|
| English bits per byte | 1.4904 | Inspected FSC validation split; one checkpoint |
| French bits per byte | 1.3676 | Inspected FSC validation split; one checkpoint |
| ARC-Easy | 0.286 | Same local evaluator; benchmark exposure not fully ruled out |
| HellaSwag | 0.267 | Same local evaluator; benchmark exposure not fully ruled out |
| Belebele English | 0.279 | Same local evaluator; benchmark exposure not fully ruled out |
| Belebele French | 0.292 | Same local evaluator; benchmark exposure not fully ruled out |
| ARC-Challenge | 0.199 | Post-hoc; no exact training 13-gram overlap found; semantic/external exposure unknown |
| WinoGrande XL validation | 0.502 | Post-hoc labeled validation split; not a confirmatory test |

The results and comparison limits are documented in [EXP-0009](../../experiments/EXP-0009-fsc-scaling-and-references.md).
Same-harness evaluation controls prompts and scoring; it does not match training corpora, token
budgets, context limits, or pretraining exposure across reference models.

## Observability and interpretability

The run logs per-layer residual magnitude, attention sink mass and entropy, gate statistics, weight
norms, and update-to-weight ratios every 500 steps. These are diagnostic observations, not causal
explanations. The repository has activation capture and intervention primitives, but this
checkpoint has no preregistered, multi-seed causal explanation of its predictions.

## Loading and architecture

The gated attention is not equivalent to standard Llama or Qwen attention. A faithful Transformers
implementation is therefore supplied as custom model code. Loading it requires `trust_remote_code=True`;
users should pin an immutable Hub revision before executing custom code. The current implementation
does not use a KV cache, so autoregressive generation is slower than optimized decoder models. A
fixed-input local parity and greedy-generation smoke report is included as
[`inference-parity-report.json`](slm-tavenel-1m-hf-parity-report.json).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "OWNER/slm-tavenel-1m"
revision = "PIN_AN_IMMUTABLE_COMMIT"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=True
)
inputs = tokenizer("La recherche scientifique", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Do not replace `OWNER` or the revision placeholder until a reviewed release exists. The included
parity report gives one exact CPU probe and its tolerance; broader device/dtype parity and evaluation
from the final Hub revision remain release checks.

## Licensing and release

No license is declared for the model weights. The repository's code license does not automatically
license data-derived weights. Public release requires a documented source-rights and privacy review,
attribution materials for included CC BY and Licence Ouverte sources, an explicit weight-license
decision, and successful inference-equivalence review. Until those steps are complete, keep any
bundle private and do not advertise it as open weights.
