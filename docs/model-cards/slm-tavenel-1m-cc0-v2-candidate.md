---
language:
- en
- fr
pipeline_tag: text-generation
library_name: transformers
license: mit
repository: https://github.com/thekester/slm-tavenel
datasets:
- PleIAs/French-Science-Commons
tags:
- research-artifact
- experimental
- open-weights-candidate
---

# SLM-Tavenel 1M — CC0 v2 pilot candidate

> **Local candidate; not published.** The weights are packaged under MIT, but public release
> remains pending signed source review and manual privacy review. This is a single-seed pilot.

## Summary

Decoder-only English/French model trained from scratch with 1,181,920 parameters, an 8,192-token
byte-level BPE vocabulary and 1,024-token context. Architecture: four layers, grouped-query
attention, QK normalization, SwiGLU, RoPE and a sigmoid head-wise attention gate.

## Training data and filtering

The source is the `french_open_science` subset of
[PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons),
pinned at revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. Selected rows were labeled CC0 and
matched CC0 across all versions returned by the HAL records API; 33 unresolved or non-CC0 rows were
excluded. This is a metadata cross-check, not independent proof of the uploader's authority to
license every work.

A second conservative screen excluded documents tagged as social sciences, life sciences,
cognitive sciences or quantitative economics, records missing discipline/title metadata, and
titles matching human-subject risk terms. From 3,450 license-screened documents, the pipeline
retained 1,543 (910 EN, 633 FR); 1,848 were excluded on discipline, 7 on title signals, and 52 on
length or quality. It redacted 2,447 email, 135 phone and 2 IP-pattern occurrences. These are
pattern matches, not people or documents. Title/discipline screening and regex redaction cannot
prove the absence of sensitive information; a manual privacy review and maintainer source sign-off
are still pending. See [`training_record/open-weight-review.md`](training_record/open-weight-review.md).

Documents were split by SHA-256 ID hash into train/dev/sealed test before tokenizer training. The
tokenizer used train documents only. Counts are 26,387,903 train tokens, 1,473,762 dev tokens and
3,399,878 sealed-test tokens. The sealed text test has not been scored.

The historical HAL audit did not verify API pagination completeness. Rerun the corrected
audit before source sign-off; the recorded training corpus has not been revalidated by this release.

## Run record

| Field | Value |
|---|---|
| Initialization | From scratch |
| Seed | 20261022 |
| Optimizer steps | 805 |
| Tokens per optimizer step | 32,768 |
| Optimizer | AdamW |
| Schedule | Warmup-stable-decay; 80 warmup steps |
| Final train loss | 3.3636 |
| Final dev loss | 3.4377 |
| Training precision | bfloat16 |
| Code revision | `37560af4bfe868e0f467b16a15e4b199a76b3bd9`, working tree dirty |
| Hardware | NVIDIA RTX 4060 Laptop GPU, 8 GB |

## Same-harness benchmark results (exploratory)

Raw accuracy (`acc`) on the repository's pinned benchmark files. The exact benchmark split and
evaluator are recorded in `training_record/mc-six-benchmarks.json`. The benchmark items were
inspected after training and are not a sealed test for this run.

| Benchmark | Split | Items | Accuracy |
|---|---|---:|---:|
| ARC-Easy | test | 2,376 | 0.281 |
| HellaSwag | validation | 10,042 | 0.264 |
| Belebele EN | test | 900 | 0.254 |
| Belebele FR | test | 900 | 0.267 |
| ARC-Challenge | test | 1,172 | 0.192 |
| WinoGrande XL | validation | 1,267 | 0.485 |

The paired item bootstrap against the 0.935M TinyStories `tiny1m` reference gives an unadjusted
delta of +0.025 on ARC-Easy (95% interval [+0.002, +0.048]); the intervals for HellaSwag and both
Belebele splits include zero. These are post-hoc comparisons across 20 model-task comparisons,
without a family-wise correction. Training corpora, tokenizers, model histories and language mixes
differ, so the comparison is descriptive and does not establish superiority. Against the previous
SLM-Tavenel 1M FSC pilot, the intervals include zero for all four tasks. The benchmark chart in the
repository identifies this candidate separately.

This run does not replicate the old FSC training and does not test the effect of the attention gate.
One seed, a small science-focused corpus and a short token budget are not sufficient for a robust
general capability claim.

## Limitations and intended use

- Experimental artifact for compact-model and bilingual scientific-text research; not a general assistant.
- No instruction tuning or safety tuning; outputs may be inaccurate or harmful.
- The source and privacy reviews are not signed. Do not describe this candidate as rights-cleared.
- No text-level manual privacy review has been completed; metadata and regex controls reduce risk but do not eliminate it.
- Same-harness scores are exploratory; the sealed text test remains unscored.
- Exact 13-gram decontamination cannot rule out semantic, translated or prior exposure to benchmark content.

## Loading

The custom Transformers architecture requires `trust_remote_code=True`. Pin an immutable repository
revision before loading. KV-cache inference is not implemented.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "OWNER/slm-tavenel-1m-cc0-v2-candidate"
revision = "PIN_AN_IMMUTABLE_COMMIT"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)
```

## License and release status

The local candidate bundle contains MIT terms for weights and inference code plus provenance and
training scripts. The MIT license does not apply to source articles. The export is intentionally
not uploaded while privacy review and maintainer source sign-off remain pending.
