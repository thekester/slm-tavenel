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

# SLM-Tavenel 1M — CC0 retraining candidate

> **Local candidate only.** The model weights are assigned MIT for this candidate, but
> public upload is paused pending the documented privacy review and maintainer sign-off. This is a
> single-seed pilot, not confirmatory evidence or a general-purpose model.

## Summary

Decoder-only bilingual English/French model trained from scratch. The model has 1,181,920
parameters, a 8,192-token tokenizer and 1,024-token context. Architecture: 4 layers, grouped-query
attention, QK normalization, SwiGLU, RoPE, tied embeddings, and a sigmoid head-wise attention gate.

## Training data and provenance

The pilot used 3,450 document-level English/French records drawn from the `french_open_science`
subset of [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons),
pinned at revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. Each selected record was labelled CC0
in the corpus and matched CC0 across every version returned by the HAL records API. 33 ambiguous
or non-CC0 IDs were excluded. The corpus is scientific text, not representative general-domain
English or French. Author metadata is preserved as supplied; this pilot's selected records include
an author field, and no identities were inferred.

The pipeline applied normalization, length and quality filters, pattern-based PII redaction,
exact/MinHash deduplication, and exact 13-gram checks against six local benchmark sets. It retained
3,328 documents and redacted 5,682 pattern matches (5,140 emails, 537 phone numbers, 3 IBANs and
2 IP addresses). These are occurrences, not counts of people or documents. The 13-gram pass removed
no benchmark matches. Pattern matching cannot detect all personal or sensitive information; a
stratified manual privacy review is still pending. A broad title-only screen matched 130 retained
records against clinical, patient, trial, interview, cohort and case-report terms. This does not
mean all such records contain sensitive data, but it confirms health and human-subject studies are
present; the pilot weights include them. See
[`training_record/open-weight-review.md`](training_record/open-weight-review.md) in the candidate bundle; the full review record is in the repository.

Document split and new tokenizer were fixed before training. The tokenizer was trained on the
train split only. The train split contains 67,046,146 tokens (32,465,659 English and 34,580,487
French); dev contains 3,277,430 tokens; the sealed test contains 5,026,546 tokens and has not been
scored. The sealed test documents were not passed to the trainer or tokenizer.

The historical HAL audit did not verify API pagination completeness. Rerun the corrected
audit before source sign-off; the recorded training corpus has not been revalidated by this release.

## Run record

| Field | Value |
|---|---|
| Initialization | From scratch |
| Seed | 20261021 |
| Steps | 2,045 |
| Training tokens per optimizer step | 32,768 |
| Optimizer | AdamW |
| Learning rate | 3e-3 to 3e-4; warmup-stable-decay |
| Sequence length | 1,024 |
| Training precision | bfloat16 |
| Final training loss | 3.0272 |
| Final dev loss | 3.1132 |
| Run code | Commit `37560af4bfe868e0f467b16a15e4b199a76b3bd9`; source tree dirty |
| Hardware | NVIDIA RTX 4060 Laptop GPU, 8 GB |

## Same-harness benchmark results (exploratory)

Raw multiple-choice accuracy on the repository's pinned local files. These scores were computed
after the pilot and are descriptive; the benchmark items are not a sealed test for this run.

| Benchmark | Split | Items | Raw accuracy | Approx. 95% item interval |
|---|---|---:|---:|---:|
| ARC-Easy | test | 2,376 | 0.290 | 0.272–0.308 |
| HellaSwag | validation | 10,042 | 0.270 | 0.261–0.279 |
| Belebele EN | test | 900 | 0.284 | 0.255–0.314 |
| Belebele FR | test | 900 | 0.316 | 0.285–0.346 |
| ARC-Challenge | test | 1,172 | 0.200 | 0.177–0.223 |
| WinoGrande XL | validation | 1,267 | 0.501 | 0.473–0.529 |

The intervals are approximate normal intervals from the evaluator's item-level standard error;
they condition on this checkpoint and do not include training-seed variability. The same-harness
comparison to TinyStories tiny1m (0.896M) gives paired, unadjusted 95% bootstrap deltas of +0.034
on ARC-Easy and +0.010 on HellaSwag, with intervals [0.012, 0.056] and [0.002, 0.018]. Its
Belebele intervals include zero. Comparisons with the old SLM-Tavenel 1M/10M runs and Pythia/SmolLM2
are also descriptive: the data, tokenizer, model size and pretraining history are not matched.
The 10M comparison intervals include zero on all four shared benchmarks. Six tasks were inspected
without a family-wise multiplicity correction; no claim of downstream-task superiority is made.

The [comparison SVG](benchmark-comparison.svg) labels the new CC0
retraining candidate separately from the older SLM-Tavenel 1M pilot. Whiskers show ±1 standard
error across items, not uncertainty across training runs.

The pilot is one run and has not been replicated across training seeds. The code tree was dirty;
the exact file hashes and source snapshot are recorded in the local provenance bundle. Do not use
this run to claim a causal benefit from the attention gate or superiority over other SLMs.

## Intended use and limitations

- Reproducibility, compact-model, and bilingual science-text research.
- No instruction tuning or safety tuning; generations can be inaccurate or harmful.
- Small science-domain corpus and small token budget; not a general assistant.
- One seed and a dirty source revision; the test set remains sealed and no benchmark result is
  reported on this card.
- Exact lexical decontamination cannot rule out semantic or translated overlap, or the model's
  prior exposure through another source.
- CC0 metadata was checked against HAL's API; that check cannot prove the submitter's authority to
  apply CC0. Privacy and source rights review remain separate from the MIT weight license.

## Loading

This architecture uses custom Transformers code and requires `trust_remote_code=True`. Pin an
immutable repository revision before loading it. The current inference code does not implement a
KV cache.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "OWNER/slm-tavenel-1m-cc0-candidate"
revision = "PIN_AN_IMMUTABLE_COMMIT"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=True
)
```

## License and release

The candidate model weights are licensed under MIT, with the license text included in the local
bundle. The custom implementation code retains the repository's MIT license. This does not
license or redistribute the training articles. The
local bundle contains record-level IDs, HAL links, hashes, run logs and preparation manifests, but
no source article text. A local fixed-input parity check returned exact logits on CPU float32,
CUDA float32 and CUDA bfloat16; greedy generation appended all eight requested tokens. This is a
narrow inference smoke check, not a generation-quality evaluation. Public release is pending
privacy review, final source sign-off, final artifact validation and maintainer confirmation of
the intended repository/account. This card does not clear the older 1M or 10M checkpoints.
