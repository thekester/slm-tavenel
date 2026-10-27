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
---

# SLM-Tavenel grown 25M | EXP-0009

> **Completed experimental base model.** This 25.72M-parameter checkpoint was grown from the trained SLM-Tavenel 10M model and then trained for 15,000 additional steps. It is one seed and one initialization path. Rights and privacy review gaps for the mixed-license source corpus are disclosed below.

The project offers the model parameters, inference/training code, and project-authored Data
Information under MIT terms. This release is described as an **OSAID 1.0 aligned research
artifact by project self-assessment**; this is not OSI certification. The assessment, including
the exact limits and rationale, is in
`docs/reviews/OSAID-1.0-EXP0009-25M-ASSESSMENT.md`. The upstream data is publicly available at the
pinned dataset revision and remains under its own per-record terms. This model package does not
relicense those records.

| Model facts | |
|---|---|
| Parameters | 25,723,648 |
| Architecture | 20-layer decoder-only transformer, head-wise attention gate |
| Context | 1,024 tokens |
| Tokenizer | 8,192-token byte-level BPE |
| Additional training | 15,000 steps, seed 1234, 491,520,000 token positions |
| Total lineage budget | Approximately 983,040,000 token positions, including the trained 10M ancestor |
| Final train / validation loss | 2.14016 / 2.21816 |
| Held-out validation BPB, EN / FR | 1.1171 / 1.0367 |

## Architecture

![SLM-Tavenel decoder block and the 1M, 10M and grown 25M model dimensions](architecture.svg)

The grown 25M checkpoint has twenty decoder blocks, width 256, four query heads and two key/value heads, and a SwiGLU width of 1,280. It uses pre-RMSNorm, grouped-query causal attention, QK normalization, RoPE, a head-wise sigmoid gate, SwiGLU and residual connections. Its tokenizer has 8,192 byte-level BPE tokens and its context window is 1,024 tokens. This checkpoint was created by expanding the trained 10M model and then continuing training; it was not trained from scratch.

## Results at a glance

The 25M checkpoint and references were scored with this repository's evaluator on the same pinned benchmark files where available. The graph compares raw accuracy and shows one-standard-error bars across items. Those bars condition on each checkpoint and do not measure variation across training runs.

![SLM-Tavenel 1M, 10M and grown 25M benchmark results compared with SLM references](scaling-benchmark-comparison.svg)

| Model | Parameters | ARC-Easy | ARC-Challenge | HellaSwag | Belebele EN | Belebele FR | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|---:|
| SLM-Tavenel 1M EXP-0010 | 1,181,920 | 0.280 | 0.196 | 0.259 | 0.264 | 0.266 | 0.493 |
| SLM-Tavenel 10M EXP-0009 | 9,978,368 | 0.306 | 0.184 | 0.275 | 0.296 | 0.319 | 0.511 |
| SLM-Tavenel grown 25M EXP-0009 | 25,723,648 | 0.319 | 0.180 | 0.273 | 0.283 | 0.309 | 0.519 |
| TinyStories tiny1m | 935,040 | 0.256 | 0.189 | 0.260 | 0.274 | 0.290 | 0.518 |
| Pythia-14M | 14,067,712 | 0.330 | 0.161 | 0.262 | 0.256 | 0.309 | 0.520 |
| Pythia-31M | 30,494,720 | not scored | 0.167 | not scored | not scored | not scored | 0.505 |
| SmolLM2-135M | 134,515,008 | 0.643 | 0.278 | 0.356 | 0.358 | 0.314 | 0.530 |

### Why the larger model does not have reliably higher benchmark accuracy

The 25M checkpoint has a higher raw score than 10M on ARC-Easy and WinoGrande, but lower raw scores on ARC-Challenge, HellaSwag and both Belebele splits. For every task, the paired 95% item-bootstrap interval for the 25M-minus-10M accuracy difference includes zero: ARC-Easy **[−0.004, +0.030]**, ARC-Challenge **[−0.023, +0.015]**, HellaSwag **[−0.006, +0.002]**, Belebele EN **[−0.033, +0.009]**, Belebele FR **[−0.032, +0.011]**, WinoGrande **[−0.018, +0.032]**. The item intervals condition on these checkpoints and do not include training-seed variation.

This does not mean added capacity had no effect. The 25M model has lower held-out BPB than 10M on the FSC validation text, but this measures next-token prediction on that text, not broad task accuracy. The 25M model also inherits the 10M initialization and receives another 491.52M token positions, so this is not a controlled test of parameter count alone. Both models use one training trajectory, and external references differ in data, tokenizer and compute. These experiments cannot yet tell which factor explains the mixed benchmark scores.

### Held-out English and French bits per byte

Lower is better. These point estimates use the same fixed FSC validation texts and tokenizer for the three SLM-Tavenel checkpoints. The paired 10M-versus-25M document-bootstrap intervals below quantify validation-document sampling conditional on these checkpoints; they do not quantify training-seed uncertainty.

| SLM-Tavenel checkpoint | English BPB | French BPB |
|---|---:|---:|
| 1M EXP-0010, selected | 1.4904 | 1.3676 |
| 10M EXP-0009 | 1.1755 | 1.0897 |
| grown 25M EXP-0009 | **1.1171** | **1.0367** |

![Paired document-bootstrap intervals for held-out BPB differences](docs/exp0009-25m-bpb-document-bootstrap.svg)

Positive values are 10M BPB minus grown-25M BPB, so they favor the 25M point estimate. Bootstrap resampling was within language, with the same sampled documents used for both checkpoints.

| Paired comparison | Documents | BPB difference | 95% paired document-bootstrap interval |
|---|---:|---:|---:|
| English, 10M minus 25M | 685 | +0.0585 | [+0.0570, +0.0602] |
| French, 10M minus 25M | 678 | +0.0530 | [+0.0520, +0.0540] |
| Equal-language mean | 685 EN; 678 FR | +0.0557 | [+0.0548, +0.0567] |

The 25M point estimate is lower by 0.0585 BPB in English and 0.0530 BPB in French than the 10M checkpoint. Because 25M inherits the trained 10M initialization and receives another 491.52M token positions, this does not isolate parameter count from initialization or extra training exposure.

This is not a parameter-count-only comparison: the 25M model inherits the 10M training history and receives another 491.52M token positions. The single run cannot estimate seed variation. External models use different corpora, tokenizers, compute and training budgets, and may have seen benchmark material. Results are descriptive and exploratory, not a general ranking.

## Intended use

Research and education on small-model training, bilingual text modeling, reproducibility and observability. This is a base model, not an instruction-tuned assistant or a reliable scientific advisor.

## Load with Transformers

The model uses custom Transformers code and requires `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "athenista/slm-tavenel-25m-exp0009"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

## Training, observability and reproducibility

The run resumed from checkpoint 3,000 in the recorded Python 3.11.9 / PyTorch 2.10.0+cu130 environment after a mismatched-runtime attempt was stopped and archived. Recomputed validation at the resume point and the step-3,500 losses matched the recorded trajectory. The configuration is not bitwise deterministic, and the original worktree was dirty, so exact source-state reproduction is not claimed. Per-step losses, validation, gradient, optimizer and observability records are included in `training_record/`. A full tracked source snapshot at the recorded Git commit, including the CLI, is in `training_record/reproduction_source/`; the run configuration and exact growth/training settings are separately recorded in `run_manifest.json`. The [resume diagnostic](docs/reviews/EXP-0009-25M-RESUME-DIAGNOSTIC-2026-10-22.md) describes the archived attempt.

## Data, licenses and privacy

Training used the `french_open_science` subset of [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. The retained 31,417-row manifest preserves source identifiers, source-declared license labels, titles, declared authors, DOI fields and hashes. No source text is included. A post-training HAL metadata audit found matching main/document-file labels for 31,262 source IDs; 72 explicit mismatches, 3 conflicts, 24 missing file-license fields and 56 not-found IDs remain unresolved. A targeted recheck on 23 October found no changes. Of 75 DOI-bearing unresolved records, Crossref metadata included the dataset-declared license URL for 30, listed only a different license or policy URL for 15, had no license field for 23, and failed for 7. These metadata do not establish rights-holder authority or identify the exact historical file version used. The reports are in `training_record/hal-file-license-audit-2026-10-22.json`, `training_record/hal-file-license-recheck-2026-10-23.json`, and `training_record/unresolved-crossref-license-discovery-2026-10-23.json`. A conservative screen now excludes all 155 unresolved records from any future release-oriented training corpus; the per-record list is `training_record/conservative-source-exclusions.jsonl`. This policy does not remove data from this already trained checkpoint or clear the other 31,262 records whose depositor authority and exact historical file versions were not independently verified. See [the conservative license policy](docs/reviews/EXP-0009-CONSERVATIVE-LICENSE-POLICY.md).

The later 23 October screen of the processed corpus found four documents with IPv4-shaped patterns, 54 replacement-character documents, and 20,808 records with at least one broad review signal; the v4 title/discipline risk filters were not part of the EXP-0009 training pipeline. A focused triage found the four IP-pattern documents are networking/security teaching or research examples, but this does not clear the broader privacy review. See `training_record/fsc-v1-privacy-signal-scan-2026-10-23.json`, the [source and privacy review record](docs/reviews/EXP-0009-25M-SOURCE-PRIVACY-REVIEW.md), and the [detailed rights and privacy triage](docs/reviews/EXP-0009-25M-DEEP-RIGHTS-PRIVACY-TRIAGE-2026-10-23.md). These are automated triage results, not a human review. Human source-rights and privacy sign-off remain pending. No rights clearance or absence-of-memorization claim is made. The MIT terms for project-authored code and weights do not relicense the source records or guarantee absence of third-party claims.

## Evaluation files

`training_record/` includes the final checkpoint provenance, source manifest, data and token manifests, training metrics, validation, benchmark outputs, paired item-bootstrap comparisons, model code, configurations, audit report, and SHA-256 inventory. Benchmark intervals quantify item-sampling uncertainty for these checkpoints only; they do not estimate training variability. Exact-overlap contamination checks do not rule out paraphrase, translation or prior exposure.

## Citation and release status

Cite this research artifact as SLM-Tavenel EXP-0009 grown 25M, seed 1234, and identify the checkpoint SHA-256 from `provenance.json`. It is available at [athenista/slm-tavenel-25m-exp0009](https://huggingface.co/athenista/slm-tavenel-25m-exp0009). The project's OSAID self-assessment is not certification by OSI. The source-rights and privacy review fields remain false; this release is not a legal clearance or privacy approval.
