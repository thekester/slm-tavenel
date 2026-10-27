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

# SLM-Tavenel 10M | EXP-0009 FSC scaling pilot

> **Experimental base model.** One completed 10M gated-attention checkpoint, trained from scratch for 15,000 steps on a bilingual science-text pilot. Project weights and code are offered under MIT. The source corpus mixes CC0, CC-BY-4.0 and Licence-Ouverte-2.0 records; the declared source licenses and unresolved rights/privacy checks are disclosed below. This is not an OSI certification or a rights clearance.

| Model facts | |
|---|---|
| Parameters | 9,978,368 |
| Languages | English and French |
| Architecture | 10-layer decoder-only transformer, head-wise attention gate |
| Context | 1,024 tokens |
| Tokenizer | 8,192-token byte-level BPE |
| Training | 15,000 steps, seed 1234, AdamW, bfloat16 |
| Training token positions | 491,520,000 |
| Final train / validation loss | 2.29647 / 2.33488 |

## Architecture

![SLM-Tavenel decoder block and the 1M, 10M and grown 25M model dimensions](architecture.svg)

This checkpoint has ten decoder blocks, width 256, four query heads and two key/value heads, with a SwiGLU width of 768. Each block uses pre-RMSNorm, grouped-query causal attention, QK normalization, RoPE, a head-wise sigmoid gate on attention outputs, and a SwiGLU feed-forward sublayer with residual connections. It shares the 8,192-token byte-level BPE vocabulary and 1,024-token context with the other shown checkpoints. The 25M model was grown from this trained 10M model by expanding depth and feed-forward width.

## Results at a glance

The checkpoint scored with the same local evaluator and pinned benchmark files used for the listed SLM references. Accuracy is close to the four-choice chance level on these tasks. The external comparisons are descriptive only because the models have different corpora, tokenizers, training budgets and possible benchmark exposure.

![Training and validation loss across the completed 10M run](training-curve.svg)

*Figure 1. Training and validation loss over optimizer steps.*

| Model | Parameters | ARC-Easy | HellaSwag | Belebele EN | Belebele FR |
|---|---:|---:|---:|---:|---:|
| SLM-Tavenel 10M | 9.98M | 0.306 | 0.275 | 0.296 | 0.319 |
| SLM-Tavenel 1M EXP-0010 (seed 421101) | 1.18M | 0.280 | 0.259 | 0.264 | 0.266 |
| TinyStories tiny1m | 0.935M | 0.256 | 0.260 | 0.274 | 0.290 |
| Pythia-14M | 14.1M | 0.330 | 0.262 | 0.256 | 0.309 |
| SmolLM2-135M | 135M | 0.643 | 0.356 | 0.358 | 0.314 |

![SLM-Tavenel 10M and same-harness SLM reference scores](benchmark-comparison.svg)

*Figure 2. Raw accuracy. Whiskers show plus or minus one standard error across questions, conditional on each checkpoint. They do not measure seed variation.*

The chart below places the three SLM-Tavenel sizes together, with the complete model names on the plot. In the paired 10M-versus-grown-25M comparison, all six 95% item-bootstrap intervals include zero. The larger checkpoint therefore has no resolved accuracy advantage on these six tasks. Its FSC validation BPB is lower, which is a separate, text-modeling result. Because the 25M model inherits the trained 10M weights and gets another 491.52M token positions, the current runs cannot separate the effect of size from extra training or initialization. One run at each size also cannot measure run-to-run training variation.

![SLM-Tavenel 1M, 10M and grown 25M compared with public SLM references.](scaling-benchmark-comparison.svg)

## Intended use

This base model is intended for research and education about small-model training, bilingual text modeling, reproducibility and training observability. It is not instruction-tuned, is not a conversational assistant, and should not be used as a reliable factual or scientific advisor.

## Load with Transformers

The model uses custom Transformers code, requires `trust_remote_code=True`, and does not implement a KV cache.

```bash
pip install -r requirements.txt
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "athenista/slm-tavenel-10m-exp0009"
revision = "9818b706a2a2467efb22848967a5842871f7422c"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)
prompt = "La recherche scientifique"
inputs = tokenizer(prompt, return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=48, do_sample=False)[0]))
```

## Training and observability

The run used a single seed (1234), an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB, a 1,024-token sequence length, micro-batch size 16, two gradient-accumulation steps and AdamW. It processed 491,520,000 token positions over 15,000 steps. The selected checkpoint contains 9,978,368 parameters. Per-step metrics, validation losses and observability records are included in `training_record/`.

The recorded training worktree was dirty. The package includes the current model implementation, architecture and training configuration, run manifest and logs, but the original uncommitted source diff was not captured at run time. Exact byte-for-byte reproduction of the training code state is therefore not claimed.

## Data and source attribution

Training data came from the `french_open_science` subset of [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), pinned to `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. The processing pipeline started with 43,011 records and retained 31,417 after quality, PII-pattern, exact-duplicate and near-duplicate processing. The retained source metadata reports 4,528 records labeled CC0-1.0, 26,368 CC-BY-4.0, and 521 Licence-Ouverte-2.0. The record-level metadata-only manifest has one row per retained document and lists titles, declared authors, identifiers, DOI fields, source IDs, declared labels and hashes. All 31,417 rows have non-empty title and author fields; 13,200 do not provide a DOI, and those omissions are left blank. The fields are supplied metadata, not independent verification of authorship or rights ownership. Source texts are not included in this repository.

CC-BY-4.0 and Licence-Ouverte-2.0 require attribution. `ATTRIBUTION.md` and `training_record/source-record-manifest.jsonl` provide source-level attribution information to support tracing. The metadata label does not independently establish that each submitter had authority to apply it. The project has not independently verified every originating publisher's terms; no blanket rights clearance is claimed. Dataset and HAL metadata also do not prove that every record is free of personal or sensitive information. Pattern-based redaction and screening were limited, not a complete privacy audit.

### HAL file-license metadata cross-check (22 October 2026)

After training, the 31,417 retained HAL identifiers were queried against HAL's current `licence_s` (main file) and `fileLicenses_s` (document files) fields. The fields and their semantics are documented in [HAL's file-license notice](https://www.ccsd.cnrs.fr/en/2026/02/no-hal-deposit-without-licence-the-essential-information-you-need-to-know/) and the [HAL Search API documentation](https://api.archives-ouvertes.fr/docs/search). All returned HAL versions were compared with the license label frozen in the French-Science-Commons training manifest. The check found 31,262 records whose returned versions all matched, 72 with explicit license mismatches, 3 with conflicting license declarations, 24 with missing file-license metadata, and 56 identifiers not found by the query. The mismatching, incomplete and missing records remain unresolved; this audit does not establish which declaration was correct when the dataset snapshot was assembled or when training occurred. The full metadata-only results and audit script are in `training_record/hal-file-license-audit-2026-10-22.json` and `training_record/scripts/audit_fsc_mixed_hal_files.py`. For each returned HAL record, the report retains `version_i`, URI, document type, main-file, document-file and annex-file license fields, plus the corpus-supplied title and declared author. It contains no source text. HAL license fields are declarations and do not prove depositor authority, publisher rights or privacy status. This post-training metadata check is not a legal clearance.

## Limitations

- One seed and one completed run; no training-seed uncertainty estimate.
- Exploratory benchmark accuracy is near chance on several tasks and does not establish general model quality.
- The training worktree was dirty and the exact uncommitted diff is unavailable.
- A post-training HAL metadata audit found unresolved license-label discrepancies or missing records for 155 of 31,417 source identifiers; see the dated audit report.
- Source license labels, attribution obligations, submitter authority and privacy risks remain incompletely reviewed.
- The source article text is not redistributed. MIT applies to the project code and model artifact as offered by the project; it does not relicense source records or guarantee the absence of third-party claims.

## Reproducibility files

`training_record/` contains the checkpoint provenance, run manifest, train/eval/observability logs, source registry, processing manifest and decisions, tokenizer and token-shard manifests, source-record metadata manifest, benchmark score record, model implementation, and configurations. The SHA-256 inventory is in `provenance.json`.

## License and citation

Model weights, project code and included Data Information are offered under MIT. Source records remain subject to their separately declared licenses. Cite this model as SLM-Tavenel EXP-0009, seed 1234, and include the immutable Hugging Face revision `9818b706a2a2467efb22848967a5842871f7422c`. The experiment report and protocol are included in `training_record/EXP-0009-fsc-scaling-and-references.md`.
