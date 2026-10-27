---
language:
- en
- fr
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: athenista/slm-tavenel-25m-exp0009
datasets:
- PleIAs/French-Science-Commons
- ashtok897/european-hplt-v1
tags:
- slm-tavenel
- experimental
- research-artifact
- base-model
- hplt-domain-adaptation
---

# SLM-Tavenel 25M EXP-0032: HPLT 10% continuation

**Experimental bilingual base model, adapted to an HPLT-derived text mixture.** This checkpoint continues the published SLM-Tavenel 25M EXP-0009 model for 1,500 steps on a frozen 90% FSC / 10% HPLT-derived token mixture. It is a separate variant; it does not replace the EXP-0009 checkpoint.

The source-rights review for the FSC lineage and page-level rights and privacy review for the HPLT-derived data remain unresolved. The model is published with that limitation disclosed for research traceability. MIT applies to this project's code and model artifact; it does not relicense either upstream dataset or establish legal clearance. No OSI certification, rights clearance, privacy approval, or absence-of-memorization claim is made.

| Property | Value |
|---|---|
| Parameters | 25,723,648 |
| Architecture | 20-layer decoder-only transformer with head-wise attention gate |
| Width / attention | 256 / 4 query heads / 2 key-value heads |
| Context | 1,024 tokens |
| Tokenizer | 8,192-token byte-level BPE; inherited unchanged from EXP-0009 |
| Parent | [SLM-Tavenel 25M EXP-0009](https://huggingface.co/athenista/slm-tavenel-25m-exp0009) |
| Continuation | 1,500 steps, 49,152,000 token positions, AdamW, seed 20261028 |
| HPLT mixture | Approximately 10% of continuation tokens; English and French balanced within the candidate subset |
| Weights | `model.safetensors`; SHA-256 in `provenance.json` |

## What changed and what did not

EXP-0032 was a paired, same-seed continuation comparison from the same EXP-0009 checkpoint. The treatment used the frozen 90/10 FSC/HPLT-derived stream; its control continued on FSC only. Both arms used the same tokenizer, model, initialization, optimizer, seed and token budget. The HPLT test split was held out from training and scored after the recipe was frozen.

The treatment reduced BPB on the held-out HPLT test documents by 0.0732 for English and 0.0650 for French relative to the matched FSC-only control. FSC validation BPB increased slightly. All six paired multiple-choice accuracy intervals include zero. These results support an HPLT-domain adaptation effect for this pair, not a general capability or accuracy improvement.

| Evaluation | Language | FSC-only control | HPLT 10% treatment | Treatment − control | 95% paired document interval |
|---|---|---:|---:|---:|---:|
| Held-out HPLT test BPB | English | 1.54603 | **1.47279** | −0.07323 | [−0.07596, −0.07037] |
| Held-out HPLT test BPB | French | 1.35556 | **1.29059** | −0.06497 | [−0.06960, −0.05989] |
| FSC validation BPB | English | 1.10299 | 1.10421 | +0.00122 | [+0.00100, +0.00143] |
| FSC validation BPB | French | 1.02416 | 1.02493 | +0.00077 | [+0.00055, +0.00098] |

Lower BPB is better. HPLT test intervals resample documents within language (13,870 English and 4,712 French); they quantify document-sample uncertainty conditional on this pair, not variation across training seeds.

![Paired HPLT development and held-out test BPB differences for the 90/10 data mixture](hplt-bpb-deltas.svg)

### Multiple-choice benchmarks

Raw accuracy from the same pinned benchmark items and evaluator. Every paired item-bootstrap 95% interval includes zero.

| Task | FSC-only | HPLT 10% | Difference | Paired 95% item interval |
|---|---:|---:|---:|---:|
| ARC-Easy | 0.3161 | 0.3194 | +0.0034 | [−0.0088, +0.0152] |
| ARC-Challenge | 0.1877 | 0.1869 | −0.0009 | [−0.0154, +0.0145] |
| HellaSwag | 0.2757 | 0.2736 | −0.0022 | [−0.0054, +0.0010] |
| Belebele English | 0.2811 | 0.2789 | −0.0022 | [−0.0144, +0.0111] |
| Belebele French | 0.3156 | 0.3078 | −0.0078 | [−0.0211, +0.0056] |
| WinoGrande | 0.5185 | 0.5170 | −0.0016 | [−0.0197, +0.0166] |

This is one fresh seed pair. It does not estimate training-seed variation, demonstrate a broad downstream improvement, or establish a scaling law. The HPLT test split was used once for the predeclared comparison and should not guide further tuning.

## Data provenance and review status

The model inherits the EXP-0009 FSC training lineage. For this continuation, 90% of token positions came from the FSC scaling stream and approximately 10% from a deterministic English/French HPLT-derived candidate stream. The HPLT source was [ashtok897/european-hplt-v1](https://huggingface.co/datasets/ashtok897/european-hplt-v1/tree/51edb798468aa71b1000e950d3ab43bb435f72cd), pinned to revision `51edb798468aa71b1000e950d3ab43bb435f72cd`. Its pinned card declares CC0-1.0 inherited from HPLT v3. HPLT describes the collection as web crawls derived from Internet Archive and Common Crawl and does not establish rights over every original page. The declared label is provenance, not independent evidence of each page's rights.

The parent FSC lineage used [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), `french_open_science`, revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`, with source-declared CC0-1.0, CC-BY-4.0 and Licence-Ouverte-2.0 records. Its remaining audit discrepancies, depositor-authority questions and unsigned human privacy review are described in the parent's card and provenance files.

The package includes the pinned mixture manifest, source revisions, selected-input hashes, processing and training settings, aggregate paired results, and SHA-256 file inventory. It excludes source text, token shards, benchmark prompts and raw per-example predictions. Provenance metadata improves traceability; it does not establish rights or privacy clearance. No independent rights or privacy sign-off is represented by this release.

## Architecture and inference

![SLM-Tavenel transformer architecture](architecture.svg)

This checkpoint uses custom Transformers code. Loading it requires `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "athenista/slm-tavenel-25m-exp0032-hplt-10pct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

## Reproduction and files

The `training_record/` directory includes EXP-0032's experiment report, training configuration, sanitized run-manifest summaries, step logs, mixture manifest, aggregate paired BPB results and checkpoint provenance. The model is not an instruction-tuned assistant.

For the broader 1M/10M/25M and public-reference benchmark comparison, see [the comparison chart](scaling-benchmark-comparison.svg). The source repository contains the preparation and evaluation scripts. No peer-reviewed paper or DOI has been assigned.

## Citation

Cite this as **SLM-Tavenel 25M EXP-0032 HPLT 10% continuation, seed 20261028**, and include the Hugging Face revision and `provenance.json` checkpoint SHA-256. Link to the [EXP-0032 report](training_record/EXP-0032-results.md) and identify the results as exploratory HPLT-domain adaptation evidence.
