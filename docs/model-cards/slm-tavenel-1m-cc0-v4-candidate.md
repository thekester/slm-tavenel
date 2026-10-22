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

# SLM-Tavenel 1M CC0 v4 pilot

> **Experimental open-weight candidate.** This is one exploratory training run. The release record does not claim that source-rights or privacy reviews are signed, and the sealed text test was not scored.

## Contents

- [Model summary](#model-summary)
- [Limitations and intended use](#limitations-and-intended-use)
- [Loading and generation](#loading-and-generation)
- [Training data and provenance](#training-data-and-provenance)
- [Training run](#training-run)
- [Evaluation](#same-harness-multiple-choice-scores-exploratory)
- [License and release status](#license-and-release-status)
- [Citation and project records](#citation-and-project-records)

## Model summary

SLM-Tavenel 1M CC0 v4 is a decoder-only English/French research model trained from scratch. It has 1,181,920 parameters (395,488 non-embedding), an 8,192-token byte-level BPE vocabulary and a 1,024-token context. Its four-layer architecture uses grouped-query attention, QK normalization, SwiGLU, RoPE and a sigmoid head-wise attention gate. It is a base language model, not an instruction-tuned chatbot.

## Limitations and intended use

This checkpoint is useful for inspecting a small bilingual training run and reproducing its recorded evaluations. It is not a general-purpose assistant. One seed and approximately 16M training tokens do not validate the attention-gate effect or establish broad capability. Scores below are exploratory and close to chance on several tasks. The source metadata and automated privacy screens do not establish legal rights or prove that all personal or sensitive information is absent. The model has no instruction or safety tuning; outputs may be inaccurate or harmful.

For the complete run record, consult `training_record/run_manifest.json`, `training_record/metrics.jsonl`, `training_record/observability.jsonl`, and `training_record/release-record.md`.

## Loading and generation

This custom Transformers implementation requires `trust_remote_code=True`. Pin an immutable repository revision before loading. KV-cache inference is not implemented.

```python
# Install the pinned dependencies from the model repository first:
# pip install -r requirements.txt
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "athenista/slm-tavenel-1m-cc0-v4-candidate"
revision = "PIN_AN_IMMUTABLE_COMMIT"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)

prompt = "La recherche scientifique"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=48, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training data and provenance

The source is the `french_open_science` subset of [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), pinned to revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. The selection contains 3,450 English/French records that the dataset labels CC0 and for which every version returned by the HAL records API also reports CC0 (1,767 EN and 1,683 FR). The API pagination count was checked. This is an independent metadata comparison, not proof that each submitter had authority to apply CC0. Missing or uncertain authorship metadata is reported as supplied; no author identity was inferred.

The v4 pipeline excluded 1,848 records on discipline metadata and 23 on title signals, then applied content-integrity, human-subject, quality, duplicate and 13-gram benchmark filters. The resulting corpus contains 1,229 documents (748 EN, 481 FR). PII rules redacted 2,579 email-shaped strings, 158 phone-shaped strings and 3 IP-pattern matches; five documents with residual email-shaped strings and two with replacement characters were excluded. A full-text check of the retained corpus found no replacement characters, email strings, configured human-subject/biometric terms or residual matches from the same PII redactor. A deterministic sample of 10 documents per language was inspected at excerpt level; the sampled excerpts showed public scholarly bylines, with no individual case narrative or non-public contact detail observed. This is a limited screen, not proof that all personal or sensitive information is absent. See `training_record/privacy-screen-report.json`, the HAL audit and processing decisions.

Training documents were separated by document ID before tokenizer training. The tokenizer used train documents only. Token counts are 15,976,924 train, 931,435 dev and 2,032,436 sealed-test tokens. The test text remains unscored and is not included in the bundle.

## Training run

| Field | Value |
|---|---|
| Initialization | From scratch |
| Seed | 421021 |
| Optimizer steps | 488 |
| Token presentations | 15,990,784 (approximately one pass over 15,976,924 unique train tokens) |
| Optimizer | AdamW |
| Schedule | Warmup-stable-decay; 48 warmup steps |
| Final training loss | 3.5985 |
| Final development loss | 3.6584 |
| Training precision | bfloat16 |
| Code revision | `02396acd15ef9bcdc453bcd53aade40819e2f69f`, working tree dirty |
| Hardware | NVIDIA RTX 4060 Laptop GPU, 8 GB |

Metrics, configuration, software versions, per-step observability and SHA-256 manifests are in `training_record/`.

## Same-harness multiple-choice scores (exploratory)

Raw accuracy from the repository evaluator and pinned benchmark files. These sets were inspected after training and are not a sealed test. For four-choice sets, chance is approximately 25%; WinoGrande has two choices and chance is 50%.

| Benchmark | Split | Items | Accuracy |
|---|---|---:|---:|
| ARC-Easy | test | 2,376 | 0.282 |
| ARC-Challenge | test | 1,172 | 0.208 |
| HellaSwag | validation | 10,042 | 0.260 |
| Belebele EN | test | 900 | 0.272 |
| Belebele FR | test | 900 | 0.278 |
| WinoGrande XL | validation | 1,267 | 0.488 |

For context, the 0.935M TinyStories `tiny1m` reference was evaluated with the same repository evaluator. The unadjusted ARC-Easy difference is +0.027 (95% item-bootstrap interval [+0.004, +0.049]); intervals for HellaSwag and both Belebele splits include zero. This post-hoc comparison spans multiple tasks without family-wise correction. The reference differs in training data, tokenizer, language mix and token budget. No superiority or causal claim follows. Historical SLM-Tavenel pilot results remain in the repository experiment archive and are not part of this release.

## License and release status

The candidate weights and inference code are marked MIT; the source articles are not included. The source notice and record-level HAL metadata audit are included. The release record keeps source-rights and manual privacy review flags unsigned. Dataset and HAL metadata report CC0 for selected records, but those records have not received independent legal clearance; the metadata match does not establish submitter authority or resolve every rights question. The artifact's availability must not be read as source-rights or privacy certification.

## Citation and project records

No paper or DOI has been assigned to this pilot. If citing the model, include its immutable Hub revision once one is released and cite the [SLM-Tavenel repository](https://github.com/thekester/slm-tavenel) and the release record shipped with the model. Historical candidate results are maintained in the repository experiment archive and are not checkpoints in this release.
