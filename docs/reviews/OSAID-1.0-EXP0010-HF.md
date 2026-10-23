# SLM-Tavenel EXP-0010 — OSAID 1.0 self-assessment

**Project assessment:** the prepared EXP-0010 package meets the core OSAID 1.0 requirements for a
model and its weights, based on the artifacts and terms listed below. This is the project's own
assessment, not OSI approval or certification. The package is still local; it becomes a public Open
Source AI offering only when these files and terms are made publicly available.

## Core requirements

| Requirement | Assessment | Evidence in this package |
|---|---|---|
| Use, study, modify and share | Met by the proposed package terms | `LICENSE` applies MIT terms to the weights, code and data information; no additional model-specific use restriction is added. MIT is OSI-approved. |
| Sufficient Data Information, under OSI-approved terms | Met | `training_record/data-card.md`, `data-pipeline.yaml`, `data-decisions.jsonl`, `data-manifest.json`, `source-record-manifest.jsonl`, `hal-license-audit.json`, `split-manifest.json`, `tokenizer-manifest.json` and `token-manifest.json` identify the sources, scope, selection, filtering, processing, languages, splits and hashes. The package declares MIT for Data Information. |
| Public or third-party data and where to obtain it | Met by record-level disclosure | The data card pins `PleIAs/French-Science-Commons` revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`, subset `french_open_science`. The 1,229 retained source records have IDs and HAL URIs in `source-record-manifest.jsonl`. Source article text remains at the public upstream repositories and is not mirrored here. No paid or nonpublic source is declared. |
| Complete preparation, training, validation/testing and inference code | Met | The package contains source snapshots, data acquisition and filtering scripts, split/tokenization code, training and evaluation scripts, configuration, run records and the Transformers inference implementation. These project files are MIT-licensed. |
| Architecture and parameters | Met | `hf_model.py`, `hf_config.py`, `config.json` and `model.safetensors` provide the model implementation and weights under MIT. The provenance inventory covers every packaged file other than the inventory file itself. |

The source dataset and HAL metadata label the selected records CC0. The audit compares 3,450
records with all versions returned by the paginated HAL API; this is a record-level account of
source metadata, not independent verification of each depositor's identity or authority. That
uncertainty is disclosed rather than hidden. For Data Information, the corpus is described as
publicly obtainable third-party material with record-level locations; the raw text is not
redistributed in this model package.

The recorded direct training dependencies are PyTorch 2.10.0 (BSD-style), NumPy 2.4.4 (BSD),
PyYAML 6.0.2 (MIT), Tokenizers 0.22.2 (Apache-2.0), Safetensors 0.8.0 (Apache-2.0) and
Transformers 4.57.6 (Apache-2.0). Their pinned requirements and upstream license sources are
recorded in `requirements-training.txt` and `requirements.txt`. CUDA 13.0 was used for the
recorded run; the training code selects CPU when CUDA is unavailable, so CUDA is an optional
acceleration backend rather than a required proprietary model component.

## Signatures and certification

Source-rights review signatures, a full privacy attestation and a project publication approval
are **not requirements listed by OSAID 1.0**. They may be maintained as separate project
governance records, but their unsigned state must not be described as an OSAID failure. The OSI
FAQ also treats safety and risk practices as separate from the definition and states that OSI does
not certify individual AI systems. The project therefore makes a self-assessment and does not
claim OSI endorsement or certification.

The package is prepared locally and has not been uploaded. After public release, archive the
immutable Hub revision and hashes. Keep the disclosed source-metadata and privacy limitations in
the public model card.

## References

- [Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition)
- [OSAID FAQ](https://opensource.org/ai/faq)
- [OSI checklist](https://opensource.org/ai/checklist), which OSI describes as a learning tool rather than an operational evaluation manual
- [OSI-approved MIT license](https://opensource.org/license/MIT)
- [French-Science-Commons pinned dataset](https://huggingface.co/datasets/PleIAs/French-Science-Commons/tree/bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156)
- [HAL Search API documentation](https://api.archives-ouvertes.fr/docs/search)
- [PyTorch 2.10.0 license](https://github.com/pytorch/pytorch/blob/v2.10.0/LICENSE), [NumPy 2.4.4 license](https://github.com/numpy/numpy/blob/v2.4.4/LICENSE.txt), [PyYAML 6.0.2 license](https://github.com/yaml/pyyaml/blob/6.0.2/LICENSE), [Tokenizers 0.22.2 license](https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE), [Safetensors 0.8.0 license](https://github.com/safetensors/safetensors/blob/v0.8.0/LICENSE), [Transformers 4.57.6 license](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE)
