# SLM-Tavenel 1M CC0 v4 training-data card

## Summary

This card describes the data information for the SLM-Tavenel 1M CC0 v4 experimental model. It
does not redistribute the source articles or claim that the source rights have received legal
clearance. The source dataset labels the selected records CC0, and the HAL metadata audit found
matching CC0 labels for all returned versions; those records remain subject to the limitations
below.

## Source and version

- Dataset: [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons)
- Subset: `french_open_science`
- Pinned dataset revision: `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`
- Source-level audit: all 3,450 English/French selected records had CC0 labels in every version
  returned by the paginated HAL records API snapshot dated 2026-10-21 (1,767 English and 1,683
  French).
- Independent reproduction: `training_record/audit_fsc_cc0_hal.py` and
  `training_record/hal-license-audit.json`.

## Selection and processing

The candidate selection began with those 3,450 records. The v4 pipeline excluded 1,848 based on
discipline metadata and 23 based on title signals. Content-integrity, human-subject, quality,
length, duplicate, and benchmark-overlap filters then retained 1,229 documents: 748 English and
481 French. All filtering decisions and source identifiers are recorded in
`training_record/data-decisions.jsonl`; the pipeline and output manifests are included alongside
the model.

Documents were split by document ID before tokenizer training. The tokenizer was fit on train
documents only. The final split has 15,976,924 train tokens, 931,435 development tokens, and
2,032,436 sealed-test tokens. The text test split has 69 English and 47 French documents and was
not used for training, tokenizer fitting, or training decisions. It is not included in the model
bundle.

## Privacy and content checks

The pipeline redacted 2,579 email-shaped strings, 158 phone-shaped strings, and 3 IP-pattern
matches. It excluded five documents with residual email-shaped strings and two containing Unicode
replacement characters. A full-text scan of retained documents found no configured replacement
characters, residual email patterns, or configured human-subject/medical/biometric risk terms.
A deterministic screen inspected short excerpts from 10 documents per language. These are
automated and limited-sample checks, not a full manual review or a guarantee that the corpus or
weights contain no personal, sensitive, or memorized information. The manual privacy attestation
is unsigned.

## Licensing and rights limits

The dataset and HAL records report CC0 for the selected source rows. The audit verifies that the
metadata agrees across the pinned dataset and the returned HAL versions; it cannot verify who
submitted each record or whether the submitter had authority to apply CC0. Some author or rights
metadata may be missing or incomplete as supplied. The package reports that information as
provided and does not infer missing authorship.

The source articles are not redistributed in the model bundle. The MIT license applies to the
model weights and included code; it is not a legal conclusion that all source rights are cleared
or that no third-party claim could affect a downstream use of the weights. The data information,
selection recipe, and audit reports are included for traceability. Source-rights review remains
unsigned.

## Reuse and reproducibility

The original dataset can be obtained from the pinned dataset repository revision above. The bundle
contains a record-level manifest with each retained record ID, source URI, declared license,
language, split, and pre/post-processing text hashes; it also contains the HAL metadata audit,
filtering decisions, split and token manifests, processing code, and SHA-256 hashes. It
intentionally omits source article text. The data-information files and project code are covered
by the package's MIT license; this does not relicense the original articles. The manifest lets a
reader inspect the stated selection and locate the same publicly available source records, subject
to their original terms and availability.
