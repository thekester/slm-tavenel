# SLM-Tavenel EXP-0010 training-data card

## Summary

This card describes the training-data source and processing for the SLM-Tavenel 1M EXP-0010
replication. It records source metadata and the data preparation recipe. It does not redistribute
the source articles or certify the rights attached to each article.

## Source and version

- Dataset: [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons)
- Subset: `french_open_science`
- Pinned dataset revision: `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`
- A paginated HAL API metadata snapshot dated 2026-10-21 reported CC0 labels for all 3,450 selected English/French source records across the returned versions: 1,767 English and 1,683 French.
- The metadata audit does not verify who submitted each record or whether they had authority to apply CC0. Source-rights review remains unsigned.

## Selection, processing and split

The processing pipeline excluded 1,848 records using discipline metadata and 23 using title signals. Integrity, human-subject, quality, length, exact/MinHash deduplication and benchmark-overlap filters produced a retained set of 1,229 documents: 748 English and 481 French. The exact rules, source record IDs and decisions are in `training_record/data-pipeline.yaml`, `training_record/data-decisions.jsonl` and `training_record/source-record-manifest.jsonl`.

The processed-data manifest records a dirty repository worktree at preprocessing time. Before packaging, the pipeline was replayed from the recorded inputs and configuration into a separate scratch directory. The resulting retained-document and filtering-decision files matched the original SHA-256 hashes exactly, as did the processing statistics. The replay configuration, manifest and comparison report are included. This demonstrates that the included pipeline can reproduce the prepared data; the original dirty-worktree marker remains visible. The subsequent EXP-0010 training runs record a clean code commit.

Documents were split by source document ID before tokenizer training. The tokenizer was trained on train documents only. Counts are 15,976,924 train tokens, 931,435 development tokens and 2,032,436 document-held-out test tokens. The text test contains 69 English and 47 French documents. The source documents are publicly available; “held out” means they were excluded from this project's training and model scoring until the locked EXP-0010 analysis, not that the text is secret from outside readers or other models' pretraining.

## Privacy and content checks

The pipeline redacted 2,579 email-shaped strings, 158 phone-shaped strings and three IP-pattern matches. It excluded five documents with residual email-shaped strings and two containing Unicode replacement characters. The retained corpus had no matches for the configured residual scan patterns. A deterministic screen reviewed 10 short excerpts per language. These checks are automated or sample-based; they are not a full human review of every document and do not prove that the corpus or learned weights contain no personal or sensitive information. The manual privacy attestation remains unsigned.

## License and attribution limits

The dataset and HAL metadata report CC0 for selected records. Some authorship and rights details may be missing or incomplete as supplied by the source metadata; no author identity is inferred. The bundle provides the record IDs, source URIs, declared license values, languages, split assignments, source and processed text hashes, HAL audit, filters, manifests and processing code. It intentionally omits source article text.

The model's MIT license applies to the model weights and included project code. It does not relicense source articles or establish that every source has been independently rights-cleared. The OSAID does not prescribe a particular legal mechanism for making parameters freely available; MIT is the mechanism selected for the model package. The package does not claim full OSI Open Source AI Definition 1.0 conformance because the authority behind the selected source records and the data-information terms have not been independently established. The traceability record helps readers locate and inspect the stated source materials, subject to their original terms and continued availability.

## Reproduction files

The package contains the pinned data-pipeline and source-registry configurations; HAL audit; processing, splitting, tokenizer and token manifests; source-record manifest; filtering decisions; privacy-screen report; and scripts used for source acquisition, filtering, splitting, encoding and training. The run environment and each paired run's configuration and metrics are recorded under `training_record/`.
