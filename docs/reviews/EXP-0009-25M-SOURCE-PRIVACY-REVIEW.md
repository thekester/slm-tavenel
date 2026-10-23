# EXP-0009 grown 25M: source and privacy review record

**Review status: open.** This document assembles evidence for a maintainer or independent reviewer. It is not a legal opinion, privacy attestation, or publication approval. No human decision or signature is recorded here.

## Exact artifact and training data

- Candidate weights: local `exports/slm-tavenel-25m-exp0009-candidate/model.safetensors`, SHA-256 `59c495cbc09558f32aef0574295b0c771f7810ac092861eeb38f664ae9449140`.
- Training trajectory: 10M mixed-source FSC checkpoint, grown to 25.72M parameters, then trained 15,000 further steps. This is not a clean retraining from the newer 1M CC0-screened dataset.
- Dataset: PleIAs/French-Science-Commons, `french_open_science`, pinned revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`.
- The 31,417 retained records have dataset-declared labels: 4,528 CC0, 26,368 CC BY 4.0, and 521 Licence Ouverte 2.0. The full metadata-only source manifest and its hash are shipped in the candidate package.
- The weights and code carry MIT terms. Those terms do not independently settle rights in source material or third-party claims about learned weights.

## Source-license evidence

The 22 October 2026 HAL API audit compared the dataset labels against current HAL file-license metadata across returned versions. It found 31,262 full metadata matches, 72 mismatches, 3 conflicts, 24 missing file-license fields, and 56 IDs not found. A targeted recheck of all 155 unresolved IDs on 23 October found **no changed outcomes**. The original complete audit and the dated recheck are included in `training_record/`.

Of the 155, 75 have a DOI. Crossref returned a license declaration whose normalized URL included the dataset's expected license for 30 records; 15 had only a different Crossref license or policy URL; 23 had no Crossref license field; and 7 lookups failed. Crossref metadata is publisher-deposited bibliographic information, can be incomplete or version-mismatched, and is discovery evidence rather than clearance. Eighty unresolved records have no DOI in the source manifest. The raw per-record HAL and Crossref observations are retained in the accompanying JSON reports without article text.

HAL's own guidance says its file fields describe the declared license of each deposited file. The same guidance distinguishes a missing license from broader reuse permission: HAL authorization permits reading, consulting, and citing, while reproduction or adaptation beyond legal exceptions requires permission from the author or rights holder. A metadata match therefore cannot substitute for confirming the exact trained file version, rights-holder authority, and attribution obligations.

The source manifest identifies the corpus revision and processed-text hashes, but does not map each processed record to the exact HAL file version whose bytes produced it. This prevents a definitive version-by-version reconciliation after the fact. The original audit did not establish depositor authority or historical license status.

## Privacy evidence

The original FSC-v1 processing pipeline normalized text, applied structured PII redaction, quality filtering, deduplication, and benchmark-overlap filtering. It recorded redactions of 64,641 email-shaped strings, 5,108 phone-shaped strings, 165 IP-pattern strings, and 7 IBANs. It did **not** apply the later v4 publication-risk metadata screen or the broad sensitive-content blocklist to this 31,417-document corpus. Those controls were developed for the smaller v4 subset and cannot be retroactively attributed to EXP-0009.

A local scan applied the v4 title/discipline signals and structured-identifier detector to all 31,417 processed FSC-v1 documents (1.945 billion characters). It recorded four residual public-IP-pattern matches, 54 documents with a replacement character, nine documents missing discipline metadata, 17,132 documents in metadata-excluded disciplines, and 12,889 documents with a configured sensitive-content term in the body. Overall 20,808 document IDs were flagged for review. The report stores counts, IDs, and hashes only, never article excerpts. The body-term flags are broad and are not a finding that each document contains personal data.

These are automated triage results, not a privacy attestation. Names, indirect identifiers, free-text sensitive details, rare combinations, and memorization are not established absent by pattern checks. The later v4 exclusion rules were not used in the EXP-0009 training pipeline. A human sample review of the flagged set remains necessary.

## What can be concluded

1. The training source set and processing are substantially traceable by pinned revision, record-level metadata, decisions, code, hashes, and training logs.
2. There are specific, unresolved source-license discrepancies. Current HAL and Crossref metadata do not make those discrepancies disappear, and neither source independently proves the submitter held the rights they declared.
3. The 25M checkpoint cannot be represented as trained only on the newer CC0-screened, privacy-filtered v4 subset. Its inherited 10M checkpoint used the older mixed-label FSC-v1 data.
4. Therefore the historic EXP-0009 25M weights cannot receive a robust positive source-rights/privacy clearance from the existing records alone. A new run on an approved, hash-pinned corpus would be needed to make a materially stronger release candidate. The maintainer must still decide the accepted residual risk and any attribution approach; this packet does not sign for them.

## Maintainer decision to complete

| Field | Reviewer entry |
|---|---|
| Reviewer name and role | Pending |
| Date | Pending |
| Exact checkpoint SHA-256 reviewed | Pending confirmation |
| Source scope examined and evidence used | Pending |
| Source-rights decision and accepted uncertainty | Pending |
| Privacy sample scope and findings | Pending human review |
| Additional exclusions required | Pending |
| Final decision: reject, conditional, or approve | Pending |
| Publication authorization | Pending; separate from OSAID self-assessment |
| Signature or link to signed decision | Pending |

## Primary references

- [HAL Search API documentation](https://api.archives-ouvertes.fr/docs/search)
- [CCSD explanation of HAL file-license fields and HAL authorization](https://www.ccsd.cnrs.fr/2026/02/pas-de-depot-hal-sans-licence-les-infos-indispensables/)
- [Creative Commons CC BY 4.0 deed](https://creativecommons.org/licenses/by/4.0/)
- [Etalab Licence Ouverte 2.0](https://www.data.gouv.fr/pages/legal/licences/etalab-2.0)
- [French Science Commons dataset card](https://huggingface.co/datasets/PleIAs/French-Science-Commons)
