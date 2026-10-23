# EXP-0009 10M: HAL file-license metadata audit

**Audit date:** 22 October 2026\
**Dataset:** PleIAs/French-Science-Commons, `french_open_science`, revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`\
**Input:** 31,417-row metadata-only source manifest\
**Full report:** `exports/slm-tavenel-10m-exp0009/training_record/hal-file-license-audit-2026-10-22.json`\
**Method:** Query HAL by retained identifier; compare each returned version?s `licence_s` main-file field and `fileLicenses_s` document-file fields with the label recorded in the pinned dataset manifest. Annex-file license metadata is recorded but not used for the verdict because annex text was not part of the training input.

## Results

| Outcome | Records | Interpretation |
|---|---:|---|
| Returned HAL version records | 31,362 | Each row retains `version_i`, URI, document type and main/document/annex license fields |
| Every returned version matches | 31,262 | Both main-file and document-file metadata match the dataset label |
| Explicit license mismatch | 72 | At least one reported license differs from the dataset label |
| Conflicting declarations | 3 | HAL returned both expected and different license values, including across versions or files |
| Missing file-license metadata | 24 | One or more required main/document fields are absent, without a different reported license |
| Identifier not found | 56 | HAL returned no record for the queried identifier |

The 155 unresolved records are not counted as verified. The audit does not decide which source was correct when the dataset snapshot was assembled. It queries present-day HAL metadata and therefore is not a historical rights record. The report preserves HAL IDs, URIs, `version_i`, document types, declared labels, observed main/document/annex license values, returned-record counts and outcomes. It contains 31,362 returned HAL version records across the 31,417 retained source identifiers, plus corpus-supplied titles and declared authors. It includes no article text. The report SHA-256 is recorded in the model package provenance.

## Scope and limitations

HAL documents `licence_s` as the main-file license and `fileLicenses_s` as licenses for deposited Document files ([HAL announcement](https://www.ccsd.cnrs.fr/en/2026/02/no-hal-deposit-without-licence-the-essential-information-you-need-to-know/); [HAL API documentation](https://api.archives-ouvertes.fr/docs/search)). The metadata are declarations. They do not establish the depositor's authority, a publisher's contractual terms, ownership of every included element, fulfillment of attribution conditions, privacy clearance or absence of memorization. No source-rights or privacy sign-off is inferred from this audit.

The pinned dataset's record-level labels and current HAL file metadata disagree or are unavailable for 155 rows. Before claiming a fully verified source set, a human reviewer would need to resolve those records against the exact dataset snapshot and authoritative source notices, and review attribution obligations. The 10M package remains an exploratory public artifact with these limits disclosed; this audit does not certify legal compliance or OSI status.
