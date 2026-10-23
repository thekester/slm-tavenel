# EXP-0010 1M: addendum to the HAL file-license audit

**Date:** 2026-10-22\
**Candidate:** `slm-tavenel-1m-exp0010-s421101`\
**Dataset revision:** `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`\
**Audit report:** `exports/slm-tavenel-1m-exp0010-release-candidate-v4/training_record/hal-license-audit.json`\
**Audit SHA-256:** `dbb8209b851b5768856ea566c44b50ceea54ccf9242fb37db6ffde4f64fa8594`

## What changed

The earlier audit compared dataset-declared CC0 rows against HAL's `licence_s` field for every
version returned by the paginated search API. HAL's current documentation distinguishes
`licence_s` for the main file from `fileLicenses_s` for the deposited Document files. The audit
script now requests and checks both fields on every returned version. The independent API
description is in the [HAL file-license announcement](https://www.ccsd.cnrs.fr/en/2026/02/no-hal-deposit-without-licence-the-essential-information-you-need-to-know/)
and [HAL API documentation](https://api.archives-ouvertes.fr/docs/search).

## Results

| Check | Count | Result |
|---|---:|---|
| Dataset rows declared CC0 | 3,483 | Input to the source audit |
| Rows whose main-file and every reported Document-file license are CC0 across returned HAL versions | 3,450 | Retained for further processing |
| Missing, mismatched or conflicting HAL records | 33 | Excluded |
| Candidate documents after content, quality, privacy-pattern and split filtering | 1,229 | 748 English, 481 French |

The 3,450 accepted identifiers exactly match the previous complete-pagination audit. The accepted
records had a non-empty `fileLicenses_s` value, and all reported values were CC0. The updated
report includes the 33 exclusions and their outcomes, so the decision path can be inspected from
the initial dataset-labeled CC0 set through the retained training corpus.

This is a stronger check of file-level metadata than checking an aggregate dataset card or only a
record-level label. It still establishes what the dataset and HAL metadata report. It does not
verify that each depositor had authority to apply CC0, that all publisher agreements permit reuse,
or that copyright ownership is undisputed. CC0 itself is applied by a person or entity that must
hold the relevant rights; the [CC0 legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode.fr)
does not turn an API label into proof of that authority.

## Attribution and privacy evidence

The public metadata-only source manifest now includes one row per retained document with its HAL
ID and URI, source title, declared author, publication date, DOI where supplied, declared license,
language, split and text hashes before and after processing. All 1,229 rows provide an author field
and date; 380 provide a DOI. The other 849 DOI fields remain empty. No source article text is included.
These are source-supplied metadata, not independent confirmation of authorship or rights ownership.

The content pipeline removed records based on discipline, title and text-risk signals and redacted
common email, phone and IP patterns. The final scan found no configured residual matches. A fixed
sample of 20 short excerpts was reviewed. This is useful screening evidence, but it is not a
corpus-wide privacy review, anonymization assessment, or model-memorization test. The human privacy
attestation remains unsigned.

## Current decision status

| Gate | Evidence | Status |
|---|---|---|
| Source identification and reproducibility | Pinned revision, filtering decisions, source manifest, hashes and replay record | Documented |
| File-level license metadata | Paginated HAL check of main and Document-file license fields | Passed for 3,450 source IDs; 33 excluded |
| Authority to apply the listed licenses | Not inferable from HAL metadata alone | Human rights review pending |
| Privacy review | Automated filters and 20-excerpt sample | Human review pending; no anonymity claim |
| OSAID 1.0 | Package self-assessment, data information, code and weights | Not OSI certified or endorsed |

Do not change `source_rights_review_signed`, `manual_privacy_review_signed`, or the release-decision
fields to `true` based on this automated addendum. The source and privacy review dossier remains the
record of outstanding maintainer decisions: [EXP-0010 source and privacy review](EXP-0010-SOURCE-PRIVACY-REVIEW.md).

The 10M and grown 25M FSC runs use the earlier mixed-license FSC pilot, not this 1,229-document
CC0-filtered EXP-0010 corpus. Completing those runs can improve the scale comparison, but it does
not convert their training provenance into the EXP-0010 source record or clear their source-rights
and privacy limitations.
