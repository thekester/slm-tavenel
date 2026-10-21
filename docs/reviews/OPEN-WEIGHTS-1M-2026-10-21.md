# Open-weight 1M retraining path — 2026-10-21

## Decision

The existing 1M checkpoint remains **not cleared** for public weight release. It was trained on
the broader French-Science-Commons pilot, including CC BY and Licence Ouverte records, and its
training run has one seed, a dirty source tree, and already-inspected evaluation results. A new
license file cannot retroactively clear its corpus.

The most practical route now prepared is a **new 1M training run on an English/French CC0-only
subset** of the pinned French-Science-Commons `french_open_science` release. The subset contains
only articles whose CC0 record was confirmed against every version returned by the HAL records
API. HAL records with a missing entry, a conflicting version license, or a non-CC0 license are
excluded. The selection retains stable HAL IDs and record URLs. This makes the rights evidence
substantially more specific than relying on the aggregate dataset card alone.

This is a candidate corpus and a review record, **not a legal opinion or release approval**. The
HAL API is the repository's own license metadata and can still reflect an uploader error. Privacy,
source notice, benchmark decontamination, and the final weight-license decision remain separate
gates. The user has asked for unrestricted open weights; the intended weight license for a
successfully cleared retrain is MIT, with an explicit source notice. Do not attach that
license to the existing FSC checkpoint or upload any model until the remaining gates pass.

## Source and record-level audit

- Dataset: [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons)
- Dataset revision: `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`
- Subset: `french_open_science`; the separate `hal_open_access` subset is not used.
- Candidate rows: exact dataset license `CC0`, English or French, and a HAL ID.
- Independent cross-check source: [HAL records API](https://api.archives-ouvertes.fr/docs/search),
  queried for every dataset ID and all versions returned for that ID.
- Reproduction script: [`scripts/audit_fsc_cc0_hal.py`](../../scripts/audit_fsc_cc0_hal.py).
- Local record-level report and filtered text: `data/corpus-audits/` (ignored by Git; not for Hub
  upload). The report stores record IDs, HAL URIs, license strings, outcomes and text hashes. It
  does not store article text in the audit JSON.

Historical audit result on 2026-10-21:

The original audit checked the versions returned by each bounded API response without recording
`numFound` or pagination. The counts below describe that recorded response set, not proof that
every matching version was retrieved. The audit tool now follows all result pages and rejects
incomplete responses. Rerun that audit before relying on these candidates for source sign-off;
this release does not rerun the corpus audit or change the historical training artifacts.

| Language | Dataset CC0 rows | All HAL versions also CC0 | Excluded | Kept words before quality filtering |
|---|---:|---:|---:|---:|
| English | 1,789 | 1,767 | 22 | 17,075,480 |
| French | 1,694 | 1,683 | 11 | 21,654,381 |
| **Total** | **3,483** | **3,450** | **33** | **38,729,861** |

The 33 exclusions are seven IDs absent from the API, 25 IDs whose returned license was not CC0,
and one ID with mixed CC0 and Licence Ouverte versions. This conservative rule avoids assuming
which text version is represented in the aggregate record. The English and French corpora are
balanced by document count, not by raw word count; the French side contains about 27% more words.
The training pipeline will report and, if needed, balance by the new tokenizer's actual token
counts without duplicating documents. Do not describe this corpus as general-domain or
representative: it is a small scientific-document pilot.

## Prepared data pipeline

[`configs/data/pipeline-fsc-cc0-open-weights.yaml`](../../configs/data/pipeline-fsc-cc0-open-weights.yaml)
starts from the filtered local CC0 text and runs normalization, quality filtering, pattern-based
PII redaction, exact and MinHash deduplication, and 13-gram filtering against the six pinned local
benchmark files (ARC-Easy, ARC-Challenge, HellaSwag, Belebele EN/FR, and WinoGrande). Its output
remains exploratory while the source registry entries are `candidate`.

The first local processing pass retained **3,328 documents** (1,709 English and 1,619 French) out
of 3,450. It removed 122 for minimum-length/quality/near-duplicate rules and found no exact
13-gram benchmark matches. Pattern filters redacted 5,140 email, 537 phone, 3 IBAN and 2 IP
occurrences. The new 8,192-token tokenizer was trained from train documents only. The document
split is 85% train, 5% dev and 10% sealed test by deterministic SHA-256 hash; observed counts can
differ slightly by language because documents are indivisible. After tokenization, the actual
counts are:

| Split | EN tokens | FR tokens | Total |
|---|---:|---:|---:|
| Train | 32,465,659 | 34,580,487 | 67,046,146 |
| Dev | 1,657,990 | 1,619,440 | 3,277,430 |
| Sealed test | 1,957,716 | 3,068,830 | 5,026,546 |

One local gated 1M pilot has now completed 2,045 steps on the training split (seed `20261021`),
with final training loss **3.0272** and dev loss **3.1132**. It took approximately four minutes on
an RTX 4060 Laptop GPU. This was an engineering/pipeline pilot on an unapproved candidate, not a
confirmatory run. The sealed test has not been scored. Its checkpoint and observability logs are
under the ignored local `runs/open-weight-candidate/1m-gated-s20261021-pilot/` directory. This
pilot does not count toward any future replicated study.

The pilot checkpoint was packaged locally with a separate MIT `LICENSE`, a source `NOTICE`, the
custom Transformers implementation, tokenizer, run logs, record-level license audit, data
processing/split/token manifests, and SHA-256 hashes. The bundle contains no paper text. A fixed
probe produced exact native-versus-Transformers logits (max absolute error 0) on CPU float32, CUDA
float32 and CUDA bfloat16, and greedy generation appended all eight requested tokens. See
[`scripts/prepare_open_weight_candidate.py`](../../scripts/prepare_open_weight_candidate.py),
[`scripts/verify_open_weight_candidate.py`](../../scripts/verify_open_weight_candidate.py), and
the local ignored directory `exports/slm-tavenel-1m-cc0-candidate/`. This verifies artifact loading
and parity only; it does not clear publication or establish useful generation quality.

The existing PII filter catches common email, phone, IP and IBAN patterns. It will not remove
personal names, sensitive facts, or all personal information in research papers. Before training,
review a stratified sample of kept and removed documents, report redaction/filter counts, and
exclude sensitive fields or documents where necessary. Author metadata is retained as provided;
no author is inferred when the field is missing. Missing authors alone are not a reason to reject a
record whose provenance and CC0 status are otherwise clear.

An additional title-only risk screen run after the pilot matched **130 retained titles** against
broad terms such as patient, clinical, trial, case report, interview and cohort. This is a screening
signal, not a finding that all 130 records contain sensitive personal data. It does confirm the
corpus includes human-health and human-subject research; the pilot was already trained on these
records. The match list and a stratified content review must be resolved before releasing this
checkpoint. If the maintainer chooses to exclude those areas, the corpus, tokenizer and model must
be rebuilt from the frozen source revision; filtering after training does not remove exposure from
the existing weights.

## Reproducible training and evaluation required

1. Finish the kept/removed privacy review and record the maintainer's source decision. If any
   source is rejected, remove its records and regenerate all hashes before tokenizer training.
2. Run the candidate pipeline; freeze a document-disjoint train/dev/sealed-test split. Train the
   tokenizer on train only. Do not use the already inspected FSC validation set as a final test.
3. Derive the token budget from unique retained documents and the new tokenizer. The current
   CC0-only candidate has 38.7M whitespace words before quality filtering, so the former
   491.52M-token schedule cannot be reused as though it represented new data. Do not silently
   repeat documents to inflate the budget.
4. Freeze a clean Git commit, complete configuration, data/tokenizer hashes and environment before
   training. For a robust claim about the head-wise gate, compare parameter-matched gated and
   ungated models on identical data and token budgets with at least five paired seeds. Report each
   seed and uncertainty intervals. The existing 1M run does not count as a replicate.
5. Score the sealed text test and benchmark tasks once after the model and analysis are fixed.
   Describe external SLM scores as same-harness references, not causal controls. Keep benchmark
   contamination limitations explicit.
6. Export the final checkpoint, tokenizer, inference code, configs, source/attribution notice,
   model/data cards and hash-linked run record. Verify packaged inference against the native model
   on CPU and GPU/dtypes, then run evaluations from the exact packaged artifact.
7. The local pilot bundle carries MIT for its weights and inference code. Keep it private until the
   source/privacy review and release decision are signed; keep article text out of the Hub bundle
   unless redistribution rights are separately established.

## What this does and does not establish

The audit provides a reproducible, record-level comparison between the exact CC0 subset and HAL's
current license metadata. It does not prove the submitter had authority to apply CC0, establish
absence of personal or sensitive information, or prove that model weights have no memorization.
The scientific study can disclose metadata limitations and operate at a narrow science-text scope;
that is different from claiming broad scientific consensus or open-weight legal clearance. Public
weights should be released only after the remaining project gates have documented outcomes.

## V2 privacy-risk-reduced pilot (seed 20261022)

The first CC0 pilot was not cleared. A title-only screen found 130 records matching broad
human-health or human-subject terms. A new pipeline therefore applied conservative metadata rules
before tokenizer training: it removed four discipline families (social sciences, life sciences,
cognitive sciences and quantitative economics), missing title/discipline metadata, and additional
human-subject title terms. This is risk reduction, not privacy certification; text-level manual
review remains pending.

### Maintainer release attestations

These fields must be completed by the project maintainer; the preparation work does not constitute
their signature.

| Gate | Current state | Required record |
|---|---|---|
| Source/license review | **Pending; unsigned** | Reviewer identity, date, decision on the CC0/HAL evidence and any exceptions |
| Privacy review | **Pending; unsigned** | Reviewer identity, date, sampling/inspection scope, findings, exclusions and residual-risk decision |
| Public release decision | **Pending** | Hugging Face namespace/repository and release version tied to the reviewed hashes |

Do not change the `*_signed` flags in the candidate provenance unless a maintainer records these
decisions. If a review requests any corpus changes, regenerate the v3 data/tokenizer/model artifacts
and their hashes before release.

The v2 pipeline began with the same 3,450 record-level CC0/HAL-matched rows. It removed 1,848 on
discipline metadata and 7 on title signals; subsequent quality and length filters removed 52 more,
leaving 1,543 documents (910 English, 633 French). Pattern matching redacted 2,447 email, 135 phone
and 2 IP occurrences. The corpus contains 26,387,903 train tokens, 1,473,762 dev tokens and
3,399,878 sealed-test tokens. The test text has not been scored. Data, split, tokenizer and token
manifests are hash-linked in the training record.

The v2 pilot used seed `20261022` for 805 steps (26,378,240 optimizer tokens), ending at training
loss 3.3636 and dev loss 3.4377. Its same-harness accuracies are ARC-Easy 0.281, HellaSwag 0.264,
Belebele EN 0.254, Belebele FR 0.267, ARC-Challenge 0.192 and WinoGrande 0.485. On four shared
benchmarks, paired item-bootstrap comparisons with TinyStories tiny1m show +0.025 on ARC-Easy
(unadjusted interval [+0.002, +0.048]); the other three intervals include zero. These are post-hoc
comparisons without family-wise correction and use non-matched corpora and budgets. No superiority
or gate-effect claim is supported. The 3.40M-token sealed text test remains untouched.

The local v2 export is `exports/slm-tavenel-1m-cc0-v2-candidate/`. It is a 1,181,920-parameter
MIT-labeled candidate with fixed-probe native/Transformers parity on CPU float32, CUDA float32 and
CUDA bfloat16. These checks establish artifact consistency only. Both `source_review_signed` and
`manual_review_signed` are explicitly false in its provenance. Do not upload it publicly until the
maintainer records the source review and completes or records the privacy review outcome. If either
review changes the selected documents, regenerate the corpus, tokenizer, splits, weights and
benchmark comparisons from the revised frozen pipeline.

## V3 title-screen refinement (seed 20261023)

A follow-up title audit of the v2 retained set found French wording for interviews/clinical work and
several medical or biomedical titles that the first screen did not cover. The title rules were
expanded to include French interview/clinical terms and medical, biomedical, tumor/cancer,
immunotherapy and radiotherapy terms. The same discipline exclusions were kept. On the same 3,450
HAL-matched candidate rows, v3 removed 1,848 records by discipline and 15 by title; 52 more were
removed by existing quality/length filters. The resulting corpus has 1,535 documents (906 English,
629 French), with 26,401,958 train tokens, 1,689,668 dev tokens and 2,897,112 sealed-test tokens.
The test split remains unscored. This additional metadata screen lowers exposure to the identified
categories but does not replace reading the documents; manual privacy review and signed source
review remain pending.

The v3 1M pilot used seed `20261023`, 805 steps and 26,378,240 optimizer tokens; final train/dev
losses were 3.3378/3.6974. Raw same-harness accuracies were ARC-Easy 0.293, HellaSwag 0.261,
Belebele EN 0.253, Belebele FR 0.276, ARC-Challenge 0.212 and WinoGrande 0.502. Against
TinyStories tiny1m, the paired unadjusted ARC-Easy delta was +0.037 (95% item-bootstrap interval
[+0.015, +0.060]); the intervals for HellaSwag and Belebele include zero. Five reference/internal
comparisons across four shared tasks were post-hoc and unadjusted for multiplicity. These results
remain descriptive, not evidence of superiority or a causal gate effect. The source tree was dirty
and this is one seed; the sealed text test remains untouched.
