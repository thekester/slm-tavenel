# Review of Sugatoray's LLM training datasets collection

**Reviewed:** 2026-10-23\
**Collection:** [Sugatoray / LLM Training Datasets](https://huggingface.co/collections/sugatoray/llm-training-datasets)\
**Purpose:** identify sources that could support a future release-oriented SLM-Tavenel pretraining run. This is a source-screening note, not a rights clearance or training approval.

## Recommendation

Use [PleIAs Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) as the first source to investigate further. Pin the reviewed repository revision `307910e4c5d040d6f318e6edf2a2b97849155771`. The dataset card describes document-level metadata including source identifier, collection, license, creator, language, and token count; it includes English and French; and the authors report the corpus and its curation in an ICLR 2026 oral paper ([ICLR record](https://iclr.cc/virtual/2026/poster/10011885)). These features make it more auditable for this project than an undifferentiated web crawl.

This does **not** make the aggregate dataset approved. The pinned card reports 635 license values in the displayed dataset schema and shows examples spanning CC0, public domain, MIT, CC BY-SA, and records with missing or generic license labels. The authors' description that records are public domain or permissively licensed is useful evidence about their process, but it is not an independent verification of every source record or the authority of each declaration.

The card also says PII removal was attempted with Presidio and locale-specific changes, and that some French administrative crawl records were dropped using an unreleased model to reduce indirect-identification risk. The public card therefore does not provide a fully reproducible privacy pipeline for that part. Public-domain status alone also does not remove privacy risk: the dataset viewer examples include legal records containing names, dates of birth, and addresses.

## Conservative subset rule for a new run

Before any release-oriented training, create a new derived manifest from the pinned revision. Keep only English and French records with a stable original identifier/source, a record-level license that has passed the project's review, and enough metadata to satisfy any attribution duties. For the initial subset, accept only independently supported public-domain/CC0 records and reviewed attribution-only permissions such as CC BY 4.0 or Licence Ouverte 2.0, with an attribution manifest. Exclude share-alike, NonCommercial, NoDerivatives, unknown, missing, conflicting, and generic “various” declarations. This is a conservative project filter, not a claim that every excluded license is incompatible with open-source AI.

Exclude court/legal and administrative documents that identify natural persons from the first subset. Add a reproducible privacy screen, log redactions and removals, inspect deterministic samples in both languages, and keep a human review gate for sensitive material. Reproduce benchmark decontamination against the locked evaluation sets. Publish the filter code, exact revision, selected source IDs, per-license and per-language counts, token totals, transformations, removals, attribution obligations, and hashes before claiming the corpus is reproducible.

The upstream dataset is too large to download wholesale for this screening. The review used the dataset card, visible schema and records, the pinned repository metadata, and the ICLR abstract. No corpus rows have been downloaded into SLM-Tavenel, and no training run has consumed this source.

## Other collection entries

| Dataset | Possible role | Current disposition |
|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Large English and multilingual web baselines | Comparison-only under the current project policy: ODC-By covers the compilation, while underlying page-level permissions are not established by that label. |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | English synthetic-text experiment | Separate exploratory candidate. Its card describes synthetic samples based on varied seed sources; audit seed/source terms and the applicable rights before use. It is not a bilingual clean-corpus substitute. |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Later instruction-tuning experiment | Not a base-pretraining corpus; it is an instruction/SFT dataset with underlying-source terms that need their own review. |

These are screening outcomes, not approvals. The collection is a useful discovery index; inclusion in it is not evidence that a dataset meets SLM-Tavenel's provenance, privacy, or license requirements.

## Relation to the existing 25M checkpoint

Adding a cleaner source to the registry does not change the 25M model's training history. The existing checkpoint was trained on the earlier mixture, including unresolved source material. This candidate can only inform a **new** run whose complete source set and preparation pipeline are fixed before training. It cannot retroactively cleanse or clear existing weights. The local 25M checkpoint remains outside this review and is not approved for upload by it.

## Status

- Common Corpus: **candidate**, pinned, not approved for training.
- Dataset contents downloaded for this project: **no**.
- New training started from this source: **no**.
- 25M upload or publication approval: **no change**.
