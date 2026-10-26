# Hugging Face pretraining dataset screen

**Reviewed:** 2026-10-24
**Purpose:** identify additional English/French text that could support a new SLM-Tavenel data-scaling
experiment.
**Status:** source screening only. No new dataset rows were downloaded or used for training. A Hugging
Face card or repository-level license label is not independent rights clearance.

## Shortlist and disposition

| Dataset and pinned revision | Declared scope | Screen result |
|---|---|---|
| [PleIAs/common_corpus](https://huggingface.co/datasets/PleIAs/common_corpus/tree/307910e4c5d040d6f318e6edf2a2b97849155771), `307910e4c5d040d6f318e6edf2a2b97849155771` | English, French and other languages; card reports 2.27T tokens and per-document source/license/language metadata | **Best scale candidate, still unapproved.** Derive a bilingual subset by source and exact record-level license; start with independently supported CC0/public-domain records. Exclude missing, generic, conflicting and share-alike declarations for the first release-oriented pilot. The existing [Common Corpus review](DATASET-COLLECTION-REVIEW-2026-10-23.md) records unresolved license-authority and privacy issues, including legal/administrative records and a partly unreproducible French privacy filter. |
| [OpenBallast/ballast-t2](https://huggingface.co/datasets/OpenBallast/ballast-t2/tree/e53b6dd0c2fb6a16909e30f6fdb2c5d787878cdf), `e53b6dd0c2fb6a16909e30f6fdb2c5d787878cdf` | English OpenStax textbook passages; 90,445 chunks from 61 books; CC BY 4.0 declared | **Strong small English pilot candidate.** Its card documents source repositories, per-book source commit pins and license evidence; it excludes nine titles that had changed to CC BY-NC-SA. It is English-only and education-heavy, so it cannot by itself preserve the current EN/FR balance or supply a large general-domain token increase. Verify the source manifest and attribution files before adding it. |
| [Helsinki-NLP/europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl/tree/ab45e286aef3fb5780067100cb5a1132b52b7949), `ab45e286aef3fb5780067100cb5a1132b52b7949` | 21-language parallel proceedings, including English/French; card reports 30.32M sentence fragments and 759.05M tokens | **Do not use yet.** The Hub card reports the license as `unknown`, supplies only a train split, and describes a translation corpus. Obtain and review the authoritative source terms, then deduplicate and create a held-out split before any use. |
| [PleIAs/French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books/tree/21abfb3f646a0f1846175c63709913a51abc9bcb), `21abfb3f646a0f1846175c63709913a51abc9bcb` | French books; card reports 289,000 books and 16.4B words | **Do not use for an any-purpose open-weight run yet.** The card asserts public-domain status, but also quotes BnF terms that charge for commercial reuse. The Hub API has no machine-readable license label, and the card notes OCR/formatting noise. Resolve the BnF/source-rights conflict and verify work-level provenance first. |
| [PleIAs/French-PD-diverse](https://huggingface.co/datasets/PleIAs/French-PD-diverse/tree/fca59daf897dd2c4f7979b8de6936cb48dec95ef), `fca59daf897dd2c4f7979b8de6936cb48dec95ef` | French public-domain book collection; card estimates 307,171 rows and 43.1B words | **Do not use yet.** The Hub API has no declared license field. The preview shows Internet Archive/Google Books material and OCR artifacts, so the source-by-source rights and quality trail needs work. |
| [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/tree/af9c13333eb981300149d5ca60a8e9d659b276b9), `af9c13333eb981300149d5ca60a8e9d659b276b9` | Multilingual Common Crawl web text; 1,870 language subsets; ODC-BY compilation label | **Exploratory only under current policy.** The compilation license and crawl provenance do not establish permission for each underlying page. This is not the first choice for an open-weight candidate. |
| [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f), `3ba9d605774198c5868892d7a8deda78031a781f` | English; ODC-BY compilation label; Cosmopedia v2 has 39.1M synthetic rows | **Exploratory only.** Cosmopedia samples are generated from web seed samples, and the card exposes a `seed_data` field. Audit seed-source permissions, transformations and output risks before using generated text in a release-oriented run. It is not bilingual. |
| [finaleads/french-corpus-llm-sample](https://huggingface.co/datasets/finaleads/french-corpus-llm-sample/tree/5cc70de432f485775b0506ad38edf1c6cdb9049a), `5cc70de432f485775b0506ad38edf1c6cdb9049a` | 500 French legal/finance/regulatory samples; Etalab 2.0 declared; about 2.3M estimated tokens | **Not a training-scale source.** The card says the full 2.66B-token corpus is available as a paid Snowflake product, not as the public Hugging Face dataset. The domain is narrow and contains regulatory/legal text; use would require source-level and privacy review. |

## Recommendation for the next experiment

The best route to more *distinct* bilingual tokens is a newly derived Common Corpus subset with a
reproducible source and license allowlist, not an unrestricted download of the aggregate. Before
training:

1. pin the dataset revision and list the exact Parquet files/configs used;
2. retain only English/French documents with a stable identifier, original source, and explicit
   license evidence; start with CC0/public-domain candidates and preserve a full attribution ledger;
3. exclude legal/administrative personal records and apply deterministic privacy and benchmark
   decontamination screens;
4. review deterministic samples in both languages, publish per-source/license/token counts and
   hashes, and compare the resulting language/domain mix with the existing FSC set;
5. only then tokenize and launch a matched 100M data-source experiment. Keep the model, optimizer,
   token count, validation files, and benchmark evaluator fixed between arms.

The current FSC scaling corpus contains 544,194,188 training tokens. The 100M EXP-0024 continuation
used 49,152,000 additional token positions and did not introduce a new source. Its newly added model
capacity therefore did not receive a proportional increase in distinct data. A new corpus should be
reported separately from merely replaying more epochs of FSC: evidence on data-constrained language
models finds diminishing benefit as repeated data accumulates ([Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)).

This screen authorizes neither dataset ingestion nor training. It records which source looks most
promising and what must be verified before the next run; the FSC v1 rights/privacy gaps remain open.

## Follow-up corpus build, 2026-10-24

After this initial screen, a local corpus candidate was built from the existing FSC CC0 v4 selection,
the CC-BY-only OpenBallast T2 partition, and one pinned Common Corpus shard. For Common Corpus, 673
English/French Written records passed a collection/license-label allowlist and the same quality,
privacy-pattern, deduplication and benchmark-overlap checks. The exact upstream labels are preserved,
including generic values such as `Various open science`; they are not promoted to independently
verified licenses. For OpenBallast, all 61 pinned source-book `LICENSE` files were checked at their
full commits. Attribution ledgers, deterministic splits, decisions and hashes are in the [corpus
candidate report](RELIABLE-CORPUS-CANDIDATE-V1-2026-10-24.md) and local machine-readable manifest.
The full Common Corpus aggregate was not downloaded or included. This candidate has not been used
to train a model, and the FSC human rights and privacy reviews remain unsigned.

## Follow-up: trending document/text catalog, 2026-10-25

The user supplied the [Hugging Face Datasets catalog filtered to document/text and sorted by trending](https://huggingface.co/datasets?modality=modality%3Adocument%2Cmodality%3Atext&sort=trending). Trending status was used only for discovery; it is not a quality, rights, privacy, or scientific endorsement.

| Dataset | Findings from the pinned Hugging Face card | Decision in this project |
|---|---|---|
| [ashtok897/european-hplt-v1](https://huggingface.co/datasets/ashtok897/european-hplt-v1/tree/51edb798468aa71b1000e950d3ab43bb435f72cd), `51edb798468aa71b1000e950d3ab43bb435f72cd` | 41 European languages; card estimates 50.9B tokens and says every document has HPLT WDS score >=10. It exposes URL, language, score and register fields. Its CC0 label is inherited from HPLT v3; the records come from Common Crawl/Internet Archive web crawls. The card says no PII filtering beyond HPLT defaults. | **Used as a documented growth candidate, not rights-cleared.** Ten pinned upstream train shards (7.96 GB, 2.5M source rows) were filtered locally. 232,517 EN/FR records remain, adding 334.0M SLM-Tavenel train tokens. URL/ID, revision, scores and hashes are retained. The official HPLT project says it does not own the original texts and downstream users remain responsible for applicable legal compliance. |
| [kwassl-ai/noah-text](https://huggingface.co/datasets/kwassl-ai/noah-text) | Card reports 106M English rows and about 700B tokens extracted from Common Corpus using Public Domain, CC0, MIT-0 and Unlicense labels. It uses ODC-BY for the compiled database and explicitly says no further privacy processing was done, so PII may remain. | **Not downloaded or included.** Hugging Face file-list/data API calls returned HTTP 401 in this session. The card is useful evidence for a later source review, not a completed audit. A later attempt should first check collection-level metadata, original source traces, rights, privacy, and whether streaming a limited subset is possible. |
| [SYNTH-Initiative/SYNTH](https://huggingface.co/datasets/SYNTH-Initiative/SYNTH) | Card describes a roughly 75B-token synthetic multilingual reasoning/task corpus; dataset card labels it CDLA-Permissive-2.0 and says the seed text is CC BY-SA with model outputs available for reuse. | **Separate auxiliary-data candidate, not mixed into base-text growth.** Its exercise/reasoning structure and seed terms require a separate source-specific review and matched experiment; it should not be counted as general web prose. |
| [OpenLLM-France/Luciole-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset) | Card declares CC BY-NC-SA 4.0 for the mixed corpus, while describing some public-domain/open components. | **Whole-dataset package not used** for this general-use open-weight candidate. A component-specific subset could only be considered after identifying and reviewing the actual files and component terms; the aggregate card label is not silently overridden. |

The HPLT growth outputs are documented in [GROWTH-CORPUS-CANDIDATE-V2-2026-10-25.md](GROWTH-CORPUS-CANDIDATE-V2-2026-10-25.md). The six local evaluation benchmarks were not training sources: their questions and answer choices were read only by the exact 13-token decontamination rule, and five candidate HPLT documents were removed for overlap. Both EXP-0028 training arms (50M and 100M) are complete. Final corpus-test and downstream benchmark evaluations remain pending.
