# SimpleStories source record, 10 November 2026

This record distinguishes a declared permission from independently established upstream rights. It does not approve the source in the project's registry, sign a human privacy review, or certify OSAID compliance.

| Item | Observed evidence | Status |
|---|---|---|
| Dataset and revision actually used | `SimpleStories/SimpleStories@e63b8adc3b1a1bdc7cac5b500d150b71346b0628` | Pinned and retrieved |
| Permission declared by the producer | `license: mit` in the README at that revision | Verified declaration; the repository has no standalone dataset LICENSE at that revision |
| Raw source shard | `data/train-00000-of-00007.parquet`, SHA-256 `ca33531b99f3bebb4125e82f56017c223a4900331b55bcf7cde1b0f750d88fd4` | Used locally; no raw training stories bundled in this model package |
| Producer and intended use | The dataset card names the SimpleStories project and presents the stories as material for training small interpretable models | Documented |
| Synthetic generator | The paper and local provenance name `gpt-4o-mini-2024-07-18` | Proprietary generator; original account agreement and any separate permission are unavailable |
| Generator code license | The linked generation repository carries MIT terms for its code | This is separate from rights in generated text and does not establish the terms of the producer's API contract |
| Training records | 60,000 train and 3,185 dev stories, one shard, one record per generation ID | Full metadata-only source index included as `training_record/document-index.jsonl.gz` |
| Completion families crossing train/dev | Zero | Actual local metadata checked; the earlier pilot's question/story-level split was replaced |
| Exact / case-and-whitespace-normalized text overlap | Zero | Not a semantic or near-duplicate audit |
| Email / phone / IP / IBAN regex matches | Zero on the 63,185 local texts | Narrow screening only; not a signed privacy review or a check of names/free-text personal data |

The MIT declaration provides a documented permission basis from the dataset producer. It does not prove what contractual terms governed the original generation. The current OpenAI Services Agreement assigns output to the customer subject to applicable law and also restricts some competitive-model uses. That current agreement is not evidence of the exact agreement used by this third-party producer in 2024/2025. No claim is made here that it necessarily prohibits downstream use of this dataset, or that it clears it.

The source therefore remains `candidate`. The user supplied a prior condition that weights should not be published until the SimpleStories rights question is settled. The prepared Hub repository is private pending an explicit publication decision about this documented uncertainty. No human signature or retrospective legal clearance has been fabricated.

## Sources

- [Pinned dataset declaration](https://huggingface.co/datasets/SimpleStories/SimpleStories/blob/e63b8adc3b1a1bdc7cac5b500d150b71346b0628/README.md).
- [SimpleStories paper](https://arxiv.org/abs/2504.09184), Finke et al. (2025).
- [Producer's generation code](https://github.com/simple-stories/simple_stories_generate) and [training code](https://github.com/simple-stories/simple_stories_train).
- [Current OpenAI Services Agreement](https://openai.com/policies/services-agreement/), sections 3.3 and 4.1; consulted as context, not treated as the producer's historical contract.

The unchanged corpus, tokenizer, run and export manifests remain in `training_record/`. Their original `candidate`, `dirty` and local-pilot fields are retained as historical evidence.
