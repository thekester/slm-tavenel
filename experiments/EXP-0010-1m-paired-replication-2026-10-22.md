# EXP-0010 prospective 1M paired replication

**Protocol frozen:** 2026-10-22, before scoring the sealed text split with any confirmatory run.
**Status:** implementation and training pending. The v4 pilot remains an exploratory candidate.

## Question and scope

Does a head-wise attention gate change English/French held-out bits-per-byte (BPB) in this
approximately 1.18M-parameter model, when the gated model is compared with the same architecture
without the gate under a fixed, one-pass token budget? The result applies only to this small
French-Science-Commons science-text corpus and the specified training recipe. It does not test
general language-model quality or establish that the gate explains model behavior.

## Data and release transparency

Use the existing frozen v4 CC0-metadata subset and tokenizer, without changing documents, filters,
splits, or vocabulary. The data is the `french_open_science` subset of
[PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons),
revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. All selected source rows are labeled CC0 by
the dataset and all versions returned by the paginated HAL API audit report CC0. This is a
documented metadata trail, not independent proof of depositor authority. Source-rights and manual
privacy reviews remain unsigned and must stay stated as such in any resulting release.

Pinned inputs:

- Token-shard manifest SHA-256: `58065825d94485cca526f93ec993d5cfc8bec25ad742a2f6d6c32213370d48b8`
- Split manifest SHA-256: `abc8371759211a9ade705fa251e6a5f978b34ea1d41353831066d0ab97661f47`
- Tokenizer SHA-256: `93c7b9364de2c42315571c851353862e63830bfbeb220dc14dde80e6ca82cfcb`
- Train/dev/sealed-test tokens: 15,976,924 / 931,435 / 2,032,436
- Sealed-test documents: 69 English and 47 French; grouped by source document ID before tokenization.

No article text or sealed-test score may be included in the public artifact. A document-hash-level
score file may be released if source terms and privacy review allow it.
The optimizer schedule, token budget, and architecture were inherited from the previously inspected
v4 pilot; the development split has already been viewed and is used only for routine monitoring.
This protocol treats only the document-disjoint text test as a prospective final outcome. The
source documents are public; “sealed” means held out from this project's training and model scoring,
not inaccessible to outside readers or absent from other models' pretraining.

## Model arms and run plan

Arm A uses `configs/model/slm-1m-gated.yaml`; arm B uses
`configs/model/slm-1m-ungated-match.yaml`. They share width, depth, tokenizer, optimizer, sequence
length and all other settings. The parameter difference is the gate projection only and must be
reported; it is expected to be below 1%.

Run ten paired seeds: `421101` through `421110`. For each seed, initialize the gated arm once,
copy every shared parameter tensor into the ungated arm, and leave only the gate parameters unique.
Both arms then consume the same seed-indexed batches in the same order. Do not count any earlier
checkpoint as a replicate. Train every run for 488 steps at 32,768 tokens per step (15,990,784
token presentations, approximately one pass over the 15,976,924 train tokens), with AdamW,
learning rate 0.003 to 0.0003, 48 warmup steps, warmup-stable-decay, bfloat16, and deterministic
algorithms enabled. Preserve every run manifest, log, initialization hash, final checkpoint hash,
and clean training-code revision. Failed runs and deviations remain in the record.

The prospective release checkpoint, if the study and package checks pass, is the gated arm for
seed `421101`, selected here before results are observed. Do not select a checkpoint by benchmark
score.

## Locked evaluation and decision rule

The single primary outcome is the equal-language macro-average of corpus BPB on the sealed test
documents: calculate aggregate BPB separately for English and French, then average the two values.
Define the paired effect as **ungated BPB minus gated BPB**, so a positive value favors the gated
arm. The practical-effect threshold is +0.01 BPB. Report all ten seed-pair effects and a 95%
hierarchical bootstrap interval resampling paired seeds and documents within language (10,000
draws, fixed bootstrap seed `20261022`). Support for a practically meaningful gate benefit
requires both a mean effect at least +0.01 BPB and an interval wholly above zero. Otherwise report
the result as inconclusive or as evidence against that threshold; do not claim a gate benefit.

English and French effects are reported separately as secondary descriptive outcomes. The
multiple-choice benchmarks have already been inspected for earlier models; any new scoring is
exploratory, secondary, and cannot replace the sealed BPB outcome. Use identical pinned items,
prompts and evaluator for any comparisons, report per-item counts and intervals, and make no
causal claim against external models with unmatched training histories.

## Reproducibility and explanation limits

The training process runs from a clean detached worktree at code revision
`02396acd15ef9bcdc453bcd53aade40819e2f69f`; the experiment driver, model configurations and
analysis code are separately versioned by SHA-256 in the experiment output. Record Python,
PyTorch, CUDA, GPU, and tokenizer versions. The train/dev logs and fixed-probe observability remain
diagnostics. They do not establish causal explanations; any mechanistic claim requires a separately
specified intervention study with controls and replication.

This protocol improves run-to-run evidence for a narrow claim. It cannot resolve source ownership
or provide a human privacy sign-off. Those states must remain explicit even if the scientific
comparison is conclusive.
