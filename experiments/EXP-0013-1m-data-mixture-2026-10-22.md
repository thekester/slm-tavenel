# EXP-0013: 1M paired FSC/MOT data-mixture pilot

**Status (2026-10-22):** queued after EXP-0012 completes both 10M arms and their evaluations.
MOT tokenization will use the frozen FSC v4 1M tokenizer. No GPU training starts before the 10M
queue finishes.

## Question

At the same approximately 1.18M parameter count and 15,990,784-token presentation budget, does
adding a 20% share of temporal-train MOT news change held-out text modeling relative to FSC v4
alone? This tests whether the 10M mixture hypothesis transfers to the smaller model; the 10M
result does not predict the 1M result.

## Matched arms

- **Control:** FSC CC0-screened v4 only, existing frozen tokenizer and validation/test shards,
  seed 421021, 488 optimizer steps.
- **Treatment:** same model shape, initialization seed, tokenizer, optimizer, learning-rate
  schedule, 488 steps and batch/token budget. Training stream targets 80% FSC v4 and 20% unique
  MOT v1.11 temporal-train tokens. Only as many MOT tokens as needed for this ratio are used; no
  document is repeated. The MOT prefix ends at the last complete document at or below the
  target; the manifest records the actual fraction, which can be slightly below 20%. Insufficient
  unique tokens cause preparation to stop.
- The 1M control is rerun to create a paired same-environment comparison. It is not a second
  displayed or promoted SLM-Tavenel 1M benchmark model.
- Both arms are evaluated once on FSC English/French dev and test, MOT English/French dev and
  2024+ test, plus the same six pinned multiple-choice benchmarks. The test results are descriptive
  and are not used for checkpoint selection. Benchmark outcomes are secondary and exploratory.

The paired seed controls the initial weights and data-order seed, but one pair does not estimate
training-seed variability. A positive result is a pilot signal only; confirmation requires multiple
paired seeds. A negative result also does not rule out other mixtures or schedules.

## Data, provenance and release status

The control data comes from the locally audited FSC v4 CC0 subset. Its final source/privacy review
remains unsigned. The treatment additionally uses MOT v1.11 English/French VOA material, revision
`9204cad746dfff0921e7a8f64c0cc0917bc75554`. MOT processing is documented in EXP-0012; English
author metadata is absent for 59.7% of records and remains missing. Its collection-level CC BY
label does not settle all item-level source and third-party rights. Keep raw article text local.
MOT-containing weights are not cleared for public release. This experiment must not be described
as a rights-cleared open-weight release.

Both FSC v4 and MOT training data were screened against six benchmark files. The FSC v4 test and
MOT post-2023 test remain outside training and selection. Exact filters cannot prove absence of all
benchmark exposure in pretraining sources. The FSC v4 test has already been scored in EXP-0010;
it is not a new untouched confirmatory test for EXP-0013.

## Reproduction

- Frozen FSC tokenizer: `data/tokenizers/fsc-cc0-open-weights-v4`.
- MOT train tokens: `data/tokens/exp0013-1m-mot-tokenized`.
- Mixture builder: `scripts/build_exp0013_1m_mixture.py`.
- Training configs: `configs/train/exp0013-1m-fsc-control.yaml` and
  `configs/train/exp0013-1m-fsc-mot-mixture.yaml`.
- Held-out evaluation: `scripts/evaluate_exp0013_1m.py`.
- Queue: `scripts/queue_exp0013_1m_after_exp0012.ps1`; events and outputs are kept under `runs/`.

## Research basis

Sedova et al. report repetition-aware mixture scaling across more than 2,000 runs and show that
mixture value depends on target-set size, model scale, and compute budget. This pilot tests a
simple fixed ratio; it does not apply their scaling-law optimum to SLM-Tavenel. Harness² targets
agent harnesses around frozen models and is not a direct base-weight pretraining method.

- Sedova et al. (2026), [Scaling Laws for Mixture Pretraining Under Data Constraints](https://arxiv.org/abs/2605.12715).
- Xu et al. (2026), [Harness²: Recursive Agent Harnessing for an Open World](https://arxiv.org/abs/2610.14224).
