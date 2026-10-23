# EXP-0012 and EXP-0013: paired data-mixture pilot results

**Status: complete as exploratory pilots.** Each model size has one matched FSC-only / FSC+MOT training-seed pair. The result supports a narrow description of these checkpoints and held-out sets; it does not estimate training-seed variability or confirm a general benefit from adding MOT data.

## Summary

The 10M and 1M mixture arms show lower bits per byte (BPB) on the MOT development and temporal test splits than their matched FSC-only controls, while BPB rises on the FSC evaluation splits. This is consistent with a domain trade-off in these particular runs. The six multiple-choice benchmark accuracy differences are small and every 95% paired item-bootstrap interval includes zero. These benchmark scores do not show a reliable downstream-task advantage for either mixture arm.

![Paired benchmark accuracy changes for the 10M and 1M FSC+MOT pilot arms](../docs/exp0012-13-mixture-benchmark-effects.svg)

*Each point is mixture accuracy minus its matched FSC-only control. Whiskers are 95% paired bootstrap intervals over benchmark items. They do not capture training-seed variation; each model size has one seed pair.*

## Design and evaluation

- **EXP-0012 (10M):** seed 1234, 15,000 optimizer steps per arm, 491.52 million token presentations, same architecture, tokenizer, optimizer, and schedule. The treatment uses a 20% MOT v1.11 / 80% FSC v1 training mixture.
- **EXP-0013 (1M):** seed 421021, 488 optimizer steps per arm, 15,990,784 token presentations, same architecture, frozen FSC v4 tokenizer, and optimizer. The treatment targets 20% unique MOT v1.11 training tokens.
- Both arms at each size were evaluated with the same local evaluator. Text metrics are descriptive held-out bits per byte. Multiple-choice results use the same six pinned benchmark files and exact per-item scoring records.
- The 1M FSC test was already evaluated in EXP-0010. It is document-held-out from this training run but is not a new untouched confirmatory test. The MOT test is a temporal 2024+ split; exact overlap screening does not rule out all pretraining or benchmark exposure.

## Text outcomes on development and test splits

The table reports mixture BPB minus matched-control BPB. **Negative values favor the mixture on that text set; positive values favor the FSC-only control.** BPB outcomes are single-pair point estimates and have no seed-level uncertainty interval.

### EXP-0012, 10M

| Evaluation split | FSC-only BPB | 20% MOT mixture BPB | Difference |
|---|---:|---:|---:|
| FSC English validation | 1.1777 | 1.2025 | +0.0249 |
| FSC French validation | 1.0912 | 1.1145 | +0.0233 |
| MOT 2022–2023 development | 1.6339 | 1.1875 | −0.4464 |
| MOT 2024+ temporal test | 1.6131 | 1.1719 | −0.4412 |

### EXP-0013, 1M

| Evaluation split | FSC-only BPB | 20% MOT mixture BPB | Difference |
|---|---:|---:|---:|
| FSC English development | 1.8001 | 1.8103 | +0.0103 |
| FSC English test | 1.7767 | 1.7880 | +0.0113 |
| FSC French development | 1.5728 | 1.6022 | +0.0293 |
| FSC French test | 1.5920 | 1.6186 | +0.0267 |
| MOT English development | 2.6208 | 1.8961 | −0.7247 |
| MOT French development | 2.4534 | 2.1742 | −0.2792 |
| MOT English 2024+ test | 2.6107 | 1.8714 | −0.7393 |
| MOT French 2024+ test | 2.4128 | 2.1185 | −0.2944 |

## Multiple-choice benchmark outcomes

Accuracy is raw accuracy. The paired interval resamples each benchmark's item-level correctness differences. It is conditional on these trained checkpoints and the evaluated items; it does not measure training variation. No family-wise correction was applied across the six tasks.

### EXP-0012, 10M

| Benchmark | Items | FSC-only | 20% MOT mixture | Difference, pp (95% paired item interval) |
|---|---:|---:|---:|---:|
| ARC-Easy | 2,376 | 0.307 | 0.298 | −0.84 [−2.61, +0.97] |
| ARC-Challenge | 1,172 | 0.181 | 0.169 | −1.19 [−3.24, +0.85] |
| HellaSwag | 10,042 | 0.274 | 0.270 | −0.48 [−0.99, +0.04] |
| Belebele English | 900 | 0.296 | 0.277 | −1.89 [−4.22, +0.44] |
| Belebele French | 900 | 0.303 | 0.313 | +1.00 [−1.44, +3.33] |
| WinoGrande XL validation | 1,267 | 0.520 | 0.497 | −2.29 [−5.60, +1.03] |

### EXP-0013, 1M

| Benchmark | Items | FSC-only replay arm | 20% MOT mixture | Difference, pp (95% paired item interval) |
|---|---:|---:|---:|---:|
| ARC-Easy | 2,376 | 0.278 | 0.279 | +0.04 [−1.56, +1.64] |
| ARC-Challenge | 1,172 | 0.195 | 0.200 | +0.51 [−1.45, +2.47] |
| HellaSwag | 10,042 | 0.260 | 0.261 | +0.12 [−0.42, +0.66] |
| Belebele English | 900 | 0.267 | 0.252 | −1.44 [−3.67, +0.78] |
| Belebele French | 900 | 0.274 | 0.260 | −1.44 [−3.44, +0.56] |
| WinoGrande XL validation | 1,267 | 0.494 | 0.501 | +0.71 [−1.74, +3.24] |

Every listed paired interval includes zero. The score changes are secondary, post-hoc outcomes and do not support a benchmark winner. The FSC-only replay arm in EXP-0013 is a paired experimental control; it is not a second promoted SLM-Tavenel 1M model.

## Context against other small models

The peer-model comparison remains the same-harness EXP-0010 chart: one selected SLM-Tavenel 1M candidate, the earlier exploratory 10M FSC pilot, TinyStories tiny1m, Pythia, and SmolLM2. It uses the same pinned evaluation files but compares checkpoints trained with different corpora, tokenizers, language mixes, compute budgets, and possible benchmark exposure. These mixture pilots are not added as promoted model versions to that chart. See the [benchmark comparison chart](../docs/exp0010-benchmark-comparison.svg) and [EXP-0010 results](EXP-0010-results-2026-10-22.md).

## Reproducibility and interpretation limits

The combined machine-readable analysis is [`EXP-0012-0013-mixture-pilot-results.json`](EXP-0012-0013-mixture-pilot-results.json). It records the paired seeds, checkpoint hashes, evaluation file hashes, per-task accuracies, item-bootstrap intervals, BPB values, and external same-harness reference results. Full item-level predictions remain with each arm under `runs/`.

The 10M evaluator originally scored ARC-Easy, HellaSwag, and both Belebele splits. ARC-Challenge and WinoGrande were then scored for both checkpoints in separate completion records using the same checkpoint, tokenizer, evaluator, and pinned files. The combined analysis verifies benchmark file hashes against both paired arms and the EXP-0010 references, plus item IDs, ordering, and labels, before calculating paired differences. `scripts/evaluate_exp0012_checkpoint.py` now lists all six tasks for a future full rerun.

The paired bootstrap used 20,000 draws with seed `20261022`, percentile 95% intervals, and no multiple-comparison adjustment. The experiments each use one training-seed pair, so item-level intervals cannot substitute for replication across seeds. Both FSC and MOT source-rights/privacy reviews remain unresolved; MOT-containing weights are not cleared for public release. This report makes no legal-clearance, capability, or general mixture-optimum claim.

## Run records

- [EXP-0012 protocol and data record](EXP-0012-10m-data-mixture-2026-10-22.md)
- [EXP-0013 protocol and data record](EXP-0013-1m-data-mixture-2026-10-22.md)
