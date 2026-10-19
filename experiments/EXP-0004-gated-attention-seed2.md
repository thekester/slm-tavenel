# EXP-0004: gated attention at 10M, second seed

*Status:* concluded
*Dates:* 2026-10-19
*Runs:* `runs/slm-10m-wiki-s2` (baseline) and `runs/slm-10m-wiki-gated-s2` (head-wise gate), both with
`--set seed=2`; otherwise identical to EXP-0001 and EXP-0003.

The tables label the original training seed (`1234`) as "seed 1" and the replication seed (`2`) as
"seed 2". These labels refer to training runs, not the sampling seed used by the analysis script.

## Question

EXP-0003 found, at one seed, that a head-wise attention output gate lowers validation loss by 0.025
and removes attention sinks. The protocol requires a second seed before adoption. The seed changes
the initialisation, the training data order and the validation windows.

## Seed variance

Validation loss is computed on windows drawn from the run seed, so it is comparable between arms at
the same seed but not across seeds. Across seeds, the comparison uses bits per byte on the same 500
held-out documents per language.

| Bits per byte | Baseline seed 1 | Baseline seed 2 | Difference |
|---|---|---|---|
| English | 1.3163 | 1.3151 | 0.0012 |
| French | 1.1562 | 1.1540 | 0.0022 |

## Results

| | Seed 1 baseline | Seed 1 gated | Seed 2 baseline | Seed 2 gated |
|---|---|---|---|---|
| Validation loss (same-seed windows) | 2.7495 | 2.7243 | 2.6768 | 2.6551 |
| Bits per byte, EN | 1.3163 | 1.3048 | 1.3151 | 1.3028 |
| Bits per byte, FR | 1.1562 | 1.1446 | 1.1540 | 1.1449 |
| Mean first-token attention (128 windows) | 0.111 | 0.014 | 0.080 | 0.016 |
| Max activation at position 0 | ~890 | 66 | 889 | 187 |
| ARC-Easy acc | 0.295 | 0.293 | 0.296 | 0.289 |
| HellaSwag acc | 0.275 | 0.273 | 0.272 | 0.271 |
| Belebele EN acc | 0.266 | 0.277 | 0.270 | 0.283 |
| Belebele FR acc | 0.309 | 0.319 | 0.314 | 0.299 |

Effect of the gate, per seed:

| | Seed 1 | Seed 2 |
|---|---|---|
| Validation loss | -0.025 | -0.022 |
| Bits per byte, EN | -0.0115 | -0.0123 |
| Bits per byte, FR | -0.0116 | -0.0091 |

The gain in bits per byte is 4 to 10 times the difference between seeds of the same architecture and
has the same sign and similar size at both seeds. Benchmark differences remain within noise (standard
errors 0.006 to 0.022), as at seed 1.

## Sinks at seed 2

- The baseline again forms a sink, but in different layers: layers 8 and 9 (25% and 20% of attention
  on the first token) instead of layers 4 and 6-8 at seed 1. Where sinks form is seed dependent; that
  they form is not.
- Neutralizing the three massive dimensions at position 0 removes 71% of the baseline's sink mass
  (0.080 to 0.023; 37% at seed 1) for +0.073 loss; random dimensions: no effect.
- The gated model has no sink (at most 2.9% in any layer) and the intervention has no effect.
- Seed 1 found that ablating the strongest sink heads was no worse than ablating random heads. At seed
  2 it is worse (+0.286 against +0.196 on average for random heads). The claim "sink heads are not
  special" from EXP-0002 therefore does not hold in general and is withdrawn.

## Decision

Adopt the head-wise gate for the 135M ablations (A1 at 135M confirms or reverses it at a scale where
benchmarks can resolve differences). Reasons: a consistent quality gain across two seeds, no sinks or
position-0 massive activations, +0.1% parameters. Costs to carry: no standard Hugging Face equivalent,
so open weights need custom modelling code; and a controlled throughput comparison is still pending
(`slm bench`).
