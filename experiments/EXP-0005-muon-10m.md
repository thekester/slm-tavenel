# EXP-0005: Muon optimiser at 10M (ablation O1)

*Status:* concluded, single seed; second seed queued (`runs/slm-10m-wiki-muon-s2`)
*Dates:* 2026-10-19
*Runs:* `runs/slm-10m-wiki` (AdamW, EXP-0001) and `runs/slm-10m-wiki-muon`

## Question and hypothesis

Does Muon (Jordan et al., 2024) with the update scaling of Liu et al. (2025, arXiv:2502.16982) reach
a given loss with fewer tokens than AdamW? With that scaling the same learning rate (3e-3), weight
decay (0.1) and WSD schedule can be reused, so the only change is `optimizer: muon` (Muon for the 2D
weights inside blocks, AdamW for embeddings and norms). Same seed (1234), data order and validation
windows as the baseline.

## Results

| Metric | AdamW | Muon | Difference |
|---|---|---|---|
| Validation loss, step 15,000 | 2.7495 | 2.7154 | -0.034 |
| Bits per byte, EN (500 docs) | 1.3163 | 1.3003 | -0.016 |
| Bits per byte, FR (500 docs) | 1.1562 | 1.1416 | -0.015 |
| Throughput during training | ~67k tokens/s | ~67k tokens/s | no visible overhead |

Loss during the constant-learning-rate phase:

| Step | AdamW | Muon | Steps Muon needs to match AdamW |
|---|---|---|---|
| 4,000 | 3.089 | 3.041 | 3,500 (88%) |
| 8,000 | 2.989 | 2.949 | 6,000 (75%) |
| 12,000 | 2.945 | 2.911 | 9,000 (75%) |

During the stable phase Muon matches AdamW with about 75% of the steps (about 1.33 times fewer
tokens). The step at which Muon matches AdamW's *final* loss (14,500) is not a fair measure, because
the final loss of both runs comes from the decay phase, which starts at step 12,000 in both.

Benchmarks (chance 25%): ARC-Easy 0.306 against 0.295 (+0.012, standard error of the difference
about 0.013), HellaSwag 0.273 against 0.275, Belebele EN 0.283 against 0.266, Belebele FR 0.302
against 0.309. No difference beyond noise.

The bits-per-byte gain (0.015-0.016) is larger than the gated attention gain at the same seed
(0.012) and 7 to 13 times the seed-to-seed difference measured in EXP-0004 (0.001-0.002).

## Internals

- **Sinks are still present** with Muon: 25% and 21% of attention on the first token in layers 8 and
  9, a position-0 activation of 730. Muon does not address sinks; the gate does. The two changes act
  on different things and can be combined.
- Neutralizing the massive dimensions at position 0 removes only 9% of the sink here (0.066 to
  0.060), against 37-71% in the AdamW baselines: in this run the sink depends less on those
  dimensions.
- Median update-to-weight ratio at step 14,500 (during decay): 0.00136 with Muon, 0.00160 with
  AdamW.

## Decision

Provisionally positive. Adoption waits for the second seed, following the protocol. If it confirms,
the next 10M run combines Muon and the gate before moving both to 135M, where the learning rate
should be re-tuned: Muon may tolerate a higher rate than the one inherited from AdamW.
