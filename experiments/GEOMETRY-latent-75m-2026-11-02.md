# Latent geometry of the 75M proxies (2026-11-02, exploratory, CPU only)

Prompted by the geometric-deep-learning / computational-topology discussion; method and estimator
details in `scripts/latent_geometry.py` (tested in `tests/test_latent_geometry.py`). Checkpoints: the
three 75M proxies trained on corpus v1 with identical data order and budget (98.3M tokens): AdamW CE
(EXP-0034a), AdamW with SmolLM2-360M distillation (EXP-0034a) and Muon CE (EXP-0036a). Residual
stream after every block on 16 validation windows of 512 tokens (seed 0), 4,000 token vectors per layer
(first 16 positions of each window excluded), centred.

| Layer | 0 | 2 | 4 | 6 | 8 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoNN ID, AdamW | (1.5) | 2.3 | 2.7 | 3.0 | 4.0 | 5.9 | 7.2 | 7.8 | 8.5 | 8.4 | 8.3 |
| TwoNN ID, distillation | (1.5) | 2.5 | 3.3 | 4.1 | 5.0 | 6.6 | 7.6 | 8.3 | 8.8 | 8.6 | 8.3 |
| TwoNN ID, Muon | (1.4) | 2.4 | 3.5 | 4.7 | 6.3 | 8.0 | 8.9 | 9.1 | 9.0 | 8.6 | 8.1 |

Participation ratio (linear dimension of the covariance spectrum), first / middle (layer 8) / last
layer: AdamW 49 / 79 / 30; distillation 49 / 89 / 35; Muon 66 / 110 / 27.

## Observations

- In all three models the nonlinear intrinsic dimension grows through most of the depth, peaks two to
  three layers before the output (layers 12–13) and then decreases, i.e. a rise-then-fall profile
  qualitatively like the one reported for larger transformers (Valeriani et al., 2023), with the peak
  late in these small, short-trained models.
- Muon's representations expand earlier: its intrinsic dimension is higher from layer 4 to layer 12 and
  peaks one layer earlier, and its middle-layer participation ratio is about 40% higher (110 vs 79).
  This fits the expectation that orthogonalised updates spread energy over more directions (Muon
  equalises the singular values of each update); distillation sits between the two.
- The last layer is more compressed (lower participation ratio) for Muon than for AdamW.

## Limits

One checkpoint per recipe and one sample of windows: no uncertainty estimate yet (a bootstrap over
windows is the next step). TwoNN assumes locally uniform density; at layer 0 the vectors form
near-discrete clusters (one per token), so the layer-0 values (in parentheses) are not interpretable.
Absolute dimension values depend on the estimator and sample size; only within-table comparisons are
meaningful. These are descriptive correlations with the BPB differences of EXP-0034a / EXP-0036a, not
evidence that higher intrinsic dimension causes lower loss.
