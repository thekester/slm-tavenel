# EXP-0040: byte-level H-Net vs BPE at matched data and compute (preregistered 2026-11-01)

Background: `experiments/AXIS-tokenizer-free-2026-10-31.md` (H-Net, arXiv:2507.07955; Cartesia
"From Tokens to Bytes" I/II, 2026-10-28; BLT, arXiv:2412.09871).

## Question

At the 75M-proxy scale of this project, does a one-stage H-Net reading raw UTF-8 bytes, with learned
dynamic chunking, reach lower held-out bits per byte than the same transformer reading SmolLM2 BPE
tokens, at matched training data (bytes) and approximately matched training compute? Cartesia's
controlled ladder reports a 1.28x compute advantage in-domain for one stage (about 2x on downstream
evaluations); their "10x" figure is an uncontrolled comparison and is not what this pilot tests.

## Arms

| Arm | Model | Input | Run |
|---|---|---|---|
| BPE (existing control) | `slm-75m-classic-proxy` (75.6M, of which 50.42M transformer) | SmolLM2 tokens, 2,048 per window | EXP-0036a Muon arm, `runs/EXP-0036a-75m-muon` |
| H-Net | `configs/model/hnet-75m-proxy.yaml` (57.4M: the identical 50.42M transformer as main network + 6.3M byte encoder/decoder) | bytes, 8,192 per window | `configs/train/exp0040-hnet-75m-muon.yaml` |

Matched: corpus v1 text (`data/tokens/slm-tavenel-edu-corpus-v1-bytes`, encoded by
`scripts/encode_bytes_corpus.py`), seed 20261029, Muon, WSD schedule (peak 3e-3, 100 warmup, 20%
decay), 1,500 updates. Data: 442M bytes (H-Net) vs 98.3M tokens x 4.42 bytes/token = 435M bytes
(BPE), +1.7%. Nominal training FLOPs per byte: H-Net 106.9M vs BPE 102.6M (+4.1%; about +7% with the
router and residual projections), i.e. the H-Net gets slightly more compute. Target compression
4.42 bytes per chunk (the BPE rate on this corpus), ratio-loss weight 0.03 as in the paper.

Implementation (`src/slm/hnet.py`, tests in `tests/test_hnet.py`): the paper's routing (cosine of
adjacent query/key projections, p_1 = 1), direct-selection chunking, EMA smoothing, causal expansion
with straight-through confidence, near-zero residual projection, ratio loss, width expansion by a shared
padding vector. Departures: sliding-window attention instead of Mamba-2 in the byte encoder/decoder
(the paper found attention outer layers slightly worse, which disadvantages the H-Net here); EMA
truncated to 32 chunks (error <= 2^-32); no per-stage learning-rate modulation (the paper's muP-style
rule is not reproduced; recorded as a limitation).

## Outcomes and criterion

Primary: held-out BPB on the full corpus-v1 test split (30,824 documents; English and French
separately), paired document bootstrap. Secondary: the six pinned benchmarks (acc and acc_norm; for the
byte model, log-probabilities are summed over bytes), learned bytes per chunk, wall-clock throughput.

**Criterion:** the H-Net is "better at matched data" if its overall test BPB is lower with a 95%
interval excluding zero. Because it receives ~4–7% more nominal compute, an advantage smaller than the
BPE arm's own BPB gain per 7% more compute is reported as not separable from the compute difference.
French and English effects are reported separately (the papers predict larger gains where the
tokenizer is weaker). Wall-clock cost is reported, not matched: masked/blocked attention on bytes is
slower per nominal FLOP on this GPU.

## Limitations fixed in advance

One seed; 75M scale (the paper's controlled gains are measured above ~200M); no Mamba-2; untuned
H-Net learning rates (the BPE arm's settings are reused); the comparison is not a test of the 2-stage
H-Net or of Cartesia's unreleased improved model.

## Results (2026-11-07)

Run `runs/EXP-0040-hnet-75m-muon` (1,500 updates, no instability; final grad_norm 0.07). Analysis
`runs/EXP-0040-evaluations/paired-comparison.json` (paired bootstrap, 10,000 draws; positive = H-Net better).

| Measure | BPE (EXP-0036a Muon) | H-Net (bytes) | BPE − H-Net [95% paired interval] |
|---|---:|---:|---|
| Held-out test BPB, all (30,824 docs) | 1.2592 | 1.3685 | −0.1094 [−0.1099, −0.1088] |
| BPB English (25,395) / French (5,429) | 1.2400 / 1.3901 | 1.3504 / 1.4924 | −0.1104 / −0.1023 |
| ARC-Easy acc | 0.421 | 0.334 | H-Net −0.087 [−0.108, −0.066] |
| ARC-Challenge acc | 0.172 | 0.157 | −0.015 [−0.037, +0.006] |
| HellaSwag acc | 0.264 | 0.276 | +0.011 [+0.006, +0.017] |
| Belebele EN / FR acc | 0.290 / 0.308 | 0.291 / 0.298 | +0.001 / −0.010 (intervals include 0) |
| WinoGrande acc | 0.504 | 0.501 | −0.002 [−0.035, +0.030] |
| Learned bytes per chunk (target 4.42) | — | 3.66 (step 100) → 3.59 (500) → 3.46 (1,000) → 3.30 (1,500) | — |
| Training throughput | ~11,800 tokens/s ≈ 52,000 bytes/s | ~24,000 bytes/s | H-Net ~2.2x slower in wall clock |

**Verdict: the H-Net is not better at matched data; it is clearly worse** (+8.7% BPB, interval far from
zero, in both languages; the French gap is not smaller than the English one, contrary to the
"weak-tokenizer" prediction). It is also much worse on ARC-Easy and slightly better on HellaSwag.

Deviations to report:
- **Compression target missed.** The router settled on ~3.3 bytes per chunk instead of 4.42 and kept
  drifting down, despite the ratio loss (weight 0.03, final value 1.03). The main network therefore
  processed ~34% more positions per byte than planned, i.e. the H-Net received *more* compute than the
  preregistered +4–7%, and still lost. A stronger ratio-loss weight would be needed to hold the target.
- **Micro-batch fallback.** Training ran at micro-batch 1 × 36 accumulation steps (same effective batch as
  planned) because the micro-batch-2 pilot failed on 2026-11-05 — due to a reshape bug in blocked local
  attention on CUDA (fixed the same day), not to memory. The run was resumed from checkpoints several
  times (PC shutdowns, a user pause).
- **Energy.** GPU board energy was measured for steps 500–1,500 only (logging added on 2026-11-06):
  197 Wh, about 50 W on average. The first 500 steps and the BPE control were not measured.

Interpretation: at 75M and ~440M bytes, with attention instead of Mamba-2 in the byte layers and the BPE
arm's learning rates, dynamic chunking does not pay off; the H-Net paper reports its controlled gains
from ~200M parameters upward with tuned per-stage learning rates. This pilot does not test those
conditions. Single seed. Next option (not started): 2–3 short runs with a stronger ratio loss
(e.g. 0.1–0.3) to first make the router hit 4.42, before any larger H-Net.
