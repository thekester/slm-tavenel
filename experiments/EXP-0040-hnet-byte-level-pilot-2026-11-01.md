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
