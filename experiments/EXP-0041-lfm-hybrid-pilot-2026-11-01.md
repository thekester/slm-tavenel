# EXP-0041: LFM-style convolution/attention hybrid at 75M (preregistered 2026-11-01)

## Source and claims

Liquid AI released LFM3 on 2026-10-27 (model card `LiquidAI/LFM3-1.7B`): 1.73B parameters, 24 layers
(13 gated short convolutions, 5 sliding-window attention, 6 global attention layers) found by a
hardware-in-the-loop search, 128k vocabulary, 256k context, 22T pretraining tokens plus 1.75T SFT tokens,
RL and on-policy distillation; LFM Open License v1.0 (not Apache; terms to be read before any reuse).
The "beats models six times larger" headline refers to Liquid's own 29-benchmark average, where the
post-trained LFM3-1.7B scores 49.37 against 47.24 for Qwen3.5-9B; it trails that model on several tests
(e.g. GPQA Diamond 63.4 vs 80.4, RULER-128k 65.6 vs 87.1). These are vendor numbers for a heavily
post-trained model (22T tokens), not an architecture-only comparison. The architecture is documented in
the LFM2 technical report (arXiv:2511.23404): gated short convolutions plus a few grouped-query attention
layers, where the search found no benefit from adding state-space or linear-attention operators once
a few attention layers were present; the main reported benefit is CPU/edge latency and memory.

## Question

Holding data, tokenizer, optimizer and token budget fixed, does replacing 10 of 16 attention layers by
gated short convolutions change held-out BPB and benchmark accuracy at 75M, and what does it change in
training throughput on this GPU?

## Design

| Arm | Model | Run |
|---|---|---|
| Control (existing) | `slm-75m-classic-proxy` (all attention, 50.42M non-embedding) | EXP-0036a Muon arm |
| Hybrid | `configs/model/slm-75m-lfm2-hybrid.yaml` (conv layers 0,1,3,4,6,7,9,11,13,15; attention at LFM2-350M's positions 2,5,8,10,12,14; 53.01M non-embedding, +5.1%) | `configs/train/exp0041-75m-lfm2-hybrid-muon.yaml` |

Block (`slm.model.GatedShortConv`, after LFM2): (b, c, h) = in_proj(x); y = c * causal depthwise
conv_k3(b * h); out = o_proj(y); followed by the same SwiGLU feed-forward and pre-norms. Same corpus v1,
SmolLM2 tokenizer, seed 20261029, Muon, WSD (peak 3e-3), 1,500 updates of 65,536 token positions.

Outcomes: held-out BPB on the corpus-v1 test split (EN/FR), the six pinned benchmarks (paired
intervals), validation CE on the same windows, training throughput. **Criterion:** the hybrid is
"not worse" if its BPB is within +0.5% of the control's (the hybrid has 5% more non-embedding
parameters, so a gain smaller than that is not attributed to the architecture); "better" requires
lower BPB with the interval excluding zero by more than 0.5%. Throughput is reported, not matched.
Edge/CPU latency, LFM's main design target, is not measured here.

## Limitations

One seed, 75M scale; layout copied from LFM2-350M rather than searched; no sliding-window/global split
(LFM3's tri-hybrid) at this size; the cached-generation path is not implemented for convolution layers
(scoring and training only).
