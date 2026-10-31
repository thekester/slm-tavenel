# Plan (2026-10-30): an SLM-Tavenel model next to each open peer, same harness

Goal: train one SLM-Tavenel model close in size to each open reference, evaluate both with the
same pinned six-task evaluator (and held-out BPB on our corpus), and report where the SLM-Tavenel model
is behind, level or ahead. Each comparison is descriptive (different data, tokenizers and budgets);
the controlled results are the internal ablations listed with each step.

All new models share one recipe family (EXP-0034): classic decoder (RoPE, RMSNorm, SwiGLU, GQA,
QK-norm, head-wise attention gate, no biases, tied embeddings), SmolLM2 tokenizer (49,152, Apache-2.0),
edu corpus v2 (FineWeb-Edu EN + FineWeb2-HQ FR, ODC-By, ~30% French), WSD schedule, from scratch.

## Family, peers and measured throughput (RTX 4060 Laptop 8 GB)

| SLM-Tavenel | Params (non-emb.) | Open peers (same harness) | Memory mode | Tokens/s |
|---|---:|---|---|---:|
| 124M GPT-2-matched (`slm-124m-gpt2match`) | 124.0M (95.7M) | GPT-2 124M (MIT) | plain AdamW | ~6,500 (est. from 135M) |
| 135M (`slm-135m-classic`) | 134.7M | SmolLM2-135M, Pythia-160M | plain AdamW | 6,161 |
| 362M (`slm-360m-classic`) | 362.3M (315.1M) | SmolLM2-360M, GPT-2 Medium 355M, Pythia-410M | AdamW moments on host | 3,280 |
| 550M (`slm-550m-classic`) | 550.0M (487.1M) | Qwen3-0.6B, Qwen2.5-0.5B | bf16 device, fp32 masters on host (host-moment offload alone peaks at 7.8 GB, 1,487 tokens/s) | 2,059 |
| 1.04B (`slm-1b-classic`) | 1,045.0M (969.5M) | OLMo-2-1B, (SmolLM2-1.7B ceiling) | bf16 device, fp32 masters on host | 1,261 |

Peers added on 2026-10-30 at pinned revisions in `configs/eval/peer-models-2026-10.json`:
Pythia-70M/160M/410M (Apache-2.0, open data: The Pile), GPT-2 / GPT-2 Medium (MIT), Qwen2.5-0.5B
(Apache-2.0), OLMo-2-0425-1B (Apache-2.0, open data), SmolLM2-1.7B (Apache-2.0). Gated or restricted
(not scored): MobileLLM-125M/350M, Llama-3.2-1B, Gemma-3-270M; OpenELM (Apple licence, gated Llama-2
tokenizer). Script: `scripts/evaluate_peer_manifest.ps1`.

### Peer scores (same evaluator, pinned revisions, 1,024-token cap; raw accuracy)

| Model | Params | ARC-E | ARC-C | HellaSwag | Belebele EN | Belebele FR | WinoGrande | Mean | HellaSwag acc_norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia-70M | 70M | 0.317 | 0.194 | 0.268 | 0.282 | 0.306 | 0.487 | 0.309 | 0.273 |
| GPT-2 | 124M | 0.447 | 0.185 | 0.292 | 0.310 | 0.318 | 0.517 | 0.345 | 0.316 |
| Pythia-160M | 160M | 0.386 | 0.187 | 0.285 | 0.288 | 0.309 | 0.493 | 0.324 | 0.304 |
| GPT-2 Medium | 355M | 0.486 | 0.214 | 0.333 | 0.333 | 0.316 | 0.513 | 0.366 | 0.398 |
| SmolLM2-360M | 360M | 0.699 | 0.350 | 0.432 | 0.416 | 0.351 | 0.554 | 0.467 | 0.564 |
| Pythia-410M | 410M | 0.520 | 0.213 | 0.339 | 0.340 | 0.344 | 0.530 | 0.381 | 0.406 |
| Qwen2.5-0.5B | 494M | 0.658 | 0.291 | 0.405 | 0.400 | 0.388 | 0.548 | 0.448 | 0.519 |
| Qwen3-0.6B | 600M | 0.573 | 0.316 | 0.374 | 0.392 | 0.399 | 0.539 | 0.432 | 0.468 |
| OLMo-2-0425-1B | ~1B | 0.723 | 0.391 | 0.510 | 0.459 | 0.363 | 0.620 | 0.511 | 0.687 |
| SmolLM2-1.7B | 1.7B | 0.763 | 0.433 | 0.533 | 0.476 | 0.409 | 0.614 | 0.538 | 0.714 |
| SLM-Tavenel 75M proxy, CE, 98M tokens (EXP-0034a) | 75.6M | 0.364 | 0.184 | 0.260 | 0.272 | 0.308 | 0.507 | 0.316 | 0.262 |

GPT-2's HellaSwag here (0.292 raw) is consistent with the ~0.294 commonly reported. The EXP-0036
targets are therefore HellaSwag > 0.292 and six-task mean > 0.345 at 124M parameters.

## EXP-0036: beat GPT-2 small at equal parameters

**How GPT-2 was built** (Radford et al., 2019, "Language Models are Unsupervised Multitask
Learners"): WebText (~40 GB, ~8M pages linked from Reddit posts with >= 3 karma, not released),
byte-level BPE with 50,257 tokens, 1,024-token context, 12 layers x 768 width x 12 heads for the
124M model, learned absolute position embeddings, GELU MLP (4x), LayerNorm moved to the input of
each sub-block plus a final LayerNorm, residual-branch initialisation scaled by 1/sqrt(N), batch
512 sequences. The number of training tokens is not reported precisely in the paper.

**What later work shows.** Karpathy's llm.c reproduction (2024) trained the same 124M architecture on
10B FineWeb tokens and reported HellaSwag ~0.299 versus ~0.294 for the original checkpoint, with the
caveat that data distributions differ. The modded-nanogpt speedrun (Jordan et al., 2024–2025) reaches
the same FineWeb validation loss (3.28) with roughly 0.7–2B tokens by combining RoPE, QK-norm, ReLU^2,
the Muon optimizer and several architectural tricks. FineWeb-Edu (Penedo et al., 2024) shows that
educational filtering raises knowledge/reasoning benchmarks at equal tokens. Our own EXP-0034a 75M
proxy already reaches ARC-Easy 0.364 after 98M tokens of edu data.

**Design.** `slm-124m-gpt2match` (124.0M vs 124.4M total; note it has more non-embedding parameters,
95.7M vs ~85M, because the vocabulary is smaller; a 24-layer variant matching non-embedding size is
the sensitivity check). Recipe changes relative to GPT-2, all with references: RoPE (Su et al., 2021),
RMSNorm (Zhang & Sennrich, 2019), SwiGLU (Shazeer, 2020), GQA (Ainslie et al., 2023), QK-norm,
head-wise gate (EXP-0003/0004), no biases, WSD schedule (Hu et al., 2024), educational data.
Optimizer ablation first: AdamW vs Muon on the 75M proxy (98.3M tokens each, same seed), because
Muon's token efficiency matters most at our budgets. Main run: about 1B token positions (~43 h).
Comparison: the original GPT-2 124M checkpoint on the six tasks and on our held-out BPB (bytes, so
tokenizers do not matter), plus HellaSwag as the headline task used by llm.c.

**Success criterion (fixed in advance).** "Beats GPT-2 at equal parameters" means higher accuracy on
HellaSwag and on the six-task mean, with paired item intervals reported; BPB on our test split is
reported but favours our model (in-domain data), so it is not used as the criterion.

## Proposed GPU order and duration

| Step | Work | GPU time |
|---|---|---:|
| 1 | Peer evaluations (8 new public models) | ~3–4 h |
| 2 | EXP-0036a: Muon vs AdamW, 75M proxy, 98M tokens each | ~5 h |
| 3 | EXP-0036: GPT-2-matched 124M, ~1B tokens, best optimizer | ~43 h |
| 4 | EXP-0034b: 362M, 524M tokens | ~44 h |
| 5 | EXP-0037: 550M, ~400M tokens | ~54 h |
| 6 | EXP-0035: 1B, ~300M tokens | ~66 h |

The 135M SmolLM2-shape model is covered by the 124M GPT-2-matched run (same shape, 27 vs 30 layers);
a separate 135M run is optional. Budgets are far below the peers' (GPT-2: undisclosed, SmolLM2:
2–11T, Qwen: 18–36T, OLMo-2-1B: ~4T tokens), so only the GPT-2 comparison is expected to be close.
