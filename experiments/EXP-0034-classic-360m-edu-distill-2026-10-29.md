# EXP-0034: classic 362M from scratch on educational web, with a distillation ablation

**Status (2026-11-15): EXP-0034a and EXP-0034b complete (results below).**
Corpus v2 (`data/processed/slm-tavenel-edu-corpus-v2`, same builder with five FineWeb2-HQ French
shards) has 1,996,506 train documents (587,744 French) and 2.140B train tokens with the SmolLM2
tokenizer (`data/tokens/slm-tavenel-edu-corpus-v2-smollm2`); French is estimated at about 30% of
train tokens (v2 minus v1 token totals, since the English shards are identical).
Initial status (2026-10-29): preregistered; corpus being built. Replaces the EXP-0026 498M growth run,
stopped on 2026-10-29 before its first checkpoint after EXP-0025 showed that FFN growth followed by a
49M-token continuation gives only 0.0012 BPB over a token-matched 100M control.

## Motivation

EXP-0025 indicates that, in the d_model-256 / 80-layer lineage, token budget rather than parameter
count limits quality. That lineage also uses the GPU poorly (about 1,800 token positions/s at 301M).
This experiment changes the recipe on four axes, each documented and, where affordable, ablated:

1. **Architecture**: the SmolLM2-360M / MobileLLM-350M shape (32 layers, width 960, 15 query and 5
   key-value heads, SwiGLU 2,560, tied embeddings), keeping the project's head-wise attention gate and
   QK-norm. `configs/model/slm-360m-classic.yaml`, 362,286,016 parameters. Trained from scratch.
2. **Tokenizer**: SmolLM2's 49,152-token BPE (Apache-2.0), pinned at
   `HuggingFaceTB/SmolLM2-360M@f8027fd0`, SHA-256 `9ca9acdd…e47c`
   (`data/tokenizers/smollm2-f8027fd0/manifest.json`). It compresses text better than the 8,192-token
   project tokenizer and allows logit distillation from SmolLM2.
3. **Data**: `data/processed/slm-tavenel-edu-corpus-v1` (`scripts/build_edu_corpus_v1.py`):
   - English: `HuggingFaceFW/fineweb-edu@87f09149`, two `sample/10BT` shards (classifier score >= 3);
   - French: `epfml/FineWeb2-HQ@c0c06e94`, one `fra_Latn` shard;
   - both declared ODC-By 1.0 (plus Common Crawl terms of use); PII redaction, exact dedup, exact
     13-token decontamination against the six benchmark files, registrable-domain split
     98/1/1 (train/dev/held-out test). Shard hashes and decision counts are in its manifest.
4. **Distillation** (ablated in EXP-0034a): loss `(1 - w) CE + w T^2 KL(teacher || student)` over the
   full vocabulary (Hinton et al., 2015), teacher `HuggingFaceTB/SmolLM2-360M@f8027fd0`
   (Apache-2.0), w = 0.5, T = 1.

Memory: the 362M model trains on the 8 GB RTX 4060 Laptop with activation recomputation and AdamW
moments in host memory (EXP-0025 amendment), micro-batch 2 x 2,048 tokens: 2,966 token positions/s,
6.0 GB peak (random-token benchmark). Without offload the run exceeds device memory (9.0 GB) and
drops to 1,150 token positions/s.

## EXP-0034a: distillation ablation at 75M (preregistered)

| Arm | Config | Loss |
|---|---|---|
| Control | `configs/train/exp0034a-75m-ce.yaml` | next-token cross-entropy |
| Treatment | `configs/train/exp0034a-75m-distill.yaml` | 0.5 CE + 0.5 KL from SmolLM2-360M |

75.6M-parameter proxy of the 362M shape (`configs/model/slm-75m-classic-proxy.yaml`), from scratch,
same seed (20261029), data order, WSD schedule (peak 3e-3, 100 warmup steps, 20% linear decay) and
1,500 updates of 65,536 token positions (98.3M). Primary outcome: bits per byte on the corpus's
held-out test split, paired document bootstrap. Secondary: the six pinned multiple-choice benchmarks,
paired item bootstrap. Training time is reported, because the teacher forward pass costs compute.

**Decision rule.** The 362M run uses distillation if the treatment lowers held-out BPB with a 95%
interval excluding zero and the six-benchmark mean accuracy does not decrease, after accounting for
its throughput cost; otherwise it uses cross-entropy only.

### EXP-0034a results (2026-10-30)

Both arms completed 1,500/1,500 updates (98.3M token positions) on corpus v1. The CE arm stopped at
step 350 when the PC shut down and resumed from its step-250 checkpoint. Throughput measured with an
idle CPU: 12,100 token positions/s (CE) and 5,760 (distillation), so the distillation arm costs about
2.1x the GPU time per token at 75M. Paired analysis:
`runs/EXP-0034-evaluations/exp0034a-paired-comparison.json`.

| Measure | CE | Distill | Distill effect (95% paired bootstrap) |
|---|---:|---:|---|
| Validation CE (same windows, step 1,500) | 3.840 | 3.743 | −0.097 nats |
| Held-out test BPB, all (30,824 docs) | 1.3300 | 1.2989 | −0.0310 [−0.0313, −0.0308] |
| Held-out test BPB, EN (25,395) | 1.3094 | 1.2756 | −0.0338 [−0.0339, −0.0336] |
| Held-out test BPB, FR (5,429) | 1.4710 | 1.4584 | −0.0126 [−0.0131, −0.0120] |
| ARC-Easy acc | 0.364 | 0.385 | +0.021 [+0.004, +0.037] |
| ARC-Challenge acc | 0.184 | 0.166 | −0.019 [−0.038, +0.001] |
| HellaSwag acc | 0.260 | 0.263 | +0.003 [−0.001, +0.008] |
| Belebele EN acc | 0.272 | 0.297 | +0.024 [+0.003, +0.047] |
| Belebele FR acc | 0.308 | 0.301 | −0.007 [−0.027, +0.013] |
| WinoGrande acc | 0.507 | 0.492 | −0.016 [−0.042, +0.010] |
| Six-task mean acc | 0.316 | 0.317 | +0.002 |

At equal tokens, distillation lowers held-out BPB by 2.3% (larger in English than French, matching
the teacher's mostly English pretraining). Benchmark effects are mixed: ARC-Easy and Belebele EN
improve with intervals excluding zero, ARC-Challenge and WinoGrande move the other way without
excluding zero, and the six-task mean is unchanged. With twelve contrasts, single intervals are not
treated as findings. The preregistered token-level criterion is met (BPB lower, mean accuracy not lower).

**Compute.** The gain must be set against a 2.1x GPU cost. Within the CE arm, validation CE fell from
4.341 at 49M tokens to 3.840 at 98M (including the learning-rate decay), far more than the 0.097-nat
distillation gain, so at 75M and this budget, spending the teacher's compute on more CE tokens is
likely better. A compute-matched CE arm was not run, so this is an inference from the learning curve,
not a measured comparison. Corpus v1 was also French-light (6.4% of bytes).

## EXP-0034b: classic 362M main run

Recipe derived from EXP-0034a, using expanded corpus v2, the same tokenizer and a WSD schedule. The token budget and schedule are set
after EXP-0034a from its measured throughput; the target is on the order of 0.5–1B token positions
(several days of the single GPU). Evaluation: held-out BPB, the six benchmarks with the same
evaluator, and the same benchmarks for SmolLM2-360M and Qwen3-0.6B (already scored) as descriptive
references.

### EXP-0034b result (2026-11-15)

Run `runs/EXP-0034b-360m-muon` (`configs/train/exp0034b-360m-muon.yaml`): 362.3M parameters, Muon,
WSD, corpus v2, SmolLM2 tokenizer, 8,000 updates x 65,536 = **524.3M token positions** (about half of
the 983M seen by the EXP-0036 124M). Final checkpoint `step_00008000.pt` (sha256 `281c53a7…`),
final validation loss 2.901 (training loss 2.80; late logged gradient norms around 0.13 to 0.17;
the full logged range is 0.13346 to 3.33809, not a formal stability guarantee).
Trained 11–15 November 2026 with four resumes after PC shutdowns or standby; the resume at step 7,400
reproduced the logged loss at step 7,450 to 4 decimals. **Final logged GPU-board energy counter: 2,885.6 Wh**
(evaluation, whole-system electricity and possibly recomputed/unmetered work are excluded).
Logged PyTorch allocation peak 3.46 GB (AdamW moments on host).

Benchmarks: `runs/FAMILY-evaluations/360m-muon/benchmarks.json` (same evaluator and items as the peers).
Paired comparisons by `scripts/compare_mc_paired.py` (items paired by position, 10,000 bootstrap
draws, exact McNemar): `vs-124m-paired.json`, `vs-gpt2-medium-paired.json`, `vs-pythia-410m-paired.json`,
`vs-smollm2-360m-paired.json` in the same folder.

| Task (items) | SLM 362M | SLM 124M (EXP-0036, 983M tok.) | GPT-2 Medium 355M | Pythia-410M | SmolLM2-360M |
|---|---:|---:|---:|---:|---:|
| ARC-Easy (2,376) | 0.484 | 0.484 | 0.486 | 0.520 | 0.699 |
| ARC-Challenge (1,172) | 0.212 | 0.206 | 0.214 | 0.213 | 0.350 |
| HellaSwag (10,042) | 0.279 | 0.277 | 0.333 | 0.339 | 0.432 |
| Belebele EN (900) | 0.298 | 0.316 | 0.333 | 0.340 | 0.416 |
| Belebele FR (900) | 0.302 | 0.311 | 0.316 | 0.344 | 0.351 |
| WinoGrande (1,267) | 0.512 | 0.519 | 0.513 | 0.530 | 0.554 |
| **Six-task mean** | **0.348** | 0.352 | 0.366 | 0.381 | 0.467 |

Paired differences, SLM 362M minus reference (six-task mean; HellaSwag). External references used
a 1,024-token context cap versus 2,048 for both SLM-Tavenel models; longer-item truncation can differ:

| Reference | Six-task mean [95% CI] | HellaSwag [95% CI], McNemar p |
|---|---|---|
| SLM 124M (EXP-0036) | −0.004 [−0.013, +0.004] | +0.002 [−0.003, +0.006], p = 0.46 |
| GPT-2 Medium | −0.018 [−0.028, −0.008] | −0.054 [−0.060, −0.048], p < 1e-70 |
| Pythia-410M | −0.033 [−0.044, −0.023] | −0.061 [−0.067, −0.055], p < 1e-85 |
| SmolLM2-360M | −0.119 [−0.130, −0.108] | −0.154 [−0.162, −0.146] |

Against GPT-2 Medium, only Belebele EN is also significantly lower (−0.036 [−0.063, −0.008],
p = 0.013); ARC-E, ARC-C, Belebele FR and WinoGrande are level (all |diff| ≤ 0.013, intervals
include 0).

Held-out test BPB, corpus v2, systematic 1-in-5 subsample (10,156 documents;
`edu-test-bpb.json`): **1.057** overall (EN 1.079, FR 1.026). On exactly the same documents the
EXP-0036 124M gives 1.069 (EN 1.089, FR 1.038): 362M − 124M = −0.011 [−0.012, −0.011] bits/byte
(document bootstrap, 2,000 draws; `scripts/compare_exp0034b_vs_124m_bpb.py` -> `paired-vs-124m.json`), both languages. Peers were not scored on
this split; BPB is in-domain for our models and is not a peer criterion.

**Reading.** As expected (§ caveats), the 362M is far from SmolLM2-360M (~4T tokens) and is
significantly behind GPT-2 Medium and Pythia-410M, almost entirely through HellaSwag (−5 to −6
points), the same everyday/web-narrative gap already seen at 124M (EXP-0036: edu data trades
HellaSwag for ARC). The striking result is internal: tripling parameters at about half the tokens buys
a small, significant BPB gain (−1.1%) but **no benchmark gain** over the 124M (mean −0.004, interval
includes zero). At 524M token positions (1.4 tokens per parameter, against ~20 for compute-optimal
training per Hoffmann et al., 2022), insufficient training is a plausible explanation, not an isolated causal result.
Interpretation limits: single seed; the 124M and 362M differ in both size and token count, so this is
not a clean size ablation; benchmark intervals cover items, not training variance.
Compute: final logged counter of 2,886 Wh for the 362M; the 124M's training energy was not metered (EXP-0036 predates
the meter), so an energy-per-point comparison is not available.

## Caveats fixed in advance

- A distilled student inherits the teacher's behaviour, including any benchmark exposure in its
  pretraining data. A distilled model is labelled as such, with the teacher and revision.
- SmolLM2-360M saw about 4T tokens; a 362M student seeing under 1B tokens is not expected to reach it.
- Single training seed per arm; bootstrap intervals cover items/documents, not training variance.
- Web sources: page-level rights not verified; ODC-By covers the compilation. Weights stay local
  exploratory artifacts until reviewed.

## Public research package

[SLM-Tavenel 362M on Hugging Face](https://huggingface.co/athenista/slm-tavenel-362m) includes the exact native-evaluation float32 values, tokenizer attribution, architecture/data/training/benchmark figures, compressed item/document evidence and fixed-probe export/cache checks. See the [model card](../docs/model-cards/slm-tavenel-362m/README.md) for the recipe and limitations. Source-rights and manual privacy reviews remain unsigned; publication is an owner-authorized research preview, not legal clearance or OSI certification.

The historical local-only publication condition above is superseded for this exact final checkpoint
by the owner's 15 November publication request, recorded with both native and export hashes in the
model package. This authorization does not sign the source-rights or privacy reviews.

The original convenience comparison script used dictionaries keyed by benchmark ID, which dropped repeated HellaSwag IDs. That secondary calculation is superseded: benchmark pairing now reuses the full 10,042-item positional comparison in `vs-124m-paired.json`; the BPB comparison explicitly matches all 10,156 treatment documents against the larger 124M report. The main benchmark table above already used the full positional results.
