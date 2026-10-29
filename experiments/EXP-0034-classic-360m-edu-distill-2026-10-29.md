# EXP-0034: classic 362M from scratch on educational web, with a distillation ablation

**Status (2026-10-29): preregistered; corpus being built.** Replaces the EXP-0026 498M growth run,
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

## EXP-0034b: classic 362M main run

Recipe from EXP-0034a, same corpus and tokenizer, WSD schedule. The token budget and schedule are set
after EXP-0034a from its measured throughput; the target is on the order of 0.5–1B token positions
(several days of the single GPU). Evaluation: held-out BPB, the six benchmarks with the same
evaluator, and the same benchmarks for SmolLM2-360M and Qwen3-0.6B (already scored) as descriptive
references.

## Caveats fixed in advance

- A distilled student inherits the teacher's behaviour, including any benchmark exposure in its
  pretraining data. A distilled model is labelled as such, with the teacher and revision.
- SmolLM2-360M saw about 4T tokens; a 362M student seeing under 1B tokens is not expected to reach it.
- Single training seed per arm; bootstrap intervals cover items/documents, not training variance.
- Web sources: page-level rights not verified; ODC-By covers the compilation. Weights stay local
  exploratory artifacts until reviewed.
