# Axis: platonic representations, good views and JEPA (2026-11-07)

Source: Randall Balestriero & Yann LeCun, "Proving and Improving the Platonic Representation Hypothesis
with JEPAs", arXiv:2610.39302 (v1, 29 Oct 2026, CC BY 4.0). The 11 November audit located the
[authors' code](https://github.com/galilai-group/prh-proof), linked from the paper. Builds on the Platonic
Representation Hypothesis (Huh, Cheung, Wang & Isola, ICML 2024: models of different modalities converge
toward a shared representation, measured by mutual nearest-neighbour alignment).

## Claims (as stated by the paper; not reproduced yet)

- **Theorem.** Spectral JEPAs (prediction term + second-order anti-collapse term) learn the top
  eigenvectors of the graph defined by their positive pairs. JEPAs trained separately on different
  modalities therefore learn the same representation up to a rotation when their positives are *good
  views* (independent observations of the same world state: two photos of a scene, two captions of it);
  the RMS distance after the best rotation is bounded by the kernel difference over the spectral gap.
- **Bad views.** Corruptions of a single observation (two crops, two maskings) share nuisance, which the
  JEPA spends dimensions on and which no rotation can align across modalities. Next-token prediction is
  read as a JEPA with bad views (prefix and continuation of one document): "a remarkably inefficient route
  to platonic representations".
- **Results.** Good views need 42x fewer training samples than next-token prediction to reach the same
  alignment; a 109M good-view text JEPA trained on 900M image-derived text samples aligns with vision
  encoders (DINOv2, CLIP) better than language models up to 72B parameters (e.g. ImageNet-100 zero-shot
  top-5 64% vs at most 56% for LMs up to 8B); a captioner trained on text only, connected to a frozen
  vision encoder by **one rotation** fitted on a few image-caption pairs, reaches CIDEr 0.838 on COCO; a
  scaling law of alignment predicts unseen runs within seed noise. Alignment is measured by mutual
  10-NN, linear CKA, retrieval R@1/R@5 after a fitted rotation, FOSCTTM and zero-shot accuracy.

What it does not say: it does not show that JEPA text models are better *language models* (no
perplexity or generation-quality claim), and "42x" is about reaching an alignment level, not about
benchmark accuracy.

## Relevance to SLM-Tavenel

1. **Our models are bilingual.** Parallel English/French passages are good views of the same content.
   Next-token training sees them only as unrelated documents (bad views). This gives a cheap, text-only
   test of the paper's argument on our own models.
2. **JEPA is already in the codebase** (`src/slm/jepa.py`, EXP-0038/0039) with token- and state-level
   variants, but used with bad views (next state of the same sequence).
3. **Geometry tooling exists** (`scripts/latent_geometry.py`): alignment metrics extend it naturally.

## Plan

| Step | Work | Resource | Status |
|---|---|---|---|
| 1 | Measure EN↔FR alignment of existing checkpoints (75M AdamW/distill/Muon/hybrid, 124M) on the 488 parallel Belebele passages: per-layer mutual 10-NN, R@1 cosine, R@1 after Procrustes rotation; random-init controls (`scripts/platonic_alignment.py`) | CPU | running 2026-11-07 |
| 2 | Preregister EXP-0043: 75M proxy, next-token loss + a good-view JEPA term on EN/FR parallel pairs (open-licensed parallel data, e.g. Tatoeba EN–FR CC-BY 2.0 FR, after checking terms), vs next-token alone at equal tokens; outcomes: EN↔FR alignment on held-out parallel passages (FLORES/Belebele, never trained on), held-out BPB, six benchmarks | GPU ~2 × 3 h | after decision |
| 3 | Small reproduction of the paper's direction: ~30M text JEPA on COCO caption pairs (five captions per image = good views; annotations CC BY 4.0) vs ~30M next-token model on the same captions; mutual-kNN alignment with a frozen DINOv2 (Apache-2.0) on COCO val images | GPU + image download | optional |

## Step 1 result (2026-11-09): English/French alignment of existing checkpoints

`runs/PLATONIC/belebele-en-fr-alignment.json` (`scripts/platonic_alignment.py`; 488 unique parallel
Belebele passages; best layer by mutual 10-NN; a random-init copy of each architecture as control; chance:
mutual 10-NN 0.021, R@1 0.002). A first computation on 900 items with repeated passages (deduplication on
question ids) was invalid; it is archived as `belebele-en-fr-alignment-WITH-DUPLICATES-invalid.json`.

| Model | Training | Best layer | Mutual 10-NN | R@1 cosine | R@1 after rotation |
|---|---|---:|---:|---:|---:|
| 75M AdamW CE (EXP-0034a) | corpus v1, 98M tokens | 12 / 16 | 0.342 | 0.889 | 0.887 |
| 75M AdamW + distillation (EXP-0034a) | same + SmolLM2-360M teacher | 13 / 16 | 0.327 | 0.844 | 0.857 |
| 75M Muon CE (EXP-0036a) | corpus v1, 98M tokens | 12 / 16 | 0.395 | 0.953 | 0.947 |
| 75M LFM2 hybrid, Muon (EXP-0041) | corpus v1, 98M tokens | 9 / 16 | 0.397 | 0.957 | 0.949 |
| 124M Muon (EXP-0036) | corpus v2, 983M tokens | 23 / 27 | **0.681** | **1.000** | **1.000** |
| Random-init controls (best layer is 0) | none | 0 / — | 0.087–0.124 | 0.236–0.270 | 0.148–0.221 |

Observations:
- Next-token training aligns English and French far above the random-init controls (which are themselves
  above chance through shared subwords, numbers and names, mostly at layer 0).
- Alignment rises with depth and peaks a few layers before the output (2–3 layers for the 16-layer AdamW and
  Muon models, 6 for the hybrid, 3 for the 124M), then drops in the last layers.
- At equal data, Muon (0.395) and the conv/attention hybrid (0.397) align more than AdamW (0.342);
  distillation from SmolLM2 does not help (0.327).
- The 124M is much more aligned (0.681) and retrieves the right translation for all 488 passages, with a
  sharp rise between layers 11 and 13. Its size, 10x token budget and corpus (v2, ~25% French vs v1) all
  differ, so this is not a scale effect alone; R@1 is at its ceiling on this set.
These are descriptive, single-checkpoint measurements on one passage set; no interval is computed here.

Caveats fixed now: Belebele passages are benchmark items (used here for analysis only, so any later
training on parallel data must exclude FLORES/Belebele text); our models are small and trained on ≤2B
tokens, so low absolute alignment is expected; shared subwords, numbers and names make EN and FR look
similar even without learning, hence the random-init control.
