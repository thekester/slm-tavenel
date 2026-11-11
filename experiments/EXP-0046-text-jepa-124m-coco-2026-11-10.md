# EXP-0046: a 124M text joint-embedding model on good views vs next-token training

Axis: `experiments/AXIS-platonic-representations-2026-11-07.md`. Source claim: Balestriero & LeCun (2026,
arXiv:2610.39302) report that a 109M good-view text JEPA, trained on 900M image-derived text samples, aligns
with vision encoders better than language models up to 72B parameters. This experiment tests the *direction* of
that claim at small scale; it cannot reproduce the paper's budget (590k captions here) or include a 72B model.

## Question

With the same 124M decoder, the same COCO captions and the same number of sampled images and updates, does a JEPA-inspired objective on
good views (two captions of one image) produce text representations more aligned with a frozen vision
encoder than next-token prediction on those captions?

## Arms

Both: `configs/model/slm-124m-gpt2match.yaml` (124.0M parameters), fresh initialization, seed 20261109, Muon, WSD
(peak 3e-3, 300 warmup updates, 20% decay to 3e-4), bfloat16, 9,000 updates of 256 images (two distinct captions
each: 512 captions, ~12 tokens each, ~19 passes over the 118,287 images).

The shared budget is **9,000 updates and the same caption-pair schedule**, not equal measured FLOPs, runtime,
energy or exact trainable parameter count. J has a 576-by-256 linear training projection (147,456 extra parameters),
which is removed for backbone evaluation. Its caption-group relationships also supply information that N's next-token
objective does not use. No pixels are input to either text model, but J does use caption-to-image group IDs.

| Arm | Objective | Run (`scripts/train_text_jepa.py`) |
|---|---|---|
| J, text JEPA | VICReg-style good-view loss (`slm.goodview`, variant `vicreg`, γ = 1) on a 256-d linear projection of the mean final hidden state; no next-token term | `runs/EXP-0046-124m-jepa` |
| N, control | next-token cross-entropy on the same sampled captions | `runs/EXP-0046-124m-ntp` |

Data: COCO 2017 training captions (`data/parallel/coco2017-captions-v1`: 591,753 captions, 118,287 images;
annotations CC BY 4.0; SmolLM2 tokenizer). Evaluation images never enter training (val2017).

This image-separation statement concerns our two training arms. It does not establish that DINOv2 or public language
models never saw these photos or captions during upstream pretraining. The 11 November audit also finds that 440/5,000
selected evaluation captions match training captions after case/whitespace normalization; 4,958 normalized captions are
unique within the 5,000 evaluated rows. No image-ID overlap is found. Caption-overlap sensitivity is therefore evaluated
separately, without rewriting the original primary score.

The implementation is a **causal decoder with a predictor-free VICReg-style joint-embedding objective**. The authors'
[released recipe](https://github.com/galilai-group/prh-proof) uses BERT-style text encoders and LeJEPA/SIGReg with a deeper
projection head. Their PRH preprocessing also includes quantile clipping and different layer-pair selection. This study
tests a related mechanism; its scores are not a direct replication or numerically comparable to the paper's 72B tables.

## Outcome and criteria

Evaluation (`scripts/evaluate_text_image_alignment.py`, CPU): the 5,000 COCO val2017 images, encoded by DINOv2
ViT-B/14 (`facebook/dinov2-base@f9e44c81`, CLS token, Apache-2.0), against the first caption of each image;
text representation = mean-pooled hidden state after each layer. A fixed seed-20261109 split assigns 2,500 images
to layer selection and 2,500 separate images to primary scoring; the best layer for each model is selected on the
first half, then the mutual 10-NN score and paired bootstrap are computed only on the held-out half (chance ~0.004).

- **Primary.** Held-out mutual 10-NN overlap with DINOv2 at the layer selected on the separate selection half (chance ~0.004). **J is "more aligned" than N** if
  its mean is higher by at least 0.02 with a 95% paired bootstrap interval over images excluding zero.
- **Collapse rule, fixed now.** If `gv_std_mean` stays below 0.1 after update 1,500, J is declared collapsed and reported
  as such, without retuning inside this experiment.
- **Secondary, descriptive (no criterion).** Linear CKA, R@1 and R@5 text→image after an orthogonal map (PCA-256,
  two-fold); random-init 124M control; our web-trained 124M (EXP-0036); same-protocol public language models at
  pinned revisions: GPT-2 (124M), Pythia-160M, SmolLM2-360M, Qwen2.5-0.5B, Qwen3-0.6B, OLMo-2-1B, SmolLM2-1.7B.
  Paired differences of J against each are reported. GPU energy and time of both arms.

## Implementation note (2026-11-10, before alignment results)

Renumbered from EXP-0044 to EXP-0046 on 2026-11-10, because PR #1 uses EXP-0044 for the 1M fluency diagnosis; the `out_dir` stored in the checkpoints still shows the old name.

The first J run reported about 26.8 GB allocated on the 8 GB GPU (0.04 updates/s, over 60 h per arm). Its stop log mentions
update 150, but the archived metrics extend to update 200, before any checkpoint or evaluation. It is archived as
`runs/EXP-0046-124m-jepa-ABORTED-no-activation-checkpointing`. Both arms now
use activation checkpointing, and N computes its cross-entropy in chunks. Objectives, data, sampling, seed and budget are
unchanged; J's loss at update 50 is identical with and without recomputation (0.57667), at 3.9 GB and 0.58 updates/s.

## Limitations fixed in advance

One seed per arm; COCO is a small, caption-only corpus (over 1,000x fewer texts than the paper), and its captions
are short descriptive English; N is trained on captions only and is not comparable as a language model to the
web-trained references (and J is not trained for next-token language modeling); the reference models were not trained by us in
these matched COCO conditions, and their upstream caption/evaluation exposure is unknown. J's advantage over them, if any,
mixes objective and data domain; the largest reference has 1.7B parameters, so
**this experiment cannot show that a small JEPA beats a 72B model**; DINOv2 is a single vision encoder, and the
alignment metric uses the first caption only; no ImageNet zero-shot or captioning evaluation is run.

## Protocol change before results (recorded 2026-11-11)

The primary criterion above (layer selected on 2,500 images, score and bootstrap on the other 2,500) was written into this
document and `scripts/evaluate_text_image_alignment.py` on 2026-11-10 at 16:45 by another agent working in the repository,
before the alignment evaluation started (21:25) and before any alignment number existed. It replaces the original rule,
which selected and scored the layer on the same 5,000 images. The change is stricter (no selection bias in the primary
score) and is kept; it was not separately recorded as an amendment at the time, hence this note. The 23:39 run was killed by
a shutdown; the reported run is the complete rerun of 2026-11-11 (07:31-08:50), whose numbers for the first nine models match
the interrupted run to four decimals.

**Audit clarification (11 November).** Commit `34a9277` records the protocol and evaluator on 10 November at 16:45:58
Paris time, before the first alignment evaluation, but after J's completed training (15:25) and during N's training.
That is an auditable pre-alignment specification, not independent evidence of preregistration before any training.
The earlier header claimed more than the committed chronology establishes. Local logs and retrospective descriptions
cannot independently prove the timestamp or content of an earlier uncommitted plan.

## Results (2026-11-11)

**Collapse rule:** not triggered. After update 1,500, `gv_std_mean` never fell below 0.823 (final 0.929).

**Primary criterion: met.** Held-out mutual 10-NN overlap with DINOv2 (2,500 images, chance 0.004): J 0.1792 (layer 23 of
27), N 0.1238 (layer 0). **J − N = +0.0554, 95% paired bootstrap interval [0.0493, 0.0616]**, above the pre-alignment specified
+0.02 and excluding zero under the recorded fixed-graph bootstrap. The resampled per-image scores share a nearest-neighbour
graph; this interval does not establish uncertainty over resampled graphs, story/scene families or training seeds.

| Model | Layer selected | Mutual 10-NN, held out | J − model [95% CI] | Linear CKA | R@1 / R@5 after rotation |
|---|---:|---:|---:|---:|---:|
| **J, text JEPA 124M (COCO captions)** | 23 / 27 | **0.1792** | — | 0.450 | 0.040 / 0.154 |
| N, next-token 124M (same captions) | 0 / 27 | 0.1238 | +0.0554 [0.0493, 0.0616] | 0.290 | 0.081 / 0.253 |
| Our web 124M (EXP-0036) | 0 / 27 | 0.1548 | +0.0243 [0.0180, 0.0306] | 0.367 | 0.082 / 0.251 |
| Random-init 124M | 0 / 27 | 0.0598 | (control) | 0.107 | 0.046 / 0.149 |
| GPT-2 | 0 / 12 | 0.1338 | +0.0453 [0.0393, 0.0514] | 0.296 | 0.083 / 0.246 |
| Pythia-160M | 11 / 12 | 0.1176 | +0.0616 [0.0556, 0.0676] | 0.311 | 0.063 / 0.218 |
| SmolLM2-360M | 0 / 32 | 0.1317 | +0.0475 [0.0412, 0.0536] | 0.269 | 0.081 / 0.249 |
| Qwen2.5-0.5B | 0 / 24 | 0.1331 | +0.0460 [0.0400, 0.0521] | 0.307 | 0.083 / 0.251 |
| Qwen3-0.6B | 0 / 28 | 0.1197 | +0.0595 [0.0534, 0.0656] | 0.287 | 0.083 / 0.245 |
| OLMo-2-1B | 0 / 16 | 0.1562 | +0.0230 [0.0170, 0.0291] | 0.414 | 0.094 / 0.276 |
| SmolLM2-1.7B | 22 / 24 | 0.1020 | +0.0771 [0.0712, 0.0832] | 0.103 | 0.067 / 0.207 |

Layer indices count the output of each block from 0. Bootstrap intervals are over held-out images only; they do not include
training-seed variance (one seed per arm).

**Reading.**

- J's neighbourhood structure is closer to DINOv2's than every model measured here, up to 1.7B parameters, by +0.023 to
  +0.077. This is the direction reported by Balestriero and LeCun at small scale. It does not test their 72B comparison.
- **The secondary retrieval metric points the other way.** After an orthogonal map, J retrieves the right image less often
  than N and every language model (R@1 0.040 against 0.063 to 0.094; R@5 0.154 against 0.207 to 0.276). Its R@1 is
  **below** the random-init control (0.046), while R@5 is slightly above it (0.149). J's representation reproduces DINOv2's local neighbourhoods and its global
  linear structure (highest CKA, 0.450) better than an orthogonal rotation can align individual captions to individual
  images. "More aligned" therefore holds for the metric specified before alignment evaluation, not for every alignment measure.
- For N and most language models the best layer is the first block, close to a bag of input embeddings; J's best layer is
  deep (23 of 27). The JEPA objective builds its scene-level structure in late layers, whereas next-token training on these
  short captions does not improve on early lexical averages.
- J's advantage over the references mixes objective and data; their upstream COCO exposure is not established. The more
  controlled comparison is J against N (same backbone, caption schedule and update count), +0.055. It still differs in the
  objective, the auxiliary projection and the use of caption-group information, and is not an equal-compute comparison.

**Energy and time (GPU board, RTX 4060 Laptop).** J: final logged counter 266.7 Wh, 9,000 updates (10:24-15:25 on 10 Nov).
N: final logged resumed-trajectory counter 295.1 Wh. Resume restores checkpoint energy, so this counter omits discarded
partial segments; adding their logged increments gives **at least 306.2 Wh** (11.078 Wh additional recorded energy).
About 250 logged updates were recomputed after two pauses; unlogged work and startup energy are not fully accounted for.
The aborted first J run (no activation checkpointing) logged 41.9 Wh for 200
updates. Alignment evaluation ran on CPU and is not metered.

**Outputs:** `runs/EXP-0046-evaluations/text-image-alignment.json`, `runs/EXP-0046-evaluations/dinov2-val2017.npy`,
checkpoints `runs/EXP-0046-124m-jepa/checkpoints/step_00009000.pt` and `runs/EXP-0046-124m-ntp/checkpoints/step_00009000.pt`.

**Next, not run:** a second seed per arm; other vision encoders (CLIP image tower, SigLIP); several captions per image; a
JEPA + next-token mixture to see whether a model can keep text generation and gain this structure.

## Audit follow-up (11 November 2026)

The [audit report](../docs/reviews/exp0046-audit-2026-11-11.md) checks source/token identities, caption overlaps,
protocol chronology, parameter and compute-budget scope, energy accounting, metric code and fresh J/N feature extraction.
It records corrections and the remaining replication/provenance gaps. The original result JSON is preserved. New cache
guards, per-image evidence and actual custom-loop manifests apply to future runs rather than rewriting original provenance.

Fresh extraction gives J **0.17916**, N **0.12372**, difference **+0.05544**, with the same rounded fixed-graph interval
**[0.0493, 0.0616]**. N differs by -0.00008 from the original rounded JSON value; the exact numerical cause remains
unresolved, and an unbatched first-layer check also gives 0.12372. The report does not call this exact bitwise reproduction.
On the 2,279 primary-half images without train-caption matches, neighbours recomputed within that subset give
J **0.19144** and N **0.12637**. This post-hoc sensitivity supports the local direction but does not replace independent
training-seed replication. Corrected fit-only-centering retrieval remains worse for J than N.
