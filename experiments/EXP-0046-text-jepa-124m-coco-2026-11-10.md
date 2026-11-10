# EXP-0046: a 124M text JEPA on good views vs next-token training, measured against a vision encoder (preregistered 2026-11-10, before any training run)

Axis: `experiments/AXIS-platonic-representations-2026-11-07.md`. Source claim: Balestriero & LeCun (2026,
arXiv:2610.39302) report that a 109M good-view text JEPA, trained on 900M image-derived text samples, aligns
with vision encoders better than language models up to 72B parameters. This experiment tests the *direction* of
that claim at small scale; it cannot reproduce the paper's budget (590k captions here) or include a 72B model.

## Question

With the same 124M decoder, the same COCO captions and the same training budget, does a JEPA objective on
good views (two captions of one image) produce text representations more aligned with a frozen vision
encoder than next-token prediction on those captions?

## Arms

Both: `configs/model/slm-124m-gpt2match.yaml` (124.0M parameters), fresh initialization, seed 20261109, Muon, WSD
(peak 3e-3, 300 warmup updates, 20% decay to 3e-4), bfloat16, 9,000 updates of 256 images (two distinct captions
each: 512 captions, ~12 tokens each, ~19 passes over the 118,287 images).

| Arm | Objective | Run (`scripts/train_text_jepa.py`) |
|---|---|---|
| J, text JEPA | VICReg-style good-view loss (`slm.goodview`, variant `vicreg`, γ = 1) on a 256-d linear projection of the mean final hidden state; no next-token term | `runs/EXP-0046-124m-jepa` |
| N, control | next-token cross-entropy on the same sampled captions | `runs/EXP-0046-124m-ntp` |

Data: COCO 2017 training captions (`data/parallel/coco2017-captions-v1`: 591,753 captions, 118,287 images;
annotations CC BY 4.0; SmolLM2 tokenizer). Evaluation images never enter training (val2017).

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

## Implementation note (2026-11-10, before any result)

Renumbered from EXP-0044 to EXP-0046 on 2026-11-10, because PR #1 uses EXP-0044 for the 1M fluency diagnosis; the `out_dir` stored in the checkpoints still shows the old name.

The first J run peaked at 26 GB on the 8 GB GPU (0.04 updates/s, over 60 h per arm). It was stopped at update 150, before
any checkpoint or evaluation, and archived as `runs/EXP-0046-124m-jepa-ABORTED-no-activation-checkpointing`. Both arms now
use activation checkpointing, and N computes its cross-entropy in chunks. Objectives, data, sampling, seed and budget are
unchanged; J's loss at update 50 is identical with and without recomputation (0.57667), at 3.9 GB and 0.58 updates/s.

## Limitations fixed in advance

One seed per arm; COCO is a small, caption-only corpus (over 1,000x fewer texts than the paper), and its captions
are short descriptive English; N is trained on captions only and is not comparable as a language model to the
web-trained references (and J is not trained for next-token language modeling); the reference models were not trained on captions,
so J's advantage over them, if any, mixes objective and data domain; the largest reference has 1.7B parameters, so
**this experiment cannot show that a small JEPA beats a 72B model**; DINOv2 is a single vision encoder, and the
alignment metric uses the first caption only; no ImageNet zero-shot or captioning evaluation is run.
