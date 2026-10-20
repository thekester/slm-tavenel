# EXP-0009: SLM-Tavenel scaling and same-harness references

*Status:* 1M and 10M are complete. On 2026-10-20, the user paused grown-25M at step 3,050; its latest durable checkpoint is step 3,000. Resume later from that checkpoint. The active runs use the frozen French-Science-Commons
pilot corpus; the corpus remains a candidate and results are exploratory pending rights and privacy
review. External references are contextual, not controlled training comparisons.

## Questions

1. At 1M, 10M and 25M parameters, how do fixed-token bits-per-byte and multiple-choice results
   change when corpus, tokenizer, language balance, held-out split and training token budget match?
2. How does the approximately 0.9M TinyStories checkpoint compare with SLM-Tavenel 1M on the same
   evaluator, and how do size-near Pythia-14M and larger SmolLM2-135M score?

## Internal scaling design

| Run | Parameters (approx.) | Initialization | Train tokens | Steps | Seed | State |
|---|---:|---|---:|---:|---:|---|
| 1M gated | 1.2M | scratch | 491.52M | 15,000 | 1234 | Complete |
| 10M gated | 10M | scratch | 491.52M | 15,000 | 1234 | Complete |
| 25M gated | 25M | function-preserving growth from the trained 10M checkpoint | 491.52M additional | 15,000 additional | 1234 | Paused at step 3,050; durable checkpoint at step 3,000 |
| 25M gated | 25M | scratch | 491.52M | 15,000 | 1234 | Deferred |

The completed 1M and 10M runs use the same 8,192-entry tokenizer, the same EN/FR and license-balanced frozen data, the
same 5% document-disjoint validation split, 1,024-token sequences, and 32,768 sampled tokens per
training step. The common token budget makes this an iso-token comparison; compute, time, and
tokens-per-parameter differ by scale and will be reported. The grown 25M condition inherits another
491.52M-token 10M run, so compare its downstream numbers as a different initialization path and
report the full cost. It must not be described as an iso-compute comparison with scratch.

The scratch 25M run is deferred to limit compute and GPU time. Grown-25M is paused and is not part of
the current evaluation round. Therefore this phase cannot compare
scratch and grown initialization at 25M; the grown model also inherits the completed 10M run's
training budget and data exposure. Primary outcome is English and French bits-per-byte on the fixed
held-out documents. Secondary
outcomes are raw and length-normalized accuracy on ARC-Easy, HellaSwag, Belebele EN and Belebele FR,
with per-item results and paired uncertainty intervals. The four tasks are exploratory at these
sample sizes; do not declare a model-size winner from small accuracy differences. Record checkpoint,
tokenizer, data and benchmark hashes for each run.

The internal 1M/10M pair is controlled for training token count, corpus, tokenizer, validation split
and evaluation files, but not compute or tokens per parameter; only one seed per scale is currently
available. Paired bootstrap intervals resample the observed item-level outcomes and are conditional
on these checkpoints. They do not capture run-to-run variation, benchmark construction uncertainty,
or residual contamination. External checkpoints use the same benchmark files and scoring code, but
their tokenizers, model-specific context caps, training data, token budgets, compute and contamination
histories differ. Their scores are descriptive reference points, not controlled comparisons. The FSC
pipeline's recorded exact-match filtering reduces detectable overlap but cannot exclude semantic,
translated, or unrecorded exposure. Rights review for the corpus and derivative weights remains open.

## External references

Evaluate the MIT-licensed `shibatch/tiny1m` checkpoint (896,256 parameters per its model card) at
pinned revision `fdacad8ff9dc27bd72c77d3684075a494be1ba11`, loading the `hf/` subfolder and capping
context at its 256-token limit. It was trained on English TinyStories with a 512-token vocabulary;
this makes it a near-parameter-count reference, not a matched-data or language comparison. Include
it in the same four MC benchmark files and validation-text BPB evaluation where supported, and pair
its item-level MC results with the SLM-Tavenel 1M run. The original TinyStories study evaluates
small models on synthetic children's story generation with GPT-4 scoring, a useful but distinct
task from this repository's general and bilingual benchmarks.

Also evaluate EleutherAI Pythia-14M at pinned revision
`d31f659f077f9ea500c4d7faa999893d7f28cbdf` and HuggingFaceTB SmolLM2-135M at
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` with the same local benchmark evaluator. Record their
parameter counts and revision. Report them beside SLM-Tavenel, while stating that tokenizer,
pretraining corpora, compute and contamination histories differ. They are contextual reference
points, not controlled causal comparisons. Score BPB on the same FSC validation texts when the
reference tokenizer permits it; keep MC outputs even if one reference's BPB has a format issue.

## Evaluation status and outputs

The sequential training queue is `scripts/run_fsc_scaling.ps1`; configs are under `configs/train/fsc-pilot-v1-*`.
Run outputs are under `runs/fsc-scaling/`. The current evaluation covers completed 1M and 10M
checkpoints and the three pinned references; it does not wait for paused 25M. Each evaluation records
BPB, item-level MC predictions, checkpoint/model revision, tokenizer or benchmark hashes under its
`analysis/` directory or `runs/reference/`. Paired bootstrap intervals quantify item-sampling
uncertainty conditional on these particular checkpoints and benchmark items; they do not estimate
training-seed or training-data variability. Keep comparisons exploratory and update the chart only
from validated same-file outputs. Resume 25M with `scripts/resume_fsc_25m_grown.ps1` when ready.
