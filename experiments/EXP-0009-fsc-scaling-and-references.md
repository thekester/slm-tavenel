# EXP-0009: SLM-Tavenel scaling and same-harness references

*Status:* running on 2026-10-20; 1M and 10M are complete and grown-25M has started. The active 1M, 10M and grown-25M runs use the frozen French-Science-Commons
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
| 25M gated | 25M | function-preserving growth from the trained 10M checkpoint | 491.52M additional | 15,000 additional | 1234 | Running |
| 25M gated | 25M | scratch | 491.52M | 15,000 | 1234 | Deferred |

The active runs use the same 8,192-entry tokenizer, the same EN/FR and license-balanced frozen data, the
same 5% document-disjoint validation split, 1,024-token sequences, and 32,768 sampled tokens per
training step. The common token budget makes this an iso-token comparison; compute, time, and
tokens-per-parameter differ by scale and will be reported. The grown 25M condition inherits another
491.52M-token 10M run, so compare its downstream numbers as a different initialization path and
report the full cost. It must not be described as an iso-compute comparison with scratch.

The scratch 25M run is deferred to limit compute and GPU time. Therefore this phase cannot compare
scratch and grown initialization at 25M; the grown model also inherits the completed 10M run's
training budget and data exposure. Primary outcome is English and French bits-per-byte on the fixed
held-out documents. Secondary
outcomes are raw and length-normalized accuracy on ARC-Easy, HellaSwag, Belebele EN and Belebele FR,
with per-item results and paired uncertainty intervals. The four tasks are exploratory at these
sample sizes; do not declare a model-size winner from small accuracy differences. Record checkpoint,
tokenizer, data and benchmark hashes for each run.

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

## Queue and outputs

The sequential queue is `scripts/run_fsc_scaling.ps1`; configs are under `configs/train/fsc-pilot-v1-*`.
Run outputs are under `runs/fsc-scaling/`. Evaluation should run after training and write each
checkpoint's BPB, item-level MC predictions and hashes under its `analysis/` directory. External
reference outputs belong under `runs/reference/` with model revision recorded. Update the README
comparison chart only after these outputs validate.
