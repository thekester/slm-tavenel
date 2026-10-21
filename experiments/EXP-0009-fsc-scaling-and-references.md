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

### Supplementary benchmarks (scored 2026-10-21)

Add ARC-Challenge and WinoGrande XL as separate secondary tasks; keep them out of the original
four-task chart so the two benchmark sets and their chance levels remain clear. ARC-Challenge uses
the 1,172-item `test` split from `allenai/ai2_arc` revision
`210d026faf9955653af8916fad021475a3f00453` (CC-BY-SA-4.0). WinoGrande uses the 1,267-item labeled
XL `validation` split from `allenai/winogrande` revision
`546d5aa0e42fc0fa9ad2c13260322ec969e083ee` (official repository Apache-2.0); test labels are
withheld. WinoGrande chance is 50%, versus 25% for ARC-Challenge. Both use the project's log-likelihood
scorer and retain item-level predictions.

WinoGrande uses the sentence prefix before the blank as context and scores each candidate followed
by the remaining sentence as a continuation. This includes both the candidate and suffix in the
likelihood; it is a local cloze-scoring convention, not a claim of equivalence to published harness
scores. Regenerate converted benchmark files and rerun their audit after changing this format.

Score completed SLM-Tavenel 1M/10M, TinyStories-0.9M, Pythia-14M, SmolLM2-135M and EleutherAI
Pythia-31M at pinned model revision `e556ace21b489575e94e9d50b6dad2fcc7419679` (Apache-2.0). Pythia-31M
is a useful general-domain reference near the paused 25M model size. Its current checkpoint is the
standard Pile-trained variant; it is not the former deduplicated variant. This is a broad reference,
not a controlled size experiment.

The new tasks were not part of the exact-match decontamination used to prepare the already-trained
v1 corpus. Before scoring, `scripts/audit_fsc_additional_contamination.py` checks exact lowercased
Unicode word 13-gram overlap between each task's context and choices and the FSC train documents.
This is post-hoc evidence only: it cannot establish absence of paraphrase or indirect exposure, and
it does not audit external models' training corpora. Future FSC corpus runs can use
`configs/data/pipeline-fsc-pilot-v2.yaml`, which adds both benchmark files without changing the v1
corpus or existing checkpoints. The labeled WinoGrande validation results and all comparisons remain
exploratory.

#### Results and paired comparisons

All six model runs completed on the same pinned inputs and local scorer. Raw accuracy and item counts
are in the README chart/table. The post-hoc exact 13-gram audit found zero matched evaluation items
and zero matched FSC train documents for both ARC-Challenge and WinoGrande. This check says nothing
about paraphrases, translated overlap, validation text, or external models' pretraining data.

| Paired comparison (treatment minus baseline) | ARC-Challenge | WinoGrande XL validation |
|---|---:|---:|
| SLM-Tavenel 10M vs 1M | -0.0145 [-0.0367, 0.0077] | +0.0095 [-0.0260, 0.0442] |
| TinyStories 0.9M vs SLM-Tavenel 1M | -0.0102 [-0.0384, 0.0179] | +0.0158 [-0.0213, 0.0521] |

Intervals are 95% paired item-bootstrap intervals with 20,000 draws and seed 20261020. Each includes
zero; these data do not establish a reliable difference on either task. They condition on these
checkpoints and sampled benchmark items and do not measure training-seed variation. Detailed outputs
are under `runs/fsc-additional/comparisons/`; item-level model outputs and the contamination audit
remain in the local `runs/fsc-additional/` directory.

OpenBookQA is excluded because its official Hugging Face dataset card currently declares no known
license. The FrenchBench wrapper declares MIT, but its source questions come from external websites;
the wrapper license alone does not establish content rights. We will add a further French benchmark
only after item-level provenance and an acceptable license are verified.

## Evaluation status and outputs

The sequential training queue is `scripts/run_fsc_scaling.ps1`; configs are under `configs/train/fsc-pilot-v1-*`.
Run outputs are under `runs/fsc-scaling/`. The current evaluation covers completed 1M and 10M
checkpoints and the three pinned references; it does not wait for paused 25M. Each evaluation records
BPB, item-level MC predictions, checkpoint/model revision, tokenizer or benchmark hashes under its
`analysis/` directory or `runs/reference/`. Paired bootstrap intervals quantify item-sampling
uncertainty conditional on these particular checkpoints and benchmark items; they do not estimate
training-seed or training-data variability. Keep comparisons exploratory and update the chart only
from validated same-file outputs. The supplementary ARC-Challenge/WinoGrande wave is complete and
also remains exploratory. Resume 25M with `scripts/resume_fsc_25m_grown.ps1` when ready.
