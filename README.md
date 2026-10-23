# SLM-Tavenel

An open research project on small language models, reproducible training, observable behavior and
careful evaluation under limited compute. The 1B goal is a longer-term direction; the current
release work concerns a roughly 1M-parameter bilingual research model.

**New here?** Start with the [SLM-Tavenel research overview](docs/SLM-TAVENEL-RESEARCH-OVERVIEW.md),
an article-style guide to the 1M study, its data, results, benchmark comparisons, and limitations.

## Current status

The repository contains historical 1M/10M pilots and same-harness comparisons with TinyStories,
Pythia and SmolLM2. They remain exploratory records because the models have different data,
tokenization and training budgets. Older pilot cards and benchmark tables are retained for audit;
they are not interchangeable release versions.

The current scientific study is [EXP-0010](experiments/EXP-0010-1m-paired-replication-2026-10-22.md):
ten paired seeds compare the 1.18M-parameter gated model with a matched ungated model, using shared
initialization and data order. The primary outcome is English/French bits per byte on document-held-
out text, with a predeclared hierarchical interval and decision threshold. All twenty runs, the
locked test analysis, selected-checkpoint benchmarks and exact data-pipeline replay are complete.
The predeclared gate effect is +0.0275 BPB (95% hierarchical bootstrap interval +0.0168 to +0.0395),
supporting the +0.01 practical threshold for this corpus, tokenizer and one-pass budget. This does
not establish general model quality. See the [results report](experiments/EXP-0010-results-2026-10-22.md)
and its [benchmark comparison](docs/exp0010-benchmark-comparison.svg), [paired-seed effect plot](docs/exp0010-paired-seed-effects.svg),
[training curves](docs/exp0010-training-curves.svg) and [data split chart](docs/exp0010-data-splits.svg).
The presentation structure takes inspiration from Hugging Face's [SmolLM report](https://huggingface.co/blog/smollm); these figures are original and generated from SLM-Tavenel's recorded results.

The proposed model weights and code use MIT terms. The training-data record pins
PleIAs/French-Science-Commons revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`, documents the
filtered records, split, source IDs, declared CC0 metadata, processing and hashes, and excludes
article text from the model bundle. The dataset and HAL metadata trail is not independent proof of
each depositor's authority. Source-rights review and manual privacy sign-off remain unsigned. The
OSAID allows different legal mechanisms for model parameters; this project uses MIT for weights,
code and Data Information. Our [OSAID 1.0 self-assessment](docs/reviews/OSAID-1.0-EXP0010-ASSESSMENT.md)
finds the published package meets the definition's core artifact and terms requirements. This is a
project self-assessment, not OSI certification; the [EXP-0010 package is public on Hugging Face](https://huggingface.co/athenista/slm-tavenel-1m-exp0010).
Source-rights and privacy findings are disclosed separately in the
[source and privacy dossier](docs/reviews/EXP-0010-SOURCE-PRIVACY-REVIEW.md); their unsigned
governance records are not OSAID criteria.

The data-preparation manifest records a dirty worktree at preprocessing time. Before packaging, the
pipeline was replayed into a separate scratch directory; the retained-document and filtering
decision hashes and processing statistics matched the recorded outputs exactly. The replay
configuration and report are included in the candidate package. The subsequent paired
model-training runs use a recorded clean code commit.

The [EXP-0010 1M package](https://huggingface.co/athenista/slm-tavenel-1m-exp0010) is public with selected weights, runnable Transformers code, training configurations, paired-run metrics, observability logs, source manifests, benchmark records and a SHA-256 inventory. The [EXP-0009 10M FSC pilot](https://huggingface.co/athenista/slm-tavenel-10m-exp0009) is public as an explicitly exploratory single-seed checkpoint. Its mixed source-license labels, rights and privacy review gaps, and dirty-worktree provenance are disclosed in its model card and attribution files. Read the [EXP-0010 model card](docs/model-cards/slm-tavenel-1m-exp0010-candidate.md), [data card](docs/model-cards/slm-tavenel-1m-exp0010-data-card.md), [paired protocol](experiments/EXP-0010-1m-paired-replication-2026-10-22.md), [experiment archive](experiments/) and [roadmap](docs/ROADMAP.md).

The 10M card also includes a post-training HAL file-license audit: 31,262 of 31,417 source IDs match across returned HAL versions; 72 have explicit mismatches, 3 conflicting declarations, 24 missing file-license fields and 56 were not found. Those 155 records remain unresolved. See the [audit review](docs/reviews/EXP-0009-HAL-FILE-LICENSE-AUDIT-2026-10-22.md) and [full metadata-only report](https://huggingface.co/athenista/slm-tavenel-10m-exp0009/blob/main/training_record/hal-file-license-audit-2026-10-22.json). This is not a rights clearance.

A targeted HAL recheck on 23 October found no change among those 155 records. Crossref metadata was available for 75 DOI-bearing records: 30 included the dataset-declared license URL, 15 listed only a different license or policy URL, 23 had no license field, and 7 lookups failed. These metadata do not establish rights-holder authority or the exact version used. A local text and metadata screen of the 31,417 training documents found four residual public-IP-pattern matches, 54 replacement-character documents, and 20,808 records with at least one broad privacy-review signal; the later v4 content-risk filters were not applied to these historic runs. See the [source and privacy review record](docs/reviews/EXP-0009-25M-SOURCE-PRIVACY-REVIEW.md) and its machine-readable reports. The [25M package is now public on Hugging Face](https://huggingface.co/athenista/slm-tavenel-25m-exp0009). It includes a full tracked source snapshot and an [OSAID 1.0 self-assessment](docs/reviews/OSAID-1.0-EXP0009-25M-ASSESSMENT.md), which describes OSAID alignment as a project self-assessment, not OSI certification. Source-rights and manual privacy review remain unsigned; those risks and limits are disclosed in the model card and reports.

Older wiki and FSC runs, including prior 1M pilots, remain in the repository as historical records.
The grown 25M run resumed from durable step 3,000 in the matching Python 3.11 / PyTorch 2.10 environment and completed step 15,000. Its six same-harness benchmark scores, comparison chart, paired item intervals and English/French BPB are recorded in the separate EXP-0009 report. Its inference package passed a fixed-input native-to-Transformers parity check. The model is publicly hosted as a separate EXP-0009 artifact and is not part of the EXP-0010 1M release.
## Current SLM benchmark comparison

The current comparison shows exactly one SLM-Tavenel 1M: the preselected EXP-0010 candidate,
validated with ten paired training seeds for the gate ablation. Earlier single-seed 1M pilots remain
in the experiment archive; they are not presented here as competing current versions. The exploratory
[SLM-Tavenel 10M FSC pilot](https://huggingface.co/athenista/slm-tavenel-10m-exp0009) and public
models provide context across model sizes.

![SLM-Tavenel 1M selected candidate and small-model benchmark references](docs/exp0010-benchmark-comparison.svg)

The chart reports raw accuracy on six pinned tasks with 95% Wilson intervals over items. Those
intervals do not measure training-seed variability. Pythia-31M has scores only for ARC-Challenge and
WinoGrande XL validation. The other models were scored on the corresponding identical pinned inputs
where results are available. Different training data, tokenizers, language mixes, budgets and possible
benchmark exposure make these descriptive references, not controlled comparisons. See the [EXP-0010
results report](experiments/EXP-0010-results-2026-10-22.md) for the score table, paired intervals,
provenance and limitations. OpenBookQA and FrenchBench are omitted because their question-source terms
are unresolved for redistribution.

### Separate EXP-0009 scaling comparison

![SLM-Tavenel 1M, 10M and grown 25M compared with public small-model references](docs/exp0009-25m-benchmark-comparison.svg)

This separate six-task chart includes the selected SLM-Tavenel 1M, the 10M pilot, the grown 25M
checkpoint, TinyStories tiny1m, Pythia-14M/31M and SmolLM2-135M. Every available score uses the same
pinned input files and repository evaluator. Error bars reflect item-sampling uncertainty only; they
do not represent training-seed variation. The 25M inherits the trained 10M checkpoint and receives
491.52M additional token positions, so it is not an iso-compute size comparison. These results are
exploratory; the [25M weights are available on Hugging Face](https://huggingface.co/athenista/slm-tavenel-25m-exp0009). See the [EXP-0009 results report](experiments/EXP-0009-25m-scaling-results-2026-10-22.md)
for the full scores, paired intervals, BPB point estimates, hashes and corpus limits. BPB is reported
with paired document-bootstrap intervals conditional on the two saved checkpoints; the intervals do
not estimate training-seed uncertainty.

Held-out bits-per-byte (lower is better; same FSC English/French validation text and tokenizer):

| SLM-Tavenel checkpoint | English BPB | French BPB |
|---|---:|---:|
| 1M EXP-0010 selected | 1.4904 | 1.3676 |
| 10M EXP-0009 | 1.1755 | 1.0897 |
| grown 25M EXP-0009 | **1.1171** | **1.0367** |

![Paired document-bootstrap intervals for held-out BPB differences](docs/exp0009-25m-bpb-document-bootstrap.svg)

Positive values are 10M BPB minus 25M BPB, so they favor the grown 25M point estimate.

| Comparison | Documents | BPB difference | 95% paired document interval |
|---|---:|---:|---:|
| English, 10M minus 25M | 685 | +0.0585 | [+0.0570, +0.0602] |
| French, 10M minus 25M | 678 | +0.0530 | [+0.0520, +0.0540] |
| Equal-language mean | 685 EN; 678 FR | +0.0557 | [+0.0548, +0.0567] |

The grown 25M is lower by 0.0585 BPB in English and 0.0530 in French than the 10M point estimate.
This single continuation is not an iso-compute or parameter-only experiment. The intervals quantify
validation-document sampling for these checkpoints, but not training-seed variation or source-data
uncertainty. The bootstrap used seed 20261024 and 20,000 draws; see the
bootstrap record (local artifact: `runs/fsc-scaling/comparisons/exp0009-bpb-document-bootstrap.json`) for hashes and method.

The later single-pair 10M and 1M FSC/MOT mixture pilots are reported separately in the
[paired pilot results](experiments/EXP-0012-0013-mixture-pilot-results-2026-10-22.md), with a
[paired benchmark-effect chart](docs/exp0012-13-mixture-benchmark-effects.svg). Their control arms
are experimental comparisons, not additional promoted SLM-Tavenel model versions; the results do
not estimate training-seed variability.

The 25M follow-up [EXP-0014](experiments/EXP-0014-25m-continuation-mixture-2026-10-23.md) compares
matched 49.15M-token continuations from the same published checkpoint. An 80% FSC / 20% MOT mix
reduces held-out MOT BPB by about 0.266 on both development and temporal-test text, while increasing
FSC BPB by 0.005–0.006. Its six multiple-choice changes are small and all paired item intervals
include zero. The result supports a domain-specific trade-off, not a general benchmark improvement;
the new checkpoints remain local because the source reviews are unresolved.

[EXP-0015](experiments/EXP-0015-25m-muon-continuation-2026-10-23.md) tested whether the project's
Muon optimizer transfers to a matched 25M continuation. The run stopped at step 300/1,500 with a
CUDA illegal-memory-access error before its first checkpoint. It is recorded as failed/incomplete;
the partial losses are not a performance result and no weights were published.

[EXP-0016](experiments/EXP-0016-25m-wiki-mixture-2026-10-23.md) compared a
decontaminated English/French Wikipedia mixture with the EXP-0014 FSC-only control. Its data are
retokenized with the 25M tokenizer and filtered against all six benchmark files. The matched
continuation completed 1,500/1,500 steps. At the same 49.152M-token continuation budget, the 20%
wiki mixture reduced held-out wiki BPB by 0.178 EN / 0.137 FR, while FSC BPB increased by 0.0035 EN /
0.0025 FR. MOT BPB also improved by about 0.059 on each set. Paired item intervals for all six
post-hoc multiple-choice tasks include zero, so this is a domain trade-off, not a general benchmark
gain. The 10% wiki dose-response [EXP-0017](experiments/EXP-0017-25m-wiki-10pct-2026-10-23.md) is complete: it retains about 76–78% of the 20% arm's wiki BPB reduction while cutting its FSC BPB regression by about 60–64%. The six paired multiple-choice intervals still include zero, so this is a better observed domain compromise, not a general benchmark win. See the [0/10/20% dose-response graph](docs/exp0017-wiki-dose-response.svg) and [full EXP-0017 report](experiments/EXP-0017-25m-wiki-10pct-2026-10-23.md). Wikipedia and FSC rights/privacy reviews remain unresolved; candidate checkpoints stay local and exploratory.

The preregistered [EXP-0018/0019 2x2 learning-rate sensitivity grid](experiments/EXP-0018-0019-25m-learning-rate-grid-2026-10-23.md) is complete. Both 1.5e-4 runs finished training and held-out BPB evaluation. At this fixed token budget, reducing the learning rate worsened BPB on all four held-out sets for both data recipes. Wiki at 10% still improves wiki BPB while slightly regressing FSC BPB; its wiki gain is smaller at the lower rate. This is a one-seed sensitivity study with document-bootstrap intervals only, not a confirmed optimum or a general benchmark win. See the four-cell analysis (local artifact: `runs/EXP-0018-0019-25m-learning-rate-grid-2026-10-23/analysis.json`) and full report. All checkpoints remain local; source rights/privacy reviews are unresolved.

A new local transfer pilot, [EXP-0020/0021](experiments/EXP-0020-0021-25m-mir-pilot-2026-10-23.md), is complete: the MIR arm had higher BPB than the same-seed FSC control on all four held-out sets. This one-seed result is exploratory, and the supporting paper starts at 72M parameters; MIR also took about 1.84x as long to train in this local run.

A second-seed replication of the 10% wiki recipe is training-complete in [EXP-0022](experiments/EXP-0022-25m-wiki-10pct-seed2-2026-10-23.md), matched to the completed EXP-0020 FSC control. Its final checkpoint is saved; held-out BPB evaluation remains pending after PC shutdown. No seed-2 mixture effect is reported until evaluation finishes. Two seeds are still too few to estimate seed variance robustly.

The grown 25M checkpoint is reported separately under [EXP-0009](experiments/EXP-0009-fsc-scaling-and-references.md).
It is kept out of the EXP-0010 chart because it follows a distinct inherited training path and uses a
different experiment record. Earlier 1M benchmark results and raw item records remain archived under
EXP-0009; their different dataset versions and training conditions are not mixed into this comparison.

## Legacy wiki-v0 benchmark snapshot

The table reports raw multiple-choice accuracy (`acc`) from the project evaluator. Each run used the
same prompts, scoring code and pinned dataset revisions. ARC-Easy has 2,376 items; HellaSwag 10,042;
each Belebele language split 900. Chance is about 25% for all four-choice sets.

**These scores are exploratory, not clean held-out estimates.** The models were trained on wiki-v0,
whose pipeline did not remove benchmark overlaps. wiki-v1 applies exact word 13-gram filtering to
the four benchmark files and logged five removed training documents: three matching Belebele EN and
two matching Belebele FR. This does not detect paraphrase or translation exposure, and the current
Wikipedia sources remain under review. The wiki-v0 table is retained as experiment history; a new
same-corpus replication is preregistered in [EXP-0008](experiments/EXP-0008-wiki-v1-clean-replication.md).
That replication retrains the tokenizer on the filtered training split, keeps its BPB holdout out of
tokenizer/model training, and records item-level benchmark outcomes for paired uncertainty estimates.
Do not present the old table as a clean benchmark result.

| 10M model | Training seed | ARC-Easy | HellaSwag | Belebele EN | Belebele FR |
|---|---:|---:|---:|---:|---:|
| AdamW baseline | 1234 | 0.295 | 0.275 | 0.266 | 0.309 |
| AdamW baseline | 2 | 0.296 | 0.272 | 0.270 | 0.314 |
| Head-wise gate | 1234 | 0.293 | 0.273 | 0.277 | 0.319 |
| Head-wise gate | 2 | 0.289 | 0.271 | 0.283 | 0.299 |
| Muon | 1234 | 0.306 | 0.273 | 0.283 | 0.302 |
| Muon | 2 | 0.290 | 0.270 | 0.286 | 0.319 |

![Grouped bar chart comparing SLM-Tavenel 10M benchmark scores across training runs](docs/benchmark-comparison.svg)

The small differences between these 10M runs are within the benchmarks' sampling uncertainty; these
scores do not establish a downstream-task winner. The gate's current case rests on its repeated
bits-per-byte gain and sink measurements, documented in [EXP-0003](experiments/EXP-0003-gated-attention-10m.md)
and [EXP-0004](experiments/EXP-0004-gated-attention-seed2.md).

| Existing SLM reference | Parameters | ARC-Easy | HellaSwag | Belebele EN | Belebele FR | Role |
|---|---:|---:|---:|---:|---:|---|
| [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) | 14M | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | Primary size-near reference |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 135M | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | Secondary larger SLM reference |
| [TinyStories tiny1m](https://huggingface.co/shibatch/tiny1m) | 0.896M | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | not scored in this wiki-v0 snapshot | Near-1M scale reference; English story-training corpus |

Pythia-14M is the closest established open checkpoint in size to SLM-Tavenel 10M. Its model card
identifies it as English-only and trained on the Pile, so English benchmark results are the most
like-for-like; Belebele French is a cross-lingual probe. The evaluation pins revision
`d31f659f077f9ea500c4d7faa999893d7f28cbdf`. SmolLM2-135M adds a larger, modern SLM point, with the
size difference stated rather than treated as a controlled comparison. TinyStories tiny1m gives a
closer parameter-count reference for the current 1M run; its model card reports 896,256 parameters,
MIT licensing, an English TinyStories training corpus, a 512-token vocabulary and 256-token context.
It is not a general-domain or bilingual peer. All references use this
repository's evaluator, prompts, scoring rules and pinned benchmark files; published scores from
other harnesses are not mixed into the chart. Same-harness scoring controls evaluation procedure, but
not training data, token budget or contamination history, so external results are outcome references,
not causal tests of architecture or training method. FSC's recorded filtering reduces detectable benchmark overlap, but cannot rule out paraphrase, translation or latent exposure; source and weight-release rights review also remains open. Within SLM-Tavenel, 1M versus 10M shares the corpus, tokenizer, held-out split and token budget, but uses one seed and different compute and tokens-per-parameter. The grown 25M condition has a distinct inherited training history. Paired intervals quantify uncertainty across benchmark items conditional on these checkpoints; repeated training seeds are needed to estimate training variability. Treat all current results as exploratory. See the [Pythia model card](https://huggingface.co/EleutherAI/pythia-14m)
and [Pythia repository](https://github.com/EleutherAI/pythia) for its model and evaluation details.

Detailed SLM-Tavenel run data and decisions are in [EXP-0001](experiments/EXP-0001-slm-10m-wiki.md),
[EXP-0003](experiments/EXP-0003-gated-attention-10m.md), [EXP-0004](experiments/EXP-0004-gated-attention-seed2.md)
and [EXP-0005](experiments/EXP-0005-muon-10m.md); raw evaluator outputs are stored under each run's
`analysis/mc.json`.

The `eval bpb` and `eval mc` commands accept `--hf-model` and `--hf-revision` for reference models.
The [wiki-v1 pipeline](configs/data/pipeline-wiki-v1.yaml) expands the exploratory corpus and adds
benchmark decontamination. EXP-0008 remains paused while the candidate French-Science-Commons corpus is
reviewed. A separate exploratory EXP-0009 reports the completed 1M, 10M and grown-25M FSC runs;
the grown-25M checkpoint and its source manifest are public on Hugging Face, and the 25M-from-scratch run remains deferred.
Model growth is available through `slm grow`; the [1M](configs/model/slm-1m-gated.yaml) and
[25M](configs/model/slm-25m-gated.yaml) configurations remain available for further work.

What exists today:

- **Model.** A decoder-only transformer with grouped-query attention, QK normalisation, SwiGLU and RoPE, configured at 135M, 300M and 1.04B parameters, plus ablation variants with gated attention and local/global layers. Every intermediate activation passes through a named hook point. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Causal analysis.** Capture and intervention on any hook point, head ablation, activation patching, logit lens, and attention sink measurements, also logged during training.
- **Open Data Factory.** A source registry with an enforced license policy, and a pipeline for normalisation, quality filtering, privacy redaction, exact and MinHash deduplication, and benchmark decontamination. Every removal is logged with its reason, and every kept document carries its source, revision and hash. See [docs/DATA_POLICY.md](docs/DATA_POLICY.md).
- **Reproducible training.** Batches are a function of seed and step, checkpoints carry optimiser and RNG state, and run manifests record configuration hashes, data hashes, code revision and environment. A resumed run matches an uninterrupted run exactly in deterministic mode. Optional Muon optimiser and per-document attention masking, each an ablation in the plan.
- **Evaluation and cost.** Bits per byte, multiple-choice benchmarks by log-likelihood at pinned revisions (ARC-Easy, HellaSwag, Belebele in English and French), verified task completion with cost per solved task, latency, memory, GPU energy, and a controlled throughput benchmark (`slm bench`).
- **First results.** Experiments at 10M parameters on Wikipedia are recorded in [`experiments/`](experiments/), including causal tests of attention sinks and a gated-attention ablation.
- **Open weights in standard formats.** Export to Hugging Face Llama or Qwen3 format, checked to give identical logits in `transformers`, with a provenance file linking weights to their run and data.
- **Agent traces.** A span schema linking tasks, model calls, tool calls, verification results and costs.

## Getting started

Requires Python 3.10 or newer and PyTorch 2.3 or newer. A GPU is optional for the smoke run.

```bash
pip install -e ".[dev,hf,energy]"
pytest

# Smoke run on synthetic data, from the repository root (a few minutes)
python -m slm sources audit -v
python -m slm data synth --kind prose --n 3000 --out data/raw/smoke/prose.jsonl
python -m slm data synth --kind arithmetic --n 200 --seed 1 --out data/raw/smoke/arith-eval.jsonl
python -m slm data run configs/data/pipeline-smoke.yaml
python -m slm tokenizer train --inputs data/processed/smoke/documents.jsonl --out data/tokenizer/smoke --vocab-size 4096
python -m slm data encode --tokenizer data/tokenizer/smoke --inputs data/processed/smoke/documents.jsonl --out data/tokens/smoke --val-fraction 0.05
python -m slm train configs/train/smoke.yaml
python -m slm eval bpb --checkpoint runs/smoke/checkpoints/step_00000300.pt --tokenizer data/tokenizer/smoke --data data/processed/smoke/documents.jsonl --limit 50
python -m slm eval cost --checkpoint runs/smoke/checkpoints/step_00000300.pt --tokenizer data/tokenizer/smoke
python -m slm export --checkpoint runs/smoke/checkpoints/step_00000300.pt --tokenizer data/tokenizer/smoke --out exports/smoke-hf

# Size and compute estimate of the 1B target
python -m slm model info configs/model/slm-1b.yaml --tokens 1e12

# Benchmarks at pinned revisions, and a controlled throughput measurement
python -m slm eval fetch-benchmarks --out data/benchmarks
python -m slm eval mc --checkpoint runs/smoke/checkpoints/step_00000300.pt --tokenizer data/tokenizer/smoke --benchmarks data/benchmarks/arc_easy.jsonl --limit 200
python -m slm bench configs/train/smoke.yaml --set optimizer=muon
```

Training configurations accept overrides, for example `python -m slm train configs/train/smoke.yaml --set max_steps=100 seed=7`. Multi-GPU runs use `torchrun --nproc_per_node=N -m slm train ...`.

## Repository layout

| Path | Contents |
|---|---|
| `src/slm/model.py` | Transformer with hook points, KV cache and generation |
| `src/slm/observe.py`, `src/slm/interp.py` | Capture, interventions and causal analyses |
| `src/slm/data/` | Source registry and policy, filters, deduplication, privacy, decontamination, pipeline, token shards, synthetic data |
| `src/slm/tokenizer.py` | Byte-level BPE training with a provenance manifest |
| `src/slm/train.py` | Deterministic, resumable, instrumented training loop |
| `src/slm/evaluation.py` | Bits per byte, verified tasks, cost and energy |
| `src/slm/export.py` | Hugging Face export with provenance |
| `src/slm/trace.py` | Agent trace spans |
| `configs/` | Model, training, data source and pipeline configurations |
| `docs/` | Architecture, data policy, experiment protocol, roadmap, and card templates |
| `tests/` | Unit and end-to-end tests |

## Open science as a design requirement

The ambition is to release the source code, model weights, tokenizer, configurations, training recipes, and evaluation procedures. Dataset provenance, preparation scripts, experiment logs, and both positive and negative findings should make the development process inspectable.

Openness must be documented for each component. A public download alone does not establish that an artifact meets the project's reuse requirements. The intended core corpus will require identifiable sources, recorded permissions, and a review of applicable obligations before inclusion.

The original code and documentation in this repository are released under the [MIT License](LICENSE), with copyright attributed to Théophile Avenel. Third party materials retain their applicable terms. Future model weights and datasets will document their licensing separately when released.

## Research directions

### Reasoning and adaptive computation

We plan to compare explicit reasoning, latent reasoning, and combinations of the two. [Coconut](https://arxiv.org/abs/2412.06769) and [HybridAgent](https://arxiv.org/abs/2610.18861) are starting points for this investigation. [Recurrent depth](https://arxiv.org/abs/2502.05171) and adaptive computation are additional directions for studying whether a compact model can benefit from spending more computation on difficult tasks.

The [Laguna technical report](https://arxiv.org/abs/2605.27605) provides a separate reference for architecture, agentic coding, and the integration of data, training, evaluation, and inference. Its Model Factory approach informs our interest in reproducible experiment infrastructure.

Experiments will measure answer quality, task completion, latency, memory use, and total inference cost. Reducing visible tokens is useful only when the resulting system remains effective and its total cost supports the claimed improvement.

### Attention and information flow

This track will examine how attention mechanisms affect information flow, representation similarity, and performance as context length increases. Candidate experiments include comparisons between standard dot product attention, gated alternatives, and configurations that vary across layers.

Attention sinks are a phenomenon to investigate rather than an assumed defect. [Bismuth's study of attention sinks](https://arxiv.org/abs/2610.19324), which examines [GRIT](https://arxiv.org/abs/2305.17589), motivates questions about attention scoring. Findings from graph models do not establish benefits for an autoregressive language model. Initial architecture experiments are planned at roughly 100 million to 300 million parameters before considering a transfer to the one billion parameter target.

### Tools, memory, and adaptive agent environments

The agent environment includes instructions, tools, memory, skills, scripts, plugins, execution rules, and optional specialized agents. We plan to compare the model alone, the model with a fixed environment, and the model with an environment that can propose and evaluate changes to its configuration.

[Harness²](https://arxiv.org/abs/2610.14224) motivates this research direction. A central question is whether a compact model can improve its own environment reliably and economically. Any adaptation must be evaluated for transfer to unseen tasks, regressions, and the computation consumed while searching for improvements.

Agent experiments will begin in controlled environments with explicit permissions, recorded actions, and verifiable outcomes.

### Causal explainability and observability

Readable explanations and attention visualizations do not, by themselves, establish why a model made a decision. We plan to test explanatory claims through interventions such as removing, replacing, or perturbing selected internal states and measuring changes in behavior.

Agent traces should connect task inputs, relevant execution context, tool calls, observable decisions, verification results, and resource costs. Experiments will assess whether those traces help locate failures and support recovery when tools return incorrect information, time out, or introduce conflicting instructions.

## Open Data Factory

The data pipeline is intended to be a reproducible research artifact in its own right. It will track source revisions, licenses, attribution requirements, transformations, filtering decisions, and dataset splits.

[Common Pile](https://arxiv.org/abs/2506.05209), selected [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) subsets, and [PleIAs Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) are candidates for review. Common Corpus is pinned in the [collection review](docs/reviews/DATASET-COLLECTION-REVIEW-2026-10-23.md) and remains blocked pending per-record license, provenance, and reproducible privacy checks. [FineWeb and FineWeb-Edu](https://arxiv.org/abs/2406.17557), [FineWeb2](https://arxiv.org/abs/2506.20920), [Dolma](https://arxiv.org/abs/2402.00159), [The Stack v2](https://arxiv.org/abs/2402.19173), and [Tülu 3](https://arxiv.org/abs/2411.15124) mixtures are additional candidates for comparison or selective use. These names form a research shortlist, not an approved training corpus. Inclusion will depend on review of the specific sources and subsets used.

Preparation will include quality filtering, exact and approximate deduplication, privacy review, and checks for benchmark contamination. [DataTrove](https://github.com/huggingface/datatrove) is a reference implementation for investigating these processing pipelines. The intended core corpus excludes sources with unidentified provenance or restrictions incompatible with the project's reuse goals, including NonCommercial and NoDerivatives restrictions.

We also plan to create synthetic tasks with verifiable outcomes, including code exercises with tests, simulated API calls, document retrieval, and operations on fictional files. Their records should preserve the generator, any teacher model, the generation conditions, and the verification procedure.

Data mixtures and curricula will be experimental variables. General text, science, mathematics, code, French, and English are areas of interest; instruction and agent trajectories will be considered separately for post training. No mixture is currently established as the training recipe.

## Experimental approach

Two development paths will support the research. Experiments with an existing checkpoint can test reasoning and agent methods sooner, subject to review of its terms and provenance. Training from scratch offers greater control over architecture, data, and the resulting scientific claims. Results from these paths will be reported separately.

Each experiment should define a baseline, a hypothesis, a controlled change, and a reproducible evaluation procedure. Comparisons will account for training data, parameter count, available tools, and computation budgets. Where practical, repeated runs will help distinguish consistent gains from variation.

Evaluation will measure accuracy, verified task completion, robustness, latency, peak memory, tokens, tool calls, retries, and energy where it can be measured. Cost per successfully completed task will include failed attempts and environment adaptation overhead. Training and preparation costs will also be reported so that cheaper inference is not mistaken for lower total cost.

Memory experiments will vary context length, introduce contradictory information, and test retrieval failures. Agent experiments will include tool errors and unfamiliar tasks. Final evaluation data must remain separate from training and from the tasks used to optimize the agent environment.

## Planned progression

The first milestone is a reproducible baseline with a documented data selection process and evaluation suite. Small architecture experiments and controlled agent tasks will then establish which directions justify larger runs.

Promising methods will be tested individually before being combined at the target scale. A release should include the artifacts and documentation needed to reproduce its reported results, together with known limitations and unsuccessful experiments.

## Research references

This bibliography covers the papers named in our research notes and the primary publications associated with the models and datasets discussed there. GRIT is included as background for the attention sink study. SmolTalk2 and DataTrove are documented separately as dataset and software resources. These references motivate experiments; their inclusion does not establish results for SLM-1B or approve their artifacts for use in training.

The 2026-10-23 [25M performance research scan](docs/reviews/25M-PERFORMANCE-RESEARCH-2026-10-23.md)
maps recent data-selection, pruning, batch-size and optimizer findings to bounded SLM-Tavenel experiments.
It explicitly separates benchmark-targeted training from general-model claims and records corpus review limits.

### Reasoning, architecture, and agent systems

**HybridAgent.** Jiaru Zou et al. (2026). [Hybrid Reasoning Agents](https://arxiv.org/abs/2610.18861). arXiv:2610.18861. Explores transitions between latent reasoning and textual interaction through HybridLink, with working memory across tool interactions. It motivates our experiments on reasoning efficiency, agent traces, and causal inspection. [Official code](https://github.com/togethercomputer/HybridAgent).

**Coconut.** Shibo Hao et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769). arXiv:2412.06769. Introduces continuous thoughts by feeding hidden states back as input embeddings. It provides a starting point for comparing textual and latent reasoning.

**Recurrent depth.** Jonas Geiping et al. (2025). [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171). arXiv:2502.05171. Studies additional inference computation through repeated application of a recurrent block. It motivates experiments that vary computation depth while tracking quality and resource use.

**Laguna.** Julien Abadji et al., Poolside Team (2026). [Laguna M.1/XS.2 Technical Report](https://arxiv.org/abs/2605.27605). arXiv:2605.27605. Documents mixture of experts models for agentic coding and the Model Factory used to develop them. It informs our architecture and experiment infrastructure research; its models are substantially larger than our target.

**Harness².** Ruiyao Xu et al. (2026). [Harness²: Recursive Agent Harnessing for an Open World](https://arxiv.org/abs/2610.14224). arXiv:2610.14224. Studies adaptation of an agent's execution environment within a task and across tasks. Its evaluations on WorkBuddy-Bench, JobBench, and LAB motivate comparisons between fixed and adaptive environments, including the cost of adaptation. [Official code](https://github.com/google-research/harness2).

### Attention and graph models

**Attention sinks in Graph Transformers.** Alexandre Bismuth (2026). [Why don't Graph Transformers attend to the same token?](https://arxiv.org/abs/2610.19324). arXiv:2610.19324. Investigates graph symmetry, locality, and attention scoring as factors in sink formation. It motivates controlled attention experiments without assuming that graph results transfer to language models.

**GRIT, supporting background.** Liheng Ma et al. (2023). [Graph Inductive Biases in Transformers without Message Passing](https://arxiv.org/abs/2305.17589). arXiv:2305.17589. Introduces the Graph Inductive bias Transformer examined in the attention sink study. It supplies architectural context for the graph attention mechanisms discussed in our notes.

### Data and open training recipes

**Common Pile and Comma.** Nikhil Kandpal et al. (2025). [The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text](https://arxiv.org/abs/2506.05209). arXiv:2506.05209. Documents corpus construction and the Comma models used to evaluate it. It is a central reference for studying training with documented data provenance.

**FineWeb and FineWeb-Edu.** Guilherme Penedo et al. (2024). [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557). arXiv:2406.17557. Studies filtering, deduplication, and educational text selection. It informs our data preparation ablations and quality comparisons.

**FineWeb2.** Guilherme Penedo et al. (2025). [FineWeb2: One Pipeline to Scale Them All: Adapting Pre-Training Data Processing to Every Language](https://arxiv.org/abs/2506.20920). arXiv:2506.20920. Extends data processing to multilingual corpora. It informs our proposed French and multilingual experiments.

**Dolma.** Luca Soldaini et al. (2024). [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159). arXiv:2402.00159. Documents corpus composition, curation, and data processing experiments. It provides a comparison point for transparent pretraining data development.

**The Stack v2 and StarCoder2.** Anton Lozhkov et al. (2024). [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/abs/2402.19173). arXiv:2402.19173. Describes the code corpus and models trained with it. It informs our investigation of code data, source tracking, and programming evaluation.

**Tülu 3.** Nathan Lambert et al. (2024). [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124). arXiv:2411.15124. Documents supervised fine tuning, preference optimization, reinforcement learning with verifiable rewards, and evaluation practices. It informs our post training experiments and separation of development tasks from final evaluation.

### Dataset and software resources

**SmolTalk2.** Hugging Face. [Official dataset card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2). Documents the data mixtures used for SmolLM3 post training, including reasoning, instruction following, and tool calling examples. This is the primary resource for reviewing candidate subsets and their provenance.

**DataTrove.** Hugging Face. [Official repository](https://github.com/huggingface/datatrove). Provides reusable data processing components, including filtering and deduplication. It is an implementation reference for the Open Data Factory.

The public comments discussed in our notes raise additional questions about explanation faithfulness, debugging, memory robustness, and cost per successful task. We treat those questions as proposed research hypotheses, rather than additional papers or established experimental findings.

## License and attribution

Copyright (c) 2026 Théophile Avenel.

This project's original code and documentation are available under the [MIT License](LICENSE). You may use, copy, modify, merge, publish, distribute, sublicense, and sell them, including as part of commercial or proprietary projects, subject to the license terms.

Copies or substantial portions must retain the copyright notice naming Théophile Avenel and the MIT permission notice. The license does not require a separate academic citation or a visible credit in a product interface. If you discuss this project in a publication, an acknowledgement of Théophile Avenel and a link to this repository are appreciated.

## Contributing

Useful contributions include reproducible baselines, dataset provenance reviews, controlled ablations, evaluation environments, and tools for inspecting failures. Research proposals should explain the question, the comparison, the expected resource requirements, and the evidence that would support or reject the hypothesis.

The project's guiding question remains open: what capabilities can a transparent, reproducible model of one billion parameters achieve when its data, computation, and tools are developed and evaluated together?
