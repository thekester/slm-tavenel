# EXP-0010: Scientific validation and open-weight readiness of the 1M model

**Status:** protocol design; no confirmatory training or evaluation has started. Existing 1M and
10M FSC checkpoints and all results already inspected remain exploratory. No public weight release
is approved.

This dossier covers four connected but distinct questions: whether a scientific claim replicates,
whether the evidence can be traced and reproduced, whether internal-behavior claims are supported
by interventions, and whether the resulting artifacts can be legally and technically redistributed.
Passing one does not imply passing the others.

## Purpose and claim boundary

The current 1.18M-parameter checkpoint cannot become confirmatory by adding more post-hoc
benchmarks. Its results have already been inspected, it has one training seed, its code manifest is
marked dirty, and its FSC training sources remain `candidate`. EXP-0010 therefore defines a new,
prospective experiment. It is not a re-analysis of the existing checkpoint.

The primary question is narrow: under a frozen, bilingual science-text training corpus and an
identical token budget, does head-wise attention gating change held-out English and French
bits-per-byte (BPB) relative to a parameter-matched ungated model at approximately 1.2M parameters?
This experiment will not support claims about general-purpose language ability or broad superiority
over external SLMs.

## Gate 0: data rights, provenance, and privacy

No confirmatory training may start until every source used for training passes the repository
source audit and a reviewer records corpus-level provenance, license, privacy, and any attribution
obligations. Author metadata completeness is recorded separately: a missing author field is disclosed
as supplied and is not, on its own, a reason to reject an otherwise traceable corpus for research.
The current FSC pilot does not pass: all six language/license entries are `candidate`; the audit
reports them `BLOCKED` for unverified provenance/license and unfinished privacy review. Its
manifest records `allow_unapproved: true` and `public_weights_approved: false`. CC BY and Licence
Ouverte 2.0 sources also need a documented attribution method. Keep training text and per-document
metadata out of any release artifact unless redistribution is separately authorized.

If the FSC review cannot clear the exact selected documents for the intended experiment, select a
different auditable corpus or narrow the claim to a research run. Evaluate public-weight
redistribution separately from scientific use. A declared dataset-level license alone is not a
complete rights analysis; preserve the exact source record, publisher URL where supplied, license
evidence, revision, and content hash for every included document, and document any missing metadata
without inventing it.

## Gate 1: freeze the confirmatory design

Before training, archive a signed-off protocol/configuration and a clean source-code commit. Freeze
the tokenizer, corpus revision, document IDs, filtering rules, model configurations, token budget,
training schedule, seeds, statistical analysis, and stopping rules. Record environment and dependency
versions. Do not tune against the final test split.

Create document-disjoint train, development, and sealed test sets by language and source. The
current FSC validation set has already been inspected and is development data only. Construct a new
final test set from approved records, hash it, and keep its contents and scores sealed until every
checkpoint and analysis script is fixed. Run exact benchmark filtering against training documents
before tokenizer training; also document the limits of lexical filtering and any known model-side
pretraining exposure.

## Gate 2: replicated matched training

Train two arms from scratch: (A) the head-wise-gated 1.2M model and (B) the same architecture with
the gate disabled, adjusting width only if needed to keep parameter counts within 1%. Both arms use
the same frozen tokenizer, examples, document order per paired seed, sequence length, 491.52M-token
budget, optimizer, schedule, and stopping rule. Use paired random seeds. Target 10 seeds per arm;
the minimum is 5 per arm, and any reduction must be justified by a pre-run power/sensitivity analysis
based only on pilot variance and documented before training. Never count the existing seed-1234
checkpoint as a confirmatory replicate.

## Gate 3: locked evaluation and analysis

**Primary outcomes:** English and French BPB on the sealed document-disjoint test set, reported
separately and as a predeclared equal-language macro average. Report per-seed results, paired
seed-level differences, and 95% uncertainty intervals that account for both training-seed and
document variation. Define the minimum practically meaningful effect before opening the test set.

**Secondary outcomes:** multiple-choice benchmarks with verified item provenance and redistribution
terms, untouched test labels, identical prompts/scoring, per-item predictions, and predeclared paired
bootstrap intervals. Correct for multiple secondary comparisons. Benchmarks already inspected in
EXP-0009, and ARC-Challenge/WinoGrande selected after that round, are exploratory for the current
checkpoint; the latter's 13-gram audit cannot establish absence of semantic or external exposure.

External SLMs may be reported as descriptive same-harness references only. Their training data,
tokenizers, compute, budgets, and contamination histories are unmatched, so they are not causal
controls. Do not call the 1M model superior based on these comparisons.

## Gate 4: traceability and observability

The release record must connect each reported number to a precise artifact and command. Preserve the
clean Git commit, model and tokenizer configuration hashes, corpus and split manifests, per-shard
hashes, seed, optimizer schedule, token/step counts, dependency and hardware details, checkpoint
hash, evaluation code revision, dataset revision, prompt/scoring settings, and raw item- or
document-level outputs where redistribution permits. Pin dependencies and record driver/runtime
versions, then generate `artifact_index.json` with `scripts/write_run_artifact_index.py` to bind the
checkpoint, run logs, tokenizer, dataset manifests, and related evaluation files by hash. Record
deviations and failed runs. Keep private
or non-redistributable text out of public logs; stable document IDs and hashes are sufficient to
support audit when the source terms prevent redistribution.

The current 1M run already logs residual RMS/max, attention sink mass and entropy, gate means and
closed fractions, weight norms, and update-to-weight ratios at 500-step intervals. Treat these as
diagnostic observations, not explanations. For the confirmatory runs, freeze a fixed, document-disjoint
probe set and report these measures by language/domain and seed, including distributions or quantiles
rather than only aggregate means. Record the probe IDs/revision and analysis script hash. Do not
inspect the sealed final set to select probes, layers, thresholds, or examples.

## Gate 5: bounded, intervention-based explanations

State each explanatory hypothesis before analysis (for example, whether closing a gate suppresses a
head's contribution on a defined token class). For each claim, specify the activation site, control
intervention, behavior metric, sample-selection rule, and uncertainty calculation. Compare targeted
ablation or activation patching with matched random-site and sham controls; evaluate on held-out
examples and report per-example effects, aggregation, and uncertainty. Repeat promising findings
across seeds. Correct for the number of heads/layers or label the analysis exploratory when the
targets were selected after seeing outcomes.

An attention pattern, gate value, logit lens, or generated rationale alone is a descriptive probe, not
an explanation of a decision. Intervention effects support only the narrow causal claim measured;
they do not prove a complete human-readable account of the model. The existing `slm.observe` and
`slm.interp` APIs provide capture, ablation, patching, sink/entropy, gate and residual statistics, but
EXP-0010 must add a frozen analysis recipe and reproducible outputs before making such claims.

## Gate 6: open-source and open-weight release package

The repository source code is MIT licensed, but this does not automatically license model weights,
the tokenizer, or data-derived artifacts. Before release, resolve a license for each artifact and
publish attribution/NOTICE material for all included sources. The current FSC sources are blocked, so
the present checkpoint cannot be cleared by documentation alone. Document exactly what data,
tokenizer, checkpoints, and logs are included or omitted and why.

The gated 1M architecture has no standard Llama/Qwen equivalent. A local Transformers custom-code
candidate is now implemented in `src/slm/hf_config.py` and `src/slm/hf_model.py`; generate the bundle
with `scripts/prepare_hf_candidate.py`. The one-probe CPU check in
`docs/model-cards/slm-tavenel-1m-hf-parity-report.json` found exact tokenizer IDs and logits
(maximum absolute error 0.0) and confirmed a short greedy generation. This is a narrow smoke check,
not broad dtype/device conformance or independent review. The implementation has no KV cache and
requires `trust_remote_code=True`; users should pin an immutable Hub revision before executing it.
Before upload, run broader parity and rerun the locked evaluations from the final packaged artifact.
Include the model card, data card, experiment protocol/results, provenance hashes,
environment/command record, known limitations, and a license decision. Do not label the model “open
weights” until both the rights and usability checks pass.

## Completion and release criteria

The confirmatory scientific claim is complete only when Gates 0-5 have records in the repository,
all planned seeds and locked analyses are complete, deviations are reported, and another person can
reproduce evaluation from pinned artifacts. Publish negative and inconclusive results as well as
positive ones. Open-weight readiness additionally requires Gate 6. A model card or a new license
declaration does not resolve the data-rights gate.

## Current evidence and blockers

- Existing 1M run: one seed, 15,000 steps, 491.52M tokens; useful exploratory evidence only.
- Run manifest: the tokenizer hash is present and `code.dirty` is `true`; the supplementary
  artifact index records the checkpoint hash.
- FSC source audit: all six training-source records are blocked; synthetic toy data is the only
  approved pretraining source and is unsuitable for validating language-model quality.
- Existing BPB and benchmark results have been inspected; the current held-out validation set is
  therefore development data for any follow-up, not a final untouched test.
- The existing `observability.jsonl` contains 30 training-time snapshots (initialization and every
  500 steps through step 14,500) with per-layer residual, sink/entropy, gate, weight-norm and update
  ratio summaries. It does not by itself establish causal explanations or run-to-run stability.
- Existing capture/intervention primitives are implemented in `src/slm/observe.py` and
  `src/slm/interp.py`; a preregistered probe set, controls, uncertainty analysis and multi-seed
  replication are still required for confirmatory interpretability claims.
- A local Hugging Face custom-code candidate and single-probe exact CPU parity report now exist; the
  bundle is not publicly released, broad conformance is pending, and the implementation uses no KV
  cache.
- The supplementary post-hoc benchmark queue is a separate exploratory comparison. Its outputs do
  not satisfy the confirmatory data, replication, or sealed-test gates in this protocol.

**Next required action:** select a corpus for the prospective study and finish its documented
privacy, quality, contamination, and intended-use review. MOT v1.11 remains a viable candidate for a
narrow VOA-news-domain research experiment: its English archive has no author metadata for 59.7% of
records, which must be disclosed as a corpus metadata limitation, not treated as proof of invalid
provenance. Public-weight redistribution remains a separate decision because VOA terms caution that
third-party content may have different rights. Once the corpus and intended use are recorded, freeze
the new train/dev/sealed-test documents and clean Gate 1 protocol before starting the multi-seed GPU
queue. See [the MOT source review](../docs/reviews/MOT-VOA-2026-10-21.md).
