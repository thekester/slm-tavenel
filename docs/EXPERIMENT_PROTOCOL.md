# Experiment protocol

The README asks that each experiment define a baseline, a hypothesis, a controlled change and a reproducible evaluation procedure. This protocol says how.

## Before running

Copy [templates/EXPERIMENT.md](templates/EXPERIMENT.md) to `experiments/EXP-NNNN-short-name.md` and fill in, before any result exists:

- the question and the hypothesis, phrased so that a result could reject it;
- the baseline configuration and the single setting that changes;
- the budget: parameters, tokens, steps, expected GPU-hours;
- the primary metric and the decision rule (for example "adopt if validation bits per byte improves by at least 0.005 on both seeds and sink mass decreases");
- secondary metrics and known confounders.

Writing the decision rule first prevents choosing the metric after seeing the results.

## Controls

- Same data order: training batches are drawn from a generator seeded by (seed, step, rank, micro-step), so two arms with the same seed and data see identical batches.
- Same budget in tokens and in steps. Report wall-clock and FLOPs as well, since architecture changes affect throughput.
- At least two seeds for any adopted change at ablation scale. Report both runs, not the better one.
- Evaluation sets are fixed and decontaminated before the experiment starts.

## Running

Every training run writes `run_manifest.json` (configuration hashes, data hashes, code revision, environment), `metrics.jsonl`, `eval.jsonl` and `observability.jsonl`. The code revision includes a `dirty` flag; results from a dirty tree are exploratory and must be rerun from a clean revision before being reported.

After the final checkpoint and evaluations are written, generate a supplementary integrity index with
`python scripts/write_run_artifact_index.py <run-dir> --output <index.json>`. It hashes the latest
checkpoint, run manifest, logs, matching tokenizer and data manifests. Use repeatable `--include
<path>` options to bind related evaluation outputs and comparisons to the same checkpoint. The index
identifies files; it does not establish scientific validity, source rights, or artifact availability.

Smoke and test runs set `deterministic: true`, which forces deterministic kernels so a run resumed from a checkpoint matches an uninterrupted run bit for bit (checked in the test suite). Large runs disable it for speed; their reproducibility claim is then statistical, supported by repeated seeds.

## Measurements

| Group | Metrics |
|---|---|
| Quality | validation loss, bits per byte per domain and language, downstream tasks through lm-evaluation-harness on exported weights |
| Task completion | verified success rate, attempts, tokens and seconds per solved task, counting failed attempts |
| Cost | tokens per second, time to first token, peak memory, energy (NVML) per task, training GPU-hours |
| Internals | residual norms, attention sink mass and entropy, update-to-weight ratios, results of targeted interventions |

## Explanations

A claim about why the model behaves in some way is reported as an intervention result: which site was ablated, patched or steered, on which inputs, and how the measured behaviour changed. Attention maps or generated rationales alone are reported as observations, not explanations.

## Reporting

The experiment file is completed with the results, the decision, and links to the run directories. Negative and inconclusive results are kept and reported with the same detail as positive ones.
