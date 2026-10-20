# EXP-0008: decontaminated 10M replication on wiki-v1

*Status:* nine-run replication started 2026-10-20; AdamW seed 1234 is training
*Purpose:* replace the exploratory wiki-v0 benchmark comparison with a controlled replication on the
same decontaminated corpus.

## Question and preregistered hypotheses

On a common wiki-v1 corpus with benchmark n-gram overlap removed, do head-wise gated attention or
Muon improve held-out English/French bits per byte relative to the AdamW baseline at 10M scale?
Multiple-choice accuracy is a secondary outcome; the earlier scores do not establish a winner.

## Design

| Arm | Model | Optimizer | Difference from baseline |
|---|---|---|---|
| AdamW | 10M, ungated | AdamW | reference |
| Head-wise gate | 10M, head-wise gate | AdamW | attention gate only |
| Muon | 10M, ungated | Muon | optimizer only |

Each arm uses seeds 1234, 2 and 3. A new 8,192-token tokenizer is trained on a deterministic 120,000-
document sample of filtered wiki-v1; wiki-v1 is then encoded again with that tokenizer. Within a seed,
all arms use the same wiki-v1-clean token shards, tokenizer, sequence length, batch size, optimizer
schedule, 15,000 steps and benchmark files. This is 491.52M training tokens per run (nine runs total).
Seed-matched arms receive the same deterministic batches. The combined
gate+Muon arm is omitted from the primary comparison because it would not isolate either change.

`wiki-v1` is filtered against ARC-Easy, HellaSwag and both Belebele splits using exact word 13-gram
overlap on each item context and choices. The filter's limitations are explicit: it cannot rule out
paraphrase, translation or other semantic exposure. The old wiki-v0 tokenizer is excluded because it
was learned from the unfiltered corpus. A deterministic, language-stratified set of 500
documents per language is selected from wiki-v1's existing held-out document split for BPB; its IDs
and file hashes are recorded before training.

## Outcomes and interpretation rule

- Primary: BPB on the fixed 500-document English and French held-out sets, reported for every seed
  and as seed-matched differences against AdamW.
- Secondary: raw multiple-choice accuracy on all four pinned benchmark files, plus per-item paired
  bootstrap confidence intervals for seed-matched model comparisons. These intervals describe item
  sampling conditional on each checkpoint, not all training randomness; report all three seeds separately
  as descriptive results. Three seeds and four benchmarks are not a confirmatory test of task accuracy;
  do not declare an MC winner from this experiment.
- Observability: validation loss, throughput, GPU memory, and the same attention-sink analysis used
  in the earlier experiments.
- Call a BPB result a replicated gain only if the treatment improves both languages at all three seeds.
  Treat any mixed result as inconclusive. Any future MC winner claim requires a separately powered,
  multiplicity-controlled study with more seeds.

## Data and provenance limitation

wiki-v1's current manifest records only five 13-gram-overlapping training documents removed. This
count will be independently audited and broken down by benchmark before the results are interpreted.
The underlying Wikipedia sources remain marked `candidate`; the pipeline used `--allow-unapproved`.
The source review pins the fetched shards, but confirms that page-level license choices, implications
for model-weight distribution and privacy review are unresolved. This replication therefore remains
exploratory. See [the wiki-v1 source review](../docs/reviews/WIKIPEDIA-WIKI-V1.md). Same-harness scores for external models are contextual references,
not controlled comparisons, because their training data and budgets differ.

## Run outputs

Runs will be written under `runs/wiki-v1-clean/`, with the nine run manifests, item-level MC outputs,
seed-matched comparison reports, BPB holdout manifest, contamination audit and evaluation hashes.
