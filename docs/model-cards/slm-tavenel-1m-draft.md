# SLM-Tavenel 1M — draft research model card

**Status:** internal research record; not approved for public weight release. This draft documents
the existing exploratory checkpoint and is not a claim of scientific validation.

## Summary

- Model: SLM-Tavenel 1M, decoder-only Transformer with head-wise gated attention.
- Parameters: approximately 1.18 million.
- Training: 15,000 steps, 491.52 million tokens, one seed (1234), from scratch.
- Checkpoint: `runs/fsc-scaling/1m-gated-s1234/checkpoints/step_00015000.pt`.
- Checkpoint SHA-256: `58f42147b63d27311e4ebf767d0f588e52d026026bf105cf919bfb694df0eb36`.
- Training corpus: French-Science-Commons pilot v1, bilingual English/French. The source audit
  currently blocks its sources for incomplete provenance/license and privacy review.
- Intended research use: study compact bilingual language-model training and gated-attention
  behavior. Do not treat this checkpoint as a validated general-purpose model.

## Licensing and distribution

No weight license has been assigned. The repository's MIT license covers the original code and
documentation only; it does not grant rights to training data or derived weights. The current corpus
review is unresolved, and public weight distribution is not approved. The tokenizer and any other
derived artifacts require the same explicit review.

The gated architecture does not map to standard Llama/Qwen attention. A local Transformers custom
model candidate now exists; a fixed CPU probe showed exact tokenizer/logit parity and greedy
generation. The report is [here](slm-tavenel-1m-hf-parity-report.json). This narrow smoke check
does not clear rights or replace broader parity/evaluation review. See
[EXP-0010](../../experiments/EXP-0010-confirmatory-validation-1m.md).

## Training and provenance

The run manifest is `runs/fsc-scaling/1m-gated-s1234/run_manifest.json`. It records tokenizer,
data-shard, configuration, environment and code-revision information; the code tree is marked dirty.
The original manifest did not record the checkpoint hash, so a supplementary
[artifact index](slm-tavenel-1m-artifact-index.json) records hashes for the checkpoint, run logs,
tokenizer, and dataset manifests.
The data manifest points to `data/tokens/fsc-pilot-v1-scaling`. Do not redistribute source text or
processed corpus shards unless their terms are reviewed for that specific use.

This single run does not quantify training-seed variation. Its data and evaluation outcomes have
already been inspected. They are exploratory and cannot serve as a confirmatory test set for a new
claim.

## Evaluation (exploratory)

| Measure | Result | Interpretation limit |
|---|---:|---|
| English BPB | 1.490382 | One checkpoint on the inspected FSC validation split |
| French BPB | 1.367617 | One checkpoint on the inspected FSC validation split |
| ARC-Easy accuracy | 0.286 | Same-harness descriptive result; benchmark exposure not fully ruled out |
| HellaSwag accuracy | 0.267 | Same-harness descriptive result; benchmark exposure not fully ruled out |
| Belebele English accuracy | 0.279 | Same-harness descriptive result; benchmark exposure not fully ruled out |
| Belebele French accuracy | 0.292 | Same-harness descriptive result; benchmark exposure not fully ruled out |
| ARC-Challenge accuracy | 0.199 | Post-hoc score; zero exact 13-gram matches found, but semantic/external exposure is unknown |
| WinoGrande XL validation accuracy | 0.502 | Post-hoc validation split; near chance, labels are not an untouched confirmatory set |

Scores are not evidence that this model is superior to other small language models. External
references differ in data, tokenizer, training budget, and possible pretraining exposure. See
[EXP-0009](../../experiments/EXP-0009-fsc-scaling-and-references.md) for methods and comparison
limits.

The supplementary task results and paired intervals are summarized in the supplementary section of
[EXP-0009](../../experiments/EXP-0009-fsc-scaling-and-references.md). The paired intervals include
zero for the reported 1M comparisons; they do not establish differences from the 10M or TinyStories
checkpoints.

## Observability and interpretability

Training-time logs are in `runs/fsc-scaling/1m-gated-s1234/observability.jsonl`. They record residual
RMS and maximum magnitude, attention sink mass and entropy, gate means and closed fractions, weight
norms, and update-to-weight ratios every 500 steps. These are descriptive diagnostics. They do not
establish why a prediction was made or whether an effect generalizes across seeds.

The repository implements activation capture and interventions in `src/slm/observe.py` and
`src/slm/interp.py`. No confirmatory, preregistered, multi-seed causal explanation for this 1M model
is reported here. Attention patterns or generated rationales alone are not causal explanations.

## Limitations

- One training seed and no estimate of between-run uncertainty.
- Training sources have not passed the source rights and privacy audit.
- Validation and benchmark outcomes were inspected; they are not untouched confirmatory evidence.
- Exact-overlap filtering cannot exclude paraphrase, translation, or external pretraining exposure.
- Bilingual science-domain data do not establish broad-domain language ability.
- Public-weight licensing and a supported gated-model inference/export path remain unresolved.

## Reproduction artifacts

The run directory contains `run_manifest.json`, `metrics.jsonl`, `eval.jsonl`, and
`observability.jsonl`. Reproduction of this exploratory run is distinct from confirmation: a
confirmatory evaluation needs a clean code revision, a newly frozen corpus and sealed test, paired
multi-seed gated/ungated runs, and a preregistered analysis. Requirements are tracked in
[EXP-0010](../../experiments/EXP-0010-confirmatory-validation-1m.md).

Regenerate the integrity index and bind the supplementary evaluations to this checkpoint with:

```powershell
python scripts/write_run_artifact_index.py runs/fsc-scaling/1m-gated-s1234 --output docs/model-cards/slm-tavenel-1m-artifact-index.json --include runs/fsc-additional/slm-tavenel-1m/fsc-pilot-v2-mc.json --include runs/fsc-additional/contamination-audit.json --include runs/fsc-additional/comparisons/slm-tavenel-1m-vs-slm-tavenel-10m.json --include runs/fsc-additional/comparisons/slm-tavenel-1m-vs-tiny1m.json
```
