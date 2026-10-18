# Model card: model name and version

## Summary

Parameters, architecture (link to `configs/model/...`), context length, tokenizer, base or instruct.

## Licensing

License of the weights. Obligations inherited from training data (attribution, share-alike), listed per source.

## Training data

Data card links and manifest hashes for every corpus used, in pretraining and in post training. Tokens seen, epochs per source.

## Training procedure

Training configuration and hash, code revision, hardware, wall-clock time, GPU-hours, measured MFU, energy where measured. Intermediate checkpoints released.

## Evaluation

| Benchmark | Setting | Score | Decontaminated |
|---|---|---|---|

Cost: tokens per second, time to first token, peak memory and energy per task on stated hardware. Verified task completion with cost per solved task.

## Observability

Links to `metrics.jsonl`, `eval.jsonl` and `observability.jsonl`. Intervention-based findings with links to their experiment files.

## Intended use and limitations

Intended uses, out-of-scope uses, known failure modes, languages with weaker performance, risks.

## Reproduction

Commands from source data to these weights. Hashes in `provenance.json`.

## Citation and acknowledgement

Copyright (c) 2026 Théophile Avenel. Data attributions.
