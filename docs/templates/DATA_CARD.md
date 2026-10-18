# Data card: corpus name and version

## Summary

Purpose, size in documents, tokens and bytes, languages and domains.

## Sources

| Source id | Revision | License(s) | Obligations | Documents kept | Tokens |
|---|---|---|---|---|---|

Reviewer and date for each source, from `configs/data/sources.yaml`.

## Processing

Pipeline configuration and its hash, with the steps and thresholds. Removal counts per step and reason, from `stats.json`.

## Privacy

Automatic redactions and their counts. Summary of the manual privacy review and its known limits.

## Contamination

Evaluation sets checked, n-gram size, number of documents removed per benchmark.

## Splits

Train and validation sizes; the split rule (hash of document id, salt, fraction).

## Reproduction

Commands to rebuild the corpus from the pinned sources. Hashes of `documents.jsonl`, `decisions.jsonl` and the token shards.

## Known limitations

Gaps in coverage, biases from source selection, filters with known false positives or negatives.
