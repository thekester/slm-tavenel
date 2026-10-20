# Source review: Wikipedia data used by wiki-v1

**Review date:** 2026-10-20

**Decision:** provenance pinned; license and privacy clearance incomplete; keep both sources
`candidate`. wiki-v1 and its training runs remain exploratory and are not cleared for public
weight release.

## Scope checked

wiki-v1 uses English and French text from the Hugging Face dataset repository
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), pinned to revision
[`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa).
The repository labels the dataset with both `cc-by-sa-3.0` and `gfdl`. The raw fetch records under
`data/raw/wiki/*.fetch.json` contain the repository revision, subset, shard names, per-shard SHA-256,
record counts and combined file hashes. The resolved wiki-v1 pipeline manifest records hashes of
the four JSONL inputs and the processed corpus.

| Language | Parquet shards used | Input records |
|---|---|---:|
| English | `train-00000`, `00013`, `00027`, `00006`, `00020`, `00034` of 41 | 937,729 |
| French | `train-00000`, `00008`, `00004`, `00012` of 17 | 603,447 |

The order reflects the two fetch batches; exact filenames and SHA-256 values are in the fetch
records. The pipeline kept 715,882 English and 461,019 French documents after its length, quality,
repetition, PII-pattern, deduplication and benchmark-overlap filters. These checks establish which
files were processed; they do not independently certify the rights in every article.

## License findings

The dataset publisher's card declares **CC BY-SA 3.0 and GFDL**. Wikimedia's current Terms of Use
describe contributed text as available under CC BY-SA 4.0 and GFDL and require attribution; they
describe attribution by linking to the reused article or an equivalent author-credit source. They
also describe share-alike and change-notice obligations for modified text. These current terms do
not replace the license declarations attached to this pinned 2023 dataset snapshot. The source
registry now records both publisher-declared licenses instead of the incomplete single-license
entry used when the corpus was first built.

The exact license option available for each included historical page has not been audited, including
exceptions for text imported from elsewhere. The project has also not established whether training
or distributing these model weights creates a work to which attribution or share-alike terms apply,
or how to satisfy those terms for a model trained on over one million retained articles. That
requires a documented legal assessment before redistribution; this repository review is not legal
advice. Until then, the source stays `candidate`, and the weights must not be described as cleared
for public or commercial release.

## Privacy and extraction findings

The pipeline's `pii` step is pattern-based. The wiki-v1 stats report 225 email, 2 IBAN, 58 public IP
and 192 phone-pattern redactions. It does not detect names, sensitive facts in prose, or every form
of personal information. Wikipedia includes biographies, including living people, and no residual
privacy audit of retained text was performed. Therefore the registry correctly leaves
`pii_reviewed: false`.

The dataset is already extracted text rather than raw wiki markup. This review did not validate the
upstream extraction for every article or inspect the retained documents for template/reference
artefacts. That remains a data-quality follow-up, separate from provenance pinning.

## Disposition

- Provenance: verified to the pinned dataset revision and fetched shard/input hashes.
- Publisher-declared licensing: checked and recorded as CC BY-SA 3.0 plus GFDL; the rights for each
  page and implications for model-weight distribution remain unresolved.
- Privacy: incomplete beyond the recorded pattern replacements.
- Registry: both languages remain `candidate`; license and privacy blockers remain active.
- Experiment: the already-running wiki-v1 replication is unchanged and remains exploratory. Its
  pipeline manifest predates this review and records the earlier single-license declaration; this
  review documents that discrepancy rather than rewriting the historical run manifest.

## Sources

- [Pinned Wikimedia Wikipedia dataset revision](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa)
- [Wikimedia Foundation Terms of Use, section 7: Licensing of Content](https://foundation.wikimedia.org/wiki/Terms_of_Use#7._Licensing_of_Content)
- [CC BY-SA 3.0 license deed](https://creativecommons.org/licenses/by-sa/3.0/)
- [GNU Free Documentation License 1.3](https://www.gnu.org/licenses/fdl-1.3.html)
