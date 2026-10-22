# Archived record: SLM-Tavenel 1M CC0 v4 pilot

> **Status:** historical single-seed candidate, superseded by EXP-0010. This record and its
> package are retained for audit and are not the current Hugging Face release candidate. The
> EXP-0010 package is being prepared locally and has not been uploaded.

## Scope

This record describes the single v4 checkpoint previously packaged locally for an experimental
open-weight release. It is a research artifact, not a claim of legal clearance, privacy
certification, or confirmatory scientific validation. Earlier 1M candidates and their reviews are
retained in the repository's experiment history; their checkpoints are not included in that package.

## Data source and traceability

- Dataset: [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons),
  subset `french_open_science`, pinned revision
  `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`.
- The selection started from 3,450 English/French records labeled CC0 by the dataset and also
  labeled CC0 across all versions returned by a paginated HAL records API audit (1,767 EN,
  1,683 FR). This is traceable license metadata, not proof that each depositor had authority to
  apply CC0.
- Filtering retained 1,229 documents (748 EN, 481 FR). The package records source identifiers,
  URIs, decisions, pipeline configuration, split information and hashes. Article text is not
  included.
- Automated filtering/redaction and a deterministic 20-excerpt screen are documented in
  `training_record/privacy-screen-report.json`. These checks did not constitute a full manual
  review of every document. Both source-rights review and manual privacy attestation remain
  unsigned.
- Records whose authorship details were unavailable are reported as supplied by the dataset or
  repository metadata; no author identity was inferred.

## Training and evaluation

The model has 1,181,920 parameters and was trained from scratch for 488 steps with seed 421021,
approximately one presentation of 15.98M unique training tokens. The run manifest records the
configuration, code revision, environment, metrics and observability. The source tree was dirty
at the recorded revision, so the included source files and hashes are the best available record
of the code used; the revision alone is not a complete immutable source snapshot.

Six same-harness multiple-choice results are reported in `training_record/mc-six-benchmarks.json`.
They are exploratory, near chance on most tasks, and were inspected after training. The sealed
text test was not scored. The TinyStories 0.935M comparison is descriptive: training data,
tokenizer, language mix and token budget differ, and the post-hoc task comparisons are not
corrected for multiplicity. No causal or superiority claim is supported.

## Release terms and limits

The weights and included inference code are marked MIT. The bundle provides the model parameters,
architecture/inference code, training code, and detailed data information. This supports open
inspection and reuse of those released components, subject to the stated MIT terms. It does not
establish that every training source was independently rights-cleared or that all downstream
uses of the trained weights are free of third-party claims. The record deliberately distinguishes
reported source metadata from legal conclusions.

The artifact is one small, single-seed pilot. It does not establish repeatability, broad model
quality, the causal effect of the attention gate, absence of memorized content, or absence of all
personal/sensitive information. See the model card and the machine-readable provenance for the
complete limitations and hashes.
