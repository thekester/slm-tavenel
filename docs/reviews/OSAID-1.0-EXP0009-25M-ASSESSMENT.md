# OSAID 1.0 self-assessment: SLM-Tavenel grown 25M EXP-0009

**Assessment date:** 23 October 2026\
**Artifact:** local candidate `exports/slm-tavenel-25m-exp0009-candidate/`\
**Checkpoint SHA-256:** `ac7cb9c5c418bfadd33b0063f37c7a12bcb667896db64fe08b8bea05a95b5d8d`\
**Hub status:** public at [athenista/slm-tavenel-25m-exp0009](https://huggingface.co/athenista/slm-tavenel-25m-exp0009). This is the project's self-assessment, not OSI certification, a legal opinion, or a source-rights/privacy clearance.

## Conclusion

The candidate package contains the core elements that OSAID 1.0 asks an open-source AI model release to provide: an explicit permissive offer for the project-authored weights and code, Data Information under MIT terms, the training and data-processing source code, a pinned public data source with record-level provenance, and the model parameters. On that basis, the project may describe a public release of this complete package as **OSAID 1.0 aligned by project self-assessment**. The package must be made publicly available for that claim to apply to the release.

This conclusion is separate from whether every upstream source declaration was made by an authorized rights holder, whether every document is free of personal information, or whether a court would uphold a particular use. Those uncertainties are stated in the model card and source dossier. They remain real release and reputational risks; they are not themselves signature fields in OSAID 1.0. Do not describe this project as “OSI-approved” or “OSI-certified.” OSI does not certify individual models.

OSI's definition requires the freedoms to use, study, modify and share, and identifies Data Information, Code and Parameters as the preferred form for modifying a machine-learning system. Its FAQ expressly distinguishes public training data from data that must be redistributed: for public data, the release must disclose full details on where to obtain it. The FAQ also says the definition does not mandate bit-for-bit reproduction. This candidate points to the pinned, publicly available corpus and includes the record manifest and processing recipe rather than republishing the source articles. See [OSAID 1.0](https://opensource.org/ai/open-source-ai-definition) and the [OSI FAQ, especially its data and reproducibility sections](https://opensource.org/ai/faq).

## Requirements and package evidence

| OSAID element | Candidate evidence | Assessment |
|---|---|---|
| Use, study, modify and share | The model card and `LICENSE` explicitly offer the project-authored package and parameters under MIT terms. The model is a research base model with no use restrictions. | Present as the project's offer; this does not settle third-party claims over training sources. |
| Data Information | Pinned source `PleIAs/French-Science-Commons`, subset, revision, 31,417-record metadata manifest, source IDs, declared license labels, selection and processing decisions, splits, hashes, token manifests, benchmark decontamination recipe and limitations are included. Source text is publicly obtainable from the cited dataset and is not redistributed in this model bundle. | Sufficiently detailed to inspect the declared data and rebuild a substantially equivalent training set. Exact historical HAL file versions are not mapped for every document, so bit-identical data reconstruction is not claimed. |
| Data Information terms | Project-authored provenance and documentation are included in the package under MIT. The public upstream data remains under its own per-record terms; this package does not relicense it. | Present for project-authored Data Information; upstream terms and attribution obligations are disclosed separately. |
| Complete Code | `training_record/reproduction_source/` contains the tracked source tree at the run's recorded Git commit `95da2cf260923bb93944f2880678a4240f419274`, including the CLI, model, growth, training, data pipeline, configurations and scripts. The run manifest records its full training configuration and environment; processing and token manifests record their resolved configurations and input/output hashes. | Present for reconstructing the method. The original worktree was dirty, so the package does not claim byte-for-byte reproduction of the historical checkout or exact final weights. OSAID does not require bitwise reproducibility. |
| Code terms | Project code is offered under MIT; `pyproject.toml`, dependency details and the source snapshot are included. | Present. |
| Parameters | `model.safetensors` is included, and `LICENSE` explicitly offers the project-authored model package under MIT terms. | Present in the local package; it becomes an actual public offer only when released. |
| Transparency about limits | The card discloses the source-license audit, 155 unresolved HAL records, attribution needs, privacy scans, unsigned human reviews, one seed, inherited 10M initialization, benchmark limits and dirty-worktree status. | Present. Do not convert these disclosures into claims of legal clearance, privacy certification or robust general performance. |

## Rights and privacy findings remain disclosed

The 31,417 training records have source-declared labels: 4,528 CC0, 26,368 CC BY 4.0, and 521 Licence Ouverte 2.0. The HAL audit found 31,262 metadata matches and 155 unresolved records: 72 mismatches, 3 conflicts, 24 missing file-license fields, and 56 not found. Crossref discovery did not establish rights-holder authority or the exact historical files used. Attribution materials for CC BY and Licence Ouverte records are included. See the [HAL audit](EXP-0009-HAL-FILE-LICENSE-AUDIT-2026-10-22.md), [recheck and rights triage](EXP-0009-25M-DEEP-RIGHTS-PRIVACY-TRIAGE-2026-10-23.md), and [conservative source policy](EXP-0009-CONSERVATIVE-LICENSE-POLICY.md).

The privacy scan surfaced broad review signals on 20,808 document IDs and four IPv4-pattern matches. Focused triage found the four matches in networking/security education or research examples, but the broader privacy review and a model memorization assessment remain open. `manual_privacy_review_signed` remains false. The technical checks and their limits are described in the [25M source and privacy dossier](EXP-0009-25M-SOURCE-PRIVACY-REVIEW.md). No claim is made that the corpus or model is anonymous or free of memorized passages.

The words `source_rights_review_signed` and `manual_privacy_review_signed` remain false in provenance. The publication authorization is recorded separately from those reviews. They track project governance and risk acceptance; they are not prerequisites named by OSAID. Do not represent the unsigned source or privacy reviews as complete.

## Scientific scope

This is one 25.72M-parameter run, seed 1234, grown from a trained 10M checkpoint and then trained for 15,000 additional steps (491.52M token positions). The comparisons and BPB intervals are conditional on these saved checkpoints and sampled items; they do not estimate training-seed variability. External models use different data, tokenizers and budgets. These results support an inspectable research artifact, not a robust scaling law or a general claim of benchmark superiority. See the [EXP-0009 results report](../../experiments/EXP-0009-25m-scaling-results-2026-10-22.md).

## Release checklist

For the public release, the exact directory was frozen, the 243-file SHA-256 inventory was verified, and the previously passed inference-parity report was checked against the unchanged weights, tokenizer and inference code. The Hub card includes this assessment and the rights/privacy limitations. No source text is included. The source-rights and manual-privacy signoffs remain false.
