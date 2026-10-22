# EXP-0012: 10M specialist/general-text mixture pilot

**Status (2026-10-22):** data preparation completed. An initial control in Python 3.12/PyTorch 2.4 was stopped at step 400 after clear loss divergence. Its logs are retained but excluded from analysis. The matched control has restarted in the reference Python 3.11/PyTorch 2.10 environment; the 20% MOT arm and paired held-out evaluations are queued behind it.

## Research motivation

The most relevant article-level idea for base-model quality is repetition-aware pretraining mixtures:
scarce specialist text may benefit from being mixed with generic text rather than repeated alone.
The cited study reports this across more than 2,000 runs but does not establish the best ratio for
SLM-Tavenel. This pilot transfers only the testable data-mixture idea, not its claimed optimum.

Harness², LongHarness, and continual-agent papers change prompts, memory, tools, or orchestration
around a frozen model. They are agent-system hypotheses; they do not directly improve base-model
weights or justify using agent-task scores as language-model benchmarks.

## Planned comparison

- Historical reference: completed EXP-0009 9.98M gated model, FSC v1 balanced science corpus,
  15,000 steps and seed 1234.
- Matched control: rerun the same FSC-only arm in the current environment using
  `configs/train/exp0012-10m-fsc-control-py311.yaml`.
- Treatment: `configs/train/exp0012-10m-fsc-mot-mixture-py311.yaml`; same model shape, tokenizer,
  initialization seed, optimizer, learning-rate schedule, sequence length, total 491.52M
  token-presentation budget, and FSC validation shards; mix 20% MOT v1.11 news tokens with FSC.
  The assembled training set contains 544,194,188 FSC tokens and 136,048,547 MOT tokens (actual
  MOT fraction 0.2000); the 284,384,289 eligible MOT train tokens provide enough unique text, so
  no MOT article is repeated to fill the mixture.
- Primary development diagnostic: FSC validation loss, evaluated on the same validation shards as
  EXP-0009. Secondary diagnostic: a document-disjoint, temporal MOT validation split.
- Benchmarks: same pinned files and evaluator; report only as exploratory item-level outcomes. The
  previously scored benchmark sets are not used to choose a checkpoint.
- After both arms finish, evaluate each once on FSC EN/FR validation, MOT 2022–2023 development,
  and the temporally held-out MOT 2024+ test. The test is reported only after both pre-registered
  arms finish; it is not used for checkpoint selection. The comparison is queued by
  `scripts/run_exp0012_py311_queue.ps1` and recorded by `scripts/evaluate_exp0012_checkpoint.py`.

This is a single paired seed. It is more controlled than comparing against an old run because both
arms are rerun with the same current software and hardware, but it cannot establish a robust gain.
A positive outcome must be followed by paired runs with multiple seeds and a new untouched final
text-test set before being called confirmatory.

## Data scope and release boundary

FSC v1 is the existing balanced French/English science corpus and remains rights/privacy candidate
data. The generic component is MOT v1.11 English and French VOA news, pinned to repository commit
`9204cad746dfff0921e7a8f64c0cc0917bc75554`. The archive audit records 321,258 English records,
885.5M English text characters, and 42,407 French records; English author metadata is absent for
59.7% of records. Missing authors remain missing. Explicit agency bylines are excluded; that check
does not prove the remaining content is exclusively VOA-authored. The corpus-level CC BY 4.0 label
and source terms do not settle rights for every record or derived public weights.

Both corpus components remain research candidates. The new run, if completed, is not approved for
public weight release. Keep source text local and out of model bundles. Report provenance and
uncertainty plainly.

## Recipe

Data processing is frozen in `configs/data/pipeline-exp0012-mot-v1.11.yaml`; extraction and the
temporal split are recorded by `scripts/extract_mot_articles.py` and
`scripts/split_exp0012_mot_temporal.py`. The shared tokenizer is the FSC v1 train-only 8,192-token
tokenizer so the data mixture is the treatment. Tokenizer versions 0.22.2 (used by the original
FSC tokens) and 0.23.3 produced identical token-ID fingerprints over the first 128 MOT train
documents. MOT records were quality-filtered, PII-redacted (920 email, 607 phone and 1 IP
redactions), exact/near-deduplicated (247 exact and 6,684 near-duplicate removals), and checked
against all six benchmark files (7 overlaps removed). The temporal split is train through 2021,
development 2022–2023, and test from 2024 onward. It contains 292,963 English and 40,418 French
train articles, 13,518 dev articles, and 8,280 test articles. No MOT test text enters training or
checkpoint selection. FSC validation shards are copied unchanged to both arms.

The FSC benchmark decontamination inherited from v1 included ARC-Easy, HellaSwag, and both Belebele
sets; it did not include ARC-Challenge or WinoGrande. Therefore all six-task results remain
exploratory, even though the MOT portion was checked against all six. The run environment will be
recorded by each manifest; the historical EXP-0009 run used a different PyTorch version and is not
the matched control for inference.

The discarded control attempt started at 2026-10-22 07:41 Europe/Paris in Python 3.12/PyTorch 2.4
and reached loss 10.37 with gradient norm 25.32 by step 400; it was stopped before the first
checkpoint. Its metrics and manifest remain under
`runs/EXP-0012-10m-fsc-control-2026-10-22/` to preserve the audit trail. It is not a completed
model or a valid comparison arm.

The matched control restarted at 2026-10-22 07:46 in Python 3.11/PyTorch 2.10, matching the
reference run's software versions. At step 250 it measured 73,422 tokens/s and loss 4.09193;
the reference run's step-250 loss was 4.09412. `scripts/run_exp0012_py311_queue.ps1` runs the FSC control,
then the 20% MOT treatment, then evaluates both checkpoints; queue events are logged in
`runs/EXP-0012-py311-queue.log`. This runner sets `resume: false` in both configs. Before the
restart, the older PyTorch 2.10 run measured about 52,000 tokens/s. The current run's first interval
implies about 1 h 53 m per 15,000-step arm, about 3 h 46 m for both arms, plus evaluation. Actual
duration may differ as throughput changes during training.

## Release-time implementation correction

The original mixture builder resized the news stream to the exact token target before checking
its last document boundary, so the recorded mixture may end with a partial final article. The
updated builder trims to the last EOS at or before the target and rejects empty streams,
inconsistent token formats/counts and nonempty output directories. Future regenerated mixtures
can therefore contain slightly fewer news tokens. The historical counts above and the active
training inputs were not changed by this source release. The ordering is inherited from the
stable document-ID hash sort; the builder does not use its seed argument to reshuffle records.

## References

- Sedova et al. (2026), [Scaling Laws for Mixture Pretraining Under Data Constraints](https://arxiv.org/abs/2605.12715).
- Xu et al. (2026), [Harness²: Recursive Agent Harnessing for an Open World](https://arxiv.org/abs/2610.14224).
- Chen et al. (2026), [Are Self-Evolving Agents Continual Learners?](https://arxiv.org/abs/2610.25858).
