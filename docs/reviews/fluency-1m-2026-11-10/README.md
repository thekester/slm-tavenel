# SLM-Tavenel 1M: free-generation diagnosis, 10 November 2026

**Outcome:** the reported poor generation is reproducible. This change adds an
evaluation tool and candidates to train; it does not improve or replace any
published weights. The 48 completions below are baseline observations, not a
48-item proof of a new model's superiority.

## What was verified

The audit downloaded the public HF model at commit
`77d1d8b09af0ff8583844460f383f8f1c4f9bd96`, its configuration, published training
record and tensors. It also inspected the repository's EXP-0011 continuation
report and downloaded `shibatch/tiny1m@fdacad8ff9dc27bd72c77d3684075a494be1ba11`.
This is the reference used by the existing benchmark table, not the original
Ronen Eldan GPT-Neo checkpoint named TinyStories-1M.

| Fact | Tavenel 1M | tiny1m reference |
|---|---:|---:|
| Counted unique parameters | 1,181,920 | 935,040 |
| Embedding and output tables, unique parameters | 786,432 (shared) | 131,072 (unshared) |
| Remaining parameters | 395,488 | 803,968 |
| Vocabulary | 8,192 | 512 |
| Context tokens | 1,024 | 256 |

Tavenel spends 66.54% of its capacity on the shared vocabulary table. Its
reference has about twice as many parameters outside its input/output tables,
despite fewer parameters overall. This is a design tradeoff, not evidence that
an 8,192-token vocabulary is universally wrong; smaller vocabularies increase
sequence length and alter effective context and compute.

The released Tavenel run used 15,990,784 token presentations over 1,229 retained
English/French scientific documents before splitting. It was an attention-gate
study. The next training experiment should optimize free generation explicitly.
The published EXP-0011 result is already negative for simply repeating the same
corpus: same-window validation loss 3.64577 → 3.84200. That is one recipe/seed,
not proof that all continued training fails.

The six-task multiple-choice average is 0.293 for Tavenel and 0.298 for tiny1m.
Tavenel's ARC-Easy score (0.280 vs 0.256) is a local advantage, not an overall
win; most scores remain close to the relevant chance levels. QCM likelihood and
coherent free generation measure different behavior. Existing paired intervals
remain the appropriate source for comparisons; this audit did not rerun QCMs.

## Executed diagnostics

- 8 published development prefixes, 4 English/4 French, spanning stories,
  everyday prose and scientific prose.
- Each model: one greedy completion and two sampled completions per prefix
  (temperature 0.8, top-p 0.9; seeds 421201/421202), for 24 per model, 48 total.
- Target 320 UTF-8 bytes, final-token overshoot retained. Model-specific context
  limits and a 512-token cap are enforced. No best-of selection or chat prompt.
- CPU float32, PyTorch 2.10.0+cpu, Transformers 4.57.6, Python 3.12;
  actual environment, source hashes and every token ID are in the artifacts.
- Run started from clean code commit
  `9c0483b5b66ed1d8d57e56f35e538a11dd6308f7`.

On the first English story prefix, Tavenel's greedy output switches to a
section heading and repeatedly emits a LaTeX fraction. On the parallel French
prefix, it switches to a corollary/abstract and loops through scientific
phrases. tiny1m stays closer to an English dog/girl narrative, but also repeats
and is not a fluent French baseline. These are qualitative observations from
inspection, not blind ratings. The review CSV's scores are intentionally empty.
Nonrepetition by itself must never be presented as grammatical or useful prose.

The published weight SHA-256 is
`b2ef1606aacc86808d575cc3ea85d93c689b37e9dfb933a6627ea7d8e6dbde6f`.
Loading those exact weights into the native SLM (restoring its tied
`lm_head.weight` alias) and the published HF implementation gave a maximum
absolute logit difference of **0.0** on three unpadded prefixes. This rules out
a native/HF arithmetic mismatch on these probes; it is not an exhaustive audit
of padding, caches, every deployment UI or every possible input.

## Artifacts

- [Manifest, versions and input hashes](manifest.json)
- [All 48 completions and raw token IDs](generations.jsonl)
- [Blind human review sheet](blind_review.csv)
- [Separate model-identity key](review_key.json)
- [Published-checkpoint parity diagnostic](published-checkpoint-logit-parity.json)
- [Experiment, commands, controls and adoption criteria](../../../experiments/EXP-0044-1m-fluency-2026-11-10.md)

Validation: 51 targeted tests passed across `test_fluency.py`, `test_model.py`,
`test_observe.py`, `test_hf_native_adapter.py` and `test_trace.py`; the full
48-generation CLI run completed. These checks validate the tooling and retained
model interfaces, not the efficacy of a new training recipe. GPU execution was
not tested here; GPU/CPU energy was not measured for these diagnostics.

## Next intervention

Curate a diverse, simple bilingual prose corpus with the existing source and
privacy policy; remove recurring extraction artefacts; freeze source-grouped
splits and train the tokenizer on train only. Then compare the original model
architecture with a compact-vocabulary control and a 1,166,720-parameter
capacity-reallocated candidate. Keep observability, gate hooks, hashes, seeds,
optimizer/RNG states and measured energy. The staged proposal starts with a
100M-presented-token pilot, not a guaranteed optimum or a promised percentage.

The new corpus is not yet built and no new checkpoint has been trained. This
environment had CPU inference available but no CUDA GPU or access to the user's
RTX 4060. The existing weights remain the baseline until the prospective tests
pass. No corpus approval, OSI certification or improved-model claim is implied.

## Primary records consulted

- [Pinned Tavenel model card](https://huggingface.co/athenista/slm-tavenel-1m/blob/77d1d8b09af0ff8583844460f383f8f1c4f9bd96/README.md)
- [Pinned run manifest](https://huggingface.co/athenista/slm-tavenel-1m/blob/77d1d8b09af0ff8583844460f383f8f1c4f9bd96/training_record/selected-run-manifest.json)
- [Pinned benchmark comparisons](https://huggingface.co/athenista/slm-tavenel-1m/blob/77d1d8b09af0ff8583844460f383f8f1c4f9bd96/training_record/benchmark-comparisons.json)
- [EXP-0011 continuation](../../../experiments/EXP-0011-1m-low-lr-continuation-2026-10-22.md)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [Meta-Harness paper](https://arxiv.org/abs/2603.28052)
- [OSI Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition)
