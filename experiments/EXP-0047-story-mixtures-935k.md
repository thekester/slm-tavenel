# EXP-0047: improve story prediction at a fixed 935,040 parameters

## Question and decision rule

Can more varied story training improve TinyStories likelihood while retaining SimpleStories performance, without increasing the deployed model's parameter count? The published EXP-0045 export remains the frozen baseline. There is no claim of success before measurements.

The first intervention changes only the training-data pool and mixture. All three pilots share the existing 1,024-token tokenizer, four-layer architecture, initialization seed 421201, 256-token training windows, AdamW, cosine schedule and 100,007,936 presented token positions (6,104 updates). The tokenizer was learned from EXP-0045 SimpleStories train only; keeping it fixed controls this variable but may favor SimpleStories. Ratios 0%, 30% and 60% TinyStories are exploratory choices, not published optima.

| Arm | TinyStories proportion of stored training tokens | SimpleStories proportion | Budget |
|---|---:|---:|---:|
| tiny00 | 0% | 100% | 100,007,936 positions |
| tiny30 | 30% | 70% | 100,007,936 positions |
| tiny60 | 60% | 40% | 100,007,936 positions |

Each arm contains 32 million token positions sampled into random training windows. Source reservoirs are shuffled deterministically and stream prefixes are used; later tokens in a selected final document can be truncated. Expected sampled proportions differ slightly from stored proportions because a window cannot start in a shard's last 257 positions; this expectation is recorded. Compared with EXP-0045, corpus size, sampling and diversity change together. The three new arms isolate mixture at a fixed stored-pool size, not a pure one-variable comparison against the old run.

Select the minimum of the equal-corpus mean of development BPB: `(SimpleStories BPB + TinyStories BPB) / 2`. This is a declared selection utility, not a universal quality score. Report both component values and regressions. The runner also measures both fixed public models on these identical dev texts. It then trains the selected recipe from scratch with a full 300,007,424-position cosine schedule (18,311 updates), preserving the 100M checkpoint. It does not resume the decayed learning-rate tail and label it an equivalent full-budget run.

## Completed 100M pilots: development results

All rows score the same new 512-document dev partition of each source, at 935,040 parameters. These are **not** the 21,990-story official TinyStories validation or the old 3,185-story SimpleStories dev scores. Lower BPB is better.

| Model / training recipe | SimpleStories dev BPB | TinyStories dev BPB | Equal-corpus mean |
|---|---:|---:|---:|
| Published SLM-Tavenel Story 935K, EXP-0045 | 0.939755 | 1.547576 | 1.243666 |
| SLM-Tavenel Story 935K, 0% TinyStories, local 100M pilot | 0.939626 | 1.529223 | 1.234424 |
| SLM-Tavenel Story 935K, 30% TinyStories, local 100M pilot | 0.974531 | 0.986588 | 0.980559 |
| SLM-Tavenel Story 935K, 60% TinyStories, local 100M pilot | 1.014045 | 0.907132 | **0.960588** |
| Published reference shibatch/tiny1m | 1.643413 | 1.055680 | 1.349547 |

The 60% recipe was selected by the declared dev-mean rule and its 300M-token follow-up is in progress at this release. It lowers TinyStories dev BPB by about 41.4% relative to the published SLM-Tavenel baseline, while **raising SimpleStories dev BPB by about 7.9%**. It is lower than tiny1m on both of these dev sets, but this does not establish a confirmed benchmark win: one seed, dev-based selection, unknown reference exposure to source train, and no human coherence ratings. The original goal of retaining the published baseline's SimpleStories performance is not yet met by this recipe.

The [selection record](../docs/reviews/story-mixture-pilots-2026-11-10/selection.json), [archive manifest](../docs/reviews/story-mixture-pilots-2026-11-10/archive.json), compressed per-document scores, completed training curves and frozen input hashes are archived together. `python scripts/archive_exp0047_pilots.py` verifies alignment and reconstructs each BPB before copying evidence. Raw training data, reserved test text and new weights are absent from this archive. Original dirty-worktree execution metadata is preserved; later publication does not turn it into a clean-commit run.

## Data and exclusions

- SimpleStories: `SimpleStories/SimpleStories@e63b8adc3b1a1bdc7cac5b500d150b71346b0628`, train shards 0 and 1 of 7. Declared MIT; historical generation-term uncertainties from EXP-0045 remain documented.
- TinyStories: `roneneldan/TinyStories@f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, train shard 0 of 4. Declared CDLA-Sharing-1.0 in the [pinned source card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md).
- Per source, reservoir-sample 150,000 training documents, 512 dev documents and 1,000 reserved test documents from separate hash buckets. SimpleStories splits on completion groups, using the original bucket mapping so EXP-0045 training groups cannot enter the new holdouts. TinyStories splits on normalized content hashes; its semantic generation families are unavailable.
- Exclude the entire old SimpleStories dev completion groups, old dev texts and normalized matches to all official TinyStories validation texts. Exclude old training text matches from new holdouts, including across sources. Deduplicate normalized content across retained source candidates. Normalization is NFC, case folding and whitespace collapse; this is not semantic decontamination.
- The 21,990 official TinyStories validation stories never enter training or tokenizer learning. The new test documents are not tokenized into trainer manifests or scored by the queue.
- Tiny1m may have seen source-train holdouts during its own training. Its exposure and training budget are unknown. Thus new dev/test comparisons against tiny1m cannot be called independently unseen by that reference. The test is a reserved candidate holdout, not a certified reference-unseen benchmark.

Source rows, revision IDs, split groups, normalized hashes, full input hashes and token-shard hashes remain local in `data/processed/exp0047-story-mix` and `data/tokens/exp0047-story-mix`. Source entries remain candidate; running these authorized local experiments does not sign a rights/privacy review or publish new weights.

The [10 November preflight evidence](../docs/reviews/story-mixture-preflight-2026-11-10.json) records 150,000 train, 512 dev and 1,000 reserved test documents per source, with zero retained normalized duplicates, blocked text matches, split-group leaks or old-train matches in new holdouts under the stated checks. Excluded rows include 20,250 TinyStories normalized duplicates and 1,550 TinyStories rows matching the combined excluded old-dev/official-validation pool. This aggregated exclusion counter does not distinguish the two blocked pools. It is not a semantic or legal clearance.

## Evaluation and observability

Use the existing `bits_per_byte` evaluator on CPU float32, document-start BOS/EOS for each tokenizer, nonoverlapping chunks of 256 target tokens, and aggregate NLL divided by UTF-8 bytes. Save per-document NLL, byte count and hashes. New native full-precision checkpoints and the already published rounded exports are identified explicitly; the selected candidate must later be exported and rechecked before replacing a release.

The trainer records loss, fixed-window dev loss, gradient norm, throughput, learning rate, allocated GPU memory, GPU board energy when available, and per-layer observability. The runner freezes input/code hashes and archives executed training sources. Epoch-equivalent exposure, energy and actual runtime must be read from artifacts rather than inferred as improvements. No hosted Langfuse dataset or text upload is created.

One seed and dev-based selection are exploratory. Confirmation on further training seeds, uncertainty accounting for story families, held-out confirmation and human narrative evaluation remain later stages. Human readers are currently unavailable; no model or automatic heuristic supplies human ratings. Lower BPB alone does not demonstrate coherent narratives or superiority in every respect.

The selected 100M candidate, its 300M follow-up and both references also produce ten raw generations each: five already public development prefixes, two seeds, temperature 0.8, top-k 50 and at most 160 new tokens within the 256-token window. These outputs are saved without cherry-picking and are not used for selection. Different tokenizers mean equal token limits do not guarantee equal output lengths in characters. No narrative rating is inferred from these samples.

## Run and pause

```powershell
$env:PYTHONPATH = 'src'
python scripts/build_exp0047_story_mix.py
python scripts/audit_exp0047_story_mix.py
python scripts/run_exp0047_story_mix.py
```

The serial queue starts tiny30, then tiny00 and tiny60, evaluates both public references on the new dev texts, selects, then runs the selected 300M recipe. It never evaluates reserved test texts and never uploads or replaces Hub weights. Use `--no-extend` for pilots only.

Status: `runs/EXP-0047/status.json`; training logs: `runs/EXP-0047/*.log`; corpus-build log: `runs/EXP-0047/automation/build.log`. To pause before the next stage, create `runs/EXP-0047/STOP`. The active training stage finishes first; to stop it earlier, identify the recorded training PID and let the existing resumable checkpoints provide recovery. Restart the same queue after removing the STOP file. Resuming rejects changed frozen inputs.

## Research basis and next interventions

1. More unique training data before excessive repetition: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264). Its larger-scale findings do not prescribe an epoch limit for 935K.
2. Teacher distillation after the data baseline: [Baby Llama](https://arxiv.org/abs/2308.02019), evidence at 58M rather than 935K. A specialized teacher could be discarded at inference. Direct logit distillation requires verified token-ID compatibility, beyond merely matching vocabulary sizes. This guard and a suitable teacher remain to implement; not launched here.
3. Tokenizer allocation (512/1,024/2,048 vocabulary) and deeper/narrower models under the same cap: [MobileLLM](https://arxiv.org/abs/2402.14905) supports architecture exploration at substantially larger scales. Existing 912,672- and 920,960-parameter configs are untrained bundles, not isolated depth or tokenizer ablations. Existing GQA and tied embeddings are already enabled.

These are successive hypotheses, not an automatic large sweep or a promise to beat tiny1m. Public promotion requires evidence on both corpora, raw generation examples, export checks and clearly stated remaining limitations.
