# Story 935K: cross-source evaluation, PR correction and human-review status

## Actual task status

| Task | Status | Evidence |
|---|---|---|
| Score the frozen models on TinyStories | Completed, all 21,990 official validation stories | [Summary](../model-cards/slm-tavenel-story-935k/training_record/tinystories/summary.json) |
| Copy the corrective overlay to PR #1 | Committed and pushed, head `4a06bbc7576ad04ffbf1ec002798ddd1da618cfb`; PR remains open, not merged | [PR](https://github.com/thekester/slm-tavenel/pull/1), [test and file record](pr1-corrections-2026-11-10.json) |
| Human blind test | Form and analysis tools ready; no completed human form received at this report | [Aggregate status](../model-cards/slm-tavenel-story-935k/training_record/human-review/summary.json), [reader instructions](story-blind-pilot-2026-11-10/README.md) |

## Likelihood results

Same text within each row, model-specific document-start token, nonoverlapping chunks of 256 tokens, CPU float32, and frozen public weight snapshots. Lower BPB is better. Both models have 935,040 parameters, but their tokenizers, architectures, training data and training budgets differ. Matching parameter counts does not isolate an architecture or optimizer effect.

| Evaluation corpus | Stories | Bytes | SLM-Tavenel Story 935K | TinyStories tiny1m |
|---|---:|---:|---:|---:|
| SimpleStories local dev | 3,185 | 3,944,858 | **0.937012** | 1.670079 |
| TinyStories official validation | 21,990 | 19,218,350 | 1.543090 | **1.045244** |

The same-domain advantage reverses on TinyStories. On the official validation set, Story 935K minus tiny1m is **+0.497847 BPB**, paired document-bootstrap 95% interval **[+0.491981, +0.503727]** (10,000 draws, seed 421204). This is evidence about the two fixed checkpoints' likelihood on these texts. It does not include training-seed variation, undocumented reference exposure or dependencies among synthetic story families, and is not a human coherence result.

TinyStories source: `roneneldan/TinyStories@f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, official validation parquet SHA-256 `33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab`. The source declares CDLA-Sharing-1.0 and is registered for evaluation only. No TinyStories text was added to training or redistributed in this release. Every scored text has a per-item hash. All 21,990 validation texts are unique under the exact and case/whitespace-normalized checks, and none matches the actual 60,000-story training subset under those checks. This is not a semantic decontamination guarantee.

The previous SimpleStories tiny1m result, 1.670289 BPB, used `LlamaTokenizerFast`. The explicit legacy slow tokenizer used in the new evaluation differs on nine dev documents (44 tokens out of roughly 2.15 million). We recomputed the SimpleStories reference line with that same slow tokenizer: **1.670079**. The old score remains archived; the changed number is a loader correction, not a weight update. [Both measurements and classes](../model-cards/slm-tavenel-story-935k/training_record/tinystories/simple-dev-reference-reconciliation.json) are recorded.

## Reproduce

Install the repository's HF/data extras and SentencePiece. Download only the standard runtime files of the two pinned models, then run:

```powershell
$env:PYTHONPATH = 'src'
@'
from huggingface_hub import snapshot_download
snapshot_download('athenista/slm-tavenel-story-935k',
    revision='62e64c2b677ea9fbc90cdee0fbfe66b589b78e83',
    local_dir='data/models/fluency/story-935k-100m-v2',
    allow_patterns=['config.json','generation_config.json','model.safetensors',
                    'tokenizer.json','tokenizer_config.json','provenance.json'])
snapshot_download('shibatch/tiny1m',
    revision='fdacad8ff9dc27bd72c77d3684075a494be1ba11',
    local_dir='data/models/fluency/tiny1m',allow_patterns=['hf/**'])
'@ | python -
python scripts/evaluate_story_tinystories.py
```

The evaluation downloads the pinned validation parquet, never trains, and checks the weight hashes. The original SimpleStories corpus is needed only for the overlap audit and reference-loader reconciliation; reconstruct it with the [EXP-0045 commands](../../experiments/EXP-0045-story-935k-2026-11-10.md). `scripts/check_story_reference_tokenizer.py` reproduces the tokenizer reconciliation. Evidence includes compressed per-item scores and exact copies of both executed scripts, alongside their hashes. Original dirty-worktree metadata is retained rather than rewritten as a clean execution.

## Human part still requires people

The local form anonymizes and randomizes the 15 existing pairs, saves six 0-4 ratings plus A/B/tie preference per pair, and records human/blinding declarations. Only complete, affirmatively blinded, self-declared human forms enter the descriptive report. There are no machine-generated human ratings. Pseudonyms, notes and the new A/B key remain local and are not uploaded.

The source outputs and identities were already public, so reviewers must disclose prior exposure. This is a five-prefix development pilot, not the proposed final test on 500 new prefixes. A prepared form and passing form tests do not constitute completed human evaluation. Run `scripts/serve_story_review.py` on the PC, have readers finish the form, then run `scripts/analyze_story_human_review.py` after the intended reviewers finish.

## Checks

- Main tree: 20 targeted tests passed, 10 deselected; includes byte-weighted paired bootstrap, score reconstruction, completion-group leakage, form key isolation and rejection of absent/machine review evidence.
- PR branch: 15 targeted tests passed, 10 deselected; historical pilot archives unchanged.
- Each model's 21,990 per-document NLL records reconstruct its aggregate BPB, and both item sequences align by index, text hash and byte count.
- The live local form serves 15 anonymous pairs, does not serve its key, and its JavaScript passes syntax checking.
