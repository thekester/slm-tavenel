# SLM-Tavenel model-card publication record (2026-11-06)

This directory preserves the five revised public SLM-Tavenel model cards
(`athenista/slm-tavenel-{1m,10m,25m,25m-hplt-10pct,124m}`) and their supporting measurements.
The initial revised cards and inference artifacts were uploaded on 2026-11-06. The follow-up with updated
measurements, observability and scoped safety-screening results was published on 2026-11-08. The prior public
cards are archived read-only in `../hf-published-2026-11-05/`.

Hub revisions: 1M `49644a2aadfe5b291202cdfcd8052e6a9c45377a`; 10M
`4b03f92843e485b46708f60fef92f374708ed1a5`; 25M
`a6c192c68b2b817f3a49dfaed43a150a2388b9bd`; 25M HPLT
`6ef4f4dd72fc89674d1c8c0e025fa24de896be13`; 124M
`3be27b7c858dea4cc77032a777f17a14108c9414`.

## What changed in every card

| Area | Before | Now |
|---|---|---|
| Structure | Different order and headings per card; empty "Evaluation" heading on 1M; broken table on 1M (training details); `?` instead of `−` in the EXP-0032 follow-up on 1M/10M/25M | Same 13 sections in the same order, contents line, "At a glance" table |
| At a glance | Partial | Parameters, architecture, tokenizer, context, training precision, published weight format, quantization status, data, optimizer/seed, compute, license, described revision, weight SHA-256 |
| Benchmarks | 10M showed 4 of 6 tasks; formats differed | Shared six-task table for the family and size neighbours (Pythia, GPT-2, SmolLM2), items, metric, chance, item intervals, protocol, pinned dataset revisions |
| HF structured results | No `model-index` | `model-index` with the six tasks, dataset revisions and source link (labelled as the SLM-Tavenel evaluator) |
| Inference | Not measured | Measured CPU / GPU fp32 / GPU bf16: prefill latency, total latency for 64 tokens, tokens/s, peak RAM and VRAM, with protocol |
| Reproducibility | Revision pinned on 1M/10M only; HPLT had no requirements | Revision pinned in every example; `requirements.txt` staged for HPLT; hyperparameter table (optimizer, schedule, batch, steps, precision, hardware, software) |
| Compute / energy | Hardware on some cards | GPU-hours from the published logs; energy explicitly "not measured" (no power log exists) |
| Traceability and explainability | Spread over several sections | One section: manifests, hashes, dirty-worktree status, parity check; per-layer observability figure (attention entropy, sink mass, closed gates, residual RMS) from the published logs, with what it does and does not show |
| Safety | Implicit | Basic screening reported with limitations; not a safety certification |
| Family | Follow-up paragraph repeated | Family table in every card + collection draft (`COLLECTION.md`) |

## Files published per repository

| Repository | Files |
|---|---|
| all five | `README.md`, `inference-benchmark.json`, `<repo>-observability.png/.svg` (none for HPLT) |
| `slm-tavenel-25m-hplt-10pct` | also `requirements.txt` |

The Hub copies were checked after upload: all five cards parsed with six structured evaluations each,
MIT metadata, and the text-generation task. These revisions update cards and supporting files, not weights.

## How the new numbers were produced

- `scripts/card_training_facts.py` → `training-facts.json` (steps, tokens, GPU-hours, peak memory from the published logs).
- `scripts/benchmark_card_inference.py` → `inference-benchmark.json` (one machine, 2026-11-06).
- `scripts/card_observability_figures.py` → `figures/` (per-layer diagnostics from the published `observability.jsonl`).
- `scripts/build_revised_cards.py` fills the generated sections; `scripts/check_model_cards.py` is the pre-publication checklist.

## Follow-up publication (2026-11-08)

Published to all five Hugging Face repositories. The card checker reports OK for all five. Current revisions
and exact uploaded-file SHA-256 hashes are recorded in `hf-publication-2026-11-08.json`.

| Work | Evidence |
|---|---|
| KV cache in `hf_model.py` (Transformers `DynamicCache`) + Transformers 5 compatibility fixes (tied-weights mapping, `tie_weights` signature, RoPE tables computed on use instead of meta-device buffers) | `tests/test_hf_native_adapter.py` (17 tests, pass on Transformers 4.57.6 and 5.21.1); `kv-cache-verification.json`: on all five published checkpoints, logits identical to the previously published code (max abs diff 0.0) and greedy tokens identical with and without cache; CPU speed-up ×1.9 (1M) to ×5.5 (124M), none on GPU |
| bfloat16 weight copies (not published) | Locally measured variants; excluded from this Hub update, and not part of the published model package |
| CPU int8 dynamic quantization (recipe, no file) | `quantization-measurements.json`: BPB +0.004 (1M) to +0.069 (124M), not faster than float32 on this CPU — documented as not recommended |
| Basic safety screening | `safety-eval-basic.json` (aggregate metrics only; sampled model continuations are excluded): RealToxicityPrompts challenging subset (500, English) scored by `unitary/toxic-bert`, PII-shaped output probe (200 EN/FR), GPT-2 reference |
| GPU energy logging for future training | `src/slm/energy.py`, `tests/test_energy.py`; `gpu_energy_wh` / `gpu_power_w` in `metrics.jsonl`, cumulative across resumes; not applicable to the already trained checkpoints |

Files published per repository in this update: `README.md`, `config.json`, `inference-benchmark.json`,
`kv-cache-verification.json`, `quantization-measurements.json`, and aggregate-only `safety-eval-basic.json`;
observability PNG/SVG figures were also updated where available (none for HPLT). The pinned revisions already
contain the KV-cache code used for the measurements, so no model code or weight files were uploaded. Raw
continuation samples and bfloat16 variants were excluded. Exact file hashes and revisions are in
`hf-publication-2026-11-08.json`.

## Open items (not done, need a decision or new work)

- Energy of past training runs is unknown (no power log was recorded); future runs now log GPU energy.
- No 4-bit, GGUF or bitsandbytes variant; CPU int8 dynamic quantization costs quality and is not faster.
- The safety screening is basic: no French toxicity classifier, no bias, memorization or jailbreak tests.
- Human source-rights and privacy reviews remain pending, as already disclosed.
