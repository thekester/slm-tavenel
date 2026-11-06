# SLM-Tavenel model-card publication record (2026-11-06)

This directory preserves the five revised public SLM-Tavenel model cards
(`athenista/slm-tavenel-{1m,10m,25m,25m-hplt-10pct,124m}`) and their supporting measurements.
The revised cards and listed artifacts were uploaded to Hugging Face on 2026-11-06. The prior public
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
| Safety | Implicit | Intended / out-of-scope uses; "no safety evaluation was performed" stated |
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

## Open items (not done, need a decision or new work)

- Energy of past training runs is unknown; instrument future runs (e.g. NVML power sampling) to report it.
- Quantization was measured on a 200-document held-out sample; no quantized weight variant was published.
- No safety evaluation exists; a basic toxicity/PII-leakage probe would be new work.
- KV-cache parity and speed measurements are recorded in `kv-cache-verification.json`; the Hub model-code
  revisions recorded in those cards still do not implement that cache. The newer repository implementation
  is tracked separately in the source release and should be validated before an HF code update.
- Human source-rights and privacy reviews remain pending, as already disclosed.
