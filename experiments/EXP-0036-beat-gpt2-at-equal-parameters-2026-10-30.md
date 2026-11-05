# EXP-0036: beat GPT-2 small at equal parameters

**Status (2026-11-04): complete. EXP-0036a: Muon beats AdamW (−5.3% BPB). Main 124M run: better than
GPT-2 on ARC-Easy, worse on HellaSwag; the preregistered criterion (HellaSwag and six-task mean) is not
met (see "main run: result").**
Background, design and success criterion: `experiments/PLAN-peer-family-2026-10-30.md`.

## EXP-0036a: Muon vs AdamW on the 75M proxy

Treatment `configs/train/exp0036a-75m-muon.yaml` (Muon for block matrices with the RMS-matched update
of Liu et al., 2025; AdamW for embeddings and norms; see `src/slm/optim.py`). Control: the EXP-0034a CE
arm (`configs/train/exp0034a-75m-ce.yaml`), identical model, corpus v1, seed 20261029, data order, WSD
schedule (peak 3e-3), and 1,500 updates of 65,536 token positions (98.3M). Only the optimizer differs.
Analysis: `runs/EXP-0036-evaluations/exp0036a-paired-comparison.json` (10,000 bootstrap draws).

| Measure | AdamW | Muon | Muon effect (95% paired bootstrap) |
|---|---:|---:|---|
| Validation CE, step 1,000 (same windows) | 4.176 | 3.923 | −0.253 nats |
| Validation CE, step 1,500 | 3.840 | 3.656 | −0.184 nats |
| Held-out test BPB, all (30,824 docs) | 1.3300 | 1.2592 | −0.0708 [−0.0711, −0.0706] |
| Held-out test BPB, EN / FR | 1.3094 / 1.4710 | 1.2400 / 1.3901 | −0.0694 / −0.0809 |
| ARC-Easy acc | 0.364 | 0.421 | +0.057 [+0.040, +0.074] |
| ARC-Challenge acc | 0.184 | 0.172 | −0.012 [−0.032, +0.008] |
| HellaSwag acc | 0.260 | 0.264 | +0.005 [+0.000, +0.009] |
| Belebele EN acc | 0.272 | 0.290 | +0.018 [−0.004, +0.040] |
| Belebele FR acc | 0.308 | 0.308 | 0.000 [−0.022, +0.021] |
| WinoGrande acc | 0.507 | 0.504 | −0.004 [−0.028, +0.021] |
| Six-task mean acc | 0.316 | 0.327 | +0.011 |
| Throughput (token positions/s) | ~12,100 | ~11,400 | about 6% slower |

Muon lowers held-out BPB by 5.3% at equal tokens (both languages), more than twice the
EXP-0034a distillation gain (−2.3%) and at about 6% extra time instead of 2.1x. ARC-Easy improves
clearly (+5.7 points, interval well away from zero); the other tasks move little. Single training
seed; the AdamW control was resumed once from its step-250 checkpoint (deterministic data order).
This reverses the small-scale picture of EXP-0015/EXP-0033 (25M, continuation, tiny gain): here the
model is trained from scratch, with a 49k vocabulary and educational data.

**Decision:** Muon is used for the GPT-2-matched 124M (`configs/train/exp0036-124m-gpt2match-muon.yaml`).
Larger family members need Muon with host offload (pending implementation) to adopt it.

## EXP-0036 main run: result (2026-11-04)

Run `runs/EXP-0036-124m-gpt2match-muon`, final checkpoint `step_00015000.pt` (sha256 `bbaaff8a…`),
983M token positions, no instability (grad_norm 0.10–0.15 throughout). Resumed several times from
checkpoints after PC shutdowns (deterministic data order; at most ~250 updates recomputed each time). Benchmarks with the same evaluator and items
as the GPT-2 reference (`runs/reference/peer-models-2026-10/gpt-2/mc.json`); paired comparison by
`scripts/compare_mc_paired.py` (items paired by position, since HellaSwag repeats ids; 10,000 paired
bootstrap draws; exact McNemar test): `runs/EXP-0036-evaluations/exp0036-vs-gpt2-paired.json`
(accuracy) and `…-paired-accnorm.json` (length-normalised accuracy).

| Task (items) | GPT-2 124M | SLM-Tavenel 124M | Difference [95% paired bootstrap] | McNemar p |
|---|---:|---:|---|---:|
| ARC-Easy (2,376) | 0.447 | 0.484 | +0.036 [+0.017, +0.054] | 0.0002 |
| ARC-Challenge (1,172) | 0.185 | 0.206 | +0.021 [0.000, +0.044] | 0.065 |
| HellaSwag (10,042) | 0.292 | 0.277 | −0.015 [−0.021, −0.010] | <0.001 |
| Belebele EN (900) | 0.310 | 0.316 | +0.006 [−0.021, +0.032] | 0.75 |
| Belebele FR (900) | 0.318 | 0.311 | −0.007 [−0.038, +0.024] | 0.73 |
| WinoGrande (1,267) | 0.517 | 0.519 | +0.002 [−0.030, +0.036] | 0.92 |
| **Six-task mean** | 0.345 | 0.352 | +0.007 [−0.003, +0.017] | – |

With length-normalised accuracy the picture is the same: HellaSwag 0.316 vs 0.289 (−0.028
[−0.035, −0.021]), six-task mean 0.338 vs 0.340 (+0.002 [−0.007, +0.012]).
Held-out test BPB on corpus v2 (50,780 documents): 1.067 overall (EN 1.085, FR 1.040); GPT-2 was not
scored on this split, and as fixed in advance BPB is not a criterion (in-domain for our model).

**Verdict against the preregistered criterion: not met.** The six-task mean is higher but its interval
includes zero, and HellaSwag is significantly *lower* (−1.5 points). The model is clearly better on
ARC-Easy (+3.6 points) and marginally on ARC-Challenge, i.e. on school-science knowledge, which is
what FineWeb-Edu filtering favours; it is worse on HellaSwag, whose everyday/web-narrative
continuations (ActivityNet, WikiHow) are close to WebText and under-represented in educational data.
This matches the FineWeb-Edu report (Penedo et al., 2024: edu filtering trades HellaSwag for ARC/MMLU)
and llm.c, where reaching GPT-2's HellaSwag at 124M took ~10B general FineWeb tokens; we used 0.98B
tokens, half of them French, so only ~0.5B English tokens versus GPT-2's undisclosed (likely several
billion) WebText tokens.

Limits: single seed; one checkpoint; the 124M has more non-embedding parameters than GPT-2 (95.7M vs
~85M) but a smaller vocabulary and a bilingual budget; the planned 24-layer sensitivity variant was not
run. **Next options** (not started, need a decision): (a) continue training on a general-web/English
mixture (e.g. add FineWeb or DCLM) for ~1B more tokens, rerunning the same comparison; (b) accept the
result and report it as "better on knowledge tasks, worse on HellaSwag" at equal parameters.

## EXP-0036b: general-web continuation (preregistered 2026-11-05, before any run)

Both outcomes are kept: the EXP-0036 result above stands as the "edu data, ~1B tokens" answer, and
EXP-0036b asks whether the HellaSwag gap is a data-distribution effect that general web text closes.

- **Data** (`data/processed/slm-tavenel-general-web-v1`, built by `scripts/build_edu_corpus_v1.py
  --en-source fineweb --en-files 2 --fr-files 1`): FineWeb `sample/10BT` shards 000–001
  (`HuggingFaceFW/fineweb@9bb295dd`, ODC-By 1.0, Common Crawl terms; no educational filter) plus one
  FineWeb2-HQ French shard as replay (the same shard as in v2, so French is not new data), with the same
  filters, PII redaction, exact dedup, 13-gram benchmark decontamination and domain split as v1/v2.
- **Training** (`configs/train/exp0036b-124m-general-web-continuation.yaml`): initialised from the EXP-0036
  step-15,000 checkpoint, fresh Muon state, re-warmed WSD schedule (300 warmup updates, peak 1.5e-3, half
  the original, 20% decay to 1.5e-4), 15,000 updates x 65,536 token positions (983M; total budget about
  2B token positions), seed 20261105. Queue: `scripts/run_exp0036b_queue.ps1`.
- **Comparison**: the same evaluator and items; `scripts/compare_mc_paired.py` against GPT-2 and against
  EXP-0036 (paired bootstrap, McNemar); full edu v2 test BPB to measure forgetting.
- **Criterion vs GPT-2 (unchanged)**: higher accuracy on HellaSwag and on the six-task mean. Because the
  model has now seen twice the tokens, a win is reported as "at equal parameters, ~2B tokens", not as a
  token-matched result (GPT-2's token count is undisclosed).
- **Secondary, vs EXP-0036**: HellaSwag change (expected up), ARC-Easy change (forgetting of edu knowledge
  if its 95% interval is below zero), edu v2 BPB change. All reported whatever the direction.
- Limits fixed in advance: single seed; continuation confounds "more tokens" with "general data" (no
  edu-only continuation arm of equal length is run, for GPU-time reasons).

## EXP-0036 main run (setup)

`slm-124m-gpt2match` (124,038,144 parameters vs 124.4M for GPT-2), corpus v2, seed 20261031,
Muon, peak 3e-3, 300 warmup updates, 20% linear decay. Comparison with the original GPT-2 checkpoint
(`openai-community/gpt2@607a30d7`, scored with the same evaluator: HellaSwag 0.292, six-task mean 0.345).
