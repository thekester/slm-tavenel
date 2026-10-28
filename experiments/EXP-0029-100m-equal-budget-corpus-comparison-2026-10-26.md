# EXP-0029: 100M equal-budget old-corpus versus HPLT-rich comparison

**Status (2026-10-28): complete.** Both arms trained 1,500/1,500 steps and were scored. The HPLT arm stopped at step 900 on 2026-10-26 (PC shutdown) and resumed from its step-500 checkpoint; its losses at steps 550–650 reproduce the first launch to within 2e-4. First-launch logs are kept in the run directory. Paired analysis: `runs/EXP-0029-evaluations/paired-comparison.json`.

## Results

| Measure | Old-corpus control | HPLT-rich | Difference (95% paired bootstrap) |
|---|---:|---:|---|
| HPLT candidate test BPB (18,582 documents) | 1.4780 | 1.3122 | −0.1659 BPB for HPLT-rich [−0.1726, −0.1586] |
| ARC-Easy acc (2,376) | 0.311 | 0.316 | +0.005 [−0.012, +0.023] |
| ARC-Challenge acc (1,172) | 0.178 | 0.179 | +0.001 [−0.018, +0.020] |
| HellaSwag acc (10,042) | 0.276 | 0.272 | −0.004 [−0.009, +0.001] |
| Belebele EN acc (900) | 0.280 | 0.280 | 0.000 [−0.023, +0.023] |
| Belebele FR acc (900) | 0.304 | 0.316 | +0.011 [−0.008, +0.031] |
| WinoGrande acc (1,267) | 0.508 | 0.524 | +0.016 [−0.009, +0.041] |

At an equal 49.152M-token continuation budget from the same 100M parent, the HPLT-rich mixture lowers
held-out HPLT test BPB by about 0.166 (11%), with a document-bootstrap interval well away from zero.
This is an in-domain result: the test split comes from the same candidate corpus as the treatment's
training data. All six raw-accuracy intervals include zero, and every score is near chance. Of the
twelve accuracy/normalized-accuracy contrasts, one interval excludes zero (Belebele FR `acc_norm`,
−0.024 [−0.046, −0.003]); with twelve unadjusted comparisons, this single contrast is not treated as a
finding. Conclusion: a large in-domain modeling gain, no measured multiple-choice change at 100M.
Training-seed variance is not estimated.

The HPLT-rich step-1,500 checkpoint is the parent of the EXP-0025 301M growth (float64 function check:
max logit difference 0.0; float32 2.48e-5).

## Original design (2026-10-26)

This experiment isolates the effect of the corpus used for an additional continuation, using a common 100M SLM-Tavenel parent checkpoint and its recorded tokenizer. The first attempted training start was stopped after discovering the previous HPLT token shards used a different tokenizer; no results from that attempt are used.

## Question

With the same 100M starting checkpoint and the same additional training budget, does continuing on the HPLT-rich candidate change held-out HPLT bits-per-byte or multiple-choice benchmark accuracy compared with continuing on the earlier FSC pilot corpus?

## Matched design

| Arm | Initialization | Continuation data | Steps | Token positions |
|---|---|---|---:|---:|
| Control | EXP-0024 100M, step 1,500 | `fsc-pilot-v1-scaling` | 1,500 | 49,152,000 |
| Treatment | EXP-0024 100M, step 1,500 | HPLT-rich growth candidate v2, re-encoded with the EXP-0024 tokenizer | 1,500 | 49,152,000 |

Both arms use the same parent, seed, tokenizer, sequence length, micro-batch, gradient accumulation, optimizer, learning-rate schedule, and step count; only `data_dir` and output paths differ. The previous EXP-0028 HPLT treatment is invalidated because it used a different tokenizer from its parent. Both EXP-0029 checkpoints will be scored with the same six pinned benchmark files and the same sealed HPLT candidate test split. The HPLT candidate is a mixture with prior candidate data, not HPLT-only.

This is a matched, single-seed exploratory comparison. The item/document bootstrap intervals quantify uncertainty over the evaluated examples only; they do not measure training-seed variance. Results cannot clear corpus rights/privacy status or establish performance for other datasets.

## Reproduction

- Control config: `configs/train/exp0029-100m-old-corpus-control.yaml`
- Run/evaluation queue: `scripts/run_exp0029_data_comparison.ps1`
- Paired analysis: `scripts/analyze_exp0029_data_comparison.py`
- HPLT re-encoding: `scripts/encode_growth_candidate_v2_fsc_tokenizer.py`
- HPLT treatment: `configs/train/exp0029-100m-hplt-treatment.yaml`

The corrected HPLT token manifest is `data/tokens/slm-tavenel-growth-candidate-v2-fsc-tokenizer/manifest.json`. It records tokenizer SHA-256 `558bab3421bc9ee280f2aa9d0c6a82a9bc482e2ecb72a75837e9c3ee7d2dbd39`, matching the EXP-0024 parent. It contains 361,146,182 train tokens, 18,210,914 dev tokens and 20,681,103 test tokens. Training hashes are checked against the initialization lineage before each arm starts.

The old FSC corpus has unresolved source review issues, and HPLT page-level rights/privacy are not fully reviewed. Both arms remain local exploratory artifacts.
