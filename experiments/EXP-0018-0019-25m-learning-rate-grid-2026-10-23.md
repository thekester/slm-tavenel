# EXP-0018/0019: 25M recipe-by-learning-rate sensitivity grid

**Status (2026-10-23, complete):** both low-rate cells completed 1,500/1,500 steps; BPB evaluation and paired-document bootstrap analysis are complete. Final training validation loss was 2.24332 for EXP-0018 (FSC only) and 2.24480 for EXP-0019 (10% wiki). This is a one-seed exploratory sensitivity study; corpus rights/privacy reviews remain unresolved and all checkpoints remain local.

## Motivation

EXP-0016/0017 compared the FSC-only and 10% wiki recipes at one peak learning rate (3e-4). The 10% arm improved held-out wiki BPB and modestly regressed FSC BPB, but that fixed-rate result does not show whether the ranking persists after recipe-specific optimization. Wang et al. (2026) report that data-recipe rankings can flip with small hyperparameter changes, and that lower-LR proxy runs better tracked their tuned larger runs. This experiment adds a predeclared half-rate cell to each recipe.

## Preregistered 2×2 design

| Data recipe | Peak LR 3e-4 | Peak LR 1.5e-4 |
|---|---|---|
| FSC only | Existing EXP-0014 control | EXP-0018, new |
| 90% FSC / 10% wiki | Existing EXP-0017 treatment | EXP-0019, new |

- Both new cells initialize from the same grown-25M step-15,000 checkpoint (SHA-256 `ac7cb9c5c418bfadd33b0063f37c7a12bcb667896db64fe08b8bea05a95b5d8d`) and reuse seed `20261023`, the 25M architecture, tokenizer, AdamW, 1,500 steps, 32,768 positions/update, 49.152M positions, weight decay, warmup, decay fraction, precision, and FSC validation split used by the existing high-rate cells.
- The sole change in EXP-0018/19 is scaling the whole learning-rate schedule by 0.5: peak 1.5e-4 and minimum 1.5e-5. The paired low-rate cells use their matched data recipe; no test-set metric will select the checkpoint.
- Primary outcomes: per-document BPB on FSC EN/FR validation and wiki EN/FR holdout. Calculate the recipe effect at each learning rate and the recipe×LR interaction with paired document bootstrap intervals (20,000 draws, seed `20261023`). Intervals are conditional on the four checkpoints and do not estimate training-seed uncertainty.
- The existing multiple-choice benchmarks are not retuned or used as a selection signal in this grid: they have been repeatedly inspected. If the likelihood result supports a follow-up, use a fresh, untouched suite and a new seed for confirmatory downstream evaluation.
- Report the complete 2×2 results whether or not the low LR wins. Keep weights, data and evaluations local.

## Limits

This is a two-rate, one-seed sensitivity grid, not an exhaustive hyperparameter search or confirmation of the proxy-model paper. EXP-0014/17 were completed before this grid was preregistered; the follow-up decision was informed by their results. The 10%/20% wiki mixture arms are related nested samples. Source-order effects, seed variability, semantic contamination, and rights/privacy remain unresolved. A lower held-out BPB in this grid does not establish benchmark or general-language superiority.

## Results

BPB is bits per byte; lower is better. Recipe effect is the 10%-wiki checkpoint minus the FSC-only checkpoint at the same learning rate. Negative values on wiki holdouts and positive values on FSC validation mean the 10% mixture trades a small FSC regression for a wiki-domain gain.

| Held-out set | FSC only, LR 3e-4 | FSC only, LR 1.5e-4 | 10% wiki, LR 3e-4 | 10% wiki, LR 1.5e-4 | Wiki recipe effect at 3e-4 | Wiki recipe effect at 1.5e-4 |
|---|---:|---:|---:|---:|---:|---:|
| FSC EN validation | 1.10320 | 1.10393 | 1.10456 | 1.10500 | +0.00136 [0.00117, 0.00154] | +0.00107 [0.00093, 0.00121] |
| FSC FR validation | 1.02415 | 1.02486 | 1.02507 | 1.02570 | +0.00093 [0.00069, 0.00115] | +0.00084 [0.00067, 0.00099] |
| Wiki EN holdout | 1.63794 | 1.63892 | 1.49982 | 1.52043 | −0.13812 [−0.14814, −0.12917] | −0.11849 [−0.12726, −0.11069] |
| Wiki FR holdout | 1.42268 | 1.42340 | 1.31844 | 1.33165 | −0.10425 [−0.11504, −0.09492] | −0.09175 [−0.10104, −0.08369] |

Intervals in the last two columns are paired held-out-document bootstrap 95% CIs (20,000 draws, seed 20261023). The low-minus-high LR effect within the FSC-only recipe was +0.00072 EN / +0.00072 FR on FSC and +0.00097 EN / +0.00072 FR on wiki; all four paired document intervals excluded zero. Within the 10%-wiki recipe, halving LR raised BPB by +0.00044 EN / +0.00063 FR on FSC and +0.02061 EN / +0.01321 FR on wiki, also with all four intervals excluding zero. Thus the lower LR did not improve any of these fixed-budget held-out likelihood outcomes.

The recipe-by-LR interaction was small but its paired document interval excluded zero on each set: lowering LR reduced the FSC cost of adding wiki by 0.00029 EN [−0.00035, −0.00022] and 0.00009 FR [−0.00017, −0.00001], while weakening the wiki BPB gain by 0.01963 EN [+0.01837, +0.02104] and 0.01250 FR [+0.01102, +0.01419]. At this budget, LR 3e-4 is better for held-out BPB on every set for either recipe. The 10% mixture still expresses a domain trade-off rather than a general winner.

These confidence intervals quantify document sampling conditional on these four checkpoints. They do not measure training-seed variation. The grid has only two LR values, and the high-rate cells predate this follow-up; it is neither an exhaustive search nor a confirmatory comparison. Multiple-choice test scores were not used for selection.

Reproducibility artifacts: analysis JSON (local artifact: `runs/EXP-0018-0019-25m-learning-rate-grid-2026-10-23/analysis.json`), EXP-0018 evaluation (local artifact: `runs/EXP-0018-25m-fsc-lr-low-2026-10-23/evaluation-primary-text.json`), EXP-0019 evaluation (local artifact: `runs/EXP-0019-25m-wiki-10pct-lr-low-2026-10-23/evaluation-primary-text.json`). Checkpoint SHA-256 values: EXP-0018 `6de7813b6aa2148f7bc7e96b426ef06338eff4b7631fecdb949894122db98695`; EXP-0019 `c8b536fba7d32c1756b94eae2847674f5c4a7fa434c8f3bb6298fbd8cb2b5d7e`.

## Sources

- Wang et al. (2026), [Can Small Training Runs Reliably Guide Data Curation?](https://arxiv.org/abs/2512.24503)
- [EXP-0014/16/17 data-mixture results](EXP-0017-25m-wiki-10pct-2026-10-23.md)
- [25M performance research review](../docs/reviews/25M-PERFORMANCE-RESEARCH-2026-10-23.md)
