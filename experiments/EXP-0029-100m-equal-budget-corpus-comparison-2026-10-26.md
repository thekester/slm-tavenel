# EXP-0029: 100M equal-budget old-corpus versus HPLT-rich comparison

**Status: old-corpus control training active; tokenizer-compatible HPLT treatment follows.** This experiment isolates the effect of the corpus used for an additional continuation, using a common 100M SLM-Tavenel parent checkpoint and its recorded tokenizer. The first attempted training start was stopped after discovering the previous HPLT token shards used a different tokenizer; no results from that attempt are used.

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
