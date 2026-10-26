# EXP-0028: matched 50M/100M continuation on the HPLT growth corpus

**Status: both training arms complete; post-training evaluations pending.**
**Data:** local [growth corpus candidate v2](../docs/reviews/GROWTH-CORPUS-CANDIDATE-V2-2026-10-25.md); not legally cleared.
**Benchmark data:** evaluation/decontamination only; never used as training text.

## Question

After both arms receive the same FSC continuation in EXP-0024, does another equal 49.152M-token continuation on the new HPLT-rich corpus improve held-out loss and bits-per-byte, and does the 100M arm retain an advantage over the 50M control?

## Design

| Arm | Starting checkpoint | Parameters | Added updates | Added token positions |
|---|---|---:|---:|---:|
| Control | EXP-0024 50M, step 1,500 | 49.35M | 1,500 | 49,152,000 |
| Treatment | EXP-0024 100M, step 1,500 | 96.60M | 1,500 | 49,152,000 |

The starting runs were matched in EXP-0024: same seed, FSC data, token exposure and optimizer schedule; the 100M treatment came from a function-preserving expansion of the 50M checkpoint. EXP-0028 keeps the same optimizer, sequence length, token positions per update, seed and schedule in both arms. The token shards are built from corpus v2: 334.0M HPLT train tokens and 41.3M tokens from the prior candidate, so this is an HPLT-dominated *mixture*, not an HPLT-only experiment.

Both arms use seed 20261025, 1,024-token sequences, micro-batch 2, accumulation 16, AdamW, BF16, 1,500 steps, peak LR `3e-4`, minimum LR `3e-5`, 50 warmup steps, and a 20% WSD decay. The 100M treatment costs more FLOPs. One seed per arm makes this exploratory; it does not estimate seed variance or establish a general scaling law.

## Evaluation

Training validation uses the corpus v2 `dev` split, which holds out source domains. The corpus `test` split remains held out for final corpus loss/BPB. The six project benchmarks are never training data; after both runs finish, they are scored with the pinned evaluator and reported as downstream tests. Their source contamination screen already removed five HPLT candidate documents that had exact normalized 13-token overlap with benchmark prompts or choices.

Report train/validation loss curves, final held-out loss and BPB in English and French, domain/source-level results where sample size permits, benchmark accuracies with item-level paired intervals, throughput, GPU peak memory, and complete data/model/config hashes. Label every result exploratory because HPLT page-level rights and human privacy review remain open.

## Reproduction

- Tokenization: `scripts/encode_growth_candidate_v2.py`
- 50M control: `configs/train/exp0028-50m-hplt-growth.yaml`
- 100M treatment: `configs/train/exp0028-100m-hplt-growth.yaml`
- Corpus manifest: `data/tokens/slm-tavenel-growth-candidate-v2/manifest.json`

Estimated training time from EXP-0024 measured throughput: approximately **2 h 48 min** for the 50M arm and **4 h 39 min** for the 100M arm, around **7 h 30 min** total serial GPU time, before final BPB and benchmark evaluation. The estimate assumes the same token throughput; actual times may differ.

## Progress (2026-10-25)

The 50M control completed all 1,500 updates (49,152,000 token positions). Final train loss was **3.41143** and final domain-held-out `dev` loss was **3.41239**. Validation loss fell from 4.60019 at step 250 to 3.41239 at step 1,500. The final checkpoint is `runs/EXP-0028-50m-hplt-growth-s20261025/checkpoints/step_00001500.pt`.

The 100M treatment completed 1,500 updates (49,152,000 token positions) on 2026-10-25. Final train loss was **3.35951** and domain-held-out `dev` loss was **3.35769**; peak recorded GPU memory was **6.17 GiB**. The matched 50M control's `dev` loss was **3.41239**, a descriptive difference of 0.05470 nats/token in favor of the 100M checkpoint. With one seed per arm, this does not estimate seed variation or establish a general scaling effect.

External benchmark scores and final corpus `test` loss/BPB are still pending. No training process remains active. Benchmark datasets remain evaluation/decontamination-only and were not used as training text.
