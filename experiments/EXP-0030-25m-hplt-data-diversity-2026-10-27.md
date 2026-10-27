# EXP-0030: 25M continuation on an HPLT-rich corpus

**Status:** complete (2026-10-27 15:43 Europe/Paris). Single-seed exploratory data comparison. The candidate HPLT corpus is not cleared for publication.

## Question

From the same trained 25M parent, does continuing on the HPLT-rich candidate-v2 stream improve held-out likelihood on development text from both the original FSC domain and the HPLT-rich domain, compared with an equal-token continuation on the original FSC scaling stream?

This targets the prior result directly: 25M improved held-out FSC BPB over 10M, but did not resolve an accuracy gain on six post-hoc multiple-choice tasks. Earlier Wikipedia mixtures shifted BPB toward Wikipedia and away from FSC, while MIR and a lower learning rate worsened held-out BPB in their matched pilots. This experiment tests corpus diversity with fixed optimization settings.

## Matched design

| Arm | Initialization | Continuation data | Steps | Token positions |
|---|---|---|---:|---:|
| Control | EXP-0009 grown 25M, step 15,000 | prior FSC scaling corpus | 1,500 | 49,152,000 |
| Treatment | same checkpoint | tokenizer-compatible HPLT-rich candidate v2 | 1,500 | 49,152,000 |

Both arms use the same 25,723,648-parameter model, tokenizer SHA-256 `558bab3421bc9ee280f2aa9d0c6a82a9bc482e2ecb72a75837e9c3ee7d2dbd39`, seed `20261027`, sequence length, batch shape, AdamW settings, learning-rate schedule and code. The intended change is the continuation corpus. Each config is resumable from its latest 500-step checkpoint after a machine interruption.

The primary comparison is paired bits per byte on fixed **development** documents: English and French FSC validation text, plus the HPLT candidate-v2 `dev` split by language. The HPLT `test` split is kept sealed for a later confirmation. Six pinned multiple-choice evaluations are secondary descriptive outcomes and will not select the treatment. They have been inspected in earlier experiments and cannot provide a fresh confirmatory benchmark result.

## Results

Lower BPB is better. Both arms completed 1,500 updates, equivalent to 49.152M token positions each. The FSC-only continuation uses the exact same validation documents as the saved parent checkpoint. The first table compares that continuation descriptively with the parent; only one continuation seed is available, and the older parent report stores aggregate rather than per-document scores, so no new paired interval is claimed for that comparison.

| FSC validation | EXP-0009 parent | FSC control | Change vs parent |
|---|---:|---:|---:|
| English BPB | 1.11707 | **1.10317** | -0.01390 |
| French BPB | 1.03674 | **1.02421** | -0.01252 |

The HPLT-rich treatment is compared with the equal-token FSC control using paired document bootstrap intervals (10,000 draws). Positive differences mean the HPLT-rich treatment has worse BPB; negative differences favor it.

| Development set | Control BPB | HPLT-rich BPB | Treatment minus control | 95% paired document interval |
|---|---:|---:|---:|---:|
| FSC English | 1.10317 | 1.15112 | +0.04795 | [+0.04628, +0.04963] |
| FSC French | 1.02421 | 1.06653 | +0.04232 | [+0.04076, +0.04390] |
| HPLT English | 1.59057 | 1.40809 | -0.18248 | [-0.19130, -0.17362] |
| HPLT French | 1.36608 | 1.23624 | -0.12984 | [-0.13729, -0.12223] |

Thus, this HPLT-dominated continuation shifts held-out likelihood toward HPLT and away from FSC. It does not improve both domains. The benchmark results are also small and inconclusive: every paired item-bootstrap interval includes zero.

| Benchmark | FSC control accuracy | HPLT-rich accuracy | Difference | 95% paired item interval |
|---|---:|---:|---:|---:|
| ARC-Easy | 0.316 | 0.321 | +0.005 | [-0.010, +0.020] |
| ARC-Challenge | 0.177 | 0.177 | +0.001 | [-0.017, +0.019] |
| HellaSwag | 0.276 | 0.272 | -0.004 | [-0.008, +0.001] |
| Belebele EN | 0.292 | 0.297 | +0.004 | [-0.016, +0.024] |
| Belebele FR | 0.309 | 0.308 | -0.001 | [-0.019, +0.017] |
| WinoGrande XL | 0.511 | 0.516 | +0.006 | [-0.020, +0.030] |

No downstream-task winner is established. The FSC control's BPB is modestly lower than the parent on the same documents, but its raw multiple-choice scores are not evidence of a broad accuracy gain. The HPLT-rich recipe is not selected as the general 25M continuation.

Machine-readable scores and hashes are in local artifacts `runs/EXP-0030-evaluations/paired-development-bpb.json`, `runs/EXP-0030-evaluations/fsc-control/`, and `runs/EXP-0030-evaluations/hplt-treatment/`. The six benchmark files have the same pinned SHA-256 values in both arm outputs.

## Interpretation limits

This is one seed per arm. Document bootstrap intervals describe sampling on the fixed development documents; they do not estimate seed-to-seed variation. The existing parent is already trained on FSC data, and the treatment candidate contains both retained v1 and HPLT text, so this measures a continuation recipe, not pretraining from scratch. A result that improves HPLT BPB may trade off against FSC BPB. HPLT metadata declare CC0 by inheritance, but page-level rights, provenance and privacy are not independently cleared. All checkpoints remain local research artifacts.

The design follows the small proxy-experiment principle from [DataDecide](https://arxiv.org/abs/2504.11393): use controlled small-scale runs and continuous likelihood metrics to compare data recipes. DataDecide's scales and corpora differ substantially, so it motivates the method but does not predict a gain here.

## Reproduction

- Control: `configs/train/exp0030-25m-fsc-control.yaml`
- Treatment: `configs/train/exp0030-25m-hplt-treatment.yaml`
- Queue: `scripts/run_exp0030_25m_data_diversity.ps1`
- Development likelihood evaluation: `scripts/evaluate_exp0030_25m_dev.py`
- Paired BPB and benchmark bootstrap: `scripts/analyze_exp0030_25m_data_diversity.py`
- Evaluation recovery runner: `scripts/evaluate_exp0030_25m_pair.ps1`
- Starting checkpoint: `runs/fsc-scaling/25m-gated-grown-s1234/checkpoints/step_00015000.pt`
