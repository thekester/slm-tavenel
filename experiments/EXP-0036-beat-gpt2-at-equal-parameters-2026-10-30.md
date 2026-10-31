# EXP-0036: beat GPT-2 small at equal parameters

**Status (2026-10-30 23:40): EXP-0036a (optimizer ablation) complete; the GPT-2-matched 124M Muon run
started at 23:13 (15,000 updates, 983M token positions, about 47 h at 5,760 token positions/s).**
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

## EXP-0036 main run (in progress)

`slm-124m-gpt2match` (124,038,144 parameters vs 124.4M for GPT-2), corpus v2, seed 20261031,
Muon, peak 3e-3, 300 warmup updates, 20% linear decay. Comparison with the original GPT-2 checkpoint
(`openai-community/gpt2@607a30d7`, scored with the same evaluator: HellaSwag 0.292, six-task mean 0.345).
