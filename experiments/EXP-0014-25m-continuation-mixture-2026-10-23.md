# EXP-0014: 25M continuation data-mixture pilot

**Status (2026-10-23):** both 1,500-step training arms and all evaluations completed. This is a single-seed, local-only exploratory pilot. It does not replace or modify the public EXP-0009 checkpoint.

## Question and motivation

Does adding a modest amount of broad English/French news text during continued training improve MOT prediction without materially worsening the existing FSC validation score? The 25M checkpoint scores near chance on several four-choice benchmarks, but its results are not uniformly poor: it is stronger than the 10M model on ARC-Easy and WinoGrande, similar on HellaSwag and lower on the other reported tasks; its held-out FSC BPB is measurably lower than 10M. Small-model comparison remains confounded by corpora, tokenizer exposure and compute.

The prior EXP-0012 10M pilot reported a large MOT BPB improvement from an 80% FSC / 20% MOT mixture, a small FSC BPB regression, and no clear multiple-choice benchmark improvement: all paired item intervals included zero. EXP-0014 transfers this mixture question to the grown 25M checkpoint; it does not assume a benchmark gain. Data-mixture research motivates testing a mixture, while scaling-law work cautions that data/parameter allocation and data quality affect outcomes. These papers do not prescribe this exact ratio for SLM-Tavenel.

## Paired protocol

- Both arms initialize from the same local step-15,000 EXP-0009 checkpoint, SHA-256 `ac7cb9c5c418bfadd33b0063f37c7a12bcb667896db64fe08b8bea05a95b5d8d`.
- Same 25M architecture, tokenizer, seed, AdamW optimizer, 1,500 steps, 32,768 token positions per step, total 49.152M token positions per arm, learning-rate schedule, BF16 precision, and held-out FSC validation split.
- Control uses FSC v1 scaling shards only. Treatment uses the existing 80% FSC / 20% MOT v1.11 training mixture. The mixture's tokenizer SHA-256 matches the 25M tokenizer, and its FSC validation split has the same token count as the control.
- Learning rate is 3e-4 peak to 3e-5 floor, with 50 warmup steps and final 20% decay. This low-rate continuation is a pilot setting, not a tuned optimum.
- Primary diagnostics: FSC validation loss and held-out FSC English/French BPB. Secondary diagnostics: MOT 2022–2023 development and 2024+ temporal-test BPB. The latter is scored once after both arms finish. Six existing multiple-choice tasks are secondary exploratory outcomes; intervals will be conditional on task items and these two checkpoints, not training seeds.
- Training artifacts, source text and checkpoints remain local. FSC rights/privacy and MOT source-rights/privacy reviews are unresolved, so these arms are not cleared for public weight release.

## Interpretation

One pair cannot estimate training-seed variability or establish a benchmark winner. The primary useful signal is whether MOT text prediction improves while FSC performance remains acceptable. A later confirmatory claim would require more paired seeds and a new untouched test set. This pilot's old benchmark files have already been used in previous work and cannot serve as untouched confirmatory data.

The original 25M is not uniformly worse than the 10M: its published same-harness scores are higher on ARC-Easy and WinoGrande, close on HellaSwag, and lower on ARC-Challenge and both Belebele splits. Against Pythia-14M, its six scores are in a similar range, while SmolLM2-135M is materially stronger on most tasks at about 5.2 times the parameter count. Those are descriptive peer comparisons, not controlled training comparisons: pretraining sources, tokenizers, data and compute differ.

The grown 25M lineage received about 983M token positions across its 10M ancestry and 25M continuation, roughly 38 positions per final parameter. This is above the familiar approximately 20-token-per-parameter compute-optimal ratio reported by Hoffmann et al.; that scaling result was derived at much larger scales and is only a rough reference here. It suggests that simply extending this same narrow corpus is unlikely to be the main fix. Broader, better-filtered training data is a more plausible next hypothesis, but source rights/privacy need review and any new run still requires a matched control. The mixture pilot tests only news data and does not show a general benchmark improvement.

## Interim control result

The FSC-only continuation's held-out BPB point estimates are 1.1032 (FSC EN), 1.0241 (FSC FR), 1.5436 (MOT 2022–2023 development), and 1.5241 (MOT 2024+ temporal test). The original 25M report gives 1.1171/1.0367 on the FSC validation texts; the small lower values after this extra 49.152M-token continuation are descriptive and do not have a run-level uncertainty estimate.

Against the original 25M checkpoint, the control's benchmark changes are small. Paired item bootstrap intervals (20,000 resamples, seed 20261023) all include zero: ARC-Easy −0.08 pp [−1.14, +0.97], ARC-Challenge +0.09 pp [−1.11, +1.28], HellaSwag +0.10 pp [−0.16, +0.35], Belebele EN +0.56 pp [−0.67, +1.78], Belebele FR +0.11 pp [−1.22, +1.44], and WinoGrande +0.32 pp [−1.34, +1.97]. These intervals describe item sampling conditional on these checkpoints only. They provide no evidence that short continuation alone improved downstream benchmark performance.

## Results

Lower BPB is better. The figures below are aggregate point estimates over each held-out text set; per-document BPB outputs were not retained, so no document-bootstrap interval is available for EXP-0014.

| Held-out text set | FSC-only continuation | 20% MOT mixture | Mixture minus control |
|---|---:|---:|---:|
| FSC English validation | 1.1032 | 1.1092 | +0.0060 |
| FSC French validation | 1.0241 | 1.0293 | +0.0052 |
| MOT 2022–2023 development | 1.5436 | 1.2777 | −0.2658 |
| MOT 2024+ temporal test | 1.5241 | 1.2576 | −0.2666 |

The mixture improves prediction on the held-out news domain by about 0.266 BPB, including the later temporal test. It modestly worsens FSC BPB. This is evidence of a domain-specific trade-off in these checkpoints, not a broad language-model improvement or an estimate of seed-level uncertainty.

| Benchmark | FSC-only continuation | 20% MOT mixture | Difference, pp (95% paired item interval) |
|---|---:|---:|---:|
| ARC-Easy | 0.318 | 0.317 | −0.08 [−1.43, +1.26] |
| ARC-Challenge | 0.181 | 0.178 | −0.26 [−1.79, +1.28] |
| HellaSwag | 0.274 | 0.273 | −0.16 [−0.56, +0.23] |
| Belebele English | 0.289 | 0.280 | −0.89 [−2.56, +0.78] |
| Belebele French | 0.310 | 0.303 | −0.67 [−2.11, +0.89] |
| WinoGrande XL | 0.522 | 0.519 | −0.32 [−2.68, +2.05] |

All paired 95% item intervals include zero. These post-hoc scores do not show a general benchmark gain; the intervals condition on these checkpoints and items, not training seeds. No correction for the six comparisons was applied. Relative to the original published 25M checkpoint, the mixture's six point scores are also similar; all paired item intervals include zero. The original 25M, control, and mixture use identical pinned item IDs and labels under the repository evaluator.

The complete raw results, per-item predictions, input hashes and paired bootstrap output are in `runs/EXP-0014-25m-fsc-control-2026-10-23/evaluation.json` (local artifact: `runs/EXP-0014-25m-fsc-control-2026-10-23/evaluation.json`), `runs/EXP-0014-25m-fsc-mot-mixture-2026-10-23/evaluation.json` (local artifact: `runs/EXP-0014-25m-fsc-mot-mixture-2026-10-23/evaluation.json`), and `runs/EXP-0014-25m-mixture-analysis-2026-10-23.json` (local artifact: `runs/EXP-0014-25m-mixture-analysis-2026-10-23.json`). Analysis is reproducible with [`scripts/analyze_exp0014_25m_mixture.py`](../scripts/analyze_exp0014_25m_mixture.py). Bootstrap settings: 20,000 draws, seed `20261023`, paired resampling of item-level correctness differences.

## Research references

- Hoffmann et al. (2022), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- Xie et al. (2023), [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429).
- Penedo et al. (2024), [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557).
