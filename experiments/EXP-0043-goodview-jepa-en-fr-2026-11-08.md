# EXP-0043: good views for a bilingual SLM — JEPA on English/French translation pairs (preregistered 2026-11-08, before any run)

Axis: `experiments/AXIS-platonic-representations-2026-11-07.md`. Source of the hypothesis: Balestriero &
LeCun (2026, arXiv:2610.39302): spectral JEPAs trained on *good views* (independent observations of the
same content) learn aligned representations; next-token prediction behaves like a JEPA with *bad views*.

## Question

At equal exposure to the same English/French translation pairs, does learning them as **good views**
(a spectral-JEPA term pulling a sentence and its translation together) give a 75M bilingual model more
aligned English/French representations than learning them through **next-token prediction** (each
sentence alone), without hurting language modeling?

## Arms (75M proxy, 1,500 updates, 98.3M main-corpus token positions each)

All arms: `slm-75m-classic-proxy`, corpus v1, SmolLM2 tokenizer, seed 20261029, Muon, WSD (peak 3e-3,
100 warmup, 20% decay), identical main-corpus batches (same seed and data order as EXP-0036a).

| Arm | Extra per update (128 Tatoeba EN–FR pairs, weight 0.1) | Config / run |
|---|---|---|
| R, reference (existing) | none | EXP-0036a Muon arm |
| C, control (bad views) | next-token loss on the 256 sentences | `exp0043-75m-parallel-ntp.yaml` |
| G, treatment (good views) | `L = mean‖z_en − z_fr‖² + 1.0 · ‖Cov(z) − I‖²_F / 64`, z = linear projection (64 dims) of the mean final hidden state | `exp0043-75m-goodview-v1-collapsed.yaml` |

Pairs: `data/parallel/tatoeba-en-fr-v1` (`scripts/build_parallel_pairs.py`): 271,265 training pairs from
Tatoeba (CC-BY 2.0 FR, via `sentence-transformers/parallel-sentences-tatoeba@cec1343a`), exact dedup,
5 pairs removed for overlap with Belebele passages, 2,000-pair dev split; ~8 English / ~15 French tokens
per sentence. 1,500 updates × 128 pairs = 192,000 pairs seen (~0.7 epoch). Code: `src/slm/goodview.py`,
`tests/test_goodview.py`. The auxiliary weight (0.1), projection size (64) and γ (1.0) are not tuned.

## Outcomes and criteria

Evaluation set: the 488 parallel English/French Belebele passages (FLORES-200; never trained on),
`scripts/platonic_alignment.py`, mean-pooled residual stream per layer.

- **Primary — alignment.** Mutual 10-NN overlap between the English and French passage representations
  at each model's best layer. **G is "more aligned" than C** if its mean is higher by at least 0.03 with
  a 95% paired bootstrap interval over passages excluding zero.
- **Guard — language modeling.** Held-out corpus-v1 test BPB (30,824 documents): G must not be worse than C
  by more than 1% (relative). If G passes the primary criterion but fails the guard, the result is
  "alignment at a language-modeling cost".
- **Secondary (reported whatever the direction).** R@1 English→French by cosine and after a Procrustes
  rotation (per layer); the six benchmarks (paired item intervals), including Belebele EN/FR accuracy;
  C vs R and G vs R (effect of seeing parallel data at all); training throughput and GPU energy.

## Amendment 2026-11-08 (14:40, before any alignment or comparison result of the treatment): v1 collapsed

The v1 treatment (`runs/EXP-0043-75m-goodview-v1-collapsed`, loss variant `cov`) collapsed: from step ~250
to the end, `gv_anticollapse` stayed at 0.9999 and `gv_invariance` at ~0.001-0.009, i.e. the projected
sentence embeddings were (nearly) the same vector for every sentence. Two design faults, mine: the
invariance was summed over the 64 dimensions while the anti-collapse term was divided by 64 (a 64x
imbalance that makes collapse the cheaper optimum), and the ‖Cov − I‖² gradient vanishes as the embeddings
shrink, so nothing pushes back once they are small. The arm therefore does not test good views.

Fix (`goodview_variant: vicreg`, `configs/train/exp0043-75m-goodview-v2.yaml`): VICReg-style terms with
the weights of Bardes et al. (2022) divided by 25 — mean-squared invariance + γ · (mean hinge(1 − std)
over dimensions + 0.04 · sum of squared off-diagonal covariances / 64), γ = 1, std over the 256 sentences
of the batch with ε = 1e-4. The standard-deviation hinge keeps a scale-free gradient near collapse (test
`test_vicreg_variant_pushes_away_from_collapse_but_cov_variant_does_not`). Everything else is unchanged
(weight 0.1, batch 128 pairs, projection 64, seed, data). Run: `runs/EXP-0043-75m-goodview-v2`.

The primary criterion and guard are unchanged and apply to **v2 vs C**. v1 is kept and reported as a
descriptive negative control (a collapsed auxiliary objective) in the alignment table; its benchmark
and BPB evaluations, already computed, are reported but not used for any criterion. Monitoring rule for
v2, fixed now: if `gv_std_mean` stays below 0.1 after step 500, the run is declared collapsed and
reported as such (no further retuning within EXP-0043).

## Results (2026-11-09)

Arms completed without instability; v2 did not collapse (`gv_std_mean` 0.77–0.97 from step 250 to the end,
`gv_invariance` ~0.05). A first alignment computation was invalid (900 items: the 488 passages were
repeated because deduplication was on question ids); it is archived as
`belebele-alignment-WITH-DUPLICATES-invalid.json` and all numbers below come from the corrected script
(488 unique passages; chance: mutual 10-NN 0.021, R@1 0.002). Files in `runs/EXP-0043-evaluations/`.

| Measure | R, reference (no parallel data) | C, control (next-token on pairs) | G, good views v2 | G1, v1 collapsed (descriptive) |
|---|---:|---:|---:|---:|
| **Mutual 10-NN, best layer** | 0.3945 (L12) | 0.4045 (L13) | **0.4092** (L12) | 0.2822 (L0) |
| R@1 cosine at that layer | 0.953 | 0.920 | 0.965 | 0.779 |
| R@1 after rotation at that layer | 0.947 | 0.924 | 0.928 | 0.789 |
| Mutual 10-NN, last layer | 0.3695 | 0.4027 | 0.3289 | 0.2633 |
| Held-out test BPB, all (30,824 docs) | 1.2592 | 1.2653 | 1.2611 | not used |
| BPB English / French | 1.2400 / 1.3901 | 1.2472 / 1.3886 | 1.2420 / 1.3917 | not used |
| Six-task mean accuracy | 0.327 | 0.327 | 0.327 | not used |
| GPU energy (training) | not measured | 231 Wh | 153 Wh | 159 Wh |
| Throughput (token positions/s) | ~11,800 | ~4,750 at the end (GPU memory at 8.2 GB) | ~10,200 | ~8,400 |

Paired comparisons, mutual 10-NN at each model's best layer (95% bootstrap over passages):
G − C **+0.0047 [−0.0094, +0.0189]**; G − R +0.0148 [+0.0004, +0.0291]; C − R +0.0100 [−0.0035, +0.0232].
Held-out BPB: G is 0.0042 lower than C (−0.33%, [0.0040, 0.0043]; English lower by 0.0052, French higher
by 0.0030), G is 0.0019 higher than R, and C is 0.0061 higher than R (all intervals exclude zero). No
benchmark differs significantly between G and C, nor between either arm and R.

**Verdict against the preregistered primary criterion: not met.** G exceeds C by 0.0047, far below the
required 0.03, with an interval that includes zero. The guard is satisfied (G's BPB is better than C's).
Learning translation pairs as good views did not produce measurably more aligned English/French
representations than learning the same sentences through next-token prediction, in this setting.

Descriptive observations, not criteria: G's mean is only marginally above R (lower bound of the interval
just above zero, one of several comparisons, uncorrected). G retrieves translations slightly better by
cosine at its best layer (0.965 vs 0.920) but not after rotation (0.928 vs 0.924), and its alignment falls
at the output layer (0.329 vs 0.403 for C). A collapsed auxiliary objective (v1) is clearly harmful to
alignment (0.282), which shows the measure is sensitive. Retrieval is already near its ceiling (R@1 of
0.92–0.97) in all non-collapsed arms, so these 488 passages leave little room to detect gains on it.

Interpretation limits: the auxiliary term sees ~8–15-token everyday sentences at weight 0.1, while the
main objective sees 98M tokens of web text; the evaluation uses long passages; 192,000 pairs is small
next to the paper's budgets. This is a null result for this configuration, not a test of the paper's
theorem nor of its 42x claim, which concern alignment across modalities at much larger scale.
Reported energy and throughput of C are affected by GPU memory saturation.

## Limitations fixed in advance

One seed per arm; 75M scale and ~100M tokens; short, everyday Tatoeba sentences versus long evaluation
passages; untuned auxiliary hyper-parameters; the "best layer" is chosen per model by the same rule for
every arm; alignment is measured between two languages of the same model, not across modalities as in
the paper; Belebele passages also appear in the benchmark suite (analysis only, no training on them).
