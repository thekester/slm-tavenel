# EXP-0002: causal tests of attention sinks in the 10M baseline

*Status:* concluded
*Dates:* 2026-10-19
*Model:* EXP-0001 final checkpoint (`runs/slm-10m-wiki/checkpoints/step_00015000.pt`)
*Script:* `scripts/analyze_sinks.py` (results in `runs/slm-10m-wiki/analysis/sinks*.json`)

## Question

EXP-0001 logged attention sinks (up to 32% of attention on the first token in layers 4, 6-8 on the
2-sequence training probe; 19-24% on the 128 held-out windows used here) and
residual activations near 900 against a median of 2-5. Are these causally involved in the model's
predictions, and is one the cause of the other?

## Method

Held-out validation windows of 256 tokens, loss measured on positions 1-255. Replication: 32 windows
(seed 0) and 128 windows (seed 1). Values below are from the 128-window run; the 32-window run agrees
in sign and order of magnitude for every effect. Effects are loss changes in nats (base loss 2.892).

1. Localise massive activations: per-layer maximum by position and dimension.
2. Remove attention to the first token in each layer, either dropping its mass ("removed") or
   redistributing it to the other keys ("renormalized").
3. Zero-ablate the 5 heads with the highest sink mass; control: 10 draws of 5 random other heads.
4. Replace the massive dimensions with their typical (median) value, at position 0 only or everywhere,
   in every residual stream except the last; control: as many random dimensions at position 0.

## Results

**Massive activations are concentrated in three dimensions.** Dimensions 162, 79/238 and 3 dominate
the residual stream. Dimension 162 reaches about 880 at position 0 from layer 3 to layer 7, against a
median absolute activation of 2-5. It is also large (above 100) at 219 of 8,160 later positions, almost
all of them newline (114) and period (82) tokens; its median elsewhere is 31.

**The sink is weakly involved in the loss.** Removing attention to the first token in every layer
costs +0.027 (removed) or +0.013 (renormalized), under 1% of the loss. The largest single-layer effect
is layer 4, the strongest sink layer (sink mass 0.23): +0.030 when its sink mass is redistributed.

**Sink heads are not more important than other heads** (seed 1 only; at seed 2 they are, see
EXP-0004, so this result does not generalise). Ablating the 5 strongest sink heads costs
+0.174, against +0.186 on average (max +0.385) for random sets of 5 heads. No difference beyond the
spread of the control.

**The position-0 massive activations cause part of the sink.** Neutralizing dimensions {3, 162, 238} at
position 0 only reduces mean sink mass from 0.111 to 0.070 (-37%) and costs +0.054. The same operation
on random dimensions costs +0.0002. Neutralizing these dimensions at every position costs +0.396: they
also carry information the model relies on, mainly at delimiter tokens.

## Interpretation

In this 10M model, the first token acts as an attention sink whose existence depends partly on a few
residual dimensions that take extreme values at position 0. The sink itself is cheap to remove, which
fits the view of a sink as a default "attend nowhere" outlet rather than a carrier of information. The
pattern of massive activations on the first token and on delimiters matches what has been reported in
much larger language models (Sun et al., 2024, arXiv:2402.17762). That paper's conclusions are not
assumed here; this experiment only shows the same pattern at 10M.

Limits: one model, one seed of training, 32,768 evaluation tokens per run, zero and median ablations
only (other baselines such as mean or resampling ablations could give different magnitudes).

## Follow-up

EXP-0003 (ablation A1, gated attention at 10M) tests whether an output gate, which lets heads output
nothing without a sink, removes the sink and the massive activations at equal or better loss.
