# EXP-0003: ablation A1 at 10M, head-wise gated attention

*Status:* concluded, single seed (provisional)
*Dates:* 2026-10-19
*Runs:* `runs/slm-10m-wiki` (baseline, EXP-0001) and `runs/slm-10m-wiki-gated` (treatment)

## Question and hypothesis

Ablation A1 (docs/ARCHITECTURE.md): does a head-wise sigmoid gate on the attention output
(Qiu et al., 2025, arXiv:2505.06708) remove attention sinks and massive activations without hurting
quality? Hypothesis: if a head can output nothing by closing its gate, it no longer needs to park its
attention on the first token, so the sink and the position-0 massive activations that feed it
(EXP-0002) should not form.

## Design

| | Baseline | Treatment |
|---|---|---|
| Model | `configs/model/slm-10m.yaml` | `configs/model/ablations/slm-10m-gated-headwise.yaml` |
| Only difference | `attn_gate: none` | `attn_gate: headwise` (+10,240 parameters, +0.1%) |
| Train config | `configs/train/slm-10m-wiki.yaml` | `configs/train/slm-10m-wiki-gated.yaml` (same except model and output directory) |
| Data, seed, batches | wiki-v0, seed 1234, identical batch sequence | same |

The decision rule was not written before EXP-0001 started; this run is a first comparison, and the
adoption criterion of the protocol (two seeds) is not met yet.

## Results

**Quality: equal or slightly better.**

| Metric | Baseline | Gated | Difference |
|---|---|---|---|
| Validation loss, step 15,000 | 2.7495 | 2.7243 | -0.025 |
| Bits per byte, EN (500 docs) | 1.3163 | 1.3048 | -0.011 |
| Bits per byte, FR (500 docs) | 1.1562 | 1.1446 | -0.012 |
| Loss on the 128 analysis windows | 2.892 | 2.870 | -0.022 |

The gated model's validation loss was within 0.01 of the baseline until step 9,500 and lower at every
evaluation from step 10,000 onward.

**Sinks: absent.**

| Metric (128 held-out windows) | Baseline | Gated |
|---|---|---|
| Mean attention on first token | 0.111 | 0.014 |
| Highest layer mean | 0.242 (layer 7) | 0.031 (layer 9) |
| Removing first-token attention everywhere, renormalized | +0.013 | +0.010 |

During training, the baseline's sink mass grew from about 3% to 24-32% in layers 4 and 6-8; the
gated model stayed at 1-4% in every layer throughout.

**Massive activations: no position-0 spike.** In the baseline, dimension 162 reached about 880 at
position 0 in layers 3-7. In the gated model the maximum in layers 0-7 is 17-66, with no difference
between position 0 and other positions. The last two layers still reach 157-284, at later positions,
not at position 0. Neutralizing the gated model's top dimensions at position 0 has no effect (+0.002,
sink mass unchanged), whereas in the baseline it cost +0.054 and removed 37% of the sink.

**Mechanism: gates close selectively.** Mean gate values are 0.1-0.5 and gates are input dependent:
in most heads the gate is below 0.1 on 10-60% of tokens, and in layer 3 head 1 on 81% of tokens. Heads
switch off through the gate instead of through a sink, as the hypothesis predicts.

**Heads.** Ablating the 5 heads with the most first-token attention costs +0.241 against +0.208 on
average for random heads (max +0.373): no special role, as in the baseline.

**Cost.** +0.1% parameters. Wall-clock 128.3 min against 121.5 min (+6%), but both runs were on a
laptop GPU subject to thermal throttling, and the gated run shared the machine with the CPU-side
analyses of EXP-0002 for part of its duration; the throughput difference is not attributable to the gate without a controlled benchmark.

## Decision

Provisionally adopt the head-wise gate for the 135M ablations, pending:

1. a second training seed for both arms at 10M (the protocol requires two seeds before adoption);
2. confirmation at 135M (A1 in the ablation plan), where sinks and massive activations are known to
   be stronger;
3. a controlled throughput benchmark of the gate.

## Interpretation and limits

The result supports the hypothesis at 10M: the gate gives heads another way to output nothing, sinks
and the position-0 massive activations do not form, and quality does not suffer. The observed quality
gain (-0.025 nats) is in the direction reported by Qiu et al. but is from one seed; its size relative to
seed variance is unknown. A gated model also changes what the interpretability tools see: "attention to
the first token" no longer indicates an inactive head, the gate value does, and `hook_gate` should be
read alongside `hook_pattern`.

Exported weights: the gated architecture has no standard Hugging Face equivalent, so `slm export`
refuses it (by design). Adopting the gate means shipping custom modelling code or upstreaming a variant.
