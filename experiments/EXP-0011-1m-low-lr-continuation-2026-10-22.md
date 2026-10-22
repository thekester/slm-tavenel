# EXP-0011: exploratory 1M low-learning-rate continuation

**Status:** completed exploratory training diagnostic; not a confirmatory evaluation and not a new benchmark candidate.

## Motivation

The selected EXP-0010 gated checkpoint has 1,181,920 parameters and was trained for 15,990,784
token presentations (about 13.5 tokens per parameter). Across the seed-421101 gated run's
fixed validation checks, loss fell from 4.70252 at step 122 to 3.64577 at step 488. The final
checkpoint therefore did not show a validation-loss plateau under that recipe. This motivates a
small, low-learning-rate continuation pilot; it does not establish that more repeated data will
improve downstream ability.

## Frozen comparison

- Starting checkpoint: EXP-0010 gated seed 421101 at step 488, retained unchanged.
- Training data, tokenizer, architecture and splits: identical to EXP-0010 FSC CC0 candidate v4.
- Additional budget: 488 steps × 32,768 tokens = 15,990,784 token presentations, approximately
  one more pass over the 15,976,924-token training split.
- Optimizer: AdamW, reset for this continuation; learning rate 0.0005 to 0.00005, 24 warmup
  steps, warmup-stable-decay schedule, deterministic mode, seed 421111.
- Output: `runs/EXP-0011-1m-low-lr-continuation-2026-10-22/`.

This is a single continuation without a matched control and is not evidence for a causal method
effect. The EXP-0010 sealed test and multiple-choice results are not used for checkpoint selection.
The validation split is a development diagnostic and has already been inspected in EXP-0010.
Repeated exposure to the same small corpus may improve in-domain BPB while harming broader
generalization; the pilot cannot settle that trade-off. It also does not change the unresolved
source/privacy sign-offs for the FSC data or authorize weight distribution.

## Next decision

The continuation completed all 488 steps (15,990,784 additional token presentations). It reached a
training loss of 3.58382. Its own fixed-window validation losses were 3.74049, 3.76106 and 3.75907
at steps 122, 244 and 366. To remove the difference in validation-window seeds, both final
checkpoints were rescored on the exact same 20 batches (validation seed 421102): EXP-0010 was
3.64577 and EXP-0011 was 3.84200, a regression of 0.19623. Machine-readable results are in
`runs/EXP-0011-1m-low-lr-continuation-2026-10-22/same-windows-validation.json` (local, ignored by Git);
the reproducible evaluator is [`eval_exp0011_same_windows.py`](../scripts/eval_exp0011_same_windows.py).

**Decision:** do not promote the continuation. The result argues against naively repeating this
narrow 16M-token corpus at the tested learning rate. It is one seed and one recipe, so it does not
show that all extra training or data repetition will hurt. Keep EXP-0010 as the sole displayed
SLM-Tavenel 1M benchmark model. A more promising next experiment is a diverse, audited general-text
and French-text mixture, with the same tokenizer/model/token budget across arms, a fresh held-out
test, and development-only selection. Recent mixture-scaling work also motivates testing target
data mixed with generic text rather than assuming one-source repetition is beneficial
([Sedova et al., 2026](https://arxiv.org/abs/2605.12715)); its findings are not yet validated for
this 1.18M model. Harness² may separately help an agent use tools, but it changes the surrounding
system rather than the language-model weights or intrinsic benchmark results
([Xu et al., 2026](https://arxiv.org/abs/2610.14224)).
