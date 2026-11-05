# Axis: a compiled computer inside an SLM-Tavenel model (Qwenputer study, 2026-11-05)

Status: study and proposal only; nothing implemented or run.

## What Qwenputer is

Qwenputer (Divyansh Agrawal, v0.1.0, October 2026; code MIT, weights Apache-2.0 as a modification of
Qwen/Qwen3.5-4B; `github.com/divagr18/qwenputer`) inserts 11 transformer layers in front of Qwen3.5-4B's
32 layers. None of their weights are learned: a compiler sets every one. Together they implement a
32-bit RISC-V CPU (RV32IM + Zba + a bignum coprocessor):

- **Memory = Gated DeltaNet state.** Qwen3.5 mixes full attention with Gated DeltaNet layers (Yang et
  al., 2025, ICLR, "Gated Delta Networks"), whose per-head state matrix is updated per token by the
  gated delta rule. With gate alpha = 1 (no decay), write strength beta = 1 and one-hot keys, a write
  overwrites exactly one row and a one-hot query reads it back: an exact addressable RAM. Low address
  bits pick the head, high bits the row; one byte per row stored as bits; registers, program counter and
  call stack get dedicated heads. alpha = 0 for one token clears a head; alpha = 1/2 halves it (counter).
- **ALU = MLPs as exact lookup tables**, exact in bf16 (the arithmetic circuit has ~0.7M nonzero weights).
- **Execution = trace tokens** (`ADDR`, `VAL`, `STEP`): each instruction is written out as a few tokens;
  Qwen's own state is frozen while the machine runs, then Qwen writes the `<tool_response>` itself.
  Triggered by Qwen's native `<tool_call>` syntax, or inline (`48213 × 9917 =`, CHIP-8 version).
- **Cost**: 18.8 GB checkpoint (config widened, mostly zeros), ~3,200–5,500 trace tokens/s on an L4;
  programs need thousands to millions of trace tokens.
- **Reported results** (author's, greedy, 40 items per set): DeepMind `mul_big` 55.0% (base) ->
  97.5% (Qwenputer) vs 100.0% for the base model with a host Python tool.

Lineage: compiling programs into transformer weights (RASP, Weiss et al., 2021; Tracr, Lindner et al.,
2023), looped transformers as programmable computers (Giannou et al., 2023), linear attention as fast
weight memory (Schlag et al., 2021). Tool use as the alternative: Toolformer (Schick et al., 2023).

## What it means for SLM-Tavenel

1. **No training is needed for the computer layers.** They can be added to a finished checkpoint, so
   the 1B (part 3) does not have to be retrained; the cost is inference memory and trace-token time.
2. **The RAM needs a delta-rule recurrent layer.** Our family is pure attention (the EXP-0041 hybrid uses
   short convolutions, which have no addressable state). Two ways: (a) the inserted computer block brings
   its own Gated DeltaNet-style layers (they are compiled, not trained, so they need not exist in the
   base model); (b) build a Qwen3.5-like attention + Gated DeltaNet hybrid for the 1B, which changes the
   1B design and needs a fast delta-rule kernel (flash-linear-attention uses Triton, fragile on Windows).
   (a) is the realistic route on this machine.
3. **Triggering.** Our base models are not instruction-tuned and know no `<tool_call>` format, so the
   inline arithmetic route (the computer layers detect `digits op digits =` in the token stream) is the
   one that fits. Tokenizer constraint: SmolLM2's tokenizer splits numbers into single digits, which makes
   reading operands from the token stream simpler than with multi-digit tokens.
4. **Honest baseline.** By the author's own numbers a host tool is at least as accurate and much faster;
   the scientific interest is self-containment (the weights alone compute exactly), not accuracy.

## Proposed plan (needs a decision; GPU only at step 3)

| Step | Work | Where | Time |
|---|---|---|---|
| 1 | Delta-rule RAM primitive in `src/slm`: a gated delta-rule layer (pure PyTorch recurrence) with alpha = beta = 1 and one-hot keys; tests for exact write/read/clear in bf16 | CPU | ~0.5 day |
| 2 | Compiled inline arithmetic block: digit reader (attention over the operand digits), exact add/multiply lookup MLPs with carry handled digit by digit through the RAM, answer writer; tests on exhaustive small cases | CPU | ~2–3 days |
| 3 | EXP-0042 (to preregister): block attached to the 124M then the 1B; DeepMind Mathematics (Saxton et al., 2019, Apache-2.0) `arithmetic__add_or_sub`, `mul`, `mul_big`-style items, >= 500 per set; arms: base, base + compiled block, base + host tool; plus the six benchmarks to check the block changes nothing else | GPU, after part 3 | ~0.5 day |

A full RISC-V CPU is out of scope at first: inline exact arithmetic tests the same two primitives (RAM
and lookup ALU) at a fraction of the engineering. Limits: the block is hand-compiled for one task family;
our models would still have to *produce* the right expression, which a 1B base model trained on ~2–4B
tokens does unreliably; Qwenputer's numbers are unreplicated author measurements on 40 items.
