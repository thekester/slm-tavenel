# Axis: a compiled computer inside an SLM-Tavenel model (Qwenputer study, 2026-11-05)

Status: user decision 2026-11-05: route (B) first (LFM3-VL-2B + compiled computer, computer as a
separate compiled stream exchanging tokens with LFM3), then (A) a small own vision-language model.
Step 1 done: `src/slm/computer.py` (gated delta-rule scan, `DeltaRAM`), `tests/test_computer.py` (exact
under 2,000 random reads/writes in bf16; clear/halve gates; scan equals the textbook recurrence).
Step 2 (data path) done 2026-11-05: `LookupMLP` (exact table MLP, one ReLU AND-unit per input
combination) and `CompiledCalculator` (decimal add and schoolbook multiply; operands/result as digit bytes
in `DeltaRAM`, every digit operation through a compiled table, Qwenputer-style ADDR/VAL/STEP trace).
Tests: every table entry exact in fp32 and bf16; addition on a two-digit grid and 50 random 12-digit
pairs; multiplication on a two-digit grid, 30 random 8-digit pairs and 48213 x 9917; RAM state stays
exactly 0/1. **Still host-controlled:** the loops (which cell to read next) run in Python. Qwenputer also
compiles the controller (program counter and decode in layers); compiling ours (program counter in RAM,
next-address and opcode decode as further lookup tables, one layer pass per trace token) is step 2b.

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

## Target named by the user (2026-11-05): LFM3-VL-2B with a compiled computer inside

**LFM3-VL-2B** (`LiquidAI/LFM3-VL-2B@71d8e745`, 2026): 1.92B parameters = LFM3 language backbone
(1.73B; 24 layers: 13 gated short convolutions, 5 sliding-window attention (window 256), 6 global
attention; hidden 2048, 16 heads / 4 KV heads, 128k vocabulary, 128k context) + in-house 164M vision
encoder (14 layers: 8 gated depthwise 2-D convolutions, 6 bidirectional attention; 16x16 patches, native
resolution, 2x2 patch merge) + 25M projector. Base LM pre-trained on 22T tokens (datasets not named),
then SFT, vision RL and on-policy distillation. XML tool calls (`<tool_call><function=...>`).
Requires `transformers>=5.21` with remote code (installed here: 4.57.6). Checkpoint 3.84 GB (bf16).
**Licence: LFM Open License v1.0**, not OSI open source: free for research and for commercial use by
entities under USD 10M annual revenue (the threshold also applies to modified models); above it a paid
licence is required. Training data undisclosed.

Two readings of the goal:

- **(A) Our own model of that kind, from scratch.** Not reachable on this machine: the backbone alone
  had 22T tokens (our 1B gets ~2-4B), plus image/video data and RL. A scaled-down own analogue is
  possible later (e.g. a ~0.5B "SLM-Tavenel-VL": our 362M LFM-style hybrid if EXP-0041 supports it + a
  small vision encoder, trained on open image-text data), compared with SmolVLM-256M/500M (Apache-2.0).
- **(B) LFM3-VL-2B + compiled computer, a derivative of Liquid's weights (Qwenputer recipe).** Feasible:
  nothing is trained, inference of a 2B in bf16 fits in 8 GB. Specific issues:
  1. LFM3 has no delta-rule layer, so the RAM must live in the inserted computer layers (compiled
     gated delta-rule heads), unlike Qwenputer which reuses Qwen3.5's own Gated DeltaNet layers.
  2. Inserted layers must leave LFM3 unchanged when idle. Qwenputer widens the hidden size and keeps
     the computer in extra dimensions; LFM3's RMS norms would see those dimensions, so either norms are
     patched to ignore them or the computer runs as a separate compiled stream that only exchanges
     tokens with LFM3 (simpler, still weights-only, but further from "inside the forward pass").
  3. Trigger: LFM3-VL already emits XML tool calls, so the computer reads
     `<tool_call><function=calc>...` and LFM3 writes the response, exactly as in Qwenputer.
  4. Environment: a separate virtual environment with `transformers>=5.21` so the training pipeline
     (4.57.6) is not disturbed.
  5. The result inherits the LFM Open License (derivative), recorded as such; it is not an SLM-Tavenel
     model trained by us.

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
