# Research axis (2026-10-31): tokenizer-free byte-level models (H-Net, BLT)

**Status: literature note and pilot proposal; nothing implemented or run.**

## Sources and what they actually claim

- **H-Net** — Hwang, Wang & Gu, "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
  arXiv:2507.07955 (CMU / Cartesia, July 2025). An autoregressive U-Net over raw bytes: a small encoder,
  a learned dynamic-chunking module (boundaries predicted from adjacent-representation similarity,
  ~4.5–5 bytes per chunk on English, close to BPE), a main network over chunks, and a decoder.
  Compute- and data-matched, a one-stage byte-level H-Net matches or exceeds a BPE Transformer
  (v2: at sizes above 1B); two stages scale better with data and match a token Transformer about twice
  its size; gains are larger where tokenizers are weak (code, Chinese, DNA) and character-level
  robustness improves.
- **Cartesia follow-up (blog "From Tokens to Bytes" I and II, 2026-10-28)**: a 2-stage ~3B H-Net reaches
  SmolLM3's six-task average with about 1T training tokens versus ~11T, reported as "10x less compute
  and data". This is the company's own, not peer-reviewed comparison, across different model sizes
  and training data; its controlled scaling experiments report smaller multipliers (about 1.3x
  in-domain for one stage, 2x on code, up to 3–5x out of domain). No weights released at the time.
- **BLT** — Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens",
  arXiv:2412.09871 (Meta / UW, ACL 2025). Byte patches with boundaries from a small model's next-byte
  entropy; FLOP-controlled study up to 8B parameters and 4T bytes: training-FLOP parity with Llama 3 and
  up to ~50% inference-FLOP savings, better scaling at fixed inference cost when patch and model size grow
  together.

Reading for this project: the controlled evidence supports "byte-level models can match BPE at equal
compute and are more robust", with larger gains in weak-tokenizer settings. The "10x" figure should be
treated as an uncontrolled vendor comparison until independently reproduced.

## Why it matters here

- French and mixed EN/FR text, where an English-centric BPE is less efficient, is the setting where
  the papers report larger gains.
- Our main held-out metric, bits per byte, is already tokenizer-independent, so a byte model and the
  BPE family compare directly.
- It removes the tokenizer from data provenance (one less external artifact to pin).

## Feasibility and a pilot (proposal, not preregistered)

H-Net's reference implementation uses Mamba-2 layers in the byte encoder/decoder, whose CUDA kernels are
poorly supported on Windows; the licences of the H-Net and BLT code must be checked before reuse. A
pilot can instead implement a one-stage H-Net in this codebase with attention (sliding-window) encoder
and decoder layers and the paper's routing, smoothing and ratio loss, at the 75M proxy scale:

- arm BPE: the EXP-0036a 75M Muon proxy (exists);
- arm bytes: one-stage dynamic-chunking model, matched training FLOPs and the same corpus v1 bytes;
- outcomes: held-out BPB (EN and FR separately), the six benchmarks, chunk statistics, throughput.

Byte sequences are ~4–5x longer, so activation memory and the 8 GB GPU are the main risk; a memory
pilot comes first. Placement in the GPU calendar is a user decision (currently booked until ~10/11).
