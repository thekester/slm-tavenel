# Architecture and compute plan

This document describes the baseline model implemented in [`src/slm/model.py`](../src/slm/model.py) and configured in [`configs/model/slm-1b.yaml`](../configs/model/slm-1b.yaml). Every choice below is a starting point. The ablation plan at the end lists the choices that are tested at 135M and 300M parameters before the 1B run is frozen.

## Baseline at the 1B target

| Setting | Value |
|---|---|
| Parameters | 1,044,476,416 (968,978,944 excluding embeddings) |
| Layers | 28 |
| Model width | 1,536 |
| Query heads / key-value heads | 12 / 4 (head dimension 128) |
| Feed-forward | SwiGLU, hidden size 6,144 |
| Vocabulary | 49,152, byte-level BPE, tied input and output embeddings |
| Context | 4,096 tokens, RoPE with θ = 100,000 |
| Normalisation | Pre-norm RMSNorm, plus RMSNorm on queries and keys per head |
| Weights in bf16 | 2.09 GB |

`python -m slm model info configs/model/slm-1b.yaml --tokens 1e12` reproduces these numbers.

For comparison, some openly documented models of similar size:

| Model | Layers | Width | Heads (Q/KV) | FFN | Vocabulary |
|---|---|---|---|---|---|
| SLM-1B baseline | 28 | 1,536 | 12 / 4 | 6,144 | 49,152 |
| Llama 3.2 1B | 16 | 2,048 | 32 / 8 | 8,192 | 128,256 |
| Qwen3-0.6B | 28 | 1,024 | 16 / 8 | 3,072 | 151,936 |
| SmolLM2-1.7B | 24 | 2,048 | 32 / 32 | 8,192 | 49,152 |

## Rationale

**Depth over width.** MobileLLM ([Liu et al., 2024](https://arxiv.org/abs/2402.14905)) reports that small models benefit from more, narrower layers at a fixed budget. The baseline uses 28 layers of width 1,536 rather than 16 of width 2,048. The trade-off is lower throughput per parameter, which the cost metrics will show.

**Grouped-query attention** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)). Four key-value heads for twelve query heads divide the KV cache by three, which matters for local deployment at long context.

**QK normalisation.** RMSNorm on queries and keys bounds attention logits and is used for training stability in OLMo 2 and Qwen3. It also makes the model exactly Qwen3-shaped, which lets exported weights run in standard tooling (see below).

**SwiGLU** ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)) and **RoPE** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) are the current defaults in open models of this size.

**Tied embeddings.** At 49,152 × 1,536 the embedding matrix is 7% of the parameters; untying it would add another 7% for little expected benefit at this scale.

**Vocabulary of 49,152.** A smaller vocabulary leaves more parameters for the transformer, but may cost French and code efficiency. Tokenizer fertility for French, English and code is measured with `slm.tokenizer.fertility` and compared with a larger vocabulary at 135M (ablation T1). Digits are split individually, and 32 special tokens are reserved for chat, reasoning and tool calls so that post training does not resize the embeddings.

**Warmup-stable-decay schedule** ([Hu et al., 2024](https://arxiv.org/abs/2404.06395); [Hägele et al., 2024](https://arxiv.org/abs/2405.18392)). A constant learning rate followed by a short decay lets one run branch into several decay phases, for example to test data mixtures for the final annealing, without retraining from scratch.

## Observability built into the model

Every intermediate value an explanation could refer to passes through a named `HookPoint`, an identity module that costs nothing when unused:

| Hook | Shape | Meaning |
|---|---|---|
| `hook_embed` | B × T × d | token embeddings |
| `blocks.{i}.hook_resid_pre`, `hook_resid_mid`, `hook_resid_post` | B × T × d | residual stream around block *i* |
| `blocks.{i}.hook_attn_out`, `hook_mlp_out` | B × T × d | what each sublayer writes to the residual stream |
| `blocks.{i}.attn.hook_q`, `hook_k`, `hook_v` | B × H × T × D | after RoPE |
| `blocks.{i}.attn.hook_pattern` | B × H × T × S | attention probabilities (requires `set_attention_capture(True)`) |
| `blocks.{i}.attn.hook_gate` | B × T × H × (1 or D) | output gate values, gated variants only; in a gated model a closed gate, not first-token attention, marks an inactive head (`interp.gate_statistics`, logged in training) |
| `blocks.{i}.attn.hook_z` | B × T × H × D | per-head output before the output projection |
| `blocks.{i}.mlp.hook_hidden` | B × T × ffn | SwiGLU hidden activations |

[`slm.observe`](../src/slm/observe.py) reads and replaces these values; [`slm.interp`](../src/slm/interp.py) builds the analyses on top: logit lens, head ablation, activation patching, attention sink mass and entropy, residual norms. During training the same functions run on a fixed probe batch and are logged to `observability.jsonl`.

## Open weights in standard formats

With global attention and no output gate, the architecture is exactly Qwen3 (with QK normalisation) or Llama (without). [`slm.export`](../src/slm/export.py) writes Hugging Face `safetensors`, configuration and tokenizer files, plus a `provenance.json` linking the weights to the run manifest and data hashes. The test suite checks that the exported model gives the same logits in `transformers` as in this code. Experimental variants without a standard equivalent are refused rather than approximated.

## Compute budget

Training FLOPs per token follow the PaLM approximation, 6N plus the attention term, and are computed by `SLM.flops_per_token`.

| Run | Tokens | FLOPs | H100 GPU-hours at 40% MFU |
|---|---|---|---|
| 135M ablation arm | 10B | 1.2 × 10¹⁹ | about 9 |
| 300M transfer arm | 30B | 9.1 × 10¹⁹ | about 64 |
| 1B pretraining | 1T | 8.4 × 10²¹ | about 5,900 |
| 1B pretraining | 2T | 1.7 × 10²² | about 11,800 |

The GPU-hour column assumes 989 dense bf16 TFLOPS per H100 and 40% model FLOPs utilisation. Measured MFU is logged when `peak_tflops` is set, and the reported cost will use measured values.

The data constraint matters as much as compute. The core corpus is restricted to sources with recorded permissions (see [DATA_POLICY.md](DATA_POLICY.md)). The Common Pile authors trained 7B Comma models on 1T and 2T tokens of openly licensed text, which suggests that a 1T-token budget is reachable under this constraint, with some repetition of high-quality subsets. The token budget will be fixed once the approved corpus size is known.

## Ablation plan

Each arm changes one setting relative to `slm-135m.yaml`, with the same data order, seed and budget (`configs/train/ablation-135m.yaml`). Winners are confirmed at 300M before transfer.

| Id | Question | Configuration | Primary measurements |
|---|---|---|---|
| A1 | Does a head-wise sigmoid gate after attention ([Qiu et al., 2025](https://arxiv.org/abs/2505.06708)) reduce attention-sink mass without hurting loss? | `ablations/gated-headwise-135m.yaml` | validation loss, sink mass per layer, activation outliers |
| A2 | Do local/global layer patterns keep quality while reducing long-context cost? | `ablations/local-global-135m.yaml` | loss at 2k, long-context probes, KV memory, latency |
| A3 | Is QK normalisation needed for stability at our learning rates? | `qk_norm: false` | loss spikes, maximum logits, final loss |
| A4 | Depth versus width at equal parameters | 30 × 576 versus 15 × 832 | loss, tokens per second |
| T1 | Vocabulary 49,152 versus 98,304 | tokenizer retrained | bits per byte (tokenizer independent), French and code fertility |
| D1 | Data mixture for the decay phase | WSD branches from one checkpoint | bits per byte per domain, downstream tasks |
| D2 | Does restricting attention to the current document inside packed windows help? (as in Llama 3) | `doc_mask: true` | whole-document bits per byte, benchmarks, throughput |
| O1 | Does Muon ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/); [Liu et al., 2025](https://arxiv.org/abs/2502.16982)) reach a given loss with fewer tokens than AdamW? | `optimizer: muon` | loss versus tokens, final loss, step time |

Status of the first runs at 10M parameters is tracked in [`experiments/`](../experiments/): A1 shows consistent loss improvements across two training seeds (EXP-0003 and [EXP-0004](../experiments/EXP-0004-gated-attention-seed2.md)), supporting its selection for the 135M ablations. Confirmation at 135M and a controlled throughput comparison remain pending. O1 is provisionally positive at one seed ([EXP-0005](../experiments/EXP-0005-muon-10m.md)); D2 remains a separate investigation (EXP-0006). The [gated attention plus Muon configuration](../configs/train/slm-10m-wiki-gated-muon.yaml) prepares a combined follow-up (EXP-0007).

Attention-sink results from graph transformers are treated as motivation only, as the README states: the A1 hypothesis is tested here, not assumed.
