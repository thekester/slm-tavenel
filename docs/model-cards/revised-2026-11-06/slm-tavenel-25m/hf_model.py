"""Transformers implementation of SLM-Tavenel for Hub-hosted custom code.

A small, auditable inference implementation. Generation uses a KV cache (Transformers
``DynamicCache``): keys and values of past positions are stored after RoPE and QK
normalization, so each new token attends to them without recomputing the prefix. With
``use_cache=False`` the computation is the original full-sequence one. Positions are
absolute (no left-padding support), as in training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

import transformers

from .hf_config import SlmTavenelConfig

# Transformers 5 describes tied weights as {target: source}; 4.x as a list of target names.
_TRANSFORMERS_5 = int(transformers.__version__.split(".")[0]) >= 5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float64)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: SlmTavenelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.window = config.window_pattern[layer % len(config.window_pattern)] if config.window_pattern else 0
        self.gate_mode = config.attn_gate
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.o_proj._slm_output = True
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        if config.attn_gate == "headwise":
            self.gate_proj = nn.Linear(config.d_model, config.n_heads, bias=False)
        elif config.attn_gate == "elementwise":
            self.gate_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        else:
            self.gate_proj = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: torch.Tensor | None = None, cache: Cache | None = None,
                offset: int = 0) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = apply_rope(q, cos[offset:offset + length], sin[offset:offset + length])
        k = apply_rope(k, cos[offset:offset + length], sin[offset:offset + length])
        if cache is not None:
            k, v = cache.update(k, v, self.layer)  # all positions so far, before head repetition
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        total = k.shape[2]
        mask = None
        if self.window or attention_mask is not None or total != length:
            q_pos = torch.arange(total - length, total, device=x.device)
            k_pos = torch.arange(total, device=x.device)
            causal = k_pos[None, :] <= q_pos[:, None]
            if self.window:
                causal &= (q_pos[:, None] - k_pos[None, :]) < self.window
            mask = causal[None, None, :, :]
            if attention_mask is not None:
                mask = mask & attention_mask[:, None, None, -total:].to(torch.bool)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        y = y.transpose(1, 2)
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x))
            gate = gate.view(batch, length, self.n_heads, 1 if self.gate_mode == "headwise" else self.head_dim)
            y = y * gate
        return self.o_proj(y.reshape(batch, length, self.n_heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, config: SlmTavenelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=False)
        self.down_proj._slm_output = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: SlmTavenelConfig, layer: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = Attention(config, layer)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: torch.Tensor | None = None, cache: Cache | None = None,
                offset: int = 0) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask, cache, offset)
        return x + self.mlp(self.mlp_norm(x))


class SlmTavenelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SlmTavenelConfig
    base_model_prefix = ""
    _tied_weights_keys = {"lm_head.weight": "embed.weight"} if _TRANSFORMERS_5 else ["lm_head.weight"]
    _no_split_modules = ["Block"]
    supports_gradient_checkpointing = False
    _supports_cache_class = True
    _supports_static_cache = False

    def __init__(self, config: SlmTavenelConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # RoPE tables are computed on first use rather than stored as buffers: Transformers 5 builds the
        # model on the meta device and would leave non-persistent buffers uninitialized.
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self.post_init()

    def rope(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope is None or self._rope[0].device != device:
            cos, sin = rope_tables(self.config.head_dim, self.config.max_seq_len, self.config.rope_theta)
            self._rope = (cos.to(device), sin.to(device))
        return self._rope

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def tie_weights(self, *args, **kwargs):  # Transformers 5 passes extra keyword arguments
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def _init_weights(self, module):
        std = self.config.init_std
        out_std = std / math.sqrt(2 * self.config.n_layers)
        if isinstance(module, nn.Linear):
            is_output = getattr(module, "_slm_output", False)
            nn.init.normal_(module.weight, mean=0.0, std=out_std if is_output else std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
                labels: torch.LongTensor | None = None, use_cache: bool = False,
                past_key_values: Cache | None = None, return_dict: bool | None = None,
                **kwargs) -> CausalLMOutputWithPast:
        cache = past_key_values if use_cache or past_key_values is not None else None
        if use_cache and cache is None:
            cache = DynamicCache()
        offset = cache.get_seq_length() if cache is not None else 0
        if offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"Input length exceeds max_seq_len={self.config.max_seq_len}")
        x = self.embed(input_ids)
        cos, sin = self.rope(x.device)
        for block in self.blocks:
            x = block(x, cos, sin, attention_mask, cache, offset)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None,
                                      use_cache=True, **kwargs):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, past_key_values.get_seq_length():]  # only the new tokens
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": use_cache}
