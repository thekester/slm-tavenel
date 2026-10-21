"""Transformers implementation of SLM-Tavenel for Hub-hosted custom code.

This intentionally uses full-sequence attention without a KV cache. It favors a
small, auditable inference implementation over fast decoding; max_seq_len is 1024
for the current 1M checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .hf_config import SlmTavenelConfig


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
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = apply_rope(q, cos[:length], sin[:length])
        k = apply_rope(k, cos[:length], sin[:length])
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        mask = None
        if self.window or attention_mask is not None:
            causal = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()
            if self.window:
                positions = torch.arange(length, device=x.device)
                causal &= (positions[:, None] - positions[None, :]) < self.window
            mask = causal[None, None, :, :]
            if attention_mask is not None:
                mask = mask & attention_mask[:, None, None, :].to(torch.bool)
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
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        return x + self.mlp(self.mlp_norm(x))


class SlmTavenelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SlmTavenelConfig
    base_model_prefix = ""
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["Block"]
    supports_gradient_checkpointing = False

    def __init__(self, config: SlmTavenelConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        cos, sin = rope_tables(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def tie_weights(self):
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
                return_dict: bool | None = None, **kwargs) -> CausalLMOutput:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"Input length exceeds max_seq_len={self.config.max_seq_len}")
        x = self.embed(input_ids)
        cos = self.rope_cos.to(device=x.device)
        sin = self.rope_sin.to(device=x.device)
        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
