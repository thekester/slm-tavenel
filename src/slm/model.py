"""Decoder-only transformer with named hook points for observation and intervention.

Architecture: pre-norm RMSNorm, rotary position embeddings, grouped-query attention,
optional per-head QK normalisation, optional sigmoid output gate on attention,
per-layer sliding windows, SwiGLU feed-forward, tied embeddings.

Every intermediate quantity that an explanation might refer to passes through a
``HookPoint``. A hook point is an identity module, so it costs nothing when no hook
is registered; ``slm.observe`` uses forward hooks on these modules to read or
replace activations. Hook names are module paths, for example
``blocks.3.attn.hook_z`` (per-head attention output, shape B x T x H x D).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig


class HookPoint(nn.Module):
    """Identity module marking a site that can be captured or intervened on."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
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
    t = torch.arange(max_len, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half convention (matches Hugging Face Llama/Qwen3 weights without permutation)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)


@dataclass
class KVCache:
    keys: list[torch.Tensor | None] = field(default_factory=list)
    values: list[torch.Tensor | None] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> "KVCache":
        return cls([None] * n_layers, [None] * n_layers)

    @property
    def seq_len(self) -> int:
        k = self.keys[0] if self.keys else None
        return 0 if k is None else k.shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer] is not None:
            k = torch.cat([self.keys[layer], k], dim=2)
            v = torch.cat([self.values[layer], v], dim=2)
        self.keys[layer], self.values[layer] = k, v
        return k, v


def attention_mask(q_len: int, k_len: int, offset: int, window: int, device: torch.device) -> torch.Tensor:
    """Boolean mask (True = may attend): causal, optionally restricted to a sliding window."""
    q_pos = torch.arange(offset, offset + q_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    allowed = k_pos <= q_pos
    if window > 0:
        allowed &= (q_pos - k_pos) < window
    return allowed


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window = cfg.window_for(layer)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.gate_mode = cfg.attn_gate
        if cfg.attn_gate == "headwise":
            self.gate_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)
        elif cfg.attn_gate == "elementwise":
            self.gate_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        else:
            self.gate_proj = None
        # When True, attention probabilities are computed explicitly and exposed at
        # hook_pattern (B x H x Tq x Tk). Slower; meant for analysis, not training.
        self.capture_pattern = False
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_gate = HookPoint()
        self.hook_z = HookPoint()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        offset: int = 0,
        cache: KVCache | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = apply_rope(q, cos[offset : offset + T], sin[offset : offset + T])
        k = apply_rope(k, cos[offset : offset + T], sin[offset : offset + T])
        q, k, v = self.hook_q(q), self.hook_k(k), self.hook_v(v)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        S = k.shape[2]
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        simple_causal = self.window == 0 and offset == 0 and S == T and doc_mask is None
        if self.capture_pattern:
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            mask = attention_mask(T, S, offset, self.window, x.device)
            if doc_mask is not None:
                mask = mask & doc_mask
            scores = scores.masked_fill(~mask, float("-inf"))
            probs = self.hook_pattern(scores.softmax(dim=-1))
            y = probs.to(v.dtype) @ v
        else:
            mask = None
            if not simple_causal:
                mask = attention_mask(T, S, offset, self.window, x.device)
                if doc_mask is not None:
                    mask = mask & doc_mask
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)

        y = y.transpose(1, 2)  # B x T x H x D
        if self.gate_proj is not None:
            g = torch.sigmoid(self.gate_proj(x))
            g = g.view(B, T, self.n_heads, 1 if self.gate_mode == "headwise" else self.head_dim)
            y = y * self.hook_gate(g)
        y = self.hook_z(y)
        return self.o_proj(y.reshape(B, T, self.n_heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.hook_hidden = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hook_hidden(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(h)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x, cos, sin, offset=0, cache=None, doc_mask=None):
        x = self.hook_resid_pre(x)
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.attn_norm(x), cos, sin, offset, cache, doc_mask)))
        x = x + self.hook_mlp_out(self.mlp(self.mlp_norm(x)))
        return self.hook_resid_post(x)


def document_ids(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Index of the document each token belongs to; the end-of-text token closes its document."""
    is_eos = (input_ids == eos_id).long()
    return torch.cumsum(is_eos, dim=1) - is_eos


def document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """B x 1 x T x T boolean mask: True where query and key belong to the same document."""
    return (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])


def _chunk_loss_sum(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, weight).float()
    return F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")


def chunked_cross_entropy(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Mean cross-entropy over rows of h, recomputing each chunk's logits in the backward pass."""
    from torch.utils.checkpoint import checkpoint

    h = h.reshape(-1, h.size(-1))
    targets = targets.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        total = total + checkpoint(_chunk_loss_sum, h[i : i + chunk], weight, targets[i : i + chunk], use_reentrant=False)
    return total / (targets != -100).sum().clamp_min(1)


def _chunk_distill_sums(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                        teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = F.linear(h, weight).float()
    ce = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
    student = F.log_softmax(logits / temperature, dim=-1)
    teacher = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    kl = (teacher.exp() * (teacher - student)).sum(-1)
    return torch.stack([ce, (kl * (targets != -100)).sum()])


def chunked_distillation_loss(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                              teacher_logits: torch.Tensor, temperature: float = 1.0,
                              chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean next-token cross-entropy and mean KL(teacher || student) at ``temperature``, in chunks.

    Forward KL on the full vocabulary, as in Hinton et al. (2015); the caller scales the KL term by
    temperature**2 so its gradient magnitude does not depend on the temperature."""
    from torch.utils.checkpoint import checkpoint

    h = h.reshape(-1, h.size(-1))
    targets = targets.reshape(-1)
    teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))
    total = h.new_zeros(2, dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        total = total + checkpoint(_chunk_distill_sums, h[i : i + chunk], weight, targets[i : i + chunk],
                                   teacher_logits[i : i + chunk], temperature, use_reentrant=False)
    n = (targets != -100).sum().clamp_min(1)
    return total[0] / n, total[1] / n


class SLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.hook_embed = HookPoint()
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # Training-time memory option, not part of the architecture or its config hash.
        self.activation_checkpointing = False
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        std = self.cfg.init_std
        # Output projections writing into the residual stream are scaled by depth.
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_out = name.endswith(("o_proj", "down_proj"))
                nn.init.normal_(module.weight, 0.0, out_std if is_out else std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        if not self.cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, 0.0, std)
        if self.rope_cos.device.type != "meta":
            cos, sin = rope_tables(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta)
            self.rope_cos.copy_(cos)
            self.rope_sin.copy_(sin)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        return_logits: bool = True,
        doc_ids: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        distill_weight: float = 0.0,
        distill_temperature: float = 1.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """With ``return_logits=False`` and targets, the loss is computed in chunks and the
        full B x T x vocab logits are never stored, which bounds training memory.

        ``doc_ids`` (B x T, see ``document_ids``) restricts attention to tokens of the same
        document when several documents share a training window.

        With ``teacher_logits`` (B x T x vocab, same tokenizer), the loss becomes
        ``(1 - w) * CE + w * T**2 * KL(teacher || student)``; both terms are kept in
        ``self.last_loss_parts`` for logging."""
        T = input_ids.shape[1]
        offset = cache.seq_len if cache is not None else 0
        if offset + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {offset + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        if doc_ids is not None and cache is not None:
            raise ValueError("document masking is for training and scoring, not cached generation")
        doc_mask = document_mask(doc_ids) if doc_ids is not None else None
        x = self.hook_embed(self.embed(input_ids))
        recompute = self.activation_checkpointing and self.training and cache is None and torch.is_grad_enabled()
        for block in self.blocks:
            if recompute:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, self.rope_cos, self.rope_sin, offset, None, doc_mask, use_reentrant=False)
            else:
                x = block(x, self.rope_cos, self.rope_sin, offset, cache, doc_mask)
        h = self.final_norm(x)
        if targets is not None and not return_logits and teacher_logits is not None:
            ce, kl = chunked_distillation_loss(h, self.lm_head.weight, targets, teacher_logits, distill_temperature)
            self.last_loss_parts = {"ce": ce.detach(), "kl": kl.detach()}
            return None, (1 - distill_weight) * ce + distill_weight * distill_temperature**2 * kl
        if targets is not None and not return_logits:
            return None, chunked_cross_entropy(h, self.lm_head.weight, targets)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    # ---- introspection -------------------------------------------------------------

    def hook_points(self) -> dict[str, HookPoint]:
        return {name: m for name, m in self.named_modules() if isinstance(m, HookPoint)}

    def set_attention_capture(self, enabled: bool) -> None:
        for block in self.blocks:
            block.attn.capture_pattern = enabled

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs per token (forward + backward), PaLM appendix B approximation."""
        c = self.cfg
        n = self.num_params(non_embedding=True) + c.vocab_size * c.d_model  # lm_head matmul
        attn = 0.0
        for layer in range(c.n_layers):
            span = min(seq_len, c.window_for(layer) or seq_len)
            attn += 12 * c.n_heads * c.head_dim * span
        return 6 * n + attn

    # ---- generation ----------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        cache = KVCache.empty(self.cfg.n_layers)
        logits, _ = self(input_ids, cache=cache)
        out = input_ids
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    kth = torch.topk(next_logits, top_k).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                probs = next_logits.softmax(dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            if out.shape[1] >= self.cfg.max_seq_len:
                break
            logits, _ = self(nxt, cache=cache)
        return out
