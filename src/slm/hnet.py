"""One-stage H-Net: a byte-level language model with learned dynamic chunking (EXP-0040).

Reimplementation of Hwang, Wang & Gu (2025), "Dynamic Chunking for End-to-End Hierarchical Sequence
Modeling", arXiv:2507.07955 (reference code: github.com/goombalab/hnet, MIT), with this repository's
decoder blocks. Bytes are embedded and processed by a small encoder; a routing module predicts chunk
boundaries; the main network (identical in shape to the BPE models' transformer) runs on the selected
chunk vectors; their outputs are smoothed, expanded back to byte positions and decoded to next-byte logits.

Formulas (paper, section 2):
  routing      p_t = (1 - cos(W_q x_t, W_k x_{t-1})) / 2, p_1 = 1, b_t = 1[p_t >= 0.5]
  chunking     keep the encoder outputs and probabilities where b_t = 1
  smoothing    z_bar_t = P_t z_t + (1 - P_t) z_bar_{t-1} over chunks
  dechunking   byte t reads chunk sum(b_1..b_t); scaled by STE(c_t), c_t = p_t if b_t else 1 - p_t,
               STE(c) = c + stopgrad(1 - c) (forward value 1)
  residual     decoder input = dechunked + Linear(encoder output), the projection initialised near 0
  ratio loss   L = N/(N-1) * ((N-1) F G + (1-F)(1-G)), F = mean b, G = mean p, weight alpha = 0.03
  widths       D_outer <= D_main; expand by appending a shared trainable vector, reduce by truncation;
               an RMSNorm closes each of the encoder, main network and decoder.

Departure from the paper, recorded as a limitation: the encoder and decoder use sliding-window
attention blocks instead of Mamba-2 (whose kernels are not available on this Windows setup); the paper's
ablation found attention outer layers slightly worse than Mamba-2, which disadvantages this H-Net.
The EMA is evaluated with a truncated window of ``ema_window`` chunks: every selected chunk has
P >= 0.5, so contributions older than k chunks are at most 2**-k (2**-32 by default).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig
from slm.model import Block, RMSNorm, rope_tables


@dataclass
class HNetConfig:
    arch: str = "hnet"
    vocab_size: int = 257
    max_seq_len: int = 8192
    d_outer: int = 384
    d_main: int = 512
    encoder_layers: int = 2
    decoder_layers: int = 2
    outer_heads: int = 6
    outer_kv_heads: int = 2
    outer_ffn: int = 1024
    outer_window: int = 256
    main_layers: int = 16
    main_heads: int = 8
    main_kv_heads: int = 4
    main_ffn: int = 1536
    rope_theta: float = 100000.0
    qk_norm: bool = True
    attn_gate: str = "headwise"
    target_ratio: float = 4.42
    ratio_loss_weight: float = 0.03
    ema_window: int = 32
    norm_eps: float = 1e-6
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.arch != "hnet":
            raise ValueError("HNetConfig.arch must be 'hnet'")
        if self.d_outer > self.d_main:
            raise ValueError("H-Net widths must not decrease inwards (d_outer <= d_main)")
        if self.target_ratio <= 1:
            raise ValueError("target_ratio must exceed 1")

    def outer(self, layers: int) -> ModelConfig:
        return ModelConfig(vocab_size=self.vocab_size, d_model=self.d_outer, n_layers=layers, n_heads=self.outer_heads,
                           n_kv_heads=self.outer_kv_heads, ffn_hidden=self.outer_ffn, max_seq_len=self.max_seq_len,
                           rope_theta=self.rope_theta, qk_norm=self.qk_norm, attn_gate=self.attn_gate,
                           window_pattern=[self.outer_window], norm_eps=self.norm_eps, init_std=self.init_std)

    def main(self) -> ModelConfig:
        return ModelConfig(vocab_size=self.vocab_size, d_model=self.d_main, n_layers=self.main_layers,
                           n_heads=self.main_heads, n_kv_heads=self.main_kv_heads, ffn_hidden=self.main_ffn,
                           max_seq_len=self.max_seq_len, rope_theta=self.rope_theta, qk_norm=self.qk_norm,
                           attn_gate=self.attn_gate, norm_eps=self.norm_eps, init_std=self.init_std)


def ratio_loss(b: torch.Tensor, p: torch.Tensor, target: float) -> torch.Tensor:
    f = b.float().mean()
    g = p.float().mean()
    n = target
    return n / (n - 1) * ((n - 1) * f * g + (1 - f) * (1 - g))


def ema_over_chunks(z: torch.Tensor, P: torch.Tensor, window: int) -> torch.Tensor:
    """z_bar_t = P_t z_t + (1 - P_t) z_bar_{t-1}, unrolled over the last ``window`` chunks.

    z: B x C x D, P: B x C (with P >= 0.5 on real chunks and P_0 = 1)."""
    C = z.size(1)
    # S_t = sum_{j<=t} log(1 - P_j); the weight of chunk t - lag in z_bar_t is P_{t-lag} exp(S_t - S_{t-lag}).
    S = torch.log1p(-P.float().clamp(max=1 - 1e-6)).cumsum(1)
    out = torch.zeros_like(z, dtype=torch.float32)
    for lag in range(min(window, C)):
        weight = P[:, : C - lag].float() * (S[:, lag:] - S[:, : C - lag]).exp()
        out[:, lag:] = out[:, lag:] + weight.unsqueeze(-1) * z[:, : C - lag].float()
    return out.to(z.dtype)


class HNet(nn.Module):
    def __init__(self, cfg: HNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        outer_enc, outer_dec, main = cfg.outer(cfg.encoder_layers), cfg.outer(cfg.decoder_layers), cfg.main()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_outer)
        self.encoder_blocks = nn.ModuleList(Block(outer_enc, i) for i in range(cfg.encoder_layers))
        self.encoder_norm = RMSNorm(cfg.d_outer, cfg.norm_eps)
        self.router_q = nn.Linear(cfg.d_outer, cfg.d_outer, bias=False)
        self.router_k = nn.Linear(cfg.d_outer, cfg.d_outer, bias=False)
        self.width_pad = nn.Parameter(torch.zeros(cfg.d_main - cfg.d_outer))
        self.main_blocks = nn.ModuleList(Block(main, i) for i in range(cfg.main_layers))
        self.main_norm = RMSNorm(cfg.d_main, cfg.norm_eps)
        self.residual_proj = nn.Linear(cfg.d_outer, cfg.d_outer, bias=False)
        self.decoder_blocks = nn.ModuleList(Block(outer_dec, i) for i in range(cfg.decoder_layers))
        self.decoder_norm = RMSNorm(cfg.d_outer, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_outer, cfg.vocab_size, bias=False)
        cos0, sin0 = rope_tables(outer_enc.head_dim, cfg.max_seq_len, cfg.rope_theta)
        cos1, sin1 = rope_tables(main.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos_outer", cos0, persistent=False)
        self.register_buffer("rope_sin_outer", sin0, persistent=False)
        self.register_buffer("rope_cos_main", cos1, persistent=False)
        self.register_buffer("rope_sin_main", sin1, persistent=False)
        self.activation_checkpointing = False
        self.keep_hidden = False
        self.last_chunk_stats: dict[str, float] = {}
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        std = self.cfg.init_std
        for blocks in (self.encoder_blocks, self.main_blocks, self.decoder_blocks):
            out_std = std / math.sqrt(2 * len(blocks))
            for name, module in blocks.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0.0, out_std if name.endswith(("o_proj", "down_proj")) else std)
                elif isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)
        for module in (self.router_q, self.router_k, self.lm_head):
            nn.init.normal_(module.weight, 0.0, std)
        nn.init.normal_(self.embed.weight, 0.0, std)
        nn.init.normal_(self.residual_proj.weight, 0.0, 1e-4)  # "initialised close to 0"
        for norm in (self.encoder_norm, self.main_norm, self.decoder_norm):
            nn.init.ones_(norm.weight)

    def _run(self, blocks: nn.ModuleList, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        recompute = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        for block in blocks:
            if recompute:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, cos, sin, 0, None, None, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return x

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, return_logits: bool = True,
                **_: object) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cfg = self.cfg
        B, T = input_ids.shape
        if T > cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {cfg.max_seq_len}")
        x = self._run(self.encoder_blocks, self.embed(input_ids), self.rope_cos_outer, self.rope_sin_outer)
        x = self.encoder_norm(x)

        # Routing: boundary where adjacent representations disagree.
        q, k = self.router_q(x), self.router_k(x)
        cos_sim = F.cosine_similarity(q[:, 1:].float(), k[:, :-1].float(), dim=-1)
        p = torch.cat([torch.ones(B, 1, device=x.device), (0.5 * (1 - cos_sim)).clamp(0, 1)], dim=1)
        b = p >= 0.5

        # Chunking: gather boundary positions into a right-padded chunk sequence.
        counts = b.sum(1)
        C = int(counts.max())
        order = torch.argsort((~b).to(torch.int8), dim=1, stable=True)[:, :C]  # boundary indices first, in order
        chunk_x = x.gather(1, order.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        chunk_p = p.gather(1, order)
        valid = torch.arange(C, device=x.device)[None, :] < counts[:, None]
        chunk_p = torch.where(valid, chunk_p, torch.ones_like(chunk_p))

        h = torch.cat([chunk_x, self.width_pad.to(chunk_x.dtype).expand(B, C, -1)], dim=-1)
        h = self.main_norm(self._run(self.main_blocks, h, self.rope_cos_main, self.rope_sin_main))
        h = h[..., : cfg.d_outer]

        # Dechunking: smooth over chunks, expand causally, scale by STE(confidence), add the residual.
        z_bar = ema_over_chunks(h, chunk_p, cfg.ema_window)
        idx = (b.long().cumsum(1) - 1).clamp(min=0)
        z = z_bar.gather(1, idx.unsqueeze(-1).expand(-1, -1, z_bar.size(-1)))
        c = torch.where(b, p, 1 - p)
        ste = c + (1 - c).detach()
        z = z * ste.unsqueeze(-1).to(z.dtype) + self.residual_proj(x)

        z = self.decoder_norm(self._run(self.decoder_blocks, z, self.rope_cos_outer, self.rope_sin_outer))
        if self.keep_hidden:
            self.last_hidden = z
        r_loss = ratio_loss(b, p, cfg.target_ratio)
        self.last_chunk_stats = {"bytes_per_chunk": float(T * B / counts.sum()), "ratio_loss": float(r_loss.detach())}
        logits = self.lm_head(z)
        if targets is None:
            return logits, None
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        self.last_loss_parts = {"ce": ce.detach(), "ratio": r_loss.detach()}
        loss = ce + cfg.ratio_loss_weight * r_loss if self.training else ce
        return (None if not return_logits else logits), loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel() + self.lm_head.weight.numel()
        return n

    def flops_per_token(self, seq_len: int) -> float:
        """Approximate training FLOPs per byte (6 x parameters touched per position)."""
        outer = sum(p.numel() for blocks in (self.encoder_blocks, self.decoder_blocks) for p in blocks.parameters())
        main = sum(p.numel() for p in self.main_blocks.parameters())
        return 6 * (outer + main / self.cfg.target_ratio + self.lm_head.weight.numel())

    def config_dict(self) -> dict:
        return asdict(self.cfg)
