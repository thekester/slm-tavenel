"""Optimisers: AdamW, and Muon for hidden weight matrices.

Muon (Jordan et al., 2024, https://kellerjordan.github.io/posts/muon/) applies SGD momentum and then
replaces each matrix update by an approximation of its nearest orthogonal matrix, computed with a
few Newton-Schulz iterations. Liu et al. (2025, arXiv:2502.16982, "Muon is Scalable for LLM
Training") add decoupled weight decay and scale the update by 0.2 * sqrt(max(rows, cols)) so that
its RMS matches a typical AdamW update; the same learning rate and weight decay can then be used
for both optimisers. That variant is implemented here.

Muon is applied to the 2D weights inside transformer blocks only. Embeddings, the output head (tied
to the embeddings), norms and other vectors use AdamW, as in both references.
"""

from __future__ import annotations

from typing import Any

import torch

from slm.model import SLM


@torch.no_grad()
def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T for G = U S V^T with the quintic iteration of Jordan et al. (2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16 if g.is_cuda else torch.float32)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.1, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError("Muon only handles 2D parameters")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = newton_schulz(g, group["ns_steps"])
                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)


class CombinedOptimizer:
    """Several optimisers driven as one: shared ``param_groups`` view, step, zero_grad, state."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self) -> None:
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for o, s in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(s)


def muon_params(model: SLM) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into (2D matrices inside blocks -> Muon, everything else -> AdamW)."""
    muon, other, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (muon if name.startswith("blocks.") and p.dim() == 2 else other).append(p)
    return muon, other


def build_optimizer(model: SLM, name: str, lr: float, weight_decay: float, betas: tuple[float, float],
                    eps: float, momentum: float = 0.95, fused: bool = False):
    from slm.train import param_groups

    if name == "adamw":
        return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps, fused=fused)
    if name == "muon":
        muon, other = muon_params(model)
        decay = [p for p in other if p.dim() >= 2 and p is not model.embed.weight]
        no_decay = [p for p in other if p.dim() < 2 or p is model.embed.weight]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        adamw = torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=betas, eps=eps, fused=fused)
        return CombinedOptimizer([Muon(muon, lr=lr, weight_decay=weight_decay, momentum=momentum), adamw])
    raise ValueError(f"unknown optimizer {name!r}")
