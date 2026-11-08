"""Good-view joint-embedding objective on parallel English/French sentences (EXP-0043).

Balestriero & LeCun (2026, arXiv:2610.39302) show that spectral JEPAs -- a prediction term plus a
second-order anti-collapse term -- learn the top eigenvectors of the graph defined by their positive
pairs, and that next-token prediction behaves like a JEPA with *bad views* (parts of one document). A
sentence and its translation are *good views* of the same content. This module adds, beside the usual
next-token loss on the main corpus, an auxiliary objective on a batch of translation pairs:

  z = P(mean of the final hidden states over the sentence's tokens)    (P: linear projection, k dims)
  L = mean ||z_en - z_fr||^2 + gamma * ||Cov(z) - I||_F^2 / k          (Cov over both languages)

i.e. a predictor-free spectral JEPA with a VICReg/Barlow-style covariance anti-collapse term. The
control arm sees the same sentences through next-token prediction instead (``parallel_ntp_loss``), so
both arms are exposed to the same parallel text and differ only in how it is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelPairs:
    """Tokenized translation pairs: ``pairs.npz`` with ``en``, ``fr`` (N, L) int32 right-padded arrays and
    ``en_len``, ``fr_len`` (N,) lengths, plus ``manifest.json``."""

    def __init__(self, directory: str | Path, split: str = "train"):
        directory = Path(directory)
        data = np.load(directory / f"{split}.npz")
        self.en, self.fr = data["en"], data["fr"]
        self.en_len, self.fr_len = data["en_len"], data["fr_len"]
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.pad_id = int(self.manifest["pad_id"])

    def __len__(self) -> int:
        return len(self.en)

    def sample(self, rng: np.random.Generator, batch: int) -> tuple[np.ndarray, ...]:
        idx = rng.integers(0, len(self), size=batch)
        width_en, width_fr = int(self.en_len[idx].max()), int(self.fr_len[idx].max())
        return self.en[idx, :width_en], self.en_len[idx], self.fr[idx, :width_fr], self.fr_len[idx]


def masked_mean(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = (torch.arange(h.size(1), device=h.device)[None, :] < lengths[:, None]).to(h.dtype)
    return (h * mask[..., None]).sum(1) / lengths[:, None].to(h.dtype)


def goodview_loss(z_en: torch.Tensor, z_fr: torch.Tensor, gamma: float,
                  variant: str = "vicreg") -> tuple[torch.Tensor, dict[str, float]]:
    """``variant="cov"`` (EXP-0043 v1): sum-of-squares invariance + gamma * ||Cov - I||_F^2 / k. Its
    anti-collapse gradient vanishes at z = 0, so the all-zero embedding is a stationary point and the
    first run collapsed onto it. ``variant="vicreg"`` (amendment, Bardes et al., 2022 weights divided by 25):
    mean-squared invariance + gamma * (mean hinge(1 - std) + 0.04 * sum of squared off-diagonal
    covariances / k); the standard-deviation hinge keeps a non-vanishing gradient near collapse."""
    z_en, z_fr = z_en.float(), z_fr.float()
    z = torch.cat([z_en, z_fr])
    z = z - z.mean(0)
    cov = z.T @ z / (z.size(0) - 1)
    k = cov.size(0)
    if variant == "cov":
        invariance = (z_en - z_fr).pow(2).sum(-1).mean()
        anti_collapse = (cov - torch.eye(k, device=cov.device)).pow(2).sum() / k
        stats = {}
    elif variant == "vicreg":
        invariance = (z_en - z_fr).pow(2).mean()
        std = torch.sqrt(z.var(0) + 1e-4)
        variance = F.relu(1.0 - std).mean()
        off_diagonal = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum() / k
        anti_collapse = variance + 0.04 * off_diagonal
        stats = {"gv_std_mean": std.mean().item()}
    else:
        raise ValueError(f"unknown good-view loss variant {variant!r}")
    loss = invariance + gamma * anti_collapse
    return loss, {"gv_invariance": invariance.item(), "gv_anticollapse": anti_collapse.item(), **stats}


def parallel_ntp_loss(logits: torch.Tensor, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy on right-padded sentences (padding positions excluded)."""
    targets = ids[:, 1:].clone()
    valid = torch.arange(targets.size(1), device=ids.device)[None, :] < (lengths[:, None] - 1)
    targets[~valid] = -100
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)


class GoodViewHead(nn.Module):
    """Linear projection to the k-dimensional joint-embedding space (registered as ``jepa_predictor`` so it
    is excluded from parameter counts and dropped when checkpoints are loaded for evaluation)."""

    def __init__(self, d_model: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, dim, bias=False)
        nn.init.normal_(self.proj.weight, 0.0, d_model ** -0.5)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)
