#!/usr/bin/env python3
"""Geometry of hidden representations, layer by layer (CPU analysis, no training).

For a checkpoint and a token-shard directory, collect residual-stream vectors after every block on a
fixed sample of validation windows, then estimate per layer:
  * TwoNN intrinsic dimension (Facco et al., 2017, Sci. Rep. 7:12140): with r1, r2 the distances of a
    point to its two nearest neighbours and mu = r2 / r1, the slope of -log(1 - F(mu)) against log(mu)
    through the origin, fitted on the 90% smallest mu values as in the paper;
  * participation ratio (sum lambda)^2 / sum lambda^2 of the covariance spectrum: a linear dimension;
  * mean norm of the centred vectors.
Valeriani et al. (2023, "The geometry of hidden representations of large transformer models") report
that intrinsic dimension rises then falls across layers; this script measures the same profile for
checkpoints of this repository (optimizers, objectives, tokenizations, training time).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.data.shards import TokenShards  # noqa: E402
from slm.interp import capture  # noqa: E402
from slm.runlog import sha256_file  # noqa: E402
from slm.train import load_model  # noqa: E402


def two_nn(x: torch.Tensor, discard: float = 0.1) -> float:
    """Facco et al.'s estimator: mu follows a Pareto law, so -log(1 - F(mu)) = d log(mu); d is the slope
    of a least-squares line through the origin over the empirical CDF, the largest ``discard`` fraction
    of mu (sensitive to noise) being left out of the fit."""
    x = torch.unique(x, dim=0)  # repeated tokens give identical vectors (zero distances) at early layers
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    r = d.topk(2, largest=False).values
    mu = (r[:, 1] / r[:, 0].clamp_min(1e-12)).sort().values.double()
    n = len(mu)
    keep = int(n * (1 - discard))
    log_mu = torch.log(mu[:keep])
    y = -torch.log1p(-torch.arange(1, keep + 1, dtype=torch.float64) / n)
    return float((log_mu * y).sum() / log_mu.pow(2).sum())


def participation_ratio(x: torch.Tensor) -> float:
    eig = torch.linalg.eigvalsh(torch.cov(x.T)).clamp_min(0)
    return float(eig.sum() ** 2 / (eig.pow(2).sum() + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="token shard directory (val split is used)")
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--points", type=int, default=4000, help="token vectors per layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_model(args.checkpoint, "cpu").eval()
    if not hasattr(model, "blocks"):
        raise SystemExit("per-layer hooks are implemented for the SLM decoder only")
    val = TokenShards(args.data, "val")
    x, _ = val.sample(np.random.default_rng(args.seed), args.windows, args.seq_len)
    with torch.no_grad(), capture(model, "blocks.*.hook_resid_post") as acts:
        model(torch.from_numpy(x))
    rng = np.random.default_rng(args.seed + 1)
    # Skip the first 16 positions of every window (start-of-window transients) and sample tokens.
    flat_index = np.array([w * args.seq_len + t for w in range(args.windows) for t in range(16, args.seq_len)])
    pick = rng.choice(flat_index, size=min(args.points, len(flat_index)), replace=False)
    layers = []
    for layer in range(model.cfg.n_layers):
        h = acts[f"blocks.{layer}.hook_resid_post"].float().reshape(-1, model.cfg.d_model)[torch.from_numpy(pick)]
        centred = h - h.mean(0)
        layers.append({"layer": layer, "two_nn_intrinsic_dimension": two_nn(centred),
                       "participation_ratio": participation_ratio(centred),
                       "mean_centred_norm": float(centred.norm(dim=1).mean())})
        print(layers[-1], flush=True)
    result = {"kind": "slm.latent_geometry", "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": sha256_file(args.checkpoint), "data": str(args.data),
              "windows": args.windows, "seq_len": args.seq_len, "points": len(pick), "seed": args.seed,
              "layers": layers}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
