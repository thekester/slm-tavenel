#!/usr/bin/env python3
"""Evaluate an Othello sequence model as a world model (EXP-0038).

1. Legal-move rate: the model's most likely next move (end-of-game excluded) is legal, after every
   prefix of held-out games (Li et al., 2023 report this as the first sign of a world model).
2. Linear probes of the board: from the residual stream after each layer, a linear classifier predicts
   every square as empty / mover's / opponent's (mover-relative labels, Nanda et al., 2023). Probes are
   fit on the first half of the validation games and tested on the second half (in distribution) and
   on the greedy-policy test games (far transfer). Chance for a square is about 0.33; a majority-class
   baseline is reported for reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.interp import capture  # noqa: E402
from slm.runlog import sha256_file  # noqa: E402
from slm.train import load_model  # noqa: E402
from slm.worlds import othello  # noqa: E402


@torch.no_grad()
def activations(model, games: list[np.ndarray], device: torch.device, batch: int = 256):
    """Per-layer residual stream at every move position, legal-move hits, and board labels."""
    layers = model.cfg.n_layers
    feats = [[] for _ in range(layers + 1)]
    labels, legal_hits, legal_total = [], 0, 0
    for i in range(0, len(games), batch):
        chunk = [list(map(int, g)) for g in games[i : i + batch]]
        length = max(len(g) for g in chunk)
        ids = torch.zeros((len(chunk), length), dtype=torch.long)
        for r, g in enumerate(chunk):
            ids[r, : len(g)] = torch.tensor(g) + 1
        with capture(model, "blocks.*.hook_resid_post") as acts:
            logits, _ = model(ids.to(device))
        embed = model.embed(ids.to(device))
        for r, g in enumerate(chunk):
            n = len(g)
            feats[0].append(embed[r, :n].float().cpu())
            for layer in range(layers):
                feats[layer + 1].append(acts[f"blocks.{layer}.hook_resid_post"][r, :n].float().cpu())
            labels.append(torch.from_numpy(othello.states(g).astype(np.int64)))
            scores = logits[r, :n].float().cpu()
            scores[:, othello.EOS] = -float("inf")
            pred = scores.argmax(-1) - 1
            for t, legal in enumerate(othello.legal_after(g)):
                if legal:
                    legal_total += 1
                    legal_hits += int(int(pred[t]) in legal)
    return [torch.cat(f) for f in feats], torch.cat(labels), legal_hits / max(1, legal_total)


def fit_probe(x: torch.Tensor, y: torch.Tensor, epochs: int = 8, lr: float = 1e-2) -> torch.nn.Linear:
    torch.manual_seed(0)
    probe = torch.nn.Linear(x.size(1), 64 * 3)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    mean, std = x.mean(0), x.std(0) + 1e-6
    probe.register_buffer("mean", mean)
    probe.register_buffer("std", std)
    xs = (x - mean) / std
    for _ in range(epochs):
        for idx in torch.randperm(len(xs)).split(4096):
            loss = F.cross_entropy(probe(xs[idx]).view(-1, 64, 3).reshape(-1, 3), y[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe


@torch.no_grad()
def probe_accuracy(probe, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = probe((x - probe.mean) / probe.std).view(-1, 64, 3).argmax(-1)
    return float((pred == y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/worlds/othello-v1")
    parser.add_argument("--games", type=int, default=4000, help="validation games (half fit, half test)")
    parser.add_argument("--transfer-games", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device).eval()
    val = np.load(args.data / "val_games.npy", allow_pickle=True)[: args.games]
    test = np.load(args.data / "test_games.npy", allow_pickle=True)[: args.transfer_games]
    half = len(val) // 2
    fit_x, fit_y, legal_fit = activations(model, list(val[:half]), device)
    id_x, id_y, legal_id = activations(model, list(val[half:]), device)
    tr_x, tr_y, legal_tr = activations(model, list(test), device)
    majority = torch.mode(fit_y, dim=0).values
    layers = []
    for layer in range(len(fit_x)):
        probe = fit_probe(fit_x[layer], fit_y)
        layers.append({"layer": layer, "in_distribution": probe_accuracy(probe, id_x[layer], id_y),
                       "far_transfer_greedy": probe_accuracy(probe, tr_x[layer], tr_y)})
        print(layers[-1], flush=True)
    result = {
        "kind": "slm.othello_world_model_eval",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_manifest_sha256": sha256_file(args.data / "manifest.json"),
        "games": {"probe_fit": half, "in_distribution": len(val) - half, "far_transfer_greedy": len(test)},
        "legal_move_rate": {"in_distribution": legal_id, "far_transfer_greedy": legal_tr, "probe_fit_games": legal_fit},
        "majority_class_square_accuracy": {"in_distribution": float((majority == id_y).float().mean()),
                                           "far_transfer_greedy": float((majority == tr_y).float().mean())},
        "probe_square_accuracy_by_layer": layers,
        "probe": "linear, standardised features, 3 classes x 64 squares, AdamW lr 1e-2, 8 epochs, seed 0",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
