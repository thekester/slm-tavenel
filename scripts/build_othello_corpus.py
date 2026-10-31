#!/usr/bin/env python3
"""Generate the synthetic Othello corpus for EXP-0038 (world-model pilot).

Train/val: games with uniformly random legal moves (the synthetic setting of Li et al., 2023).
Test (far transfer): games from a greedy most-flips policy, a different play style never trained on.
Each game is one fixed-length row of 64 tokens (moves + 1, then end-of-game padding), so training
windows start at the opening position. Games are also saved as move lists for state probes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import code_revision, sha256_file  # noqa: E402
from slm.worlds import othello  # noqa: E402

ROW = 64


def make_games(args: tuple[int, int, str]) -> list[list[int]]:
    seed, n, policy = args
    rng = np.random.default_rng(seed)
    return [othello.random_game(rng, policy) for _ in range(n)]


def generate(n: int, policy: str, seed: int, workers: int, chunk: int = 2000) -> list[list[int]]:
    jobs = [(seed * 1_000_003 + i, min(chunk, n - i * chunk), policy) for i in range((n + chunk - 1) // chunk)]
    games: list[list[int]] = []
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(make_games, jobs):
            games.extend(part)
    return games


def to_rows(games: list[list[int]]) -> np.ndarray:
    rows = np.full((len(games), ROW), othello.EOS, dtype=np.uint8)
    for i, moves in enumerate(games):
        tokens = othello.encode(moves)
        rows[i, : len(tokens)] = tokens
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/worlds/othello-v1")
    parser.add_argument("--train-games", type=int, default=1_000_000)
    parser.add_argument("--val-games", type=int, default=10_000)
    parser.add_argument("--test-games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plan = {"train": (args.train_games, "uniform", args.seed), "val": (args.val_games, "uniform", args.seed + 1),
            "test": (args.test_games, "greedy", args.seed + 2)}
    splits, seen = {}, set()
    for split, (n, policy, seed) in plan.items():
        games = generate(n, policy, seed, args.workers)
        overlap = sum(tuple(g) in seen for g in games) if split != "train" else 0
        if split == "train":
            seen = {tuple(g) for g in games}
        rows = to_rows(games)
        name = f"{split}_00000.bin"
        rows.reshape(-1).tofile(args.out / name)
        np.save(args.out / f"{split}_games.npy", np.array([np.array(g, dtype=np.uint8) for g in games], dtype=object),
                allow_pickle=True)
        splits[split] = {"tokens": int(rows.size), "docs": len(games), "policy": policy, "seed": seed,
                         "games_also_in_train": overlap, "mean_moves": float(np.mean([len(g) for g in games])),
                         "shards": [{"file": name, "tokens": int(rows.size), "sha256": sha256_file(args.out / name)}]}
        print(split, splits[split]["docs"], "games", round(time.time() - t0), "s", flush=True)
    manifest = {
        "kind": "slm.tokens.othello", "dtype": "uint8", "eos_id": othello.EOS, "vocab_size": othello.VOCAB_SIZE,
        "row_length": ROW, "token_map": "0 = end of game / padding, 1..64 = square index + 1 (a1 = 1, h8 = 64)",
        "splits": {"train": splits["train"], "val": splits["val"], "test": splits["test"]},
        "notes": "Synthetic, rights-free data generated locally; test split uses a different (greedy) policy.",
        "code": code_revision(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
