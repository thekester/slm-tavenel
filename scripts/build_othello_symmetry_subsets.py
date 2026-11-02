#!/usr/bin/env python3
"""Low-data Othello corpora for the EXP-0038 symmetry arms (amendment of 2026-11-02).

  othello-v1-50k       the first 50,000 training games of othello-v1, unchanged;
  othello-v1-50k-sym4  the same 50,000 games, each also under the three non-identity symmetries that
                       preserve the opening (rotate180, transpose, anti_transpose): 200,000 rows.
Validation and test splits are copied from othello-v1, so all arms are evaluated on identical games.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_othello_corpus import ROW, to_rows  # noqa: E402
from slm.runlog import code_revision, sha256_file  # noqa: E402
from slm.worlds import othello  # noqa: E402

SOURCE = ROOT / "data/worlds/othello-v1"
N_GAMES = 50_000


def write(out: Path, games: list[list[int]], note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = to_rows(games)
    rows.reshape(-1).tofile(out / "train_00000.bin")
    source = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        shutil.copyfile(SOURCE / f"{split}_00000.bin", out / f"{split}_00000.bin")
        shutil.copyfile(SOURCE / f"{split}_games.npy", out / f"{split}_games.npy")
    splits = {"train": {"tokens": int(rows.size), "docs": len(games),
                        "shards": [{"file": "train_00000.bin", "tokens": int(rows.size),
                                    "sha256": sha256_file(out / "train_00000.bin")}]},
              "val": source["splits"]["val"], "test": source["splits"]["test"]}
    manifest = {**{k: source[k] for k in ("kind", "dtype", "eos_id", "vocab_size", "row_length", "token_map")},
                "splits": splits, "derived_from": {"path": "data/worlds/othello-v1",
                                                   "manifest_sha256": sha256_file(SOURCE / "manifest.json")},
                "notes": note, "code": code_revision()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    games = [list(map(int, g)) for g in np.load(SOURCE / "train_games.npy", allow_pickle=True)[:N_GAMES]]
    write(ROOT / "data/worlds/othello-v1-50k", games, "First 50,000 training games of othello-v1.")
    augmented = [othello.transform(g, name) for g in games for name in othello.SYMMETRIES]
    write(ROOT / "data/worlds/othello-v1-50k-sym4", augmented,
          "The same 50,000 games under the 4 opening-preserving symmetries (identity included).")
    print(len(games), len(augmented), ROW)


if __name__ == "__main__":
    main()
