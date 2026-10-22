"""Build an 80/20 FSC/MOT training stream with the FSC validation shards unchanged."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from slm.runlog import code_revision, environment_info, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]


def load_stream(directory: Path, split: str) -> tuple[np.ndarray, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dtype = np.dtype(manifest["dtype"])
    pieces = []
    for shard in manifest["splits"][split]["shards"]:
        path = directory / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        piece = np.fromfile(path, dtype=dtype)
        if len(piece) != shard["tokens"]:
            raise ValueError(f"shard token count mismatch: {path}")
        pieces.append(piece)
    if not pieces or not sum(len(piece) for piece in pieces):
        raise ValueError(f"empty {split} token stream: {directory}")
    return np.concatenate(pieces), manifest


def trim_at_eos(tokens: np.ndarray, target: int, eos: int) -> np.ndarray:
    if target <= 0:
        raise ValueError("target token count must be positive")
    ends = np.flatnonzero(tokens[:target] == eos)
    if not len(ends):
        raise ValueError("no document boundary before target token count")
    return tokens[: int(ends[-1]) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--news-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Mixture output directory must be empty to preserve existing runs")
    fsc_dir = ROOT / "data/tokens/fsc-pilot-v1-scaling"
    fsc = json.loads((fsc_dir / "manifest.json").read_text(encoding="utf-8"))
    mot = json.loads((args.mot_tokens / "manifest.json").read_text(encoding="utf-8"))
    if fsc["tokenizer_sha256"] != mot["tokenizer_sha256"]:
        raise ValueError("FSC and MOT token shards must use the same frozen tokenizer")
    if (fsc["dtype"] != mot["dtype"] or fsc["vocab_size"] != mot["vocab_size"]
            or fsc["eos_id"] != mot["eos_id"]):
        raise ValueError("FSC and MOT token formats do not match")
    if not 0.0 < args.news_fraction < 1.0:
        raise ValueError("news fraction must be between zero and one")

    fsc_train, _ = load_stream(fsc_dir, "train")
    fsc_val, _ = load_stream(fsc_dir, "val")
    mot_train, _ = load_stream(args.mot_tokens, "train")
    if mot_train[-1] != fsc["eos_id"]:
        raise ValueError("MOT stream must end at an EOS document boundary before repetition")
    target_news = int(round(len(fsc_train) * args.news_fraction / (1 - args.news_fraction)))
    repeats = math.ceil(target_news / max(1, len(mot_train)))
    news = trim_at_eos(np.resize(mot_train, target_news), target_news, int(fsc["eos_id"]))
    actual_fraction = len(news) / (len(fsc_train) + len(news))
    args.out.mkdir(parents=True, exist_ok=True)

    train_shards = []
    fsc_dtype = np.dtype(fsc["dtype"])
    for i, shard in enumerate(fsc["splits"]["train"]["shards"]):
        src = fsc_dir / shard["file"]
        name = f"train_fsc_{i:05d}.bin"
        dest = args.out / name
        shutil.copyfile(src, dest)
        train_shards.append({"file": name, "tokens": shard["tokens"], "sha256": sha256_file(dest),
                             "component": "fsc-v1-balanced-science"})
    news_path = args.out / "train_mot_00000.bin"
    news.astype(fsc_dtype, copy=False).tofile(news_path)
    train_shards.append({"file": news_path.name, "tokens": int(len(news)), "sha256": sha256_file(news_path),
                         "component": "mot-v1.11-news", "repetitions": repeats})

    val_shards = []
    for i, shard in enumerate(fsc["splits"]["val"]["shards"]):
        src = fsc_dir / shard["file"]
        name = f"val_fsc_{i:05d}.bin"
        dest = args.out / name
        shutil.copyfile(src, dest)
        val_shards.append({"file": name, "tokens": shard["tokens"], "sha256": sha256_file(dest)})

    manifest = {
        "kind": "slm.tokens",
        "dtype": fsc["dtype"],
        "eos_id": fsc["eos_id"],
        "vocab_size": fsc["vocab_size"],
        "tokenizer_sha256": fsc["tokenizer_sha256"],
        "tokenizer_manifest_sha256": sha256_file(ROOT / "data/tokenizer/fsc-pilot-v1-train-only/manifest.json"),
        "mix": {
            "target_news_fraction": args.news_fraction,
            "actual_news_fraction": actual_fraction,
            "fsc_unique_train_tokens": int(len(fsc_train)),
            "mot_train_tokens_before_repetition": int(len(mot_train)),
            "mot_tokens_in_mix": int(len(news)),
            "mot_repetitions": repeats,
            "requested_seed": args.seed,
            "seed_used_for_reordering": False,
            "note": "FSC files and FSC validation shards are copied unchanged; MOT tokens are sorted by stable document-ID hash before encoding.",
        },
        "inputs": [
            {"path": str(fsc_dir / "manifest.json"), "sha256": sha256_file(fsc_dir / "manifest.json")},
            {"path": str(args.mot_tokens / "manifest.json"), "sha256": sha256_file(args.mot_tokens / "manifest.json")},
        ],
        "splits": {
            "train": {"tokens": sum(s["tokens"] for s in train_shards), "docs": None, "shards": train_shards},
            "val": {"tokens": sum(s["tokens"] for s in val_shards), "docs": fsc["splits"]["val"]["docs"], "shards": val_shards},
        },
        "code": code_revision(),
        "environment": environment_info(),
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "provenance.json", {
        "schema": "slm.exp0012_data_provenance.v1",
        "experiment": "EXP-0012",
        "fsc": {"source": "PleIAs/French-Science-Commons", "revision": "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156",
                "subset": "french_open_science", "licenses": ["CC0-1.0", "CC-BY-4.0", "Licence-Ouverte-2.0"],
                "status": "candidate; rights and privacy review incomplete"},
        "mot": {"source": "MOT v1.11 / VOA English and French archives", "revision": "9204cad746dfff0921e7a8f64c0cc0917bc75554",
                "license_label": "CC-BY-4.0 at collection level; source terms and third-party items unresolved",
                "metadata_audit": "runs/corpus-audits/mot-v1.11-metadata-audit.json",
                "author_missingness": "English author metadata absent for 59.7%; missing values are not inferred"},
        "public_weight_release_approved": False,
        "raw_article_text_distributable": False,
    })
    print(json.dumps({"train_tokens": manifest["splits"]["train"]["tokens"],
                      "fsc_tokens": len(fsc_train), "mot_tokens": len(news),
                      "actual_news_fraction": actual_fraction, "mot_repetitions": repeats,
                      "val_tokens": manifest["splits"]["val"]["tokens"],
                      "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
