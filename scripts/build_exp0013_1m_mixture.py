"""Build matched 1M FSC-only and FSC/MOT token-stream inputs for EXP-0013."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from slm.runlog import code_revision, environment_info, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]


def read_stream(directory: Path, split: str) -> tuple[np.ndarray, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    pieces = []
    for shard in manifest["splits"][split]["shards"]:
        path = directory / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        array = np.fromfile(path, dtype=np.dtype(manifest["dtype"]))
        if len(array) != shard["tokens"]:
            raise ValueError(f"token count mismatch: {path}")
        pieces.append(array)
    if not pieces or not sum(len(piece) for piece in pieces):
        raise ValueError(f"no token shards for {split} in {directory}")
    return np.concatenate(pieces), manifest


def end_on_document_boundary(tokens: np.ndarray, target: int, eos: int) -> np.ndarray:
    if target <= 0:
        raise ValueError("target token count must be positive")
    if target > len(tokens):
        raise ValueError("Not enough unique MOT tokens for the requested mixture fraction")
    ends = np.flatnonzero(tokens[:target] == eos)
    if not len(ends):
        raise ValueError("MOT token stream has no complete document before target fraction")
    return tokens[: int(ends[-1]) + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsc", type=Path, default=ROOT / "data/tokens/fsc-cc0-open-weights-v4/training")
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--news-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.0 < args.news_fraction < 1.0:
        raise ValueError("news fraction must be between zero and one")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output must be empty: {args.out}")

    fsc_manifest = json.loads((args.fsc / "manifest.json").read_text(encoding="utf-8"))
    mot_manifest = json.loads((args.mot / "manifest.json").read_text(encoding="utf-8"))
    for key in ("tokenizer_sha256", "dtype", "vocab_size", "eos_id"):
        if fsc_manifest[key] != mot_manifest[key]:
            raise ValueError(f"FSC and MOT token formats differ at {key}")

    fsc_train, _ = read_stream(args.fsc, "train")
    mot_train, _ = read_stream(args.mot, "train")
    # Validate held-out inputs against their recorded hashes before copying them.
    for split in ("val", "test"):
        read_stream(args.fsc, split)
    target_mot = int(round(len(fsc_train) * args.news_fraction / (1.0 - args.news_fraction)))
    if mot_train[-1] != fsc_manifest["eos_id"]:
        raise ValueError("MOT token stream does not end on a document boundary")
    mot = end_on_document_boundary(mot_train, target_mot, int(fsc_manifest["eos_id"]))
    actual_fraction = len(mot) / (len(fsc_train) + len(mot))

    args.out.mkdir(parents=True, exist_ok=True)
    train_shards = []
    for index, shard in enumerate(fsc_manifest["splits"]["train"]["shards"]):
        src = args.fsc / shard["file"]
        name = f"train_fsc_{index:05d}.bin"
        shutil.copyfile(src, args.out / name)
        train_shards.append({"file": name, "tokens": shard["tokens"], "sha256": sha256_file(args.out / name),
                             "component": "fsc-cc0-open-weights-v4"})
    mot_path = args.out / "train_mot_00000.bin"
    mot.astype(np.dtype(fsc_manifest["dtype"]), copy=False).tofile(mot_path)
    train_shards.append({"file": mot_path.name, "tokens": int(len(mot)), "sha256": sha256_file(mot_path),
                         "component": "mot-v1.11-temporal-train", "repetition_count": 1,
                         "selection": "prefix of deterministic SHA-256 document-ID order; ends at complete document"})

    split_records = {"train": {"tokens": sum(s["tokens"] for s in train_shards), "docs": None,
                                "shards": train_shards}}
    for split in ("val", "test"):
        records = []
        for index, shard in enumerate(fsc_manifest["splits"][split]["shards"]):
            src = args.fsc / shard["file"]
            name = f"{split}_fsc_{index:05d}.bin"
            shutil.copyfile(src, args.out / name)
            records.append({"file": name, "tokens": shard["tokens"], "sha256": sha256_file(args.out / name)})
        split_records[split] = {"tokens": sum(s["tokens"] for s in records),
                                "docs": fsc_manifest["splits"][split].get("docs"), "shards": records}

    manifest = {
        "kind": "slm.tokens",
        "dtype": fsc_manifest["dtype"], "eos_id": fsc_manifest["eos_id"],
        "vocab_size": fsc_manifest["vocab_size"], "tokenizer_sha256": fsc_manifest["tokenizer_sha256"],
        "tokenizer_manifest_sha256": sha256_file(ROOT / "data/tokenizers/fsc-cc0-open-weights-v4/manifest.json"),
        "mix": {"target_mot_fraction": args.news_fraction, "actual_mot_fraction": actual_fraction,
                "fsc_unique_train_tokens": int(len(fsc_train)), "mot_available_unique_tokens": int(len(mot_train)),
                "mot_tokens_used": int(len(mot)), "mot_repetitions": 1,
                "selection_order": "deterministic temporal train stream sorted by stable SHA-256 document-ID hash",
                "val_and_test": "copied unchanged from FSC v4; MOT dev/test remain external evaluation-only files"},
        "inputs": [{"path": str(args.fsc / "manifest.json"), "sha256": sha256_file(args.fsc / "manifest.json")},
                   {"path": str(args.mot / "manifest.json"), "sha256": sha256_file(args.mot / "manifest.json")}],
        "splits": split_records, "code": code_revision(), "environment": environment_info(),
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "provenance.json", {
        "schema": "slm.exp0013_data_provenance.v1",
        "experiment": "EXP-0013",
        "fsc": {"source": "PleIAs/French-Science-Commons", "revision": "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156",
                "subset": "audited CC0 subset; FSC v4 integrity/privacy screen",
                "status": "candidate; final source/privacy sign-offs remain open"},
        "mot": {"source": "MOT v1.11 / VOA English and French archives",
                "revision": "9204cad746dfff0921e7a8f64c0cc0917bc75554",
                "license_label": "CC-BY-4.0 at collection level; item-level source terms and third-party content unresolved",
                "metadata_audit": "runs/corpus-audits/mot-v1.11-metadata-audit.json",
                "author_metadata_missingness": "59.7% of English records; missing values remain missing"},
        "public_weight_release_approved": False,
        "raw_article_text_distributable": False,
    })
    print(json.dumps({"fsc_tokens": int(len(fsc_train)), "mot_tokens": int(len(mot)),
                      "actual_mot_fraction": actual_fraction, "train_budget_tokens": int(len(fsc_train)),
                      "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
