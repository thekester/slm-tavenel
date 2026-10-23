"""Assemble the fixed 80/20 FSC/wiki token streams for the EXP-0016 treatment."""

from __future__ import annotations

import json
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm.runlog import code_revision, environment_info, sha256_file, write_json  # noqa: E402

FSC = ROOT / "data/tokens/fsc-pilot-v1-scaling"
WIKI = ROOT / "data/processed/exp0016-wiki"
OUT = ROOT / "data/tokens/exp0016-25m-wiki-mixture"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_shards(source: Path, manifest: dict, split: str, prefix: str, component: str,
                out_dir: Path):
    records = []
    for index, shard in enumerate(manifest["splits"][split]["shards"]):
        src = source / shard["file"]
        if sha256_file(src) != shard["sha256"]:
            raise ValueError(f"input shard hash mismatch: {src}")
        name = f"{prefix}_{index:05d}.bin"
        dst = out_dir / name
        shutil.copyfile(src, dst)
        records.append({"file": name, "tokens": int(shard["tokens"]), "sha256": sha256_file(dst),
                        "component": component})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-source", type=Path, default=WIKI)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    wiki_dir = args.wiki_source if args.wiki_source.is_absolute() else ROOT / args.wiki_source
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output must be empty to avoid overwriting a prior run: {out_dir}")
    fsc = read_json(FSC / "manifest.json")
    wiki = read_json(wiki_dir / "manifest.json")
    token_format = {
        "tokenizer_sha256": wiki["tokenizer"]["sha256"],
        "vocab_size": wiki["tokenizer"]["vocab_size"],
        "dtype": wiki["dtype"],
        "eos_id": wiki["tokenizer"]["eos_id"],
    }
    for key, value in token_format.items():
        if fsc[key] != value:
            raise ValueError(f"FSC and wiki token formats do not match at {key}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_shards = copy_shards(FSC, fsc, "train", "train_fsc", "fsc-pilot-v1-balanced-science", out_dir)
    for lang in ("en", "fr"):
        src = wiki_dir / f"train_wiki_{lang}.bin"
        row = wiki["training_shards"][lang]
        if sha256_file(src) != row["sha256"]:
            raise ValueError(f"wiki token shard hash mismatch: {src}")
        name = f"train_wiki_{lang}.bin"
        shutil.copyfile(src, out_dir / name)
        train_shards.append({"file": name, "tokens": int(row["tokens"]),
                             "sha256": sha256_file(out_dir / name), "component": f"wikipedia-{lang}-wiki-v1"})

    val_shards = copy_shards(FSC, fsc, "val", "val_fsc", "fsc-pilot-v1-heldout", out_dir)
    total_train = sum(x["tokens"] for x in train_shards)
    wiki_train = sum(x["tokens"] for x in train_shards if x["component"].startswith("wikipedia-"))
    actual_fraction = wiki_train / total_train
    manifest = {
        "kind": "slm.tokens",
        "dtype": fsc["dtype"], "eos_id": fsc["eos_id"], "vocab_size": fsc["vocab_size"],
        "tokenizer_sha256": fsc["tokenizer_sha256"],
        "tokenizer_manifest_sha256": fsc.get("tokenizer_manifest_sha256"),
        "mix": {"target_wiki_fraction": wiki["sampling"]["target_wiki_fraction"], "actual_wiki_fraction": actual_fraction,
                "fsc_unique_train_tokens": int(fsc["splits"]["train"]["tokens"]),
                "wiki_unique_train_tokens": int(wiki_train),
                "language_split": {lang: int(wiki["sampling"]["tokens_written_by_language"][lang])
                                   for lang in ("en", "fr")},
                "selection": "deterministic SHA-256 sample; benchmark decontaminated; no document repeated"},
        "inputs": [{"path": (FSC / "manifest.json").relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(FSC / "manifest.json")},
                   {"path": (wiki_dir / "manifest.json").relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(wiki_dir / "manifest.json")}],
        "splits": {"train": {"tokens": total_train, "docs": None, "shards": train_shards},
                   "val": {"tokens": sum(x["tokens"] for x in val_shards),
                           "docs": fsc["splits"]["val"].get("docs"), "shards": val_shards}},
        "code": code_revision(), "environment": environment_info(),
    }
    write_json(out_dir / "manifest.json", manifest)
    experiment_number = out_dir.name[3:7] if out_dir.name.lower().startswith("exp") else "unknown"
    experiment_id = f"EXP-{experiment_number}" if experiment_number.isdigit() else out_dir.name
    write_json(out_dir / "provenance.json", {
        "schema": "slm.wiki_mixture_data_provenance.v1", "experiment": experiment_id,
        "fsc": {"source": "PleIAs/French-Science-Commons", "status": "candidate; inherited EXP-0009 sources not cleared"},
        "wiki": wiki["source"], "public_weight_release_approved": False,
        "raw_article_text_distributed": False,
    })
    print(json.dumps({"out": out_dir.relative_to(ROOT).as_posix(), "train_tokens": total_train,
                      "wiki_tokens": wiki_train, "actual_wiki_fraction": actual_fraction,
                      "val_tokens": manifest["splits"]["val"]["tokens"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
