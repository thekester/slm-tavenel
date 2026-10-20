"""Join final FSC document, tokenizer, and token-shard manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(balanced: Path, tokenizer: Path, tokens: Path, val_fraction: float = 0.05) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    tokenizer_manifest = read(tokenizer / "manifest.json")
    tokenizer_hash = sha256_file(tokenizer / "tokenizer.json")
    if tokenizer_manifest["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("tokenizer file does not match its manifest")
    splits = {}
    for language in ("en", "fr"):
        splits[language] = {}
        for split in ("train", "val"):
            doc_path = balanced / f"{language}-{split}.jsonl"
            doc_hash = sha256_file(doc_path)
            token_manifest = read(tokens / language / split / "manifest.json")
            if token_manifest["tokenizer_sha256"] != tokenizer_hash:
                raise ValueError(f"{language}/{split}: tokenizer differs from the frozen tokenizer")
            inputs = token_manifest["inputs"]
            if len(inputs) != 1 or inputs[0]["sha256"] != doc_hash:
                raise ValueError(f"{language}/{split}: encoded input does not match the document file")
            if token_manifest["val_fraction"] != 0 or any(
                token_manifest["splits"]["val"][key] for key in ("docs", "tokens", "shards")
            ):
                raise ValueError(f"{language}/{split}: encode each pre-split file with --val-fraction 0.0")
            split_info = token_manifest["splits"]["train"]
            if split_info["docs"] <= 0 or split_info["tokens"] <= 0:
                raise ValueError(f"{language}/{split}: encoded split is empty")
            splits[language][split] = {
                "documents_path": str(doc_path), "documents_sha256": doc_hash,
                "documents": split_info["docs"], "tokens": split_info["tokens"],
                "token_shard_manifest": str(tokens / language / split / "manifest.json"),
                "token_shard_manifest_sha256": sha256_file(tokens / language / split / "manifest.json"),
            }
    balance = {}
    for split in ("train", "val"):
        en, fr = splits["en"][split]["tokens"], splits["fr"][split]["tokens"]
        balance[split] = {"en_tokens": en, "fr_tokens": fr, "absolute_difference": abs(en - fr),
                          "difference_fraction_of_mean": abs(en - fr) / ((en + fr) / 2)}
    manifest = {
        "kind": "slm.fsc_final_token_balanced_corpus",
        "pipeline_manifest": "data/processed/fsc-pilot-v1/manifest.json",
        "selection_script": "scripts/balance_fsc_by_tokens.py",
        "selection_unit": "whole documents, matched within license class, language, and train/validation split by frozen-tokenizer token count",
        "tokenizer_manifest": str(tokenizer / "manifest.json"),
        "tokenizer_manifest_sha256": sha256_file(tokenizer / "manifest.json"),
        "tokenizer_sha256": tokenizer_manifest["tokenizer_sha256"],
        "split_salt": "slm-split-v1", "validation_fraction": val_fraction,
        "splits": splits, "language_token_balance": balance,
        "caveat": "Residual differences remain because document boundaries are retained. The upstream preparation must exclude validation documents from tokenizer training; this manifest join checks input and tokenizer hashes but does not re-audit that separation.",
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Document split fraction used before encoding")
    args = parser.parse_args()
    balanced, tokenizer, tokens = Path(args.balanced_dir), Path(args.tokenizer), Path(args.tokens_dir)
    manifest = build_manifest(balanced, tokenizer, tokens, args.val_fraction)
    out = balanced / "final_token_balance_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["language_token_balance"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
