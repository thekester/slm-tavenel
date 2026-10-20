"""Balance FSC documents by frozen-tokenizer budget, license, and split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer


def split_of(doc_id: str, val_fraction: float) -> str:
    digest = int.from_bytes(hashlib.sha256(f"slm-split-v1:{doc_id}".encode()).digest()[:8], "big")
    return "val" if digest / 2**64 < val_fraction else "train"


def license_key(source: str) -> str:
    return source.removeprefix("fsc-en-").removeprefix("fsc-fr-")


def balance(inputs: list[Path], tokenizer_path: Path, output_dir: Path, val_fraction: float, seed: int) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    groups: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    docs_seen = 0
    for path in inputs:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                language = row["source"][4:6]
                lic = license_key(row["source"])
                split = split_of(row["id"], val_fraction)
                tokens = len(tokenizer.encode(row["text"]).ids) + (1 if eos_id is not None else 0)
                groups[(language, lic, split)].append((row["id"], tokens))
                docs_seen += 1

    selections: dict[tuple[str, str, str], set[str]] = {}
    report = {}
    for lic in sorted({key[1] for key in groups}):
        report[lic] = {}
        for split in ("train", "val"):
            en = groups[("en", lic, split)]
            fr = groups[("fr", lic, split)]
            target = min(sum(n for _, n in en), sum(n for _, n in fr))
            for language, items in (("en", en), ("fr", fr)):
                ordered = sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{lic}:{split}:{language}:{item[0]}".encode()).digest())
                selected, total = set(), 0
                for doc_id, count in ordered:
                    if total + count <= target:
                        selected.add(doc_id)
                        total += count
                selections[(language, lic, split)] = selected
                report[lic].setdefault(split, {})[language] = {
                    "available_documents": len(items), "available_tokens": sum(n for _, n in items),
                    "selected_documents": len(selected), "selected_tokens": total,
                    "discarded_documents": len(items) - len(selected),
                }
            report[lic][split]["target_tokens_per_language"] = target
            report[lic][split]["token_difference"] = abs(
                report[lic][split]["en"]["selected_tokens"] - report[lic][split]["fr"]["selected_tokens"]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    handles = {}
    try:
        for lang in ("en", "fr"):
            for split in ("train", "val"):
                path = output_dir / f"{lang}-{split}.jsonl"
                handles[(lang, split)] = path.open("w", encoding="utf-8", newline="\n")
                outputs[(lang, split)] = {"path": str(path), "documents": 0}
        for path in inputs:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    language = row["source"][4:6]
                    lic = license_key(row["source"])
                    split = split_of(row["id"], val_fraction)
                    if row["id"] in selections[(language, lic, split)]:
                        handles[(language, split)].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        outputs[(language, split)]["documents"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for info in outputs.values():
        path = Path(info["path"])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        info["sha256"] = digest.hexdigest()
        info["bytes"] = path.stat().st_size
    manifest = {
        "kind": "slm.fsc_token_balanced_sample", "inputs": [str(p) for p in inputs],
        "tokenizer": str(tokenizer_path), "tokenizer_sha256": hashlib.sha256((tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path).read_bytes()).hexdigest(),
        "selection": "deterministic SHA-256 order within license, language, and hash-based train/validation split; document boundaries retained",
        "split_salt": "slm-split-v1", "val_fraction": val_fraction, "seed": seed,
        "documents_scanned": docs_seen, "license_balance": report,
        "outputs": [{"language": key[0], "split": key[1], **value} for key, value in sorted(outputs.items())],
        "note": "Per-group document boundaries prevent exact token equality; residual differences are recorded. Tokenizer was trained only on the train split.",
    }
    (output_dir / "token_balance_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    result = balance([Path(p) for p in args.inputs], Path(args.tokenizer), Path(args.output_dir), args.val_fraction, args.seed)
    print(json.dumps(result["license_balance"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
