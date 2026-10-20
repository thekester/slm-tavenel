"""Create a deterministic, document-disjoint train/validation split for FSC."""

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


def split(path: Path, output_dir: Path, val_fraction: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {name: {"documents": 0, "whitespace_words": 0, "characters": 0} for name in ("train", "val")}
    paths = {name: output_dir / f"{path.stem}-{name}.jsonl" for name in stats}
    with path.open(encoding="utf-8") as source, paths["train"].open("w", encoding="utf-8", newline="\n") as train, paths["val"].open("w", encoding="utf-8", newline="\n") as val:
        outputs = {"train": train, "val": val}
        for line in source:
            row = json.loads(line)
            digest = int.from_bytes(hashlib.sha256(f"slm-split-v1:{row['id']}".encode()).digest()[:8], "big")
            group = "val" if digest / 2**64 < val_fraction else "train"
            outputs[group].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats[group]["documents"] += 1
            stats[group]["whitespace_words"] += len(row["text"].split())
            stats[group]["characters"] += len(row["text"])
    manifest = {"kind": "slm.fsc_document_split", "input": str(path), "input_sha256": sha256_file(path),
                "split_hash": "sha256('slm-split-v1:' + document_id), first 64 bits mapped to [0,1)",
                "val_fraction": val_fraction, "files": {name: {"path": str(paths[name]), "sha256": sha256_file(paths[name]), "bytes": paths[name].stat().st_size, **stats[name]} for name in stats}}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    results = [split(Path(path), Path(args.output_dir), args.val_fraction) for path in args.inputs]
    manifest = {"kind": "slm.fsc_stratified_document_splits", "val_fraction": args.val_fraction, "splits": results}
    out = Path(args.output_dir) / "split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
