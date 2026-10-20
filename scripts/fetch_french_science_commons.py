"""Fetch a deterministic, evenly spaced sample of pinned French-Science-Commons shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DATASET = "PleIAs/French-Science-Commons"
REVISION = "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156"
SUBSET = "french_open_science"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(count: int, output_dir: Path, revision: str = REVISION) -> dict:
    repo_files = HfApi().list_repo_files(DATASET, repo_type="dataset", revision=revision)
    paths = sorted(
        (p for p in repo_files if p.startswith(f"{SUBSET}/") and p.endswith(".parquet")),
        key=lambda p: int(Path(p).stem.rsplit("_", 1)[1]),
    )
    if not paths:
        raise RuntimeError(f"no parquet shards found under {SUBSET!r} at {revision}")
    if count < 1 or count > len(paths):
        raise ValueError(f"count must be in [1, {len(paths)}]")

    if count == 1:
        chosen = [paths[len(paths) // 2]]
    else:
        indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
        chosen = [paths[i] for i in indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, repo_path in enumerate(chosen, start=1):
        local = Path(hf_hub_download(
            DATASET, repo_path, repo_type="dataset", revision=revision,
            local_dir=output_dir / "hf-cache",
        ))
        target = output_dir / "shards" / repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        # The Hub cache and destination may be on the same volume; copy explicitly so the
        # selected raw input set is self-contained and easy to hash/rebuild.
        local_hash = sha256(local)
        if not target.exists() or sha256(target) != local_hash:
            import shutil
            shutil.copyfile(local, target)
        files.append({"repo_path": repo_path, "path": str(target), "size_bytes": target.stat().st_size,
                      "sha256": local_hash})
        print(f"[{i}/{count}] {repo_path} ({target.stat().st_size:,} bytes)", flush=True)

    manifest = {
        "dataset": DATASET,
        "revision": revision,
        "subset": SUBSET,
        "selection": "evenly spaced shard indices over the numerically sorted subset file list",
        "available_shards": len(paths),
        "selected_shards": files,
    }
    with (output_dir / "download_manifest.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output-dir", default="data/raw/french-science-commons-pilot")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    fetch(args.count, Path(args.output_dir), args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
