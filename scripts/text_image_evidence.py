"""Input-bound caches and score evidence for text/image alignment evaluations."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def image_fingerprint(rows: list[dict], directory: Path, encoder: tuple[str, str], extractor_sha256: str | None = None) -> dict:
    files = [{"file_name": row["file_name"], "sha256": file_hash(directory / row["file_name"])} for row in rows]
    software = {}
    for package in ("torch", "transformers", "Pillow", "numpy"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = None
    return {"encoder": list(encoder), "ordered_rows_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "ordered_image_files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "images": len(rows), "software": software,
            "extractor_sha256": extractor_sha256,
            "cache_helper_sha256": file_hash(Path(__file__)),
            "feature": "CPU float32, pinned processor defaults, last hidden-state CLS"}


def save_image_cache(path: Path, embeddings: np.ndarray, fingerprint: dict) -> None:
    if embeddings.ndim != 2 or len(embeddings) != fingerprint["images"] or not np.isfinite(embeddings).all():
        raise ValueError("Invalid image cache shape/values")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    metadata = {"fingerprint": fingerprint, "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
                "embeddings_sha256": file_hash(path), "created_by": "full encoder extraction on fingerprinted inputs"}
    path.with_suffix(path.suffix+".manifest.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")


def load_image_cache(path: Path, fingerprint: dict) -> np.ndarray:
    sidecar = path.with_suffix(path.suffix+".manifest.json")
    if not sidecar.exists():
        raise ValueError("Legacy DINO cache has no input manifest. Preserve it and choose a new --cache path for a fully fingerprinted extraction.")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata["fingerprint"] != fingerprint or metadata["embeddings_sha256"] != file_hash(path):
        raise ValueError("Image cache inputs, ordering, encoder or contents changed")
    embeddings = np.load(path)
    if list(embeddings.shape) != metadata["shape"] or str(embeddings.dtype) != metadata["dtype"] or len(embeddings) != fingerprint["images"] or not np.isfinite(embeddings).all():
        raise ValueError("Image cache shape/dtype/finite check failed")
    return embeddings
