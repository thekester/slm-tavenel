"""Run manifests and append-only JSONL logs.

A manifest records what is needed to reproduce or audit a run: resolved
configuration and its hash, code revision, software versions, hardware, and the
hashes of the input artifacts.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def code_revision(repo: str | Path = ".") -> dict[str, Any]:
    """Read-only query of the code revision; never modifies the repository."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [],
    }
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["cudnn"] = torch.backends.cudnn.version()
    for pkg in ("numpy", "tokenizers", "safetensors"):
        try:
            info[pkg] = __import__(pkg).__version__
        except ImportError:
            pass
    return info


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_tokenizer_compatible(checkpoint: str | Path, tokenizer_sha256: str) -> list[dict[str, str]]:
    """Reject a tokenizer that differs from any recorded checkpoint ancestor."""
    current = Path(checkpoint).resolve()
    seen: set[Path] = set()
    lineage: list[dict[str, str]] = []
    while current not in seen:
        seen.add(current)
        run_dir = current.parent.parent if current.parent.name in {"checkpoints", "initialization"} else current.parent
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            provenance_path = current.with_suffix(current.suffix + ".provenance.json")
            if provenance_path.is_file():
                provenance = read_json(provenance_path)
                current_hash = provenance.get("tokenizer_sha256")
                if current_hash:
                    lineage.append({"checkpoint": str(current), "tokenizer_sha256": str(current_hash)})
                    if str(current_hash).lower() != tokenizer_sha256.lower():
                        raise ValueError(
                            f"Tokenizer mismatch in growth checkpoint {current}: records {current_hash}, "
                            f"but requested tokenizer is {tokenizer_sha256}"
                        )
                source = provenance.get("source_checkpoint")
                if source:
                    current = Path(source).resolve()
                    continue
            break
        manifest = read_json(manifest_path)
        current_hash = manifest.get("tokenizer_sha256") or manifest.get("data", {}).get("tokenizer_sha256")
        if current_hash:
            lineage.append({"checkpoint": str(current), "tokenizer_sha256": str(current_hash)})
            if str(current_hash).lower() != tokenizer_sha256.lower():
                raise ValueError(
                    f"Tokenizer mismatch in checkpoint lineage: {current} records {current_hash}, "
                    f"but requested tokenizer is {tokenizer_sha256}"
                )
        compatibility = manifest.get("data", {}).get("tokenizer_compatibility") or manifest.get("tokenizer_compatibility") or {}
        for ancestor in compatibility.get("source_checkpoint_lineage", []):
            ancestor_hash = ancestor.get("tokenizer_sha256")
            if ancestor_hash and str(ancestor_hash).lower() != tokenizer_sha256.lower():
                raise ValueError(
                    f"Tokenizer mismatch in recorded source lineage for {current}: records {ancestor_hash}, "
                    f"but requested tokenizer is {tokenizer_sha256}"
                )
        initialized = manifest.get("initialized_from") or {}
        grown = initialized.get("grown_from") or {}
        grown_hash = grown.get("tokenizer_sha256")
        if grown_hash:
            lineage.append({"checkpoint": str(current), "tokenizer_sha256": str(grown_hash)})
            if str(grown_hash).lower() != tokenizer_sha256.lower():
                raise ValueError(
                    f"Tokenizer mismatch in grown checkpoint lineage: {current} records {grown_hash}, "
                    f"but requested tokenizer is {tokenizer_sha256}"
                )
        source = initialized.get("path") or grown.get("source_checkpoint")
        if not source:
            break
        current = Path(source).resolve()
    return lineage


class JsonlLogger:
    """Append-only JSONL writer. Each record gets a wall-clock timestamp."""

    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record: Any) -> None:
        if not self.enabled:
            return
        record = {"time": round(time.time(), 3), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
