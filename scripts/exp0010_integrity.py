"""Validate the immutable code and input identities recorded for EXP-0010."""

import hashlib
from pathlib import Path
import subprocess

EXPECTED_COMMIT = "02396acd15ef9bcdc453bcd53aade40819e2f69f"
TOKENIZER_SHA256 = "93c7b9364de2c42315571c851353862e63830bfbeb220dc14dde80e6ca82cfcb"
SPLIT_MANIFEST_SHA256 = "abc8371759211a9ade705fa251e6a5f978b34ea1d41353831066d0ab97661f47"


def validate_code_root(code_root: Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=code_root, text=True).strip()
    if commit != EXPECTED_COMMIT or dirty:
        raise ValueError(f"Expected a clean code worktree at {EXPECTED_COMMIT}")


def validate_hash(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Pinned artifact hash differs: {path}")
