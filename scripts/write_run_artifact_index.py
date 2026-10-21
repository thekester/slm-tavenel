"""Write a hash index for a completed training run's checkpoint and logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = run_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints.glob("step_*.pt"):
        match = re.fullmatch(r"step_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No step_*.pt checkpoint found in {checkpoints}")
    return max(candidates, key=lambda item: item[0])[1]


def last_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                last = json.loads(line)
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing run_manifest.json and checkpoints/")
    parser.add_argument("--output", type=Path, help="Index JSON output (default: <run_dir>/artifact_index.json)")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=Path,
        help="Additional related artifact to hash; may be repeated",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = latest_checkpoint(run_dir)
    metrics = last_jsonl_record(run_dir / "metrics.jsonl")
    train = manifest.get("train_config", {})

    files = [checkpoint, manifest_path]
    for name in ("metrics.jsonl", "eval.jsonl", "observability.jsonl"):
        path = run_dir / name
        if path.is_file():
            files.append(path)
    data = manifest.get("data", {})
    provenance = data.get("provenance") or {}
    for relative in (
        data.get("provenance_path"),
        provenance.get("token_shards", {}).get("manifest"),
    ):
        if relative:
            path = Path(relative)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                files.append(path.resolve())
    expected_tokenizer_hash = data.get("tokenizer_sha256")
    if expected_tokenizer_hash:
        for path in Path("data/tokenizer").rglob("tokenizer.json"):
            if sha256_file(path) == expected_tokenizer_hash:
                files.append(path.resolve())
                break
    for path in args.include:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Additional artifact not found: {resolved}")
        files.append(resolved)
    files = list(dict.fromkeys(files))
    artifacts = []
    for path in files:
        try:
            recorded_path = path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            recorded_path = path.as_posix()
        artifacts.append(
            {
                "path": recorded_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    script = Path(__file__).resolve()
    record = {
        "schema": "slm.run_artifact_index.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "index_script": script.name,
        "index_script_sha256": sha256_file(script),
        "status": "integrity index only; does not validate scientific claims or redistribution rights",
        "run": {
            "name": run_dir.name,
            "code_commit": manifest.get("code", {}).get("commit"),
            "code_dirty": manifest.get("code", {}).get("dirty"),
            "seed": train.get("seed"),
            "steps_completed": metrics.get("step") if metrics else None,
            "tokens_seen": metrics.get("tokens") if metrics else None,
            "parameters": manifest.get("params"),
            "model_config_hash": manifest.get("model_config_hash"),
            "train_config_hash": manifest.get("train_config_hash"),
            "tokenizer_sha256": expected_tokenizer_hash,
        },
        "artifacts": artifacts,
        "limitations": [
            "Artifact hashes establish file identity, not scientific validity or data rights.",
            "The run manifest may omit the checkpoint hash; this index binds the selected checkpoint to its recorded logs.",
        ],
    }
    output = args.output or (run_dir / "artifact_index.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote artifact index for {run_dir.name} ({len(artifacts)} files) to {output}")


if __name__ == "__main__":
    main()
