"""Validate cached FSC outputs against the exact evaluation inputs."""

import json
from pathlib import Path

from slm.runlog import sha256_file


def input_signature(model: dict, benchmarks: list[str], validation: dict[str, str]) -> dict:
    return {
        "model": model,
        "benchmarks": [{"path": path, "sha256": sha256_file(path)} for path in benchmarks],
        "validation": {lang: {"path": path, "sha256": sha256_file(path)}
                       for lang, path in validation.items()},
    }


def complete(paths: list[Path], signature: dict) -> bool:
    try:
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if not all(isinstance(result, dict) and result.get("evaluation_inputs") == signature
                   for result in results):
            return False
        mc, *bpb = results
        tasks = [Path(entry["path"]).stem for entry in signature["benchmarks"]]
        return bool(tasks) and all(
            mc.get("results", {}).get(task, {}).get("per_item") for task in tasks
        ) and all("bits_per_byte" in result and result.get("tokens", 0) > 0 for result in bpb)
    except (OSError, ValueError, TypeError, AttributeError):
        return False
