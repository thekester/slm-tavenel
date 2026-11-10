"""Archive completed pilot scores and verify byte-weighted reconstruction.

Only hashes, source row IDs, likelihoods and training metrics are copied. No
corpus text, reserved-test scores or new model weights enter this evidence set.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import sha256_file, write_json


def main() -> None:
    source = ROOT / "runs/EXP-0047"
    target = ROOT / "docs/reviews/story-mixture-pilots-2026-11-10"
    target.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source / "selection.json").read_text())
    alignments, archive_files = {}, {}
    for key, result in selection["results"].items():
        folder = source / "dev-evaluations" / key
        for corpus, summary in result["corpora"].items():
            path = folder / f"{corpus}-items.jsonl"
            if sha256_file(path) != summary["items_sha256"]:
                raise ValueError(f"Item file changed: {path}")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            keys = [(r["index"], r["text_sha256"], r["bytes"]) for r in rows]
            if corpus in alignments and keys != alignments[corpus]:
                raise ValueError("Models were not scored on identical documents/order")
            alignments[corpus] = keys
            reconstructed = sum(r["nll"] for r in rows) / math.log(2) / sum(r["bytes"] for r in rows)
            if len(rows) != summary["documents"] or not math.isclose(reconstructed, summary["bits_per_byte"], rel_tol=1e-12):
                raise ValueError("Per-document evidence does not reconstruct summary")
            dest = target / f"{key}-{corpus}-items.jsonl.gz"
            dest.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            archive_files[dest.name] = {"sha256": sha256_file(dest), "uncompressed_sha256": sha256_file(path),
                                        "documents": len(rows), "reconstructed_bpb": reconstructed}
        if "100m" in key:
            for name in ("metrics.jsonl", "eval.jsonl"):
                src = source / key / name
                dest = target / f"{key}-{name}"
                shutil.copyfile(src, dest)
                archive_files[dest.name] = {"sha256": sha256_file(dest)}
    for name in ("selection.json", "frozen-inputs.json"):
        shutil.copyfile(source / name, target / name)
        archive_files[name] = {"sha256": sha256_file(target / name)}
    write_json(target / "archive.json", {
        "kind": "slm.exp0047.completed_100m_pilots", "files": archive_files,
        "archive_script_sha256": sha256_file(__file__),
        "scope": "512 development stories per corpus; one training seed; selection on dev; reference training exposure unknown; no human coherence claim",
        "training_code": "Original run's frozen-inputs.json retains dirty-worktree provenance; Git publication is subsequent to execution",
        "followup": "300M run was still in progress when this completed-pilot evidence was archived",
        "reserved_test": "unscored and not included", "new_weights_published": False,
    })
    print(f"Verified and archived {len(archive_files)} files in {target}")


if __name__ == "__main__":
    main()
