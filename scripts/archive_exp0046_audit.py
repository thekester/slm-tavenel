"""Copy completed EXP-0046 audit evidence without images, captions or model weights."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import sha256_file, write_json


def main() -> None:
    local = ROOT / "runs/EXP-0046-audit-2026-11-11"
    target = ROOT / "docs/reviews/exp0046-audit-2026-11-11"
    report = json.loads((local / "report.json").read_text())
    if report["audit_script_sha256"] != sha256_file(ROOT / "scripts/audit_exp0046.py"):
        raise ValueError("Executed audit source changed; preserve its original bytes before archiving")
    if report["original_result_sha256"] != sha256_file(ROOT / "runs/EXP-0046-evaluations/text-image-alignment.json"):
        raise ValueError("Original result changed")
    target.mkdir(parents=True, exist_ok=True)
    sources = {name: local / name for name in ("report.json", "data-audit.json", "dino-anchor-audit.json",
                                               "training-and-log-audit.json", "J-primary-items.npy", "N-primary-items.npy", "N-first-layer-check.json")}
    sources["executed-audit.py"] = ROOT / "scripts/audit_exp0046.py"
    sources["original-feature-extraction-audit.py"] = local / "original-inputs/audit_exp0046-r5.py"
    sources["unbatched-first-layer-check.py"] = ROOT / "scripts/check_exp0046_first_layer.py"
    for name in ("train_text_jepa.py", "evaluate_text_image_alignment.py", "build_coco_caption_pairs.py",
                 "run_exp0046_queue.ps1", "text-image-alignment.json"):
        sources[f"observed-original-{name}"] = local / "original-inputs" / name
    for name, mode in (("J", "jepa"), ("N", "ntp")):
        sources[f"{name}-training-metrics.jsonl"] = ROOT / f"runs/EXP-0046-124m-{mode}/metrics.jsonl"
        sources[f"{name}-feature-metadata.json"] = local / mode / "metadata.json"
    evidence = {}
    for name, source in sources.items():
        dest = target / name
        shutil.copyfile(source, dest)
        evidence[name] = {"sha256": sha256_file(dest), "bytes": dest.stat().st_size}
    write_json(target / "archive.json", {"kind": "slm.exp0046.audit_evidence", "files": evidence,
               "archive_script_sha256": sha256_file(__file__),
               "scope": "Observed original artifacts, fresh J/N CPU re-extraction and post-hoc diagnostics; not independent training replication",
               "original_code": "Observed at audit start. No original immutable execution manifest exists; these copies do not retroactively establish original runtime bytes.",
               "large_local_artifacts": "Checkpoint weights and all-layer text features remain in runs/. Image files and caption text are not copied."})
    print(f"Archived {len(evidence)} verified audit files: {target}")


if __name__ == "__main__":
    main()
