"""Re-query only unresolved FSC source IDs from a prior HAL file-license audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_fsc_mixed_hal_files import LICENSES, query_hal


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(records: list[dict], expected: str) -> str:
    if not records:
        return "not_found"
    observed = [
        str(value)
        for record in records
        for value in [record.get("licence_s"), *(record.get("fileLicenses_s") or [])]
        if value
    ]
    has_expected = expected in observed
    has_mismatch = any(value != expected for value in observed)
    has_missing = any(not row.get("licence_s") or not row.get("fileLicenses_s") for row in records)
    if not has_mismatch and not has_missing:
        return "all_versions_file_license_match"
    if has_mismatch and has_expected:
        return "license_conflict"
    if has_mismatch:
        return "license_mismatch"
    return "missing_file_license_metadata"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior_audit", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-pause", type=float, default=0.25)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100 or args.request_pause < 0:
        parser.error("--batch-size must be in [1,100] and --request-pause nonnegative")

    prior = json.loads(args.prior_audit.read_text(encoding="utf-8"))
    source_rows = [json.loads(line) for line in args.source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_by_id = {str(row["source_doc_id"]): row for row in source_rows}
    unresolved = [row for row in prior["records"] if row["outcome"] != "all_versions_file_license_match"]
    unresolved_ids = [row["hal_id"] for row in unresolved]
    returned: dict[str, list[dict]] = {hal_id: [] for hal_id in unresolved_ids}

    batches = [unresolved_ids[i:i + args.batch_size] for i in range(0, len(unresolved_ids), args.batch_size)]
    for index, batch in enumerate(batches):
        for record in query_hal(batch, args.timeout, args.retries):
            hal_id = str(record.get("halId_s", ""))
            if hal_id in returned:
                returned[hal_id].append(record)
        print(f"rechecked {min((index + 1) * args.batch_size, len(unresolved_ids))}/{len(unresolved_ids)} IDs", flush=True)
        if index + 1 < len(batches):
            time.sleep(args.request_pause)

    records = []
    for prior_row in unresolved:
        hal_id = prior_row["hal_id"]
        source = source_by_id[hal_id]
        new_rows = returned[hal_id]
        expected = LICENSES[source["license_declared"]]
        outcome = classify(new_rows, expected)
        records.append({
            "dataset_record_id": source.get("id"),
            "hal_id": hal_id,
            "dataset_license_declared": source["license_declared"],
            "expected_license_uri": expected,
            "prior_outcome_2026_10_22": prior_row["outcome"],
            "current_outcome": outcome,
            "changed_outcome": outcome != prior_row["outcome"],
            "source_title": source.get("title"),
            "source_author_declared": source.get("author"),
            "source_doi_declared": source.get("doi"),
            "hal_version_details": [
                {
                    "uri": row.get("uri_s"),
                    "version": row.get("version_i"),
                    "document_type": row.get("docType_s"),
                    "main_file_license": row.get("licence_s"),
                    "document_file_licenses": row.get("fileLicenses_s") or [],
                    "annex_file_licenses": row.get("fileAnnexeLicenses_s") or [],
                }
                for row in new_rows
            ],
        })

    counts = Counter(row["current_outcome"] for row in records)
    changed = sum(row["changed_outcome"] for row in records)
    result = {
        "kind": "slm.fsc_hal_unresolved_record_recheck",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": prior["dataset"],
        "dataset_revision": prior["dataset_revision"],
        "dataset_subset": prior["dataset_subset"],
        "source_api": prior["source_api"],
        "prior_audit_path": str(args.prior_audit),
        "prior_audit_sha256": sha256(args.prior_audit),
        "source_manifest_path": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "prior_audit_date": prior["created_utc"],
        "rechecked_records": len(records),
        "changed_outcome_count": changed,
        "current_counts": dict(counts),
        "limitations": [
            "A current HAL metadata query is not a historical snapshot and does not establish depositor authority or publisher permissions.",
            "The prior audit compares all API-returned versions; the source manifest does not identify which exact file version produced each training text hash.",
            "This targeted query rechecks the previously unresolved IDs only; other prior matches were not re-queried.",
            "Privacy, attribution compliance, and model memorization are outside this audit.",
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rechecked_records": len(records), "changed_outcome_count": changed, "current_counts": dict(counts), "report": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
