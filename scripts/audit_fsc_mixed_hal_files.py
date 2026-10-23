"""Cross-check retained mixed-license FSC source rows against HAL file-license metadata.

The output is metadata and hashes only; it never contains article text and is not a legal opinion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://api.archives-ouvertes.fr/search/"
LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "Licence-Ouverte-2.0": "https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def query_hal(ids: list[str], timeout: int, retries: int) -> list[dict]:
    query = "halId_s:(" + " OR ".join(ids) + ")"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            records: list[dict] = []
            total: int | None = None
            while total is None or len(records) < total:
                params = urlencode({
                    "q": query,
                    "fl": "halId_s,version_i,licence_s,fileLicenses_s,fileAnnexeLicenses_s,uri_s,docType_s",
                    "wt": "json", "rows": str(len(ids)), "start": str(len(records)),
                })
                request = Request(f"{API}?{params}", headers={"User-Agent": "SLM-Tavenel mixed-source license audit/1.0"})
                with urlopen(request, timeout=timeout) as response:
                    payload = json.load(response)["response"]
                found = int(payload["numFound"])
                if total is not None and found != total:
                    raise ValueError("HAL result count changed during pagination")
                total = found
                page = payload["docs"]
                if len(records) + len(page) > total or (not page and len(records) < total):
                    raise ValueError("Incomplete or inconsistent HAL response")
                records.extend(page)
            return records
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"HAL API request failed after {retries} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="metadata-only retained source-record manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-pause", type=float, default=0.15)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100 or args.request_pause < 0:
        parser.error("--batch-size must be in [1,100] and --request-pause must be nonnegative")

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id: dict[str, dict] = {}
    for row in rows:
        hal_id = str(row.get("source_doc_id", "")).strip()
        declared = str(row.get("license_declared", "")).strip()
        if not hal_id or declared not in LICENSES:
            raise ValueError(f"Missing HAL source ID or unrecognized license label: {row.get('id')}")
        if hal_id in by_id:
            raise ValueError(f"Duplicate HAL source ID in retained manifest: {hal_id}")
        by_id[hal_id] = row

    ids = list(by_id)
    api_by_id: dict[str, list[dict]] = {hal_id: [] for hal_id in ids}
    batches = [ids[i:i + args.batch_size] for i in range(0, len(ids), args.batch_size)]
    for index, batch in enumerate(batches):
        for record in query_hal(batch, args.timeout, args.retries):
            hal_id = str(record.get("halId_s", ""))
            if hal_id in api_by_id:
                api_by_id[hal_id].append(record)
        if index % 25 == 0 or index + 1 == len(batches):
            print(f"audited {min((index + 1) * args.batch_size, len(ids))}/{len(ids)} source IDs", flush=True)
        if index + 1 < len(batches):
            time.sleep(args.request_pause)

    audits = []
    for hal_id, source in by_id.items():
        expected = LICENSES[source["license_declared"]]
        records = api_by_id[hal_id]
        main_licenses = sorted({str(r.get("licence_s") or "(missing)") for r in records})
        document_licenses = sorted({str(x) for r in records for x in (r.get("fileLicenses_s") or [])})
        annex_licenses = sorted({str(x) for r in records for x in (r.get("fileAnnexeLicenses_s") or [])})
        if not records:
            outcome = "not_found"
        else:
            observed = [
                str(value)
                for record in records
                for value in [record.get("licence_s"), *(record.get("fileLicenses_s") or [])]
                if value
            ]
            has_expected = any(value == expected for value in observed)
            has_mismatch = any(value != expected for value in observed)
            has_missing = any(
                not record.get("licence_s") or not record.get("fileLicenses_s")
                for record in records
            )
            if not has_mismatch and not has_missing:
                outcome = "all_versions_file_license_match"
            elif has_mismatch and has_expected:
                outcome = "license_conflict"
            elif has_mismatch:
                outcome = "license_mismatch"
            else:
                outcome = "missing_file_license_metadata"
        audits.append({
            "dataset_record_id": source.get("id"),
            "source_revision": source.get("source_revision"),
            "source_title": source.get("title"),
            "source_author_declared": source.get("author"),
            "source_doi_declared": source.get("doi"),
            "hal_id": hal_id,
            "hal_uri": sorted({r.get("uri_s") for r in records if r.get("uri_s")}),
            "dataset_license_declared": source["license_declared"],
            "expected_license_uri": expected,
            "hal_main_file_licenses": main_licenses,
            "hal_document_file_licenses": document_licenses,
            "hal_annex_file_licenses": annex_licenses,
            "hal_versions": len(records),
            "hal_version_details": [
                {
                    "uri": record.get("uri_s"),
                    "version": record.get("version_i"),
                    "document_type": record.get("docType_s"),
                    "main_file_license": record.get("licence_s"),
                    "document_file_licenses": record.get("fileLicenses_s") or [],
                    "annex_file_licenses": record.get("fileAnnexeLicenses_s") or [],
                }
                for record in records
            ],
            "outcome": outcome,
        })

    counts = Counter(r["outcome"] for r in audits)
    result = "PASS" if counts.get("all_versions_file_license_match", 0) == len(rows) else "BLOCKED"
    report = {
        "kind": "slm.fsc_mixed_hal_file_license_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PleIAs/French-Science-Commons",
        "dataset_revision": (next(iter({r.get("source_revision") for r in rows}))
                             if len({r.get("source_revision") for r in rows}) == 1 else None),
        "dataset_subset": "french_open_science",
        "input_manifest": str(args.manifest),
        "input_manifest_sha256": sha256(args.manifest),
        "source_rows": len(rows),
        "unique_hal_ids": len(ids),
        "hal_api_version_records": sum(len(record["hal_version_details"]) for record in audits),
        "source_license_counts": dict(Counter(r.get("license_declared") for r in rows)),
        "source_api": API,
        "source_api_query_fields": ["halId_s", "version_i", "licence_s", "fileLicenses_s", "fileAnnexeLicenses_s", "uri_s", "docType_s"],
        "field_interpretation": "HAL documents licence_s as the main-file license and fileLicenses_s as licenses for Document files. Every returned version must match the dataset-declared license in both fields. Outcomes separate explicit mismatches, mixed/conflicting values, missing fields and IDs not found.",
        "counts": dict(counts),
        "result": result,
        "limitations": [
            "HAL and dataset license fields are declarations and do not establish depositor authority, publisher agreements or copyright ownership.",
            "This audit compares HAL metadata as returned on the audit date; it is not a historical snapshot of source rights at training time.",
            "Privacy review, attribution compliance and model memorization are outside this license-field audit.",
        ],
        "records": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": result, "source_rows": len(rows), "counts": dict(counts), "report": str(args.output)}, indent=2))
    return 0 if result == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
