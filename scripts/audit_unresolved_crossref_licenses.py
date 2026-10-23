"""Collect Crossref-deposited license metadata for unresolved FSC records with DOIs.

This is a discovery aid only: Crossref metadata is not an authoritative legal clearance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API = "https://api.crossref.org/works/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_work(doi: str, timeout: int, retries: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(
                API + quote(doi, safe=""),
                headers={"User-Agent": "SLM-Tavenel provenance audit/1.0"},
            )
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            return payload.get("message", {})
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code == 404:
                break
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    return {"_lookup_error": str(last_error) if last_error else "not found"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hal_recheck", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-pause", type=float, default=0.2)
    args = parser.parse_args()

    report = json.loads(args.hal_recheck.read_text(encoding="utf-8"))
    rows = [row for row in report["records"] if row.get("source_doi_declared")]
    results = []
    for index, row in enumerate(rows, 1):
        doi = row["source_doi_declared"]
        message = get_work(doi, args.timeout, args.retries)
        licenses = message.get("license") or []
        results.append({
            "dataset_record_id": row["dataset_record_id"],
            "hal_id": row["hal_id"],
            "doi": doi,
            "dataset_license_declared": row["dataset_license_declared"],
            "hal_current_outcome": row["current_outcome"],
            "crossref_title": (message.get("title") or [None])[0],
            "crossref_publisher": message.get("publisher"),
            "crossref_license_declarations": [
                {
                    "url": item.get("URL"),
                    "content_version": item.get("content-version"),
                    "applies_to": item.get("applies-to"),
                    "start": item.get("start"),
                }
                for item in licenses
            ],
            "lookup_error": message.get("_lookup_error"),
        })
        if index % 20 == 0 or index == len(rows):
            print(f"looked up {index}/{len(rows)} DOIs", flush=True)
        if index < len(rows):
            time.sleep(args.request_pause)

    result = {
        "kind": "slm.fsc_unresolved_crossref_license_discovery",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_api": API,
        "input_recheck": str(args.hal_recheck),
        "input_recheck_sha256": sha256(args.hal_recheck),
        "doi_records_queried": len(results),
        "records_with_crossref_license_metadata": sum(bool(row["crossref_license_declarations"]) for row in results),
        "records_without_crossref_license_metadata": sum(not row["crossref_license_declarations"] and not row["lookup_error"] for row in results),
        "lookup_errors": sum(bool(row["lookup_error"]) for row in results),
        "limitations": [
            "Crossref license fields are publisher-deposited bibliographic metadata and may be incomplete, stale, or apply to a different version.",
            "This report is a discovery aid; it does not verify rights-holder authority, the exact file used for training, or reuse permission for model weights.",
            "Records without DOIs are outside this lookup.",
        ],
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("doi_records_queried", "records_with_crossref_license_metadata", "records_without_crossref_license_metadata", "lookup_errors")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
