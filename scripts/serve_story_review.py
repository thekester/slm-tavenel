"""Serve a local human review form. No model identities or review key are exposed over HTTP.

Human ratings are collected, never generated. The source is the existing 30-output development pilot;
five prefixes and 15 paired comparisons do not constitute the planned 500-prefix confirmatory study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/model-cards/slm-tavenel-story-935k/training_record/generation-comparison/generations.jsonl"
HTML = ROOT / "docs/reviews/story-blind-pilot-2026-11-10/reviewer.html"
TRAITS = ("grammar", "coherence", "relevance")


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(directory: Path, source: Path = SOURCE) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if (directory / "meta.json").exists():
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta["source_sha256"] != digest:
            raise ValueError("Existing review bundle uses different outputs; choose a new directory")
        return meta
    grouped = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        grouped.setdefault((r["id"], r["temperature"], r["seed"]), []).append(r)
    cases, key = [], {}
    for rows in grouped.values():
        if len(rows) != 2 or len({r["model"] for r in rows}) != 2 or rows[0]["prompt"] != rows[1]["prompt"]:
            raise ValueError("Each case must contain exactly two different models on the same prompt")
        secrets.SystemRandom().shuffle(rows)
        case_id = secrets.token_hex(8)
        cases.append({"id": case_id, "prompt": rows[0]["prompt"],
                      "variants": {label: {"completion": r["completion"], "stop_reason": r["stop_reason"]}
                                   for label, r in zip(("A", "B"), rows)}})
        key[case_id] = {"prompt_id": rows[0]["id"], "temperature": rows[0]["temperature"],
                        "seed": rows[0]["seed"], "models": {label: r["model"] for label, r in zip(("A", "B"), rows)}}
    secrets.SystemRandom().shuffle(cases)
    meta = {"kind": "slm.human_blind_pilot", "source_sha256": digest, "pairs": len(cases),
            "unique_prefixes": len({k[0] for k in grouped}), "human_ratings_generated": False,
            "scope": "Previously public development outputs, not a sealed test; disclose prior exposure",
            "key_canonical_json_sha256": hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()}
    write(directory / "cases.json", cases)
    write(directory / "key.json", key)
    write(directory / "meta.json", meta)
    return meta


def rater_id(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError("Use a pseudonym of 1-40 letters, digits, underscores or hyphens")
    return value


def validate_rating(data: dict, case_ids: set[str]) -> dict:
    rater_id(data.get("rater"))
    if data.get("case_id") not in case_ids or data.get("human") is not True or type(data.get("blind")) is not bool:
        raise ValueError("Valid case and explicit human/blinding declarations are required")
    if data.get("preference") not in {"A", "B", "tie"}:
        raise ValueError("Choose A, B or tie")
    for label in ("A", "B"):
        for trait in TRAITS:
            score = data.get("ratings", {}).get(label, {}).get(trait)
            if type(score) is not int or not 0 <= score <= 4:
                raise ValueError("All six trait scores must be integers from 0 to 4")
    if not isinstance(data.get("notes", ""), str) or len(data.get("notes", "")) > 2000:
        raise ValueError("Notes must be at most 2,000 characters")
    return {k: data[k] for k in ("case_id", "human", "blind", "preference", "ratings")} | {
        "notes": data.get("notes", ""), "saved_at": time.time()}


def handler(directory: Path, token: str):
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    case_ids = {r["id"] for r in cases}

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = HTML.read_text(encoding="utf-8").replace("__TOKEN__", token).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            elif url.path == "/api/cases":
                self.send_json({"cases": cases})
            elif url.path == "/api/state":
                try:
                    who = rater_id(parse_qs(url.query).get("rater", [""])[0])
                    path = directory / f"ratings-{who}.json"
                    self.send_json(json.loads(path.read_text(encoding="utf-8")) if path.exists()
                                   else {"rater": who, "ratings": {}, "finished": False})
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, 400)
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            if (self.headers.get("X-Review-Token") != token or
                    self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                return self.send_json({"error": "Invalid local form request"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                who = rater_id(data.get("rater")); path = directory / f"ratings-{who}.json"
                state = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                         else {"rater": who, "ratings": {}, "finished": False})
                if self.path == "/api/rating":
                    rating = validate_rating(data, case_ids)
                    state["ratings"][data["case_id"]] = rating
                    state["finished"] = False
                elif self.path == "/api/finish":
                    if set(state["ratings"]) != case_ids:
                        raise ValueError("Rate all cases before finishing")
                    state["finished"] = True; state["finished_at"] = time.time()
                else:
                    return self.send_json({"error": "Not found"}, 404)
                state["reviewer_type"] = "human_self_declared"
                write(path, state)
                self.send_json({"saved": len(state["ratings"]), "finished": state["finished"]})
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "runs/EXP-0045/human-blind-pilot-2026-11-10")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    meta = prepare(args.directory)
    server = HTTPServer(("127.0.0.1", args.port), handler(args.directory, secrets.token_urlsafe(24)))
    url = f"http://127.0.0.1:{server.server_port}"
    write(args.directory / "server.json", {"url": url, "pairs": meta["pairs"], "bound_to": "127.0.0.1"})
    print(url, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
