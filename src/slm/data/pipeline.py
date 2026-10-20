"""Provenance-tracked document pipeline.

Reads JSONL inputs declared against the source registry, applies configured steps
in order, and writes:

- ``documents.jsonl``: kept documents with source, revision and original-text hash;
- ``decisions.jsonl``: one line per removed document with the step and reason;
- ``stats.json``: counts per step and reason;
- ``manifest.json``: resolved configuration, input and output hashes, license
  decisions, code revision and software environment.

Example configuration: ``configs/data/pipeline-smoke.yaml``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm.config import config_hash, load_yaml
from slm.data import filters
from slm.data.contamination import ContaminationIndex
from slm.data.dedup import ExactDedup, MinHashLSH
from slm.data.pii import redact_doc
from slm.data.registry import Registry, evaluate
from slm.runlog import code_revision, environment_info, sha256_file, write_json

FILTERS = {
    "min_length": filters.min_length,
    "gopher_quality": filters.gopher_quality,
    "repetition": filters.repetition,
    "blocklist": filters.blocklist,
}


def open_text(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Step:
    """A pipeline step: returns (doc, None) to keep or (None, (reason, detail)) to drop."""

    name: str

    def __call__(self, doc: dict[str, Any]) -> tuple[dict[str, Any] | None, tuple[str, Any] | None]:
        raise NotImplementedError


class FilterStep(Step):
    def __init__(self, name: str, fn: filters.Filter) -> None:
        self.name, self.fn = name, fn

    def __call__(self, doc):
        reason = self.fn(doc)
        return (doc, None) if reason is None else (None, (reason, None))


class NormalizeStep(Step):
    name = "normalize"

    def __call__(self, doc):
        return filters.normalize(doc), None


class PiiStep(Step):
    name = "pii"

    def __init__(self) -> None:
        self.totals: Counter = Counter()

    def __call__(self, doc):
        doc, counts = redact_doc(doc)
        counts = {k: v for k, v in counts.items() if v}
        if counts:
            doc["pii_redactions"] = counts
            self.totals.update(counts)
        return doc, None


class ExactDedupStep(Step):
    name = "exact_dedup"

    def __init__(self) -> None:
        self.dedup = ExactDedup()

    def __call__(self, doc):
        earlier = self.dedup.check(doc["id"], doc["text"])
        return (doc, None) if earlier is None else (None, ("exact_duplicate", {"of": earlier}))


class MinHashStep(Step):
    name = "minhash_dedup"

    def __init__(self, max_words: int | None = None, **kwargs: Any) -> None:
        if max_words is not None and (not isinstance(max_words, int) or max_words < 1):
            raise ValueError("max_words must be a positive integer")
        self.lsh = MinHashLSH(**kwargs)
        self.max_words = max_words

    def __call__(self, doc):
        text = doc["text"]
        if self.max_words is not None:
            text = " ".join(text.split()[: self.max_words])
        hit = self.lsh.check(doc["id"], text)
        if hit is None:
            return doc, None
        return None, ("near_duplicate", {"of": hit[0], "jaccard_estimate": round(hit[1], 4)})


class DecontaminateStep(Step):
    name = "decontaminate"

    def __init__(self, eval_files: list[dict[str, Any]], n: int = 13, min_matches: int = 1) -> None:
        self.index = ContaminationIndex(n=n, min_matches=min_matches)
        self.eval_files = []
        for spec in eval_files:
            self.index.add_jsonl(spec["path"], spec.get("fields", ["text"]), spec.get("id_field", "id"))
            self.eval_files.append({**spec, "sha256": sha256_file(spec["path"])})

    def __call__(self, doc):
        hits = self.index.matches(doc["text"])
        if not hits:
            return doc, None
        top = sorted(hits.items(), key=lambda kv: -kv[1])[:5]
        return None, ("benchmark_overlap", {"items": dict(top)})


def build_step(spec: dict[str, Any]) -> Step:
    if len(spec) != 1:
        raise ValueError(f"each step must have exactly one key, got {list(spec)}")
    (kind, params), = spec.items()
    params = params or {}
    if kind in FILTERS:
        return FilterStep(kind, FILTERS[kind](**params))
    if kind == "normalize":
        return NormalizeStep()
    if kind == "pii":
        return PiiStep()
    if kind == "exact_dedup":
        return ExactDedupStep()
    if kind == "minhash_dedup":
        return MinHashStep(**params)
    if kind == "decontaminate":
        return DecontaminateStep(**params)
    raise ValueError(f"unknown step {kind!r}")


def run(config_path: str | Path, allow_unapproved: bool = False) -> dict[str, Any]:
    """Run a pipeline configuration and return its manifest."""
    cfg = load_yaml(config_path)
    registry = Registry.load(cfg["registry"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    for spec in cfg["inputs"]:
        reserved = {"id", "text", "source", "source_revision", "source_doc_id", "license", "input_sha256"}
        conflicts = reserved.intersection(spec.get("preserve_fields", []))
        if conflicts:
            raise ValueError(f"preserve_fields cannot overwrite pipeline fields: {sorted(conflicts)}")
        source = registry[spec["source"]]
        decision = evaluate(source)
        if not decision.usable and not allow_unapproved:
            raise PermissionError(
                f"source {source.id} is not usable for training: {'; '.join(decision.blockers)}. "
                "Use --allow-unapproved only for exploratory runs; it is recorded in the manifest."
            )
        inputs.append((spec, source, decision))

    steps = [build_step(s) for s in cfg.get("steps", [])]
    stats: dict[str, Any] = {"docs_in": 0, "docs_out": 0, "chars_in": 0, "chars_out": 0}
    dropped: dict[str, Counter] = defaultdict(Counter)
    per_source: dict[str, Counter] = defaultdict(Counter)

    docs_path, decisions_path = out_dir / "documents.jsonl", out_dir / "decisions.jsonl"
    with open(docs_path, "w", encoding="utf-8", newline="\n") as docs_f, open(
        decisions_path, "w", encoding="utf-8", newline="\n"
    ) as dec_f:
        for spec, source, _ in inputs:
            text_field, id_field = spec.get("text_field", "text"), spec.get("id_field", "id")
            for i, rec in enumerate(read_jsonl(spec["path"])):
                text = rec[text_field]
                src_id = str(rec.get(id_field, i))
                doc = {
                    "id": f"{source.id}/{src_id}",
                    "text": text,
                    "source": source.id,
                    "source_revision": source.revision,
                    "source_doc_id": src_id,
                    "license": source.license,
                    "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                for field in spec.get("preserve_fields", []):
                    doc[field] = rec.get(field)
                stats["docs_in"] += 1
                stats["chars_in"] += len(text)
                per_source[source.id]["in"] += 1
                for step in steps:
                    doc, drop = step(doc)
                    if drop is not None:
                        reason, detail = drop
                        dropped[step.name][reason] += 1
                        dec_f.write(json.dumps({"id": f"{source.id}/{src_id}", "step": step.name,
                                                "reason": reason, "detail": detail}, ensure_ascii=False) + "\n")
                        break
                else:
                    stats["docs_out"] += 1
                    stats["chars_out"] += len(doc["text"])
                    per_source[source.id]["out"] += 1
                    docs_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    stats["dropped"] = {k: dict(v) for k, v in dropped.items()}
    stats["per_source"] = {k: dict(v) for k, v in per_source.items()}
    pii = next((s for s in steps if isinstance(s, PiiStep)), None)
    if pii is not None:
        stats["pii_redactions"] = dict(pii.totals)
    write_json(out_dir / "stats.json", stats)

    decon = next((s for s in steps if isinstance(s, DecontaminateStep)), None)
    manifest = {
        "kind": "slm.data.pipeline",
        "name": cfg.get("name", Path(config_path).stem),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "allow_unapproved": allow_unapproved,
        "inputs": [
            {
                "source": source.id,
                "path": str(spec["path"]),
                "sha256": sha256_file(spec["path"]),
                "license": source.license,
                "revision": source.revision,
                "policy": decision.as_dict(),
            }
            for spec, source, decision in inputs
        ],
        "eval_files": decon.eval_files if decon else [],
        "outputs": {
            "documents.jsonl": sha256_file(docs_path),
            "decisions.jsonl": sha256_file(decisions_path),
        },
        "stats": stats,
        "code": code_revision(),
        "environment": environment_info(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
