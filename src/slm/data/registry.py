"""Source registry and license policy.

Every dataset that may enter training is declared in ``configs/data/sources.yaml``
with its declared license, pinned revision and review record. The policy below
decides whether a source may be used in the core corpus. It encodes the README's
requirements: identifiable sources, recorded permissions, no NonCommercial or
NoDerivatives restrictions, and a completed review before inclusion.

The policy is a gate, not legal advice: a source passing it still needs the human
review recorded in its ``review`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm.config import from_dict, load_yaml

# Licenses compatible with redistribution of derived weights and data, including
# commercial use. Identifiers follow SPDX where one exists.
PERMISSIVE = {
    "CC0-1.0",
    "PDDL-1.0",
    "public-domain",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "ISC",
    "Unlicense",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "ODC-By-1.0",
}
# Allowed, but carry obligations (share-alike) that must be documented in the release.
SHARE_ALIKE = {"CC-BY-SA-3.0", "CC-BY-SA-4.0", "ODbL-1.0", "GFDL-1.3"}
FORBIDDEN_MARKERS = ("-NC", "-ND", "NonCommercial", "NoDerivatives")

STATUSES = ("candidate", "approved", "rejected", "comparison-only")
ROLES = ("pretrain", "posttrain", "eval", "tokenizer")


@dataclass
class Review:
    provenance_verified: bool = False
    license_verified: bool = False
    pii_reviewed: bool = False
    reviewer: str | None = None
    date: str | None = None
    notes: str = ""


@dataclass
class Source:
    id: str
    name: str
    url: str
    role: str
    license: str
    status: str = "candidate"
    license_url: str | None = None
    revision: str | None = None
    subset: str | None = None
    attribution: str | None = None
    languages: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    # For aggregated collections: the licenses of the components actually used.
    component_licenses: list[str] = field(default_factory=list)
    review: Review = field(default_factory=Review)
    open_questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.review, dict):
            self.review = from_dict(Review, self.review)
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: unknown status {self.status!r}")
        if self.role not in ROLES:
            raise ValueError(f"{self.id}: unknown role {self.role!r}")


@dataclass
class Decision:
    source_id: str
    usable: bool
    obligations: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "usable": self.usable,
            "obligations": self.obligations,
            "blockers": self.blockers,
        }


def classify_license(license_id: str) -> str:
    """Return 'permissive', 'share-alike', 'forbidden' or 'unknown'."""
    if any(m.lower() in license_id.lower() for m in FORBIDDEN_MARKERS):
        return "forbidden"
    if license_id in PERMISSIVE:
        return "permissive"
    if license_id in SHARE_ALIKE:
        return "share-alike"
    return "unknown"


def evaluate(source: Source) -> Decision:
    """Decide whether a source may be used in the core training corpus."""
    blockers: list[str] = []
    obligations: list[str] = []

    licenses = source.component_licenses or [source.license]
    if source.license == "mixed" and not source.component_licenses:
        blockers.append("license is 'mixed' but component_licenses are not listed")
    for lic in licenses:
        kind = classify_license(lic)
        if kind == "forbidden":
            blockers.append(f"license {lic} has NonCommercial or NoDerivatives terms")
        elif kind == "unknown":
            blockers.append(f"license {lic} is not on the reviewed allow-list")
        elif kind == "share-alike":
            obligations.append(f"{lic}: share-alike terms apply to adapted material")
        if lic.startswith(("CC-BY", "ODC-By", "ODbL")):
            obligations.append(f"{lic}: attribution required")

    if not source.revision:
        blockers.append("no pinned revision (results would not be reproducible)")
    if not source.review.provenance_verified:
        blockers.append("provenance not verified")
    if not source.review.license_verified:
        blockers.append("license not verified by a reviewer")
    if source.role in ("pretrain", "posttrain") and not source.review.pii_reviewed:
        blockers.append("privacy review not completed")
    if source.status in ("rejected", "comparison-only"):
        blockers.append(f"status is {source.status}")

    return Decision(source.id, usable=not blockers, obligations=sorted(set(obligations)), blockers=blockers)


class Registry:
    def __init__(self, sources: list[Source]) -> None:
        ids = [s.id for s in sources]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate source ids: {sorted(dupes)}")
        self.sources = {s.id: s for s in sources}

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        data = load_yaml(path)
        return cls([from_dict(Source, s) for s in data.get("sources", [])])

    def __getitem__(self, source_id: str) -> Source:
        if source_id not in self.sources:
            raise KeyError(f"source {source_id!r} is not declared in the registry")
        return self.sources[source_id]

    def audit(self) -> tuple[list[Decision], list[str]]:
        """Evaluate every source. Errors: sources marked approved that fail the policy."""
        decisions = [evaluate(s) for s in self.sources.values()]
        errors = [
            f"{d.source_id} is marked approved but blocked: {'; '.join(d.blockers)}"
            for d in decisions
            if self.sources[d.source_id].status == "approved" and not d.usable
        ]
        return decisions, errors
