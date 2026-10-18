"""Structured agent traces.

A trace links a task to everything that happened while solving it: model calls,
tool calls, memory reads and writes, decisions, verification results, errors, and
their costs. Spans nest through ``parent_id`` and are written as JSONL, one span
per line, so a failure can be located by walking from a failed verification back
to the tool result or decision that caused it. Field names follow OpenTelemetry
span conventions (trace id, span id, parent, start/end in nanoseconds,
attributes) so traces can be forwarded to standard observability backends.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("task", "model_call", "tool_call", "memory_read", "memory_write", "decision", "verification", "environment_change", "error")


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str
    name: str
    start_ns: int
    end_ns: int | None = None
    status: str = "ok"  # ok | error | timeout
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    # Costs: tokens_in, tokens_out, latency_ms, energy_j, tool_calls, retries
    cost: dict[str, float] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def set(self, **outputs: Any) -> None:
        self.outputs.update(outputs)


class Tracer:
    def __init__(self, path: str | Path, trace_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_id = trace_id or uuid.uuid4().hex
        self._stack: list[str] = []

    @contextmanager
    def span(self, kind: str, name: str, **inputs: Any) -> Iterator[Span]:
        if kind not in KINDS:
            raise ValueError(f"unknown span kind {kind!r}")
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=self._stack[-1] if self._stack else None,
            kind=kind,
            name=name,
            start_ns=time.time_ns(),
            inputs=inputs,
        )
        self._stack.append(span.span_id)
        t0 = time.perf_counter()
        try:
            yield span
        except TimeoutError as e:
            span.status = "timeout"
            span.attributes["exception"] = repr(e)
            raise
        except Exception as e:
            span.status = "error"
            span.attributes["exception"] = repr(e)
            raise
        finally:
            self._stack.pop()
            span.end_ns = time.time_ns()
            span.cost.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 3))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(span), ensure_ascii=False, default=str) + "\n")


def load(path: str | Path) -> list[Span]:
    with open(path, encoding="utf-8") as f:
        return [Span(**json.loads(line)) for line in f if line.strip()]


def total_cost(spans: list[Span], trace_id: str | None = None) -> dict[str, float]:
    """Sum costs over spans; latency is taken from root spans only to avoid double counting."""
    totals: dict[str, float] = {}
    for s in spans:
        if trace_id and s.trace_id != trace_id:
            continue
        for k, v in s.cost.items():
            if k == "latency_ms" and s.parent_id is not None:
                continue
            totals[k] = totals.get(k, 0.0) + v
    return totals


def failure_path(spans: list[Span], span_id: str) -> list[Span]:
    """Ancestors of a span, root first: the context in which a failure happened."""
    by_id = {s.span_id: s for s in spans}
    path = []
    cur = by_id.get(span_id)
    while cur is not None:
        path.append(cur)
        cur = by_id.get(cur.parent_id) if cur.parent_id else None
    return path[::-1]
