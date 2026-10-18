import pytest

from slm import trace


def test_trace_nesting_cost_and_failure_path(tmp_path):
    path = tmp_path / "trace.jsonl"
    t = trace.Tracer(path)
    with t.span("task", "rename files", task_id="t1") as root:
        with t.span("model_call", "plan") as s:
            s.cost.update(tokens_in=120, tokens_out=40)
        with pytest.raises(TimeoutError):
            with t.span("tool_call", "list_dir", path="/tmp") as tool:
                tool.cost["tool_calls"] = 1
                raise TimeoutError("tool did not answer")
        with t.span("verification", "check names") as v:
            v.set(passed=False)
        root.set(success=False)

    spans = trace.load(path)
    assert [s.kind for s in spans] == ["model_call", "tool_call", "verification", "task"]
    root_span = spans[-1]
    assert all(s.parent_id == root_span.span_id for s in spans[:-1])
    assert spans[1].status == "timeout"
    totals = trace.total_cost(spans)
    assert totals["tokens_in"] == 120 and totals["tool_calls"] == 1
    assert [s.name for s in trace.failure_path(spans, spans[1].span_id)] == ["rename files", "list_dir"]
