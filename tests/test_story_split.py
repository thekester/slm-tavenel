"""Regression for the completion-family leakage found in the original story pilot."""
import importlib.util
import json
import sys
from pathlib import Path

import torch  # Before pyarrow on Windows.
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("story_split_v2", Path(__file__).parents[1]/"scripts/build_exp0045_simple_stories_v2.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_completion_group_retains_non_numeric_suffixes():
    assert builder.completion_group("abc-1") == builder.completion_group("abc-18") == "abc"
    assert builder.completion_group("abc-text") == "abc-text"
    assert builder.completion_group("abc") == "abc"


def test_actual_builder_splits_complete_families_and_deduplicates_ids(tmp_path, monkeypatch):
    train = next(f"train{i}" for i in range(1000) if builder.bucket(f"train{i}") >= 5)
    dev = next(f"dev{i}" for i in range(1000) if builder.bucket(f"dev{i}") < 5)
    rows = [{"story": f"Story {group} number {i}. " + "a little fox went home. "*8,
             "generation_id": f"{group}-{i}", "model": "unit-test-fixture"}
            for group in (train, dev) for i in range(2)]
    rows.append(rows[0].copy())
    source=tmp_path/'source.parquet'; out=tmp_path/'out'
    pq.write_table(pa.Table.from_pylist(rows), source)
    monkeypatch.setattr(sys, 'argv', ['build', '--parquet', str(source), '--out', str(out), '--max-train', '2'])
    builder.main()
    trace=[json.loads(s) for s in (out/'provenance.jsonl').read_text().splitlines()]
    assert len(trace)==4
    train_groups={r['completion_group'] for r in trace if r['split']=='train'}
    dev_groups={r['completion_group'] for r in trace if r['split']=='dev'}
    assert train_groups=={train} and dev_groups=={dev}
    assert not train_groups & dev_groups
