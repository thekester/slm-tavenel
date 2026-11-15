"""Guard against silent loss of documents or repeated HellaSwag item IDs."""
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
align = runpy.run_path(str(ROOT / 'scripts/compare_exp0034b_vs_124m_bpb.py'))['align_documents']
paired = runpy.run_path(str(ROOT / 'scripts/compare_mc_paired.py'))['paired']


def row(key, size=10, lang='en'):
    return {'text_sha256': key, 'bytes': size, 'language': lang, 'nll': 3.0}


def test_bpb_pairs_full_treatment_subset_and_checks_metadata():
    treatment = {'per_document': [row('b'), row('a')]}
    control = {'per_document': [row('a'), row('c'), row('b')]}
    assert [b['text_sha256'] for a, b in align(treatment, control)] == ['b', 'a']
    with pytest.raises(ValueError, match='match'):
        align(treatment, {'per_document': [row('a')]})
    with pytest.raises(ValueError, match='Duplicate'):
        align(treatment, {'per_document': [row('a'), row('b'), row('b')]})
    with pytest.raises(ValueError, match='byte counts'):
        align(treatment, {'per_document': [row('a', size=11), row('b')]})


def test_repeated_benchmark_ids_remain_distinct_observations():
    control = {'per_item': [{'id': '24', 'correct': False}, {'id': '24', 'correct': True}]}
    treatment = {'per_item': [{'id': '24', 'correct': True}, {'id': '24', 'correct': True}]}
    a, b = paired(control, treatment, 'correct')
    assert len(a) == len(b) == 2
    assert (b-a).mean() == .5
