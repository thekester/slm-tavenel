"""Reconstruct the 362M/124M BPB comparison without dropping repeated benchmark IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def align_documents(big: dict, small: dict) -> list[tuple[dict, dict]]:
    def index(report):
        result = {}
        for row in report['per_document']:
            key = row['text_sha256']
            if key in result:
                raise ValueError('Duplicate document hash; refusing ambiguous pairing')
            result[key] = row
        return result
    a, b = index(big), index(small)
    if not a or not a.keys() <= b.keys():
        raise ValueError('Every 362M document must have exactly one 124M match')
    pairs = [(row, b[key]) for key, row in a.items()]
    if any(x['bytes'] != y['bytes'] or x['language'] != y['language'] for x, y in pairs):
        raise ValueError('Paired document byte counts or languages differ')
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261115)
    args = parser.parse_args()
    folder = ROOT / 'runs/FAMILY-evaluations/360m-muon'
    paths = {'bpb_362m': folder / 'edu-test-bpb.json',
             'bpb_124m': ROOT / 'runs/EXP-0036-evaluations/124m-gpt2match-muon/edu-test-bpb.json',
             'benchmark_comparison': folder / 'vs-124m-paired.json'}
    reports = {key: json.loads(path.read_text(encoding='utf-8')) for key, path in paths.items()}
    pairs = align_documents(reports['bpb_362m'], reports['bpb_124m'])
    rng = np.random.default_rng(args.seed)
    out = {'kind': 'slm.exp0034b.paired_bpb', 'seed': args.seed, 'draws': args.draws,
           'inputs': {key: {'path': str(path.relative_to(ROOT)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                      for key, path in paths.items()},
           'bpb': {}, 'benchmark_comparison': reports['benchmark_comparison'],
           'document_coverage': {'362m': len(reports['bpb_362m']['per_document']),
                                 '124m': len(reports['bpb_124m']['per_document']),
                                 'paired': len(pairs), 'scope': 'all 362M documents; subset of full 124M report'},
           'limitations': ['Document bootstrap ignores domain clusters and training-seed variance.',
                          'Benchmark results reuse positional pairing in compare_mc_paired.py; repeated IDs are retained.']}
    for lang in ('all', 'en', 'fr'):
        rows = [p for p in pairs if lang == 'all' or p[0]['language'] == lang]
        nll = np.array([[x['nll'], y['nll']] for x, y in rows], dtype=np.float64)
        sizes = np.array([x['bytes'] for x, _ in rows], dtype=np.float64)
        if len(rows) == 0 or np.any(sizes <= 0):
            raise ValueError('Empty or invalid BPB stratum')
        scores = nll.sum(0) / math.log(2) / sizes.sum()
        diffs = []
        for start in range(0, args.draws, 32):
            ids = rng.integers(0, len(rows), size=(min(32, args.draws-start), len(rows)))
            sampled = nll[ids].sum(1) / math.log(2) / sizes[ids].sum(1)[:, None]
            diffs.extend(sampled[:, 0] - sampled[:, 1])
        out['bpb'][lang] = {'docs': len(rows), 'bpb_362m': float(scores[0]),
                             'bpb_124m': float(scores[1]), 'diff': float(scores[0]-scores[1]),
                             'ci95': np.quantile(diffs, [.025, .975]).tolist()}
    (folder / 'paired-vs-124m.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out['bpb'], indent=2))


if __name__ == '__main__':
    main()
