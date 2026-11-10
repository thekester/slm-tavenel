"""Protect byte-weighted uncertainty and the per-document scoring decomposition used in the report."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slm.evaluation import bits_per_byte

spec = importlib.util.spec_from_file_location("story_eval", Path(__file__).parents[1]/"scripts/evaluate_story_tinystories.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_bootstrap_is_paired_and_byte_weighted():
    byte_counts = np.array([1, 100])
    nll = np.array([10, 100], dtype=float)
    result = module.paired_bootstrap(nll, nll.copy(), byte_counts, draws=300)
    assert result["story_minus_tiny_ci95"] == [0, 0]
    doubled = module.paired_bootstrap(nll, nll*2, byte_counts, draws=300)
    assert doubled["story_minus_tiny_ci95"][1] < 0
    assert module.text_hash(" A  B\n", True) == module.text_hash("a b", True)


def test_item_nlls_reconstruct_the_existing_aggregate_evaluator():
    class Tok:
        def token_to_id(self, token): return 0
        def encode(self, text): return SimpleNamespace(ids=[1]*len(text))

    class Model(torch.nn.Module):
        cfg = SimpleNamespace(max_seq_len=3)
        def forward(self, x): return torch.zeros((*x.shape, 4)), None

    texts = ["a", "élong", "another longer text"]
    full = bits_per_byte(Model(), Tok(), texts, torch.device("cpu"), max_len=3)
    individual = [bits_per_byte(Model(), Tok(), [t], torch.device("cpu"), max_len=3) for t in texts]
    total_nll = sum(r["nll_per_token"]*r["tokens"] for r in individual)
    assert full["bits_per_byte"] == pytest.approx(total_nll/math.log(2)/sum(r["bytes"] for r in individual))
    assert full["bytes"] == sum(len(t.encode()) for t in texts)
