import pytest
import torch

from slm.config import load_model_config
from slm.model import SLM, attention_mask

from conftest import tiny_config


def test_forward_shapes_and_loss(tiny_model):
    x = torch.randint(0, 97, (2, 10))
    logits, loss = tiny_model(x, x)
    assert logits.shape == (2, 10, 97)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_causality(tiny_model):
    x = torch.randint(0, 97, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % 97
    a, _ = tiny_model(x)
    b, _ = tiny_model(y)
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


@pytest.mark.parametrize("overrides", [{}, {"attn_gate": "headwise"}, {"attn_gate": "elementwise"},
                                       {"window_pattern": [4, 0]}, {"qk_norm": False, "n_kv_heads": 4}])
def test_kv_cache_matches_full_forward(overrides):
    torch.manual_seed(0)
    model = SLM(tiny_config(**overrides)).eval()
    x = torch.randint(0, 97, (2, 16))
    full, _ = model(x)
    out = model.generate(x[:, :5], max_new_tokens=0)
    assert out.shape == (2, 5)
    from slm.model import KVCache

    cache = KVCache.empty(model.cfg.n_layers)
    step_logits = [model(x[:, :5], cache=cache)[0]]
    for t in range(5, 16):
        step_logits.append(model(x[:, t : t + 1], cache=cache)[0])
    torch.testing.assert_close(torch.cat(step_logits, dim=1), full, atol=1e-5, rtol=1e-4)


def test_explicit_attention_matches_sdpa():
    torch.manual_seed(0)
    model = SLM(tiny_config(window_pattern=[3, 0])).eval()
    x = torch.randint(0, 97, (2, 12))
    fast, _ = model(x)
    model.set_attention_capture(True)
    slow, _ = model(x)
    torch.testing.assert_close(fast, slow, atol=1e-5, rtol=1e-4)


def test_sliding_window_mask():
    m = attention_mask(5, 5, 0, window=2, device=torch.device("cpu"))
    expected = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(m, expected)


def test_window_limits_receptive_field():
    torch.manual_seed(0)
    model = SLM(tiny_config(n_layers=1, window_pattern=[2])).eval()
    x = torch.randint(0, 97, (1, 10))
    y = x.clone()
    y[0, 0] = (y[0, 0] + 1) % 97
    a, _ = model(x)
    b, _ = model(y)
    # with one layer and window 2, position 5 cannot see position 0
    torch.testing.assert_close(a[:, 5], b[:, 5])


def test_greedy_generation_deterministic(tiny_model):
    x = torch.randint(0, 97, (1, 4))
    assert torch.equal(tiny_model.generate(x, 8), tiny_model.generate(x, 8))


@pytest.mark.parametrize("path,low,high", [
    ("configs/model/slm-135m.yaml", 125e6, 145e6),
    ("configs/model/slm-300m.yaml", 280e6, 320e6),
    ("configs/model/slm-1b.yaml", 0.95e9, 1.15e9),
])
def test_parameter_budgets(path, low, high):
    cfg = load_model_config(path)
    with torch.device("meta"):
        model = SLM(cfg)
    assert low <= model.num_params() <= high


def test_tied_embeddings(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.embed.weight
