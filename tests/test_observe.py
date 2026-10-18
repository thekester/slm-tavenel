import math

import pytest
import torch

from slm import interp
from slm.observe import ablate_heads, capture, intervene, replace_with, resolve_sites, zero


def test_capture_sites(tiny_model):
    x = torch.randint(0, 97, (1, 6))
    with capture(tiny_model, "blocks.*.hook_resid_post") as acts:
        tiny_model(x)
    assert sorted(acts) == [f"blocks.{i}.hook_resid_post" for i in range(3)]
    assert acts["blocks.0.hook_resid_post"].shape == (1, 6, 32)
    with capture(tiny_model, "blocks.1.attn.hook_z") as acts:
        tiny_model(x)
    assert acts["blocks.1.attn.hook_z"].shape == (1, 6, 4, 8)


def test_hooks_are_removed(tiny_model):
    x = torch.randint(0, 97, (1, 6))
    base, _ = tiny_model(x)
    with intervene(tiny_model, {"blocks.0.hook_resid_post": zero()}):
        changed, _ = tiny_model(x)
    after, _ = tiny_model(x)
    assert not torch.allclose(base, changed)
    torch.testing.assert_close(base, after)


def test_pattern_requires_capture(tiny_model):
    with pytest.raises(RuntimeError):
        resolve_sites(tiny_model, "blocks.0.attn.hook_pattern")


def test_ablating_all_heads_equals_zero_attention_output(tiny_model):
    x = torch.randint(0, 97, (1, 6))
    with intervene(tiny_model, {"blocks.1.attn.hook_z": ablate_heads(range(4))}):
        a, _ = tiny_model(x)
    with intervene(tiny_model, {"blocks.1.hook_attn_out": zero()}):
        b, _ = tiny_model(x)
    torch.testing.assert_close(a, b)


def test_activation_patching_final_residual_restores_clean(tiny_model):
    clean = torch.randint(0, 97, (1, 8))
    corrupt = clean.clone()
    corrupt[0, 2] = (corrupt[0, 2] + 5) % 97
    metric = interp.target_logprob_metric(-1, int(clean[0, -1]))
    effects = interp.activation_patching(
        tiny_model, clean, corrupt, metric, ["blocks.2.hook_resid_post", "blocks.0.hook_resid_pre"]
    )
    assert effects["blocks.2.hook_resid_post"] == pytest.approx(1.0, abs=1e-4)
    # embeddings of the clean tokens at every position also fully restore the clean run
    assert effects["blocks.0.hook_resid_pre"] == pytest.approx(1.0, abs=1e-4)


def test_patching_with_wrong_shape_fails(tiny_model):
    x = torch.randint(0, 97, (1, 6))
    with pytest.raises(ValueError):
        with intervene(tiny_model, {"blocks.0.hook_resid_post": replace_with(torch.zeros(1, 5, 32))}):
            tiny_model(x)


def test_logit_lens_last_layer_matches_output(tiny_model):
    x = torch.randint(0, 97, (2, 7))
    targets = torch.randint(0, 97, (2, 7))
    lens = interp.logit_lens(tiny_model, x, targets)
    assert lens.shape == (4, 2, 7)
    logits, _ = tiny_model(x)
    expected = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    torch.testing.assert_close(lens[-1], expected)


def test_attention_statistics(tiny_model):
    x = torch.randint(0, 97, (2, 9))
    stats = interp.attention_statistics(tiny_model, x)
    assert stats["sink_mass"].shape == (3, 4)
    assert ((stats["sink_mass"] >= 0) & (stats["sink_mass"] <= 1)).all()
    assert (stats["entropy"] <= math.log(9) + 1e-5).all()
    assert not any(b.attn.capture_pattern for b in tiny_model.blocks)


def test_head_ablation_effects_shape(tiny_model):
    x = torch.randint(0, 97, (1, 6))
    eff = interp.head_ablation_effects(tiny_model, x, interp.target_logprob_metric(-1, 3), layers=[0])
    assert eff.shape == (3, 4)
    assert eff[1:].abs().sum() == 0
    assert eff[0].abs().sum() > 0
