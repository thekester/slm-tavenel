import pytest
import torch

from slm.grow import check_compatible, grow, grow_checkpoint
from slm.model import SLM
from slm.observe import capture

from conftest import tiny_config


@pytest.mark.parametrize("overrides", [{}, {"attn_gate": "headwise"}, {"qk_norm": False, "tie_embeddings": False}])
def test_growth_preserves_the_function(overrides):
    torch.manual_seed(0)
    src = SLM(tiny_config(**overrides)).eval()
    for p in src.parameters():  # move away from the initialisation so copies are non-trivial
        p.data.add_(0.05 * torch.randn_like(p))
    dst = grow(src, tiny_config(n_layers=6, ffn_hidden=96, **overrides)).eval()
    x = torch.randint(0, 97, (2, 12))
    torch.testing.assert_close(dst(x)[0], src(x)[0], atol=1e-5, rtol=1e-4)
    assert dst.num_params() > src.num_params()


def test_grown_blocks_learn_from_the_first_step():
    torch.manual_seed(0)
    src = SLM(tiny_config())
    dst = grow(src, tiny_config(n_layers=6, ffn_hidden=96))
    x = torch.randint(0, 97, (2, 12))
    _, loss = dst(x[:, :-1], x[:, 1:])
    loss.backward()
    new_block = dst.blocks[1]
    assert new_block.attn.o_proj.weight.abs().sum() == 0
    assert new_block.attn.o_proj.weight.grad.abs().sum() > 0
    assert new_block.mlp.down_proj.weight.grad.abs().sum() > 0
    assert dst.blocks[0].mlp.down_proj.weight.grad[:, 64:].abs().sum() > 0  # new MLP units


def test_new_blocks_are_identity_at_start():
    torch.manual_seed(0)
    dst = grow(SLM(tiny_config()), tiny_config(n_layers=6))
    x = torch.randint(0, 97, (1, 8))
    with capture(dst, ["blocks.1.hook_resid_pre", "blocks.1.hook_resid_post"]) as acts:
        dst(x)
    torch.testing.assert_close(acts["blocks.1.hook_resid_pre"], acts["blocks.1.hook_resid_post"])


def test_incompatible_growth_is_refused():
    with pytest.raises(ValueError):
        check_compatible(tiny_config(), tiny_config(d_model=64, n_heads=4))
    with pytest.raises(ValueError):
        check_compatible(tiny_config(), tiny_config(n_layers=5))
    with pytest.raises(ValueError):
        check_compatible(tiny_config(), tiny_config(n_layers=6, ffn_hidden=32))


def test_grow_checkpoint_and_init_from(tmp_path):
    from dataclasses import asdict

    from slm.config import ModelConfig

    torch.manual_seed(0)
    src = SLM(tiny_config())
    ckpt = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "model_config": asdict(src.cfg), "train_config": {"seed": 1}}, ckpt)
    info = grow_checkpoint(ckpt, tiny_config(n_layers=6), tmp_path / "grown.pt")
    assert info["depth_factor"] == 2 and info["source_sha256"]
    state = torch.load(tmp_path / "grown.pt", weights_only=False)
    model = SLM(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    x = torch.randint(0, 97, (1, 8))
    torch.testing.assert_close(model.eval()(x)[0], src.eval()(x)[0], atol=1e-5, rtol=1e-4)
