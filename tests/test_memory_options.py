import pytest
import torch

from slm.config import TrainConfig
from slm.model import SLM
from slm.optim import CPUOffloadAdamW, build_optimizer
from slm.train import param_groups

from conftest import tiny_config


def _grads(model: SLM, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y, return_logits=False)
    loss.backward()
    return loss.detach(), [p.grad.clone() for p in model.parameters()]


def test_activation_checkpointing_preserves_loss_and_gradients():
    torch.manual_seed(0)
    model = SLM(tiny_config(attn_gate="headwise")).train()
    x, y = torch.randint(0, 97, (2, 24)), torch.randint(0, 97, (2, 24))
    loss_plain, grads_plain = _grads(model, x, y)
    model.activation_checkpointing = True
    loss_ckpt, grads_ckpt = _grads(model, x, y)
    torch.testing.assert_close(loss_ckpt, loss_plain)
    for a, b in zip(grads_plain, grads_ckpt):
        torch.testing.assert_close(a, b)


def test_activation_checkpointing_is_inactive_in_eval_mode():
    torch.manual_seed(0)
    model = SLM(tiny_config()).eval()
    model.activation_checkpointing = True
    x = torch.randint(0, 97, (1, 12))
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (1, 12, 97)


def _train_steps(model: SLM, opt, steps: int = 4) -> None:
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randint(0, 97, (2, 16), generator=gen)
        _, loss = model(x[:, :-1], x[:, 1:], return_logits=False)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


def test_offloaded_adamw_matches_adamw():
    torch.manual_seed(0)
    reference = SLM(tiny_config())
    offloaded = SLM(tiny_config())
    offloaded.load_state_dict(reference.state_dict())
    ref_opt = build_optimizer(reference, "adamw", 1e-2, 0.1, (0.9, 0.95), 1e-8)
    off_opt = build_optimizer(offloaded, "adamw", 1e-2, 0.1, (0.9, 0.95), 1e-8, offload=True)
    assert isinstance(off_opt, CPUOffloadAdamW)
    _train_steps(reference, ref_opt)
    _train_steps(offloaded, off_opt)
    for a, b in zip(reference.parameters(), offloaded.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_offloaded_adamw_state_roundtrip_and_lr_groups():
    torch.manual_seed(0)
    model = SLM(tiny_config())
    opt = CPUOffloadAdamW(param_groups(model, 0.1), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    _train_steps(model, opt, steps=2)
    for group in opt.param_groups:
        group["lr"] = 5e-4
    state = opt.state_dict()
    restored = CPUOffloadAdamW(param_groups(model, 0.1), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    restored.load_state_dict(state)
    assert [g["lr"] for g in restored.param_groups] == [5e-4, 5e-4]
    s0 = next(iter(opt.inner.state.values()))
    r0 = next(iter(restored.inner.state.values()))
    torch.testing.assert_close(s0["exp_avg"], r0["exp_avg"])
    with pytest.raises(ValueError):
        restored.load_state_dict({"state": {}, "param_groups": []})


def test_offload_requires_adamw():
    with pytest.raises(ValueError):
        TrainConfig(optimizer="muon", optimizer_offload=True)
