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


def test_master_offload_matches_fp32_adamw_and_keeps_fp32_masters():
    from slm.config import TrainConfig
    from slm.optim import MasterOffloadAdamW
    from slm.train import accumulate_gradients, build_model_and_optimizer, clip_gradients

    torch.manual_seed(0)
    reference = SLM(tiny_config())
    ref_opt = build_optimizer(reference, "adamw", 1e-2, 0.1, (0.9, 0.95), 1e-8)
    cfg = TrainConfig(optimizer="adamw", lr=1e-2, master_weights_offload=True)
    torch.manual_seed(0)
    model, opt = build_model_and_optimizer(tiny_config(), cfg, torch.device("cpu"))
    assert isinstance(opt, MasterOffloadAdamW)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.rope_cos.dtype == torch.float32
    names = {id(p): n for n, p in model.named_parameters()}
    ref = dict(reference.named_parameters())
    pairs = [(ref[names[id(p)]], h) for p, h in zip(opt.device_params, opt.host_params)]
    for a, h in pairs:
        torch.testing.assert_close(a.detach(), h)  # masters start from the exact fp32 initialization

    # Exact mechanics: feed the reference fp32 AdamW the same fp32 sum of the micro-batch gradients.
    gen = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(3):
        accumulated = {}
        for _ in range(2):
            x = torch.randint(0, 97, (2, 16), generator=gen)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, loss = model(x[:, :-1], x[:, 1:], return_logits=False)
            (loss / 2).backward()
            losses.append(loss.item())
            for p in model.parameters():
                accumulated[id(p)] = accumulated.get(id(p), 0) + p.grad.float()
            accumulate_gradients(opt)
        assert all(p.grad is None for p in model.parameters())
        for p, h in zip(opt.device_params, opt.host_params):
            torch.testing.assert_close(h.grad, accumulated[id(p)])  # fp32 accumulation on the host
        for p, _ in zip(opt.device_params, opt.host_params):
            ref[names[id(p)]].grad = accumulated[id(p)].clone()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        ref_opt.step()
        ref_opt.zero_grad(set_to_none=True)
        clip_gradients(model, opt, 1.0)
        opt.step()
        opt.zero_grad()
        for a, h in pairs:
            assert h.dtype == torch.float32
            torch.testing.assert_close(a.detach(), h)
    assert losses[-1] < losses[0]
    for p, h in zip(opt.device_params, opt.host_params):
        torch.testing.assert_close(p.detach(), h.to(torch.bfloat16))

    state = opt.state_dict()
    _, fresh = build_model_and_optimizer(tiny_config(), cfg, torch.device("cpu"))
    fresh.load_state_dict(state)
    for a, b in zip(opt.host_params, fresh.host_params):
        torch.testing.assert_close(a, b)


def test_master_offload_muon_matches_muon_on_same_gradients():
    from slm.config import TrainConfig
    from slm.optim import MasterOffloadAdamW
    from slm.train import accumulate_gradients, build_model_and_optimizer

    torch.manual_seed(0)
    reference = SLM(tiny_config())
    ref_opt = build_optimizer(reference, "muon", 1e-2, 0.1, (0.9, 0.95), 1e-8)
    cfg = TrainConfig(optimizer="muon", lr=1e-2, master_weights_offload=True)
    torch.manual_seed(0)
    model, opt = build_model_and_optimizer(tiny_config(), cfg, torch.device("cpu"))
    assert isinstance(opt, MasterOffloadAdamW) and opt.muon_groups
    names = {id(p): n for n, p in model.named_parameters()}
    ref = dict(reference.named_parameters())
    pairs = [(ref[names[id(p)]], h) for p, h in zip(opt.device_params, opt.host_params)]
    gen = torch.Generator().manual_seed(2)
    for _ in range(3):
        x = torch.randint(0, 97, (2, 16), generator=gen)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss = model(x[:, :-1], x[:, 1:], return_logits=False)
        loss.backward()
        grads = {id(p): p.grad.float() for p in model.parameters()}
        accumulate_gradients(opt)
        for p in opt.device_params:
            ref[names[id(p)]].grad = grads[id(p)].clone()
        for group in opt.param_groups + ref_opt.param_groups:
            group["lr"] = 1e-2
        ref_opt.step()
        ref_opt.zero_grad()
        opt.step()
        opt.zero_grad()
        for a, h in pairs:
            torch.testing.assert_close(a.detach(), h)
    state = opt.state_dict()
    _, fresh = build_model_and_optimizer(tiny_config(), cfg, torch.device("cpu"))
    fresh.load_state_dict(state)
    for a, b in zip(opt.muon_groups[0]["buffers"], fresh.muon_groups[0]["buffers"]):
        torch.testing.assert_close(a, b)


def test_offload_requires_adamw():
    with pytest.raises(ValueError):
        TrainConfig(optimizer="muon", optimizer_offload=True)
