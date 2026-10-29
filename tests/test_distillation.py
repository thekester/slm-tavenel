import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from slm import train as train_module
from slm.config import TrainConfig, load_model_config, load_train_config
from slm.model import SLM, chunked_distillation_loss

from conftest import tiny_config


def test_chunked_distillation_matches_direct_computation():
    torch.manual_seed(0)
    h, weight = torch.randn(2, 9, 16), torch.randn(97, 16)
    targets = torch.randint(0, 97, (2, 9))
    targets[0, :3] = -100
    teacher = torch.randn(2, 9, 97)
    ce, kl = chunked_distillation_loss(h, weight, targets, teacher, temperature=2.0, chunk=4)
    logits = (h @ weight.T).reshape(-1, 97)
    flat_t = targets.reshape(-1)
    mask = flat_t != -100
    expected_ce = F.cross_entropy(logits, flat_t, ignore_index=-100)
    s = F.log_softmax(logits / 2.0, -1)
    t = F.log_softmax(teacher.reshape(-1, 97) / 2.0, -1)
    expected_kl = (t.exp() * (t - s)).sum(-1)[mask].mean()
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(kl, expected_kl)


def test_self_distillation_has_zero_kl_and_mixes_losses():
    torch.manual_seed(0)
    model = SLM(tiny_config()).train()
    x, y = torch.randint(0, 97, (2, 12)), torch.randint(0, 97, (2, 12))
    with torch.no_grad():
        own_logits, _ = model(x)
    _, plain = model(x, y, return_logits=False)
    _, mixed = model(x, y, return_logits=False, teacher_logits=own_logits, distill_weight=0.25)
    assert model.last_loss_parts["kl"].abs().item() < 1e-5
    torch.testing.assert_close(mixed, 0.75 * plain, rtol=1e-5, atol=1e-6)
    mixed.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_distillation_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(distill_teacher="org/model")
    with pytest.raises(ValueError):
        TrainConfig(distill_teacher="org/model", distill_revision="abc", distill_weight=1.5)


class _FakeTeacher(torch.nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.vocab = vocab

    def forward(self, input_ids, **kwargs):
        gen = torch.Generator().manual_seed(int(input_ids.sum()))
        return SimpleNamespace(logits=torch.randn(*input_ids.shape, self.vocab, generator=gen) * self.scale)


def test_train_loop_logs_distillation(tmp_path, monkeypatch):
    base = load_train_config("configs/train/smoke.yaml")
    vocab = load_model_config(base.model).vocab_size
    monkeypatch.setattr(train_module, "load_teacher",
                        lambda cfg, device, vocab_size: _FakeTeacher(vocab).eval().requires_grad_(False))
    cfg = load_train_config("configs/train/smoke.yaml", {
        "out_dir": str(tmp_path), "max_steps": 2, "log_interval": 1, "eval_interval": 0, "observe_interval": 0,
        "ckpt_interval": 2, "distill_teacher": "fake/teacher", "distill_revision": "0" * 40,
        "distill_weight": 0.5, "distill_temperature": 2.0,
    })
    train_module.train(cfg)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records and all("distill_kl" in r and r["distill_kl"] > 0 for r in records)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["distillation"]["teacher"] == "fake/teacher"
