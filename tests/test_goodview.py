import json

import numpy as np
import pytest
import torch

from slm.config import TrainConfig
from slm.goodview import ParallelPairs, goodview_loss, masked_mean, parallel_ntp_loss
from slm.train import load_model, train
from test_jepa import _othello_shards


def test_goodview_loss_rewards_matching_views_and_penalizes_collapse():
    torch.manual_seed(0)
    z = torch.randn(256, 8)
    for variant in ("cov", "vicreg"):
        matched, _ = goodview_loss(z, z, gamma=1.0, variant=variant)
        unmatched, _ = goodview_loss(z, torch.randn(256, 8), gamma=1.0, variant=variant)
        collapsed, parts = goodview_loss(torch.zeros(256, 8), torch.zeros(256, 8), gamma=1.0, variant=variant)
        assert matched < unmatched
        assert collapsed > matched
        assert parts["gv_invariance"] == 0.0
    _, parts = goodview_loss(torch.zeros(256, 8), torch.zeros(256, 8), gamma=1.0, variant="cov")
    assert parts["gv_anticollapse"] == pytest.approx(1.0)  # ||0 - I||^2 / k


def test_vicreg_variant_pushes_away_from_collapse_but_cov_variant_does_not():
    """The v1 'cov' loss has a vanishing gradient at z = 0 (EXP-0043 v1 collapsed); the amended loss does not."""
    def gradient(variant: str, scale: float) -> float:
        torch.manual_seed(0)
        z = (scale * torch.randn(256, 8)).requires_grad_()  # identical views: only anti-collapse acts
        loss, _ = goodview_loss(z, z.detach(), gamma=1.0, variant=variant)
        loss.backward()
        return z.grad.norm().item()

    # Shrinking the spread 10x shrinks the 'cov' anti-collapse gradient ~10x (it vanishes at z = 0), while
    # the standard-deviation hinge keeps a scale-free gradient (above its sqrt(eps) = 0.01 floor).
    assert gradient("cov", 0.05) / gradient("cov", 0.5) < 0.2
    assert gradient("vicreg", 0.05) / gradient("vicreg", 0.5) > 0.8
    with pytest.raises(ValueError):
        goodview_loss(torch.zeros(4, 2), torch.zeros(4, 2), 1.0, variant="other")


def test_masked_mean_and_ntp_loss_ignore_padding():
    h = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert masked_mean(h, torch.tensor([2])).item() == 2.0
    logits = torch.zeros(1, 3, 5, requires_grad=True)
    ids = torch.tensor([[1, 2, 4]])
    short = parallel_ntp_loss(logits, ids, torch.tensor([2]))  # only the 1 -> 2 transition counts
    assert short.item() == pytest.approx(np.log(5))


def _pairs(path, n=40, width=6):
    rng = np.random.default_rng(0)
    path.mkdir()
    for split in ("train", "dev"):
        en = rng.integers(1, 60, size=(n, width)).astype(np.int32)
        fr = rng.integers(1, 60, size=(n, width)).astype(np.int32)
        np.savez(path / f"{split}.npz", en=en, fr=fr, en_len=np.full(n, width - 1, np.int32),
                 fr_len=np.full(n, width, np.int32))
    (path / "manifest.json").write_text(json.dumps({"pad_id": 0}))


@pytest.mark.parametrize("mode", ["goodview", "ntp"])
def test_train_with_parallel_objective(tmp_path, mode):
    data = tmp_path / "othello"
    _othello_shards(data)
    _pairs(tmp_path / "pairs")
    pairs = ParallelPairs(tmp_path / "pairs")
    en, en_len, fr, fr_len = pairs.sample(np.random.default_rng(1), 8)
    assert en.shape == (8, 5) and fr.shape == (8, 6)
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("vocab_size: 65\nd_model: 32\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nffn_hidden: 64\n"
                          "max_seq_len: 64\nattn_gate: headwise\n")
    out = tmp_path / "run"
    cfg = TrainConfig(model=str(model_yaml), data_dir=str(data), out_dir=str(out), seq_len=63, micro_batch_size=4,
                      grad_accum_steps=2, max_steps=3, log_interval=1, eval_interval=3, eval_batches=1,
                      observe_interval=0, ckpt_interval=3, dtype="float32", resume=False,
                      parallel_data=str(tmp_path / "pairs"), parallel_mode=mode, parallel_batch=8, goodview_dim=4)
    train(cfg)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert all(r["parallel_loss"] > 0 for r in records)
    assert all(("gv_anticollapse" in r) == (mode == "goodview") for r in records)
    model = load_model(out / "checkpoints" / "step_00000003.pt")
    assert not any(n.startswith("jepa_predictor") for n, _ in model.named_parameters())


def test_parallel_options_are_validated():
    with pytest.raises(ValueError):
        TrainConfig(parallel_mode="goodview")  # no data
    with pytest.raises(ValueError):
        TrainConfig(parallel_data="x", parallel_mode="other")
