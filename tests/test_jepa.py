import json

import numpy as np
import torch

from slm.jepa import JEPAPredictor, jepa_loss, make_target, update_target
from slm.model import SLM
from slm.train import load_model, train
from slm.config import TrainConfig
from slm.worlds import othello

from conftest import tiny_config


def test_jepa_loss_trains_model_and_predictor_but_not_target():
    torch.manual_seed(0)
    model = SLM(tiny_config())
    model.keep_hidden = True
    predictor = JEPAPredictor(32, 97)
    target = make_target(model)
    x = torch.randint(1, 97, (2, 12))
    x[0, 8:] = 0
    model(x)
    with torch.no_grad():
        target(x)
    loss = jepa_loss(predictor, model.last_hidden, target.last_hidden, x, ignore_action=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embed.weight.grad is not None and predictor.action.weight.grad is not None
    assert all(p.grad is None for p in target.parameters())


def test_state_level_jepa_uses_action_position_and_next_state_block():
    from slm.jepa import state_jepa_loss

    torch.manual_seed(0)
    model = SLM(tiny_config(max_seq_len=16))
    model.keep_hidden = True
    predictor = JEPAPredictor(32, 97)
    target = make_target(model)
    x = torch.randint(1, 97, (3, 15))  # state of 7 tokens, action, next state of 7 tokens
    model(x)
    with torch.no_grad():
        target(x)
    loss = state_jepa_loss(predictor, model.last_hidden, target.last_hidden, x, state_tokens=7)
    loss.backward()
    assert torch.isfinite(loss) and predictor.net[0].weight.grad is not None
    assert model.embed.weight.grad is not None


def test_ema_update_moves_target_towards_model():
    torch.manual_seed(0)
    model = SLM(tiny_config())
    target = make_target(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = target.embed.weight.clone()
    update_target(target, model, decay=0.9)
    torch.testing.assert_close(target.embed.weight, before + 0.1)


def _othello_shards(path):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        tokens = othello.encode(othello.random_game(rng))
        rows.append(tokens + [othello.EOS] * (64 - len(tokens)))
    arr = np.array(rows, dtype=np.uint8).reshape(-1)
    path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        arr.tofile(path / f"{split}_00000.bin")
    manifest = {"dtype": "uint8", "eos_id": 0, "vocab_size": 65, "row_length": 64,
                "splits": {s: {"tokens": int(arr.size), "shards": [{"file": f"{s}_00000.bin", "tokens": int(arr.size),
                                                                     "sha256": "x"}]} for s in ("train", "val")}}
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_train_with_jepa_on_othello_rows(tmp_path):
    data = tmp_path / "othello"
    _othello_shards(data)
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("vocab_size: 65\nd_model: 32\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nffn_hidden: 64\n"
                          "max_seq_len: 64\nattn_gate: headwise\n")
    out = tmp_path / "run"
    cfg = TrainConfig(model=str(model_yaml), data_dir=str(data), out_dir=str(out), seq_len=63, micro_batch_size=4,
                      grad_accum_steps=2, max_steps=3, log_interval=1, eval_interval=3, eval_batches=1,
                      observe_interval=0, ckpt_interval=3, jepa_weight=1.0, dtype="float32", resume=False)
    train(cfg)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert all("jepa_loss" in r and r["jepa_loss"] > 0 for r in records)
    ckpt = out / "checkpoints" / "step_00000003.pt"
    assert "jepa_target" in torch.load(ckpt, map_location="cpu", weights_only=False)
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["jepa"]["predictor_params"] > 0
    model = load_model(ckpt)
    assert not any(n.startswith("jepa_predictor") for n, _ in model.named_parameters())

    import importlib.util
    spec = importlib.util.spec_from_file_location("othello_eval", "scripts/evaluate_othello_world_model.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    rng = np.random.default_rng(3)
    games = [np.array(othello.random_game(rng), dtype=np.uint8) for _ in range(6)]
    feats, labels, legal_rate = evaluation.activations(model.eval(), games, torch.device("cpu"), batch=4)
    assert len(feats) == 3 and feats[0].shape[0] == labels.shape[0] == sum(len(g) for g in games)
    assert 0.0 <= legal_rate <= 1.0
    probe = evaluation.fit_probe(feats[-1], labels, epochs=1)
    assert 0.0 <= evaluation.probe_accuracy(probe, feats[-1], labels) <= 1.0
