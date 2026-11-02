import torch

from slm.hnet import HNet, HNetConfig, ema_over_chunks, ratio_loss
from slm.tokenizer import EOS, ByteTokenizer


def tiny() -> HNetConfig:
    return HNetConfig(max_seq_len=64, d_outer=32, d_main=48, encoder_layers=1, decoder_layers=1, outer_heads=4,
                      outer_kv_heads=2, outer_ffn=64, outer_window=8, main_layers=2, main_heads=4, main_kv_heads=2,
                      main_ffn=96, target_ratio=4.0)


def test_ema_matches_the_recurrence():
    torch.manual_seed(0)
    z = torch.randn(2, 12, 5)
    P = torch.rand(2, 12) * 0.5 + 0.5
    P[:, 0] = 1.0
    expected = torch.zeros_like(z)
    for t in range(12):
        prev = expected[:, t - 1] if t else torch.zeros_like(z[:, 0])
        expected[:, t] = P[:, t, None] * z[:, t] + (1 - P[:, t, None]) * prev
    torch.testing.assert_close(ema_over_chunks(z, P, window=12), expected, rtol=1e-5, atol=1e-6)
    # Truncation error is bounded by 2**-window because P >= 0.5.
    assert (ema_over_chunks(z, P, window=6) - expected).abs().max() < 2**-6 * z.abs().max() * 2


def test_ratio_loss_is_minimal_at_target_compression():
    target = 4.0
    at_target = ratio_loss(torch.tensor([True, False, False, False] * 8), torch.full((32,), 0.25), target)
    too_many = ratio_loss(torch.ones(32, dtype=torch.bool), torch.full((32,), 0.9), target)
    assert abs(float(at_target) - 1.0) < 1e-6 and float(too_many) > float(at_target)


def test_hnet_forward_backward_and_causality():
    torch.manual_seed(0)
    model = HNet(tiny()).train()
    x = torch.randint(1, 257, (2, 40))
    _, loss = model(x, torch.randint(1, 257, (2, 40)), return_logits=False)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.router_q.weight.grad is not None and model.main_blocks[0].attn.q_proj.weight.grad is not None
    assert 1.0 <= model.last_chunk_stats["bytes_per_chunk"] <= 40
    # Causality: changing a future byte must not change earlier logits.
    model.eval()
    with torch.no_grad():
        a, _ = model(x)
        y = x.clone()
        y[:, 30:] = (y[:, 30:] % 256) + 1
        b, _ = model(y)
    torch.testing.assert_close(a[:, :30], b[:, :30], rtol=1e-4, atol=1e-5)


def test_local_attention_matches_masked_attention():
    import torch.nn.functional as F
    from slm.model import attention_mask, local_attention

    torch.manual_seed(0)
    for T, window in ((37, 8), (64, 16), (50, 7)):
        q, k, v = (torch.randn(2, 3, T, 4) for _ in range(3))
        mask = attention_mask(T, T, 0, window, q.device)
        dense = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(local_attention(q, k, v, window), dense, rtol=1e-5, atol=1e-5)


def test_train_load_and_score_hnet(tmp_path):
    import json

    import numpy as np

    from slm.config import TrainConfig
    from slm.train import load_model, train

    text = ("Le chat dort. The cat sleeps. " * 400).encode("utf-8")
    arr = np.frombuffer(text, dtype=np.uint8).astype(np.uint16) + 1
    data = tmp_path / "bytes"
    data.mkdir()
    for split in ("train", "val"):
        arr.tofile(data / f"{split}_00000.bin")
    manifest = {"dtype": "uint16", "eos_id": 0, "vocab_size": 257,
                "splits": {s: {"tokens": int(arr.size), "shards": [{"file": f"{s}_00000.bin", "tokens": int(arr.size),
                                                                     "sha256": "x"}]} for s in ("train", "val")}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    model_yaml = tmp_path / "hnet.yaml"
    model_yaml.write_text("arch: hnet\nmax_seq_len: 64\nd_outer: 32\nd_main: 48\nencoder_layers: 1\ndecoder_layers: 1\n"
                          "outer_heads: 4\nouter_kv_heads: 2\nouter_ffn: 64\nouter_window: 8\nmain_layers: 2\n"
                          "main_heads: 4\nmain_kv_heads: 2\nmain_ffn: 96\ntarget_ratio: 4.0\n")
    out = tmp_path / "run"
    cfg = TrainConfig(model=str(model_yaml), data_dir=str(data), out_dir=str(out), seq_len=63, micro_batch_size=4,
                      grad_accum_steps=1, max_steps=3, log_interval=1, eval_interval=3, eval_batches=1,
                      observe_interval=0, ckpt_interval=3, optimizer="muon", dtype="float32", resume=False)
    train(cfg)
    record = json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])
    assert "bytes_per_chunk" in record and "ratio_loss" in record
    model = load_model(out / "checkpoints" / "step_00000003.pt")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", "scripts/evaluate_exp0023_50m.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    scores = ev.score_documents_batched(model, ByteTokenizer(), ["Le chat dort.", "x" * 150], torch.device("cpu"),
                                        batch_size=2)
    assert all(s["tokens"] == s["bytes"] and s["nll"] > 0 for s in scores)


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    ids = tok.encode("Été 2026").ids
    assert min(ids) >= 1 and tok.decode(ids) == "Été 2026" and tok.token_to_id(EOS) == 0
