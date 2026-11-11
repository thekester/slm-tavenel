"""Protect the metric, caption-pair sampling and custom NTP gradient used in EXP-0046."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slm.config import ModelConfig
from slm.goodview import parallel_ntp_loss
from slm.model import SLM


def script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_primary_overlap_matches_sets_and_excludes_self():
    audit = script("audit_exp0046")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(80, 16)), rng.normal(size=(80, 20))
    na, nb = audit.knn(a), audit.knn(b)
    assert all(i not in neighbours for i, neighbours in enumerate(na))
    assert np.all(audit.overlap(na, na) == 1)
    brute = np.array([len(set(x) & set(y))/10 for x, y in zip(na, nb)])
    assert np.array_equal(audit.overlap(na, nb), brute)
    assert np.array_equal(audit.overlap(na, nb), audit.evaluator.mutual_knn_items(a, b))
    with pytest.raises(ValueError):
        audit.knn(np.ones((5, 3)), k=10)


def test_graph_permutation_is_equivalent_to_permuting_embeddings():
    audit = script("audit_exp0046")
    rng = np.random.default_rng(4)
    image = rng.normal(size=(80, 12))
    perm = rng.permutation(len(image))
    reindexed = np.argsort(perm)[audit.knn(image)[perm]]
    assert np.array_equal(np.sort(reindexed, axis=1), np.sort(audit.knn(image[perm]), axis=1))


def test_caption_sampler_uses_same_images_and_distinct_caption_indices(tmp_path):
    trainer = script("train_text_jepa")
    ids = np.array([[image*10+caption, 1, 0] for image in range(6) for caption in range(5)], dtype=np.int32)
    np.savez(tmp_path / "train.npz", ids=ids, lengths=np.full(30, 2), image=np.repeat(np.arange(6), 5))
    pairs = trainer.CaptionPairs(tmp_path)
    j, n = pairs.sample(np.random.default_rng(20261109), 256), pairs.sample(np.random.default_rng(20261109), 256)
    assert all(np.array_equal(a, b) for a, b in zip(j, n))
    a, _, b, _ = j
    assert np.array_equal(a[:, 0]//10, b[:, 0]//10)
    assert np.all(a[:, 0] != b[:, 0])


def test_custom_chunked_ntp_matches_value_and_gradient():
    trainer = script("train_text_jepa")
    torch.manual_seed(3)
    model = SLM(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                            n_kv_heads=2, ffn_hidden=64, max_seq_len=16))
    ids = torch.randint(1, 64, (3, 8))
    lengths = torch.tensor([8, 5, 3])
    ids[1, 5:] = 0
    ids[2, 3:] = 0
    logits, _ = model(ids)
    full = parallel_ntp_loss(logits, ids, lengths)
    full.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    chunked = trainer.chunked_ntp_loss(model, ids, lengths)
    chunked.backward()
    assert chunked.item() == pytest.approx(full.item(), rel=1e-6)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], rtol=1e-4, atol=1e-6)


def test_fit_only_rotation_recovers_synthetic_rotation():
    audit = script("audit_exp0046")
    rng = np.random.default_rng(5)
    text = rng.normal(size=(160, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    image = text @ rotation + 4.0
    result = audit.rotation_recall_fit_only(text, image, dims=8)
    assert result["r1"] == result["r5"] == 1.0


def test_custom_training_manifest_records_real_cli_and_rejects_data_change(tmp_path):
    trainer = script("train_text_jepa")
    np.savez(tmp_path / "train.npz", ids=np.ones((5, 3), dtype=np.int32))
    config = tmp_path / "model.yaml"
    config.write_text("test configuration identity")
    model = SLM(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                            n_kv_heads=2, ffn_hidden=64, max_seq_len=16))
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(data=tmp_path, model=str(config), mode="jepa", steps=9000,
                           batch=256, dim=256, seed=20261109, out=out, device="cpu")
    fingerprint = trainer.capture_run_manifest(args, model, None)
    assert fingerprint["batch_images"] == 256
    assert trainer.capture_run_manifest(args, model, None) == fingerprint
    np.savez(tmp_path / "train.npz", ids=np.zeros((5, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="changed"):
        trainer.capture_run_manifest(args, model, None)


def test_batched_block_features_match_legacy_and_ignore_right_padding(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    audit = script("audit_exp0046")
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "fox": 2, "dog": 3, "runs": 4}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    token_file = tmp_path / "tokenizer.json"
    tok.save(str(token_file))
    audit.evaluator.TOKENIZER = token_file
    model = SLM(ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                            n_kv_heads=2, ffn_hidden=64, max_seq_len=64)).eval()
    texts = ["a fox", "a dog runs", "fox", "a fox runs a dog"]
    legacy = np.stack(audit.evaluator.native_layers(model, texts))
    batched = audit.native_layers_batched(model, texts, tok, batch_size=4)
    np.testing.assert_allclose(batched, legacy, atol=1e-6, rtol=1e-5)


def test_cache_extractor_fingerprint_ignores_report_edits_but_not_pooling(tmp_path):
    audit = script("audit_exp0046")
    original = Path(audit.__file__).read_text(encoding="utf-8")
    copied = tmp_path / "extractor.py"
    copied.write_text(original+"\n# A report-only edit\n", encoding="utf-8")
    expected = audit.extraction_code_hash(Path(audit.__file__))
    assert audit.extraction_code_hash(copied) == expected
    copied.write_text(original.replace(".sum(1)/lengths[:, None]", ".mean(1)/lengths[:, None]"), encoding="utf-8")
    assert audit.extraction_code_hash(copied) != expected
