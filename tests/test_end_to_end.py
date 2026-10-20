"""Tokenizer -> shards -> training -> resume -> evaluation -> export, on CPU in seconds."""

import json

import pytest
import torch
import yaml

from slm import tokenizer
from slm.config import TrainConfig
from slm.data import synthetic
from slm.data.shards import TokenShards, encode
from slm.evaluation import bits_per_byte, measure_cost, verified_tasks
from slm.train import latest_checkpoint, load_model, lr_at, train


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    docs = root / "docs.jsonl"
    synthetic.write_jsonl(synthetic.prose(300, seed=0), docs)
    tokenizer.train([str(docs)], root / "tok", vocab_size=600, seed=0)
    encode(root / "tok", [str(docs)], root / "tokens", val_fraction=0.1, shard_tokens=20_000)
    model_cfg = root / "model.yaml"
    model_cfg.write_text(yaml.safe_dump(dict(vocab_size=640, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                                             ffn_hidden=64, max_seq_len=128)))
    return root


def make_cfg(root, out, max_steps, **kw) -> TrainConfig:
    params = dict(model=str(root / "model.yaml"), data_dir=str(root / "tokens"), out_dir=str(out),
                  seq_len=32, micro_batch_size=4, max_steps=max_steps, lr=1e-2, min_lr=1e-3,
                  warmup_steps=2, dtype="float32", log_interval=2, eval_interval=4, eval_batches=2,
                  observe_interval=3, ckpt_interval=3)
    params.update(kw)
    return TrainConfig(**params)


def test_tokenizer_roundtrip(corpus):
    tok = tokenizer.load(corpus / "tok")
    text = "La rivière mesure 123 étapes."
    assert tok.decode(tok.encode(text).ids) == text
    assert tok.encode("123").tokens == ["1", "2", "3"]
    assert tok.token_to_id(tokenizer.EOS) == 0


def test_shards_manifest(corpus):
    manifest = json.loads((corpus / "tokens" / "manifest.json").read_text())
    assert manifest["splits"]["train"]["docs"] + manifest["splits"]["val"]["docs"] == 300
    assert len(manifest["splits"]["train"]["shards"]) > 1
    import numpy as np

    data = TokenShards(corpus / "tokens")
    x, y = data.sample(np.random.default_rng(0), 3, 16)
    assert x.shape == (3, 16) and (x[:, 1:] == y[:, :-1]).all()


def test_training_is_reproducible_across_resume(corpus, tmp_path):
    torch.use_deterministic_algorithms(False)
    full = train(make_cfg(corpus, tmp_path / "full", 6))
    train(make_cfg(corpus, tmp_path / "split", 3))
    resumed = train(make_cfg(corpus, tmp_path / "split", 6))
    assert full["final_loss"] == pytest.approx(resumed["final_loss"], rel=1e-6)
    a = load_model(latest_checkpoint(tmp_path / "full"))
    b = load_model(latest_checkpoint(tmp_path / "split"))
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        torch.testing.assert_close(p, q, msg=n)

    metrics = [json.loads(l) for l in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 6
    obs = [json.loads(l) for l in (tmp_path / "full" / "observability.jsonl").read_text().splitlines()]
    assert {"resid_rms", "sink_mass_mean", "update_ratio"} <= set(obs[0])
    manifest = json.loads((tmp_path / "full" / "run_manifest.json").read_text())
    assert manifest["train_config_hash"] and manifest["data"]["tokenizer_sha256"]


def test_muon_and_doc_mask_training_reproducible_across_resume(corpus, tmp_path):
    full = train(make_cfg(corpus, tmp_path / "full", 6, optimizer="muon", doc_mask=True, lr=0.02))
    train(make_cfg(corpus, tmp_path / "split", 3, optimizer="muon", doc_mask=True, lr=0.02))
    resumed = train(make_cfg(corpus, tmp_path / "split", 6, optimizer="muon", doc_mask=True, lr=0.02))
    assert full["final_loss"] == pytest.approx(resumed["final_loss"], rel=1e-6)
    a = load_model(latest_checkpoint(tmp_path / "full"))
    b = load_model(latest_checkpoint(tmp_path / "split"))
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        torch.testing.assert_close(p, q, msg=n)


def test_resume_does_not_require_initial_weights(corpus, tmp_path):
    train(make_cfg(corpus, tmp_path / "source", 3))
    initial = tmp_path / "initial.pt"
    initial.write_bytes(latest_checkpoint(tmp_path / "source").read_bytes())
    train(make_cfg(corpus, tmp_path / "initialized", 3, init_from=str(initial)))
    manifest_path = tmp_path / "initialized" / "run_manifest.json"
    provenance = json.loads(manifest_path.read_text())["initialized_from"]
    assert provenance["path"] == str(initial) and provenance["sha256"]
    initial.unlink()
    train(make_cfg(corpus, tmp_path / "initialized", 6, init_from=str(initial)))
    state = torch.load(latest_checkpoint(tmp_path / "initialized"), weights_only=False)
    assert state["step"] == 6
    assert json.loads(manifest_path.read_text())["initialized_from"] == provenance


def test_training_reduces_loss(corpus, tmp_path):
    cfg = make_cfg(corpus, tmp_path / "learn", 60, eval_interval=60)
    train(cfg)
    evals = [json.loads(l) for l in (tmp_path / "learn" / "eval.jsonl").read_text().splitlines()]
    assert evals[-1]["val_loss"] < 0.6 * torch.log(torch.tensor(640.0)).item()


def test_evaluation_and_export(corpus, tmp_path):
    train(make_cfg(corpus, tmp_path / "run", 3))
    model = load_model(latest_checkpoint(tmp_path / "run"))
    tok = tokenizer.load(corpus / "tok")
    cpu = torch.device("cpu")
    bpb = bits_per_byte(model, tok, ["La rivière mesure le flux d'information."], cpu)
    assert bpb["bits_per_byte"] > 0 and bpb["tokens"] > 0
    cost = measure_cost(model, tok, ["The river"], 5, cpu)
    assert cost["new_tokens"] > 0
    tasks = synthetic.arithmetic(3)
    res = verified_tasks(model, tok, tasks, synthetic.verify_arithmetic, cpu, max_new_tokens=4, attempts=2,
                         temperature=1.0)
    assert res["attempts"] >= 3

    transformers = pytest.importorskip("transformers")
    from slm.export import export

    out = export(model, corpus / "tok", tmp_path / "hf", dtype=torch.float32)
    hf = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    x = torch.tensor([tok.encode("La rivière mesure").ids])
    ours, _ = model(x)
    theirs = hf(x).logits
    torch.testing.assert_close(theirs, ours, atol=1e-4, rtol=1e-4)
    hf_tok = transformers.AutoTokenizer.from_pretrained(out)
    enc = hf_tok("La rivière", return_tensors="pt")
    assert enc["input_ids"][0].tolist() == tok.encode("La rivière").ids
    hf_out = hf.generate(**enc, max_new_tokens=5, do_sample=False)
    ours_out = model.generate(enc["input_ids"], 5)
    assert hf_out[0].tolist()[: ours_out.shape[1]] == ours_out[0].tolist()[: hf_out.shape[1]]

    from slm.hf_adapter import load as load_reference

    reference, reference_tok = load_reference(str(out), cpu)
    torch.testing.assert_close(reference(x)[0], ours, atol=1e-4, rtol=1e-4)
    assert reference_tok.encode("La rivière").ids == tok.encode("La rivière").ids
    reference_bpb = bits_per_byte(reference, reference_tok, ["La rivière mesure le flux d'information."], cpu)
    assert reference_bpb["bits_per_byte"] == pytest.approx(bpb["bits_per_byte"], rel=1e-5)
    nested, nested_tok = load_reference(str(tmp_path), cpu, subfolder="hf", max_seq_len=64)
    torch.testing.assert_close(nested(x)[0], ours, atol=1e-4, rtol=1e-4)
    assert nested.cfg.max_seq_len == 64
    assert nested_tok.encode("test").ids == tok.encode("test").ids


def test_training_records_optional_dataset_provenance(corpus, tmp_path):
    import shutil
    from slm.runlog import sha256_file

    data_dir = tmp_path / "tokens"
    shutil.copytree(corpus / "tokens", data_dir)
    provenance = {"kind": "slm.dataset_provenance", "status": "candidate", "artifacts": {"example": "hash"}}
    (data_dir / "provenance.json").write_text(json.dumps(provenance))
    output = tmp_path / "run"
    train(make_cfg(corpus, output, 1, data_dir=str(data_dir)))
    recorded = json.loads((output / "run_manifest.json").read_text())["data"]
    assert recorded["manifest_sha256"] == sha256_file(data_dir / "manifest.json")
    assert recorded["provenance"] == provenance
    assert recorded["provenance_path"] == str(data_dir / "provenance.json")


@pytest.mark.parametrize("qk_norm,arch", [(False, "llama"), (True, "qwen3")])
def test_export_matches_transformers(corpus, tmp_path, qk_norm, arch):
    transformers = pytest.importorskip("transformers")
    from slm.config import ModelConfig
    from slm.export import export, hf_architecture
    from slm.model import SLM

    torch.manual_seed(0)
    model = SLM(ModelConfig(vocab_size=640, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=96,
                            max_seq_len=64, qk_norm=qk_norm, tie_embeddings=not qk_norm)).eval()
    assert hf_architecture(model) == arch
    out = export(model, corpus / "tok", tmp_path / arch, dtype=torch.float32)
    hf = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    x = torch.randint(0, 640, (2, 20))
    torch.testing.assert_close(hf(x).logits, model(x)[0], atol=1e-4, rtol=1e-4)


def test_export_refuses_nonstandard_variants():
    from slm.config import ModelConfig
    from slm.export import hf_architecture
    from slm.model import SLM

    for kw in ({"attn_gate": "headwise"}, {"window_pattern": [8, 0]}):
        model = SLM(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
                                max_seq_len=32, **kw))
        with pytest.raises(ValueError):
            hf_architecture(model)


def test_wsd_schedule():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100, decay_fraction=0.2)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(50, cfg) == 1.0
    assert lr_at(79, cfg) == 1.0
    assert lr_at(90, cfg) == pytest.approx(0.55)
    assert lr_at(99, cfg) < 0.2
