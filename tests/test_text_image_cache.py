"""Caches must reject swapped rows, revised encoders and changed image bytes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/text_image_evidence.py"
    spec = importlib.util.spec_from_file_location("text_image_evidence", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cache_binds_row_order_encoder_and_image_bytes(tmp_path):
    evidence = module()
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    rows = [{"image_id": i, "file_name": name, "caption": "A scene"} for i, name in enumerate(("a.jpg", "b.jpg"))]
    encoder = ("encoder", "pinned-revision")
    fingerprint = evidence.image_fingerprint(rows, tmp_path, encoder)
    path = tmp_path / "cache.npy"
    features = np.arange(6, dtype=np.float32).reshape(2, 3)
    evidence.save_image_cache(path, features, fingerprint)
    np.testing.assert_array_equal(evidence.load_image_cache(path, fingerprint), features)
    for wrong in (evidence.image_fingerprint(rows[::-1], tmp_path, encoder),
                  evidence.image_fingerprint(rows, tmp_path, ("encoder", "new-revision")),
                  evidence.image_fingerprint(rows, tmp_path, encoder, "changed-extractor")):
        with pytest.raises(ValueError):
            evidence.load_image_cache(path, wrong)
    (tmp_path / "a.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError):
        evidence.load_image_cache(path, evidence.image_fingerprint(rows, tmp_path, encoder))


def test_unbound_and_corrupted_caches_are_rejected(tmp_path):
    evidence = module()
    path = tmp_path / "cache.npy"
    fingerprint = {"images": 2}
    np.save(path, np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Legacy"):
        evidence.load_image_cache(path, fingerprint)
    evidence.save_image_cache(path, np.zeros((2, 4), dtype=np.float32), fingerprint)
    np.save(path, np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="contents"):
        evidence.load_image_cache(path, fingerprint)


def test_audited_evaluator_persists_scores_and_binds_cache(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/evaluate_text_image_alignment_audited.py"
    spec = importlib.util.spec_from_file_location("audited_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows = []
    for i in range(40):
        name = f"{i}.jpg"
        (tmp_path / name).write_bytes(str(i).encode())
        rows.append({"image_id": i, "file_name": name, "caption": "A caption"})
    val = tmp_path / "val.json"
    val.write_text(json.dumps(rows))
    token = tmp_path / "tokenizer.json"
    token.write_text("{}")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    image = np.random.default_rng(0).normal(size=(40, 16)).astype(np.float32)
    monkeypatch.setattr(evaluator.legacy, "TOKENIZER", token)
    monkeypatch.setattr(evaluator.legacy, "image_embeddings", lambda *a: image)
    monkeypatch.setattr(evaluator.legacy, "native_layers", lambda *a: [image.copy()])
    monkeypatch.setattr(evaluator.legacy, "rotation_recall", lambda *a: (.2, .4))
    monkeypatch.setattr(evaluator, "load_model", lambda *a: SimpleNamespace(eval=lambda: object()))
    args = SimpleNamespace(out=tmp_path / "result.json", val=val, images=tmp_path,
                           cache=tmp_path / "cache.npy", native=[f"J={checkpoint}"],
                           random_control=[], hf=[], compare=[], threads=1)
    evaluator.run(args)
    result = json.loads(args.out.read_text())
    manifest = json.loads(args.out.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert result["results"]["J"]["mutual_knn10_primary_heldout"] == 1.0
    model = manifest["models"]["J"]
    scores = np.load(model["primary_items"])
    assert len(scores) == 20 and np.all(scores == 1)
    from text_image_evidence import file_hash
    assert file_hash(Path(model["primary_items"])) == model["primary_items_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        evaluator.run(args)
