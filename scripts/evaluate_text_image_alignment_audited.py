"""Evidence-producing EXP-0046 evaluator; preserves the legacy metric definitions.

Use a fresh --out and a fresh --cache. Legacy files are never silently rebound to
new inputs. Selected features, per-image primary scores and input hashes persist.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from slm.model import SLM
from slm.runlog import code_revision, environment_info, sha256_file, write_json
from slm.train import load_model
from text_image_evidence import image_fingerprint, load_image_cache, save_image_cache

spec = importlib.util.spec_from_file_location("legacy_alignment", ROOT / "scripts/evaluate_text_image_alignment.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def pinned(specification: str) -> tuple[str, str]:
    repo, revision = specification.rsplit("@", 1)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Use a full pinned 40-character Hub revision")
    return repo, revision


def run(args) -> None:
    if args.out.exists():
        raise ValueError("Output already exists; preserve historical results and choose a fresh --out")
    torch.set_num_threads(args.threads)
    rows = json.loads(args.val.read_text(encoding="utf-8"))
    if len({r["image_id"] for r in rows}) != len(rows) or len({r["file_name"] for r in rows}) != len(rows) or len(rows) < 24:
        raise ValueError("Need unique image IDs and enough rows for split 10-NN")
    model_names = [item.split("=", 1)[0] for item in args.native+args.random_control+args.hf]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Repeated model name")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    evidence = args.out.with_suffix("")
    evidence.mkdir(parents=True, exist_ok=True)
    fingerprint = image_fingerprint(rows, args.images, legacy.DINO,
                                    sha256_file(ROOT / "scripts/evaluate_text_image_alignment.py"))
    if args.cache.exists():
        image = load_image_cache(args.cache, fingerprint)
    else:
        image = legacy.image_embeddings(rows, args.images, args.cache)
        save_image_cache(args.cache, image, fingerprint)
    metadata = {"status": "running", "code": code_revision(), "environment": environment_info(),
                "val_sha256": sha256_file(args.val), "native_tokenizer_sha256": sha256_file(legacy.TOKENIZER),
                "image_cache_sha256": sha256_file(args.cache), "image_fingerprint": fingerprint,
                "evaluator_sources": {p: sha256_file(ROOT / "scripts" / p) for p in
                                      ("evaluate_text_image_alignment.py", "evaluate_text_image_alignment_audited.py", "text_image_evidence.py")},
                "models": {}, "secondary_protocol": "legacy transductive test-fold centering, descriptive only",
                "interval_scope": "paired per-image bootstrap on a shared fixed neighbour graph; not graph/population or training-seed uncertainty"}
    manifest_path = args.out.with_suffix(".manifest.json")
    write_json(manifest_path, metadata)
    split = np.random.default_rng(legacy.LAYER_SPLIT_SEED).permutation(len(rows))
    select, test = split[:len(rows)//2], split[len(rows)//2:]
    write_json(evidence / "row-identities.json", [{"index": i, "image_id": row["image_id"],
               "caption_sha256": hashlib.sha256(row["caption"].encode()).hexdigest(),
               "partition": "selection" if i in set(select.tolist()) else "primary_test"} for i, row in enumerate(rows)])
    texts, results, items = [r["caption"] for r in rows], {}, {}

    def evaluate(name, layers, source):
        curve = [legacy.mutual_knn_items(x[select], image[select]) for x in layers]
        best = int(np.argmax([x.mean() for x in curve]))
        values = legacy.mutual_knn_items(layers[best][test], image[test])
        r1, r5 = legacy.rotation_recall(layers[best], image)
        items[name] = values
        key = hashlib.sha256(name.encode()).hexdigest()[:16]
        feature_file, score_file = evidence / f"{key}-selected.npy", evidence / f"{key}-items.npy"
        np.save(feature_file, layers[best])
        np.save(score_file, values)
        metadata["models"][name] = {**source, "selected_features": str(feature_file),
                                    "selected_features_sha256": sha256_file(feature_file),
                                    "primary_items": str(score_file), "primary_items_sha256": sha256_file(score_file)}
        results[name] = {"best_layer_selected_on_half": best, "layers": len(layers),
                         "primary_selection_images": len(select), "primary_test_images": len(test),
                         "mutual_knn10_primary_heldout": round(float(values.mean()), 4),
                         "mutual_knn10_last_layer_heldout": round(float(legacy.mutual_knn_items(layers[-1][test], image[test]).mean()), 4),
                         "selection_curve": [round(float(x.mean()), 4) for x in curve],
                         "linear_cka_best_descriptive": round(legacy.linear_cka(layers[best], image), 4),
                         "r1_rotation_descriptive": round(r1, 4), "r5_rotation_descriptive": round(r5, 4)}
        write_json(manifest_path, metadata)
        print(name, results[name]["mutual_knn10_primary_heldout"], flush=True)

    for item in args.native+args.random_control:
        name, filename = item.split("=", 1)
        path = ROOT / filename
        model = load_model(path, "cpu").eval()
        if item in args.random_control:
            torch.manual_seed(0)
            model = SLM(model.cfg).eval()
        evaluate(name, legacy.native_layers(model, texts), {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
                                                           "random_init_seed": 0 if item in args.random_control else None})
    for item in args.hf:
        name, specification = item.split("=", 1)
        repo, revision = pinned(specification)
        evaluate(name, legacy.hf_layers(repo, revision, texts), {"repo": repo, "revision": revision})
    comparisons = {}
    for item in args.compare:
        control, treatment = item.split("=", 1)
        comparisons[f"{treatment} minus {control}"] = {"difference": round(float((items[treatment]-items[control]).mean()), 4),
                                                        "ci95": legacy.bootstrap(items[control], items[treatment])}
    write_json(args.out, {"kind": "slm.text_image_alignment.audited", "images": len(rows), "dino": legacy.DINO,
                         "primary_protocol": {"layer_selection": "separate half", "split_seed": legacy.LAYER_SPLIT_SEED,
                                              "chance_mutual_knn10": 10/(len(test)-1)},
                         "results": results, "comparisons_mutual_knn10_best_layer": comparisons,
                         "evidence_manifest": str(manifest_path)})
    metadata["status"] = "completed"
    metadata["result_sha256"] = sha256_file(args.out)
    write_json(manifest_path, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=ROOT / "data/parallel/coco2017-captions-v1/val.json")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/coco2017/val2017")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/EXP-0046-evaluations/dinov2-val2017-bound-v1.npy")
    parser.add_argument("--native", action="append", default=[])
    parser.add_argument("--random-control", action="append", default=[])
    parser.add_argument("--hf", action="append", default=[])
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())
