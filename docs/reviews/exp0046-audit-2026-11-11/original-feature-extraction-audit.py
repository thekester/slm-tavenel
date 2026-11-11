"""Audit EXP-0046 without rewriting its original result or retraining models.

Re-extract J/N representations on CPU, reproduce the declared primary metric,
and add explicitly post-hoc caption-overlap and shuffled-correspondence controls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import code_revision, environment_info, sha256_file, write_json
from slm.train import load_model

spec = importlib.util.spec_from_file_location("exp0046_original_evaluator", ROOT / "scripts/evaluate_text_image_alignment.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
DATA = ROOT / "data/parallel/coco2017-captions-v1"
RAW = ROOT / "data/raw/coco2017/annotations"


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


@torch.no_grad()
def native_layers_batched(model, texts: list[str], tok, batch_size: int = 32) -> np.ndarray:
    """Same pre-final-normalization block states, without computing unused vocabulary logits.

    Causal attention means right padding cannot affect earlier valid positions.
    Mean pooling explicitly excludes padding. Numerical parity is checked on
    real anchor captions against the original single-caption extractor.
    """
    all_encoded = [tok.encode(text).ids[:64] for text in texts]
    order = np.argsort([len(ids) for ids in all_encoded], kind="stable")
    pooled = [[] for _ in model.blocks]
    for start in range(0, len(texts), batch_size):
        encoded = [all_encoded[i] for i in order[start:start+batch_size]]
        lengths = torch.tensor([len(ids) for ids in encoded])
        width = int(lengths.max())
        ids = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, tokens in enumerate(encoded):
            ids[row, :len(tokens)] = torch.tensor(tokens)
        mask = (torch.arange(width)[None, :] < lengths[:, None]).float()
        hidden = model.hook_embed(model.embed(ids))
        for layer, block in enumerate(model.blocks):
            hidden = block(hidden, model.rope_cos, model.rope_sin, 0, None, None)
            pooled[layer].append(((hidden.float()*mask[:, :, None]).sum(1)/lengths[:, None]).numpy())
    inverse = np.argsort(order)
    return np.stack([np.concatenate(parts)[inverse] for parts in pooled])


def knn(x: np.ndarray, k: int = 10) -> np.ndarray:
    if not 0 < k < len(x) or not np.isfinite(x).all():
        raise ValueError("Need finite features and 0 < k < number of images")
    normalized = evaluator.normalize(x)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    return np.argpartition(-similarities, k, axis=1)[:, :k]


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Need equally shaped neighbour indices")
    return (a[:, :, None] == b[:, None, :]).any(2).mean(1)


def permutation_control(a: np.ndarray, b: np.ndarray, image: np.ndarray,
                        draws: int = 499, seed: int = 20261111) -> dict:
    """Relabel one image-neighbour graph; no replacement or graph-independent IID claim."""
    na, nb, ni = knn(a), knn(b), knn(image)
    observed = float((overlap(na, ni) - overlap(nb, ni)).mean())
    null, levels = [], []
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        perm = rng.permutation(len(ni))
        inverse = np.argsort(perm)
        shuffled_image_graph = inverse[ni[perm]]
        j, n = overlap(na, shuffled_image_graph), overlap(nb, shuffled_image_graph)
        null.append(float((j-n).mean()))
        levels.append([float(j.mean()), float(n.mean())])
    return {"seed": seed, "draws": draws, "observed_j_minus_n": observed,
            "chance_overlap": 10/(len(ni)-1), "null_mean_overlap_j_n": np.mean(levels, axis=0).tolist(),
            "null_difference_quantiles_025_975": np.quantile(null, [.025, .975]).tolist(),
            "one_sided_monte_carlo_p": (1+sum(d >= observed for d in null))/(draws+1),
            "scope": "post-hoc shuffled-correspondence diagnostic, conditional on these checkpoints and graphs; not training-seed uncertainty"}


def rotation_recall_fit_only(text: np.ndarray, image: np.ndarray, seed: int = 0, dims: int = 256) -> dict:
    """Descriptive sensitivity: fit both centering/PCA and rotation on the fit fold only."""
    idx = np.random.default_rng(seed).permutation(len(text))
    halves = idx[:len(idx)//2], idx[len(idx)//2:]
    recalls = []
    for fit, test in (halves, halves[::-1]):
        def project(x):
            mean = x[fit].mean(0)
            _, _, vt = np.linalg.svd(x[fit]-mean, full_matrices=False)
            return (x-mean) @ vt[:dims].T

        def unit(x):
            return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

        t, i = project(text), project(image)
        u, _, vt = np.linalg.svd(unit(t[fit]).T @ unit(i[fit]))
        scores = unit(t[test]) @ (u @ vt) @ unit(i[test]).T
        rank = np.argsort(-scores, axis=1)
        hit = rank == np.arange(len(test))[:, None]
        recalls.append([float(hit[:, :1].any(1).mean()), float(hit[:, :5].any(1).mean())])
    values = np.mean(recalls, axis=0)
    return {"r1": float(values[0]), "r5": float(values[1]), "seed": seed, "dimensions": dims,
            "scope": "post-hoc descriptive cross-fitted retrieval, layer still selected by primary metric; no test-fold recentering"}


def data_audit() -> tuple[dict, np.ndarray]:
    train = json.loads((RAW / "captions_train2017.json").read_text(encoding="utf-8"))
    val = json.loads((RAW / "captions_val2017.json").read_text(encoding="utf-8"))
    rows = json.loads((DATA / "val.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text())
    for path, expected in [(RAW / "captions_train2017.json", manifest["source"]["captions_train_sha256"]),
                           (RAW / "captions_val2017.json", manifest["source"]["captions_val_sha256"]),
                           (DATA / "train.npz", manifest["train_npz_sha256"]),
                           (evaluator.TOKENIZER, manifest["tokenizer_sha256"])]:
        if sha256_file(path) != expected:
            raise ValueError(f"Input hash changed: {path}")
    train_ids = {r["id"] for r in train["images"]}
    val_ids = {r["id"] for r in val["images"]}
    if train_ids & val_ids or len(rows) != 5000 or len({r["image_id"] for r in rows}) != 5000:
        raise ValueError("Image identity overlap or unexpected evaluation set")
    first = {}
    for annotation in sorted(val["annotations"], key=lambda r: r["id"]):
        first.setdefault(annotation["image_id"], annotation["caption"].strip())
    if any(r["image_id"] not in val_ids or r["caption"] != first[r["image_id"]] for r in rows):
        raise ValueError("Evaluation rows do not match first captions by lowest annotation ID")
    by_image = defaultdict(list)
    for annotation in sorted(train["annotations"], key=lambda r: r["id"]):
        by_image[annotation["image_id"]].append(annotation["caption"].strip())
    texts, owner = [], []
    for i, image_id in enumerate(sorted(by_image)):
        texts.extend(by_image[image_id])
        owner.extend([i]*len(by_image[image_id]))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(evaluator.TOKENIZER))
    # NpzFile loads a complete array on each indexing access. Materialize once
    # before the per-caption verification loop, rather than rereading 118 MB
    # for every row.
    with np.load(DATA / "train.npz") as archive:
        encoded = {key: archive[key] for key in archive.files}
    if not np.array_equal(encoded["image"], owner):
        raise ValueError("Stored caption groups differ from annotation image groups")
    truncations = 0
    for start in range(0, len(texts), 4096):
        for i, enc in enumerate(tok.encode_batch(texts[start:start+4096]), start):
            ids = enc.ids[:48]
            truncations += len(enc.ids) > 48
            if encoded["lengths"][i] != len(ids) or not np.array_equal(encoded["ids"][i, :len(ids)], ids):
                raise ValueError("Stored training token sequence differs from pinned annotations/tokenizer")
            if np.any(encoded["ids"][i, len(ids):] != manifest["pad_id"]):
                raise ValueError("Unexpected padding token")
    train_exact = set(texts)
    train_norm = {norm(t) for t in texts}
    eval_texts = [r["caption"] for r in rows]
    shared = np.array([norm(t) in train_norm for t in eval_texts])
    split = np.random.default_rng(evaluator.LAYER_SPLIT_SEED).permutation(len(rows))
    selection, test = split[:2500], split[2500:]
    image_hashes, seen_files = {}, set()
    for row in rows:
        path = ROOT / "data/raw/coco2017/val2017" / row["file_name"]
        if path.name in seen_files:
            raise ValueError("Repeated image file in val rows")
        seen_files.add(path.name)
        image_hashes[row["file_name"]] = sha256_file(path)
    result = {"train_images": len(train_ids), "val_images": len(val_ids), "image_id_overlap": 0,
              "all_train_token_sequences_verified": len(texts), "train_caption_truncations_at48": truncations,
              "train_caption_min_tokens": int(encoded["lengths"].min()),
              "exact_eval_caption_matches_train": sum(t in train_exact for t in eval_texts),
              "normalized_eval_caption_matches_train": int(shared.sum()), "val_unique_normalized_captions": len({norm(t) for t in eval_texts}),
              "shared_in_selection_half": int(shared[selection].sum()), "shared_in_test_half": int(shared[test].sum()),
              "val_rows_sha256": sha256_file(DATA / "val.json"), "data_manifest_sha256": sha256_file(DATA / "manifest.json"),
              "val_image_file_hashes": image_hashes,
              "limits": "Image IDs and file identities verified, not perceptual near-duplicates between train and val. Caption hashes do not establish semantic independence or upstream reference exposure."}
    return result, ~shared


def checked_native_features(key: str, out: Path, texts: list[str]) -> tuple[np.ndarray, dict]:
    checkpoint = ROOT / f"runs/EXP-0046-124m-{key}/checkpoints/step_00009000.pt"
    fingerprint = {"checkpoint_sha256": sha256_file(checkpoint), "val_rows_sha256": sha256_file(DATA / "val.json"),
                   "tokenizer_sha256": sha256_file(evaluator.TOKENIZER),
                   "extractor_sha256": sha256_file(ROOT / "scripts/evaluate_text_image_alignment.py"),
                   "audit_extractor_sha256": sha256_file(__file__), "extraction_threads": torch.get_num_threads(),
                   "extraction": "length-bucketed batch32 block outputs restored to original order, no unused logits; anchors checked against legacy"}
    folder = out / key
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "features.npy").exists() and (folder / "metadata.json").exists():
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["fingerprint"] != fingerprint or metadata["features_sha256"] != sha256_file(folder / "features.npy"):
            raise ValueError("Stale audit text-feature cache")
        return np.load(folder / "features.npy", mmap_mode="r"), metadata
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["train_config"]
    backbone_params = sum(v.numel() for k, v in state["model"].items()
                          if not k.startswith("jepa_predictor.") and k != "lm_head.weight")
    head_params = sum(v.numel() for k, v in state["model"].items() if k.startswith("jepa_predictor."))
    if state["step"] != 9000 or config["seed"] != 20261109:
        raise ValueError("Unexpected checkpoint step/seed")
    metadata = {"fingerprint": fingerprint, "checkpoint_step": state["step"], "saved_train_config": config,
                "saved_model_config": state["model_config"], "backbone_unique_params": backbone_params,
                "auxiliary_training_head_params": head_params,
                "caveat": "Custom-loop CLI batch/dim/mode were not saved as a complete original run manifest. Saved generic TrainConfig micro_batch_size/seq_len are not the loop's actual batch/token cap."}
    del state
    model = load_model(checkpoint, "cpu").eval()
    if model.num_params() != backbone_params:
        raise ValueError("Checkpoint parameter count reconciliation failed")
    print(f"extracting {key}: {len(texts)} captions, {model.num_params()} backbone parameters", flush=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(evaluator.TOKENIZER))
    anchors = [0, 999, 2499, 2500, 3999, 4999]
    original_anchors = np.stack(evaluator.native_layers(model, [texts[i] for i in anchors]))
    features = native_layers_batched(model, texts, tok)
    delta = float(np.max(np.abs(features[:, anchors]-original_anchors)))
    if not np.allclose(features[:, anchors], original_anchors, rtol=1e-4, atol=1e-4):
        raise ValueError("Batched block extraction failed numerical parity with legacy extractor")
    metadata["legacy_anchor_extraction_max_abs_difference"] = delta
    if not np.isfinite(features).all() or features.shape[:2] != (27, 5000):
        raise ValueError("Unexpected text feature shape/values")
    np.save(folder / "features.npy", features)
    metadata["features_sha256"] = sha256_file(folder / "features.npy")
    write_json(folder / "metadata.json", metadata)
    del model
    gc.collect()
    return features, metadata


def verify_dino_samples(rows: list[dict], cached: np.ndarray) -> dict:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(evaluator.DINO[0], revision=evaluator.DINO[1], local_files_only=True)
    model = AutoModel.from_pretrained(evaluator.DINO[0], revision=evaluator.DINO[1], local_files_only=True).eval()
    indices = [0, 999, 2499, 2500, 3999, 4999]
    images = [Image.open(ROOT / "data/raw/coco2017/val2017" / rows[i]["file_name"]).convert("RGB") for i in indices]
    with torch.no_grad():
        actual = model(**processor(images=images, return_tensors="pt")).last_hidden_state[:, 0].numpy()
    delta = float(np.max(np.abs(actual-cached[indices])))
    if not np.allclose(actual, cached[indices], rtol=1e-4, atol=1e-4):
        raise ValueError("DINO cache does not match current pinned model/processor/image samples")
    return {"indices": indices, "max_abs_difference": delta, "scope": "Six fixed cache anchors re-embedded, not a full 5,000-image cache recomputation"}


def run(out: Path, threads: int, permutations: int) -> None:
    torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=True)
    original = ROOT / "runs/EXP-0046-evaluations/text-image-alignment.json"
    original_hash = sha256_file(original)
    print("Checking data, every training token sequence, and 5,000 image-file hashes", flush=True)
    data, clean = data_audit()
    write_json(out / "data-audit.json", data)
    rows = json.loads((DATA / "val.json").read_text(encoding="utf-8"))
    image = np.load(ROOT / "runs/EXP-0046-evaluations/dinov2-val2017.npy")
    if image.shape != (5000, 768) or not np.isfinite(image).all():
        raise ValueError("Unexpected DINO cache shape or nonfinite features")
    anchors = verify_dino_samples(rows, image)
    write_json(out / "dino-anchor-audit.json", anchors)
    split = np.random.default_rng(evaluator.LAYER_SPLIT_SEED).permutation(5000)
    select, test = split[:2500], split[2500:]
    original_result = json.loads(original.read_text())
    results, primary, selected = {}, {}, {}
    for key, label in (("jepa", "J"), ("ntp", "N")):
        features, metadata = checked_native_features(key, out, [r["caption"] for r in rows])
        curve = [float(evaluator.mutual_knn_items(x[select], image[select]).mean()) for x in features]
        best = int(np.argmax(curve))
        values = evaluator.mutual_knn_items(features[best][test], image[test])
        primary[label] = values
        selected[label] = np.array(features[best])
        if best != original_result["results"][label]["best_layer_selected_on_half"] or round(float(values.mean()), 4) != original_result["results"][label]["mutual_knn10_primary_heldout"]:
            raise ValueError(f"Original {label} primary result did not reproduce")
        clean_indices = test[clean[test]]
        result = {"selected_layer": best, "primary_mean": float(values.mean()), "reproduces_original_to_four_decimals": True,
                  "posthoc_no_train_caption_match_images": len(clean_indices),
                  "posthoc_no_train_caption_match_mean": float(evaluator.mutual_knn_items(features[best][clean_indices], image[clean_indices]).mean()),
                  "checkpoint": metadata, "selection_curve": curve}
        np.save(out / f"{label}-primary-items.npy", values)
        results[label] = result
        print(label, {k: v for k, v in result.items() if k not in ("checkpoint", "selection_curve")}, flush=True)
        del features
        gc.collect()
    difference = float((primary["J"]-primary["N"]).mean())
    ci = evaluator.bootstrap(primary["N"], primary["J"])
    null = permutation_control(selected["J"][test], selected["N"][test], image[test], permutations)
    print("Checking fit-only centering retrieval sensitivity", flush=True)
    retrieval = {key: rotation_recall_fit_only(features, image) for key, features in selected.items()}
    report = {"kind": "slm.exp0046.independent_local_audit", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "code": code_revision(), "environment": environment_info(), "threads": threads,
              "original_result_sha256": original_hash, "audit_script_sha256": sha256_file(__file__),
              "data_audit_sha256": sha256_file(out / "data-audit.json"),
              "dino_cache_sha256": sha256_file(ROOT / "runs/EXP-0046-evaluations/dinov2-val2017.npy"),
              "results": results, "j_minus_n": difference, "original_fixed_graph_bootstrap_ci95": ci,
              "original_rule_on_this_fixed_graph": difference >= .02 and ci[0] > 0,
              "permutation_control": null, "dino_cache_anchor_audit": anchors,
              "fit_only_centering_retrieval_sensitivity": retrieval,
              "limits": ["One training seed per arm, not confirmatory replication across seeds",
                         "Bootstrap resamples per-image scores in a fixed shared graph; graph/population uncertainty not established",
                         "New no-train-caption-match and permutation results are post-hoc diagnostics, not new preregistered outcomes",
                         "Public-reference aggregate values checked against original JSON/log only; their features were not independently re-extracted by this J/N audit",
                         "Original execution lacked a complete frozen manifest, provenance reconstructed at audit time cannot retroactively prove exact runtime inputs/code",
                         "No 72B test, other vision encoder or shuffled-positive training control",
                         "Original retrieval recenters test cohorts (transductive preprocessing, not label leakage); fit-only sensitivity remains descriptive"]}
    if sha256_file(original) != original_hash:
        raise ValueError("Original result changed during audit")
    write_json(out / "report.json", report)
    print(f"Audit complete: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/EXP-0046-audit-2026-11-11")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=499)
    args = parser.parse_args()
    run(args.out, args.threads, args.permutations)
