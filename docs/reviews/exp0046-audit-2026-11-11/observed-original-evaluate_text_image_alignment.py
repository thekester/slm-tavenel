#!/usr/bin/env python3
"""Text-image representation alignment on COCO val2017 (EXP-0046), on CPU.

Images: CLS token of DINOv2 ViT-B/14 (`facebook/dinov2-base@f9e44c81`, Apache-2.0). Text: the first caption of
each image; representation = mean-pooled hidden state after every layer. Per layer, against the image set:
mutual 10-NN overlap (primary, Huh et al., 2024), linear CKA, and R@1 / R@5 text->image after an orthogonal
map (both sides reduced to 256 PCA dimensions fitted on the fit half; two-fold). The best layer per model is
chosen by mutual 10-NN, by the same rule for every model. Native checkpoints are loaded with
``slm.train.load_model``; Hugging Face models at pinned revisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.model import SLM  # noqa: E402
from slm.train import load_model  # noqa: E402

DINO = ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415")
LAYER_SPLIT_SEED = 20261109
TOKENIZER = ROOT / "data/tokenizers/smollm2-f8027fd0/tokenizer.json"


def image_embeddings(rows: list[dict], image_dir: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINO[0], revision=DINO[1])
    model = AutoModel.from_pretrained(DINO[0], revision=DINO[1]).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            images = [Image.open(image_dir / r["file_name"]).convert("RGB") for r in rows[i:i + 32]]
            out.append(model(**processor(images=images, return_tensors="pt")).last_hidden_state[:, 0].numpy())
            print(f"images {i + len(images)}/{len(rows)}", flush=True)
    embeddings = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    return embeddings


@torch.no_grad()
def native_layers(model: SLM, texts: list[str]) -> list[np.ndarray]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    store: list[torch.Tensor] = []
    hooks = [b.register_forward_hook(lambda m, i, o: store.append(o if torch.is_tensor(o) else o[0])) for b in model.blocks]
    rows = []
    for text in texts:
        store.clear()
        model(torch.tensor([tokenizer.encode(text).ids[:64]]))
        rows.append(torch.stack([h[0].float().mean(0) for h in store]).numpy())
    for hook in hooks:
        hook.remove()
    stacked = np.stack(rows, 1)
    return [stacked[i] for i in range(len(stacked))]


@torch.no_grad()
def hf_layers(repo: str, revision: str, texts: list[str]) -> list[np.ndarray]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float32).eval()
    rows = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        hidden = model(**ids, output_hidden_states=True).hidden_states[1:]  # after each layer
        rows.append(torch.stack([h[0].float().mean(0) for h in hidden]).numpy())
    stacked = np.stack(rows, 1)
    return [stacked[i] for i in range(len(stacked))]


def normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def mutual_knn_items(a: np.ndarray, b: np.ndarray, k: int = 10) -> np.ndarray:
    def knn(x):
        s = x @ x.T
        np.fill_diagonal(s, -np.inf)
        return np.argpartition(-s, k, axis=1)[:, :k]
    na, nb = knn(normalize(a)), knn(normalize(b))
    return np.array([len(set(na[i]) & set(nb[i])) / k for i in range(len(a))])


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    return float(np.linalg.norm(a.T @ b, "fro") ** 2 / (np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")))


def rotation_recall(text: np.ndarray, image: np.ndarray, seed: int = 0, dims: int = 256) -> tuple[float, float]:
    idx = np.random.default_rng(seed).permutation(len(text))
    halves = idx[: len(idx) // 2], idx[len(idx) // 2:]
    r1, r5 = [], []
    for fit, test in (halves, halves[::-1]):
        def pca(x):
            mean = x[fit].mean(0)
            _, _, vt = np.linalg.svd(x[fit] - mean, full_matrices=False)
            return lambda y: (y - mean) @ vt[:dims].T
        project_t, project_i = pca(text), pca(image)
        t_fit, i_fit = normalize(project_t(text[fit])), normalize(project_i(image[fit]))
        u, _, vt = np.linalg.svd(t_fit.T @ i_fit)
        rotation = u @ vt
        s = normalize(project_t(text[test])) @ rotation @ normalize(project_i(image[test])).T
        rank = (-s).argsort(1)
        hit = rank == np.arange(len(test))[:, None]
        r1.append(hit[:, :1].any(1).mean())
        r5.append(hit[:, :5].any(1).mean())
    return float(np.mean(r1)), float(np.mean(r5))


def bootstrap(x: np.ndarray, y: np.ndarray, draws: int = 10000) -> list[float]:
    d = y - x
    means = d[np.random.default_rng(20261109).integers(0, len(d), size=(draws, len(d)))].mean(1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=ROOT / "data/parallel/coco2017-captions-v1/val.json")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/coco2017/val2017")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/EXP-0046-evaluations/dinov2-val2017.npy")
    parser.add_argument("--native", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--hf", action="append", default=[], metavar="NAME=REPO@REVISION")
    parser.add_argument("--random-control", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--compare", action="append", default=[], metavar="CONTROL=TREATMENT")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = json.loads(args.val.read_text(encoding="utf-8"))
    texts = [r["caption"] for r in rows]
    image = image_embeddings(rows, args.images, args.cache)
    results, items = {}, {}

    split = np.random.default_rng(LAYER_SPLIT_SEED).permutation(len(rows))
    select_idx, test_idx = split[:len(split) // 2], split[len(split) // 2:]

    def evaluate(name: str, layers: list[np.ndarray]) -> None:
        # Select the layer on half the images, then estimate the primary metric on the untouched half.
        selection_curve = [mutual_knn_items(x[select_idx], image[select_idx]) for x in layers]
        best = int(np.argmax([score.mean() for score in selection_curve]))
        heldout = mutual_knn_items(layers[best][test_idx], image[test_idx])
        r1, r5 = rotation_recall(layers[best], image)
        items[name] = heldout
        results[name] = {"best_layer_selected_on_half": best, "layers": len(layers),
                         "primary_selection_images": len(select_idx), "primary_test_images": len(test_idx),
                         "mutual_knn10_primary_heldout": round(float(heldout.mean()), 4),
                         "mutual_knn10_last_layer_heldout": round(float(mutual_knn_items(layers[-1][test_idx], image[test_idx]).mean()), 4),
                         "selection_curve": [round(float(score.mean()), 4) for score in selection_curve],
                         "linear_cka_best_descriptive": round(linear_cka(layers[best], image), 4),
                         "r1_rotation_descriptive": round(r1, 4), "r5_rotation_descriptive": round(r5, 4)}
        print(name, {k: v for k, v in results[name].items() if k != "selection_curve"}, flush=True)

    for item in args.native:
        name, path = item.split("=", 1)
        evaluate(name, native_layers(load_model(ROOT / path, "cpu").eval(), texts))
    for item in args.random_control:
        name, path = item.split("=", 1)
        torch.manual_seed(0)
        evaluate(name, native_layers(SLM(load_model(ROOT / path, "cpu").cfg).eval(), texts))
    for item in args.hf:
        name, spec = item.split("=", 1)
        repo, revision = spec.split("@", 1)
        evaluate(name, hf_layers(repo, revision, texts))
    comparisons = {}
    for item in args.compare:
        control, treatment = item.split("=", 1)
        d = items[treatment] - items[control]
        comparisons[f"{treatment} minus {control}"] = {"difference": round(float(d.mean()), 4), "ci95": bootstrap(items[control], items[treatment])}
        print("compare", item, comparisons[f"{treatment} minus {control}"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"kind": "slm.text_image_alignment", "images": len(rows), "dino": DINO,
                                    "primary_protocol": {"layer_selection": "first half of a fixed seed-20261109 split", "scoring": "second half, held out from layer selection", "split_seed": LAYER_SPLIT_SEED, "chance_mutual_knn10": round(10 / (len(test_idx) - 1), 4)}, "results": results,
                                    "comparisons_mutual_knn10_best_layer": comparisons}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
