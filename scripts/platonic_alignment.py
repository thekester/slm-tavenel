#!/usr/bin/env python3
"""Cross-lingual "platonic" alignment of SLM-Tavenel representations (CPU analysis, no training).

Motivation: Balestriero & LeCun (2026, arXiv:2610.39302) prove that JEPAs trained on *good views* (two
independent observations of the same world state) learn the same representation up to a rotation, and
argue that next-token prediction is a JEPA with *bad views* (prefix/continuation of one document), hence
an inefficient route to aligned ("platonic", Huh et al., 2024) representations. Parallel English/French
passages are good views of the same content, so a bilingual model's English and French representations
of the same passages measure how far next-token training has gone toward a shared representation.

Data: the 488 unique parallel passages of Belebele (`facebook/belebele@7899cdfa`, eng_Latn / fra_Latn,
CC-BY-SA-4.0; FLORES-200 passages), from the local benchmark files (analysis only, never trained on).
Representation: mean-pooled residual stream after every block (first 512 tokens).
Metrics per layer, between the English set and the French set:
  * mutual 10-NN overlap (Huh et al., 2024): for each passage, overlap of its 10 nearest neighbours among
    English passages and among French passages (cosine), averaged; chance = 10 / 487;
  * retrieval R@1 English -> French by cosine after centering each set (no fitted map);
  * R@1 after an orthogonal map (Procrustes) fitted on half the passages and tested on the other half,
    the "alignment up to a rotation" of the paper (averaged over both halves).
A randomly initialized copy of each architecture is the control for tokenizer and architecture effects
(shared numbers, names and subwords make English and French similar even without learning).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.model import SLM  # noqa: E402
from slm.train import load_model  # noqa: E402

TOKENIZER = ROOT / "data/tokenizers/smollm2-f8027fd0/tokenizer.json"
CHECKPOINTS = {
    "75M AdamW CE (EXP-0034a)": "runs/EXP-0034a-75m-ce/checkpoints/step_00001500.pt",
    "75M AdamW distill (EXP-0034a)": "runs/EXP-0034a-75m-distill/checkpoints/step_00001500.pt",
    "75M Muon CE (EXP-0036a)": "runs/EXP-0036a-75m-muon/checkpoints/step_00001500.pt",
    "75M LFM2 hybrid (EXP-0041)": "runs/EXP-0041-75m-lfm2-hybrid-muon/checkpoints/step_00001500.pt",
    "124M Muon (EXP-0036)": "runs/EXP-0036-124m-gpt2match-muon/checkpoints/step_00015000.pt",
}


def passages(language: str) -> tuple[list[str], list[str]]:
    seen, ids, texts = set(), [], []
    for line in (ROOT / f"data/benchmarks/belebele_{language}.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        key = item["id"]
        if key in seen:
            continue
        seen.add(key)
        match = re.match(r"P: (.*?)\nQ:", item["context"], re.S)
        ids.append(key)
        texts.append(match.group(1).strip() if match else item["context"])
    return ids, texts


@torch.no_grad()
def layer_means(model: SLM, tokenizer, texts: list[str]) -> np.ndarray:
    """(layers, passages, d_model) mean-pooled residual stream after each block."""
    store: list[torch.Tensor] = []
    hooks = [block.register_forward_hook(lambda m, i, o: store.append(o if torch.is_tensor(o) else o[0]))
             for block in model.blocks]
    out = []
    for text in texts:
        ids = torch.tensor([tokenizer.encode(text).ids[:512]])
        store.clear()
        model(ids)
        out.append(torch.stack([h[0].float().mean(0) for h in store]))
    for hook in hooks:
        hook.remove()
    return torch.stack(out, 1).numpy()


def normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def mutual_knn_per_item(a: np.ndarray, b: np.ndarray, k: int = 10) -> np.ndarray:
    def knn(x):
        s = x @ x.T
        np.fill_diagonal(s, -np.inf)
        return np.argsort(-s, axis=1)[:, :k]
    na, nb = knn(normalize(a)), knn(normalize(b))
    return np.array([len(set(na[i]) & set(nb[i])) / k for i in range(len(a))])


def mutual_knn(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    return float(mutual_knn_per_item(a, b, k).mean())


def paired_bootstrap(x: np.ndarray, y: np.ndarray, draws: int = 10000, seed: int = 20261108) -> list[float]:
    """95% interval of mean(y - x) resampling passages (both arms scored on the same passages)."""
    d = y - x
    means = d[np.random.default_rng(seed).integers(0, len(d), size=(draws, len(d)))].mean(1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def recall_at_1(a: np.ndarray, b: np.ndarray) -> float:
    s = normalize(a) @ normalize(b).T
    return float((s.argmax(1) == np.arange(len(a))).mean())


def procrustes_r1(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    idx = np.random.default_rng(seed).permutation(len(a))
    halves = [idx[: len(a) // 2], idx[len(a) // 2:]]
    scores = []
    for fit, test in (halves, halves[::-1]):
        a_fit, b_fit = normalize(a[fit]), normalize(b[fit])
        u, _, vt = np.linalg.svd(a_fit.T @ b_fit)
        rotation = u @ vt
        scores.append(recall_at_1(normalize(a[test]) @ rotation, b[test]))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/PLATONIC/belebele-en-fr-alignment.json")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="replaces the default checkpoint list; repeatable")
    parser.add_argument("--no-random-control", action="store_true")
    parser.add_argument("--compare", action="append", default=[], metavar="CONTROL=TREATMENT",
                        help="paired bootstrap of per-passage mutual 10-NN at each model's best layer")
    args = parser.parse_args()
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint) or CHECKPOINTS
    from tokenizers import Tokenizer

    torch.set_num_threads(4)  # leaves CPU for a concurrent GPU training run
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    en_ids, en = passages("en")
    fr_ids, fr = passages("fr")
    if en_ids != fr_ids:
        raise SystemExit("English and French passages are not aligned")
    results, per_passage = {}, {}
    for name, path in checkpoints.items():
        trained = load_model(ROOT / path, "cpu").eval()
        torch.manual_seed(0)
        models = [(name, trained)] if args.no_random_control else [(name, trained), (name + " — random init", SLM(trained.cfg).eval())]
        for label, model in models:
            e, f = layer_means(model, tokenizer, en), layer_means(model, tokenizer, fr)
            rows = [{"layer": i, "mutual_knn10": round(mutual_knn(e[i], f[i]), 4),
                     "r1_cosine": round(recall_at_1(e[i], f[i]), 4),
                     "r1_procrustes": round(procrustes_r1(e[i], f[i]), 4)} for i in range(len(e))]
            best = max(rows, key=lambda r: r["mutual_knn10"])
            per_passage[label] = mutual_knn_per_item(e[best["layer"]], f[best["layer"]])
            results[label] = {"checkpoint": path, "layers": rows, "best_layer": best}
            print(label, "best", best, flush=True)
    comparisons = {}
    for item in args.compare:
        control, treatment = item.split("=", 1)
        x, y = per_passage[control], per_passage[treatment]
        comparisons[f"{treatment} minus {control}"] = {"difference": round(float((y - x).mean()), 4),
                                                       "ci95": paired_bootstrap(x, y)}
        print("compare", item, comparisons[f"{treatment} minus {control}"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"kind": "slm.platonic_alignment", "passages": len(en),
                                    "chance_mutual_knn10": round(10 / (len(en) - 1), 4),
                                    "chance_r1": round(1 / len(en), 4), "results": results,
                                    "comparisons_mutual_knn10_best_layer": comparisons}, indent=2) + "\n",
                        encoding="utf-8")


if __name__ == "__main__":
    main()
