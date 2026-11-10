"""Score both frozen story models on the full official TinyStories validation split, on CPU.

Uses the existing bits_per_byte evaluator without changing its prefix/chunk policy. Only hashes and
per-document likelihoods are written to the evidence directory; source stories stay under data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch  # Load before pyarrow on Windows.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.evaluation import bits_per_byte
from slm.hf_adapter import HFModel, HFTokenizer
from slm.runlog import code_revision, environment_info, sha256_file

DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SOURCE_FILE = "data/validation-00000-of-00001-869c898b519ad725.parquet"
MODELS = [
    {"label": "SLM-Tavenel Story 935K", "key": "story935k", "repo": "athenista/slm-tavenel-story-935k",
     "revision": "62e64c2b677ea9fbc90cdee0fbfe66b589b78e83",
     "path": "data/models/fluency/story-935k-100m-v2",
     "weights_sha256": "8804d57668ac492454d9835841ff2a57f7867dbcbb36df4452ffedfa8813c9a8"},
    {"label": "TinyStories tiny1m", "key": "tiny1m", "repo": "shibatch/tiny1m",
     "revision": "fdacad8ff9dc27bd72c77d3684075a494be1ba11",
     "path": "data/models/fluency/tiny1m/hf",
     "weights_sha256": "f30ae85c53797b5944da7f235162756b31987233b44eac36ed449afd1b82523b"},
]


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def text_hash(text: str, normalized: bool = False) -> str:
    if normalized:
        text = " ".join(text.casefold().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paired_bootstrap(nll_a: np.ndarray, nll_b: np.ndarray, byte_counts: np.ndarray,
                     draws: int = 10000, seed: int = 421204) -> dict:
    """Resample paired documents and recompute byte-weighted ratios, not means of document BPB."""
    if not (len(nll_a) == len(nll_b) == len(byte_counts)) or not len(nll_a) or byte_counts.sum() <= 0:
        raise ValueError("Need aligned nonempty scores with positive total bytes")
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 64):
        idx = rng.integers(0, len(nll_a), size=(min(64, draws-start), len(nll_a)))
        den = byte_counts[idx].sum(1) * math.log(2)
        estimates.append(np.column_stack((nll_a[idx].sum(1)/den, nll_b[idx].sum(1)/den)))
    values = np.concatenate(estimates)
    return {"draws": draws, "seed": seed, "unit": "paired document",
            "story935k_ci95": np.quantile(values[:, 0], [.025, .975]).tolist(),
            "tiny1m_ci95": np.quantile(values[:, 1], [.025, .975]).tolist(),
            "story_minus_tiny_ci95": np.quantile(values[:, 0]-values[:, 1], [.025, .975]).tolist(),
            "limits": "Fixed checkpoints; does not include seed variation, corpus-family dependence or unknown reference training exposure"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/EXP-0045/tinystories-validation-2026-11-10")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
    import pyarrow.parquet as pq
    import transformers

    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(DATASET, SOURCE_FILE, repo_type="dataset", revision=REVISION,
                                local_dir=ROOT / "data/raw/tinystories-f54c09fd"))
    texts = pq.read_table(source, columns=["text"]).column("text").to_pylist()
    if any(not isinstance(t, str) for t in texts):
        raise ValueError("Unexpected non-text row in official validation")
    train_path = ROOT / "data/processed/exp0045-simple-stories-v2/train.jsonl"
    train = [json.loads(s)["text"] for s in train_path.read_text(encoding="utf-8").splitlines()]
    train_exact, train_normal = {text_hash(t) for t in train}, {text_hash(t, True) for t in train}
    exact, normal = [text_hash(t) for t in texts], [text_hash(t, True) for t in texts]
    audit = {"dataset": DATASET, "revision": REVISION, "split": "validation", "source_file": SOURCE_FILE,
             "declared_license": "CDLA-Sharing-1.0", "source_sha256": sha256_file(source),
             "documents": len(texts), "bytes": sum(len(t.encode()) for t in texts),
             "empty_texts": sum(not t for t in texts), "unique_exact_texts": len(set(exact)),
             "unique_case_whitespace_normalized_texts": len(set(normal)),
             "exact_overlaps_with_story_training": sum(h in train_exact for h in exact),
             "normalized_overlaps_with_story_training": sum(h in train_normal for h in normal),
             "train_sha256": sha256_file(train_path),
             "limits": "Exact/case/whitespace checks only; reference training exposure and semantic overlap are not established"}
    write(args.out / "dataset-audit.json", audit)
    fingerprint = {"source_sha256": audit["source_sha256"], "script_sha256": sha256_file(Path(__file__)),
                   "context": 256, "model_weight_hashes": {s["key"]: s["weights_sha256"] for s in MODELS}}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Existing scores have different data/code/models; choose a new output directory")
    manifest = {"kind": "slm.story_cross_source_bpb", "status": "running", "fingerprint": fingerprint,
                "code": code_revision(), "environment": {**environment_info(), "transformers": transformers.__version__},
                "models": MODELS, "data": audit, "threads": args.threads, "device": "cpu", "loaded_dtype": "float32",
                "protocol": "Existing bits_per_byte; model-specific start token; nonoverlapping 256-token chunks; identical full validation texts"}
    write(manifest_path, manifest)
    results, scores = {}, {}
    for spec in MODELS:
        folder = ROOT / spec["path"]
        if sha256_file(folder / "model.safetensors") != spec["weights_sha256"]:
            raise ValueError(f"Frozen model changed: {spec['key']}")
        token = (AutoTokenizer.from_pretrained(folder, local_files_only=True) if (folder / "tokenizer.json").exists()
                 else LlamaTokenizer(vocab_file=str(folder / "tokenizer.model"), legacy=True))
        token.model_max_length = 2**31-1  # Actual model inputs are chunked at 256, never this length.
        hf = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.float32).eval()
        if hf.config.max_position_embeddings < 256 or sum(p.numel() for p in hf.parameters()) != 935040:
            raise ValueError("Unexpected context limit or frozen model parameter count")
        model, tok = HFModel(hf, 256), HFTokenizer(token)
        score_path = args.out / f"{spec['key']}-items.jsonl"
        rows = [json.loads(s) for s in score_path.read_text(encoding="utf-8").splitlines()] if score_path.exists() else []
        for index, row in enumerate(rows):
            if row["index"] != index or row["text_sha256"] != exact[index]:
                raise ValueError("Resume item order or text hash mismatch")
        started, start = time.perf_counter(), len(rows)
        print(f"{spec['label']}: {start}/{len(texts)} resumed", flush=True)
        with score_path.open("a", encoding="utf-8") as stream:
            for i in range(start, len(texts)):
                measured = bits_per_byte(model, tok, [texts[i]], torch.device("cpu"), max_len=256)
                if not math.isfinite(measured["nll_per_token"]):
                    raise ValueError(f"Nonfinite score on item {i}")
                row = {"index": i, "text_sha256": exact[i], "bytes": measured["bytes"],
                       "tokens": measured["tokens"], "nll": measured["nll_per_token"] * measured["tokens"]}
                stream.write(json.dumps(row) + "\n"); rows.append(row)
                if (i+1) % 250 == 0 or i+1 == len(texts):
                    stream.flush()
                    seconds = time.perf_counter()-started
                    remaining = seconds/(i+1-start)*(len(texts)-i-1)
                    print(f"{spec['key']}: {i+1}/{len(texts)}, {remaining/60:.1f} min remaining for this model", flush=True)
        nll = np.array([r["nll"] for r in rows], dtype=np.float64)
        byte_counts = np.array([r["bytes"] for r in rows], dtype=np.int64)
        scores[spec["key"]] = nll
        results[spec["key"]] = {"label": spec["label"], "parameters": sum(p.numel() for p in hf.parameters()),
                                 "documents": len(rows), "bytes": int(byte_counts.sum()),
                                 "tokens": sum(r["tokens"] for r in rows),
                                 "bits_per_byte": float(nll.sum()/math.log(2)/byte_counts.sum()),
                                 "items_sha256": sha256_file(score_path)}
        del hf, model
    intervals = paired_bootstrap(scores["story935k"], scores["tiny1m"], byte_counts)
    summary = {"kind": "slm.story_cross_source_bpb", "data": audit, "context": 256, "results": results,
               "story_minus_tiny_bpb": results["story935k"]["bits_per_byte"]-results["tiny1m"]["bits_per_byte"],
               "paired_bootstrap": intervals,
               "scope": "Frozen-model descriptive cross-source likelihood; not a human coherence or overall model ranking"}
    write(args.out / "summary.json", summary)
    manifest["status"] = "completed"; write(manifest_path, manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
