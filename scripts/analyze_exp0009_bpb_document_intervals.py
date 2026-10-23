"""Paired document bootstrap for the FSC 10M vs grown-25M BPB comparison.

Scores each validation document independently, then resamples the same documents
for both checkpoints within language. This quantifies validation-document
sampling uncertainty conditional on these two checkpoints; it does not estimate
training-seed variability.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm.config import ModelConfig  # noqa: E402
from slm.evaluation import read_texts  # noqa: E402
from slm.model import SLM  # noqa: E402
from slm.tokenizer import EOS  # noqa: E402


TOKENIZER_PATH = ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json"
VALIDATION = {
    "en": ROOT / "data/processed/fsc-pilot-v1-token-balanced/en-val.jsonl",
    "fr": ROOT / "data/processed/fsc-pilot-v1-token-balanced/fr-val.jsonl",
}
CHECKPOINTS = {
    "10m": ROOT / "runs/fsc-scaling/10m-gated-s1234/checkpoints/step_00015000.pt",
    "25m_grown": ROOT / "runs/fsc-scaling/25m-gated-grown-s1234/checkpoints/step_00015000.pt",
}
OUT = ROOT / "runs/fsc-scaling/comparisons/exp0009-bpb-document-bootstrap.json"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20261024
BATCH_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> SLM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def score_documents(model: SLM, tokenizer: Tokenizer, texts: list[str], device: torch.device, label: str) -> list[dict]:
    """Compute one summed token NLL per document, batching equal-length chunks."""
    eos = tokenizer.token_to_id(EOS)
    max_len = model.cfg.max_seq_len
    rows: list[dict] = [
        {"nll": 0.0, "tokens": 0, "bytes": len(text.encode("utf-8"))}
        for text in texts
    ]
    buckets: dict[int, list[tuple[int, list[int]]]] = {}
    total_chunks = 0
    for doc_index, text in enumerate(texts):
        ids = [eos] + tokenizer.encode(text).ids
        for start in range(0, len(ids) - 1, max_len):
            chunk = ids[start : start + max_len + 1]
            buckets.setdefault(len(chunk), []).append((doc_index, chunk))
            total_chunks += 1

    completed = 0
    for chunk_len, chunks in sorted(buckets.items(), reverse=True):
        for offset in range(0, len(chunks), BATCH_SIZE):
            batch_rows = chunks[offset : offset + BATCH_SIZE]
            batch = torch.tensor([chunk for _, chunk in batch_rows], dtype=torch.long, device=device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inputs)
            log_probs = logits.float().log_softmax(-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            chunk_nll = (-token_log_probs.sum(dim=1)).to(dtype=torch.float64).cpu().tolist()
            target_count = targets.shape[1]
            for (doc_index, _), nll in zip(batch_rows, chunk_nll, strict=True):
                rows[doc_index]["nll"] += float(nll)
                rows[doc_index]["tokens"] += target_count
            completed += len(batch_rows)
            if completed % 2_000 < len(batch_rows) or completed == total_chunks:
                print(f"{label}: scored {completed:,}/{total_chunks:,} chunks", flush=True)

    return rows


def bpb(rows: list[dict]) -> float:
    nll = sum(row["nll"] for row in rows)
    nbytes = sum(row["bytes"] for row in rows)
    return nll / math.log(2) / nbytes


def bootstrap_delta(base_rows: list[dict], grown_rows: list[dict], rng: np.random.Generator) -> dict:
    if len(base_rows) != len(grown_rows):
        raise ValueError("Paired checkpoint records have different document counts")
    base_nll = np.asarray([row["nll"] for row in base_rows], dtype=np.float64)
    grown_nll = np.asarray([row["nll"] for row in grown_rows], dtype=np.float64)
    byte_counts = np.asarray([row["bytes"] for row in base_rows], dtype=np.float64)
    n_docs = len(base_rows)
    draws = rng.integers(0, n_docs, size=(BOOTSTRAPS, n_docs), endpoint=False)
    denominator = byte_counts[draws].sum(axis=1) * math.log(2)
    deltas = (base_nll[draws].sum(axis=1) - grown_nll[draws].sum(axis=1)) / denominator
    point = bpb(base_rows) - bpb(grown_rows)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "documents": n_docs,
        "delta_bpb_10m_minus_25m": float(point),
        "ci95_percentile": [float(lo), float(hi)],
        "bootstrap_draws": BOOTSTRAPS,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    texts = {lang: read_texts(path) for lang, path in VALIDATION.items()}
    if not texts["en"] or not texts["fr"]:
        raise ValueError("A validation language split is empty")

    scored: dict[str, dict[str, list[dict]]] = {scale: {} for scale in CHECKPOINTS}
    runtimes = {}
    score_reproduction = {}
    for scale, checkpoint_path in CHECKPOINTS.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = SLM(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        runtimes[scale] = {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(checkpoint_path),
            "step": int(checkpoint["step"]),
            "parameters": model.num_params(),
        }
        for lang, documents in texts.items():
            scored[scale][lang] = score_documents(model, tokenizer, documents, device, f"{scale}/{lang}")
            reference = json.loads(
                (ROOT / f"runs/fsc-scaling/{'10m-gated-s1234' if scale == '10m' else '25m-gated-grown-s1234'}/analysis/fsc-pilot-v1-bpb-{lang}.json").read_text(encoding="utf-8")
            )
            calculated = bpb(scored[scale][lang])
            reference_bpb = float(reference["bits_per_byte"])
            score_reproduction[f"{scale}/{lang}"] = {
                "saved_aggregate_bpb": reference_bpb,
                "batched_document_score_bpb": calculated,
                "absolute_difference": abs(calculated - reference_bpb),
            }
            if abs(calculated - reference_bpb) > 2e-5:
                raise ValueError(
                    f"Batched per-document score for {scale}/{lang} ({calculated:.8f}) "
                    f"does not reproduce saved aggregate ({reference_bpb:.8f})"
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    intervals = {
        lang: bootstrap_delta(scored["10m"][lang], scored["25m_grown"][lang], rng)
        for lang in ("en", "fr")
    }

    # Equal-language mean is the preregistered way the project combines English
    # and French rather than weighting whichever language has more bytes.
    en_base = np.asarray([row["nll"] for row in scored["10m"]["en"]], dtype=np.float64)
    en_grown = np.asarray([row["nll"] for row in scored["25m_grown"]["en"]], dtype=np.float64)
    en_bytes = np.asarray([row["bytes"] for row in scored["10m"]["en"]], dtype=np.float64)
    fr_base = np.asarray([row["nll"] for row in scored["10m"]["fr"]], dtype=np.float64)
    fr_grown = np.asarray([row["nll"] for row in scored["25m_grown"]["fr"]], dtype=np.float64)
    fr_bytes = np.asarray([row["bytes"] for row in scored["10m"]["fr"]], dtype=np.float64)
    en_idx = rng.integers(0, len(en_base), size=(BOOTSTRAPS, len(en_base)), endpoint=False)
    fr_idx = rng.integers(0, len(fr_base), size=(BOOTSTRAPS, len(fr_base)), endpoint=False)
    en_draw = (en_base[en_idx].sum(axis=1) - en_grown[en_idx].sum(axis=1)) / (en_bytes[en_idx].sum(axis=1) * math.log(2))
    fr_draw = (fr_base[fr_idx].sum(axis=1) - fr_grown[fr_idx].sum(axis=1)) / (fr_bytes[fr_idx].sum(axis=1) * math.log(2))
    balanced_draw = 0.5 * (en_draw + fr_draw)
    balanced_point = 0.5 * (intervals["en"]["delta_bpb_10m_minus_25m"] + intervals["fr"]["delta_bpb_10m_minus_25m"])
    balanced_lo, balanced_hi = np.quantile(balanced_draw, [0.025, 0.975])
    intervals["equal_language_mean"] = {
        "delta_bpb_10m_minus_25m": float(balanced_point),
        "ci95_percentile": [float(balanced_lo), float(balanced_hi)],
        "definition": "arithmetic mean of language-specific BPB differences, resampling documents independently within each language",
        "bootstrap_draws": BOOTSTRAPS,
    }

    output = {
        "schema": "slm.exp0009_paired_bpb_document_bootstrap.v1",
        "comparison": "10M minus grown-25M BPB; positive values favor grown-25M",
        "metric": "sum of next-token negative log likelihood divided by ln(2) times UTF-8 bytes",
        "evaluation": {
            "python_device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "analyzer_script_sha256": sha256(Path(__file__).resolve()),
            "model_code_sha256": sha256(ROOT / "src/slm/model.py"),
            "tokenizer_sha256": sha256(TOKENIZER_PATH),
            "validation": {
                lang: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "documents": len(texts[lang])}
                for lang, path in VALIDATION.items()
            },
            "batch_size": BATCH_SIZE,
            "batch_scoring_matches_saved_aggregate_tolerance_bpb": 2e-5,
            "batch_scoring_reproduction": score_reproduction,
            "checkpoints": runtimes,
        },
        "point_estimates": {
            lang: {
                "10m_bpb": bpb(scored["10m"][lang]),
                "grown_25m_bpb": bpb(scored["25m_grown"][lang]),
                "delta_10m_minus_25m": bpb(scored["10m"][lang]) - bpb(scored["25m_grown"][lang]),
            }
            for lang in ("en", "fr")
        },
        "uncertainty": {
            "method": "paired percentile bootstrap of validation documents, sampled with replacement within language",
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAPS,
            "scope": "conditional on the two saved checkpoints; does not estimate training-seed variability, data-source uncertainty or benchmark exposure",
        },
        "results": intervals,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUT.relative_to(ROOT).as_posix(), "results": intervals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
