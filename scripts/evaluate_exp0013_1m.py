"""One-shot held-out evaluation for the paired 1M data-mixture pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slm import tokenizer as slm_tokenizer
from slm.benchmarks import load_items, multiple_choice
from slm.evaluation import bits_per_byte, read_texts
from slm.runlog import sha256_file, write_json
from slm.train import load_model


ROOT = Path(__file__).resolve().parents[1]
FSC_SPLITS = {
    "fsc_en_dev": "data/processed/fsc-cc0-open-weights-v4/splits/en-dev.jsonl",
    "fsc_fr_dev": "data/processed/fsc-cc0-open-weights-v4/splits/fr-dev.jsonl",
    "fsc_en_test": "data/processed/fsc-cc0-open-weights-v4/splits/en-test.jsonl",
    "fsc_fr_test": "data/processed/fsc-cc0-open-weights-v4/splits/fr-test.jsonl",
}
MOT_SPLITS = {
    "mot_dev": "data/processed/exp0012-mot-v1.11/temporal/dev.jsonl",
    "mot_test_2024_plus": "data/processed/exp0012-mot-v1.11/temporal/test.jsonl",
}
BENCHMARKS = [
    "data/benchmarks/arc_easy.jsonl", "data/benchmarks/arc_challenge.jsonl",
    "data/benchmarks/hellaswag.jsonl", "data/benchmarks/belebele_en.jsonl",
    "data/benchmarks/belebele_fr.jsonl", "data/benchmarks/winogrande.jsonl",
]


def mot_texts(path: str | Path, language: str) -> list[str]:
    if language not in {"en", "fr"}:
        raise ValueError(f"Unsupported MOT evaluation language: {language}")
    source = "mot-v1.11-en" if language == "en" else "mot-v1.11-fr"
    with open(path, encoding="utf-8") as handle:
        return [row["text"] for line in handle if line.strip()
                if (row := json.loads(line)).get("source") == source]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    tokenizer_dir = ROOT / "data/tokenizers/fsc-cc0-open-weights-v4"
    model = load_model(checkpoint, device).eval()
    tok = slm_tokenizer.load(tokenizer_dir)

    text_scores = {}
    for name, path in FSC_SPLITS.items():
        text_scores[name] = {**bits_per_byte(model, tok, read_texts(ROOT / path), device),
                             "path": path, "sha256": sha256_file(ROOT / path)}
    for name, path in MOT_SPLITS.items():
        for language in ("en", "fr"):
            key = f"{name}_{language}"
            texts = mot_texts(ROOT / path, language)
            if not texts:
                raise ValueError(f"No {language} records in MOT evaluation split: {path}")
            text_scores[key] = {**bits_per_byte(model, tok, texts, device),
                                "path": path, "sha256": sha256_file(ROOT / path), "language": language}

    mc_scores = {}
    for path in BENCHMARKS:
        mc_scores[Path(path).stem] = {
            **multiple_choice(model, tok, load_items(ROOT / path), device, batch_items=8),
            "path": path, "sha256": sha256_file(ROOT / path),
        }
    result = {
        "kind": "slm.exp0013-paired-1m-mixture-heldout-evaluation",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "params": model.num_params(), "device": str(device),
        "evaluator": "SLM-Tavenel local bits-per-byte and log-likelihood multiple-choice evaluator",
        "text_sets": text_scores, "benchmarks": mc_scores,
        "interpretation": {
            "fsc_test_and_mot_test": "pre-registered held-out descriptive outcomes; report after both arms finish; not for selection",
            "fsc_test_history": "FSC v4 test was already evaluated in EXP-0010; it is not a new untouched confirmatory test",
            "single_seed": "paired pilot only; does not estimate training-seed variability",
            "benchmarks": "secondary exploratory scores; do not select or promote a public benchmark checkpoint from this one pilot",
            "rights": "MOT-containing weights are not cleared for public release; keep source text local",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
