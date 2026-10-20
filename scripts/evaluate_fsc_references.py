"""Score pinned external SLMs on the same FSC validation text and local MC benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from slm.benchmarks import load_items, multiple_choice
from slm.evaluation import bits_per_byte, read_texts
from slm.hf_adapter import load
from slm.runlog import sha256_file, write_json
from fsc_evaluation_cache import complete, input_signature


REFERENCES = {
    "pythia-14m": {"model": "EleutherAI/pythia-14m", "revision": "d31f659f077f9ea500c4d7faa999893d7f28cbdf",
                   "max_seq_len": 1024},
    "smollm2-135m": {"model": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                     "max_seq_len": 1024},
    "tiny1m": {"model": "shibatch/tiny1m", "revision": "fdacad8ff9dc27bd72c77d3684075a494be1ba11",
               "subfolder": "hf", "max_seq_len": 256},
}
BENCHMARKS = [
    "data/benchmarks/arc_easy.jsonl", "data/benchmarks/hellaswag.jsonl",
    "data/benchmarks/belebele_en.jsonl", "data/benchmarks/belebele_fr.jsonl",
]
VAL_FILES = {"en": "data/processed/fsc-pilot-v1-token-balanced/en-val.jsonl",
             "fr": "data/processed/fsc-pilot-v1-token-balanced/fr-val.jsonl"}


def evaluate(name: str, device: str, output_dir: Path) -> dict:
    spec = REFERENCES[name]
    model_id, revision = spec["model"], spec["revision"]
    expected = [output_dir / "fsc-pilot-v1-mc.json", output_dir / "fsc-pilot-v1-bpb-en.json",
                output_dir / "fsc-pilot-v1-bpb-fr.json"]
    signature = input_signature(dict(spec), BENCHMARKS, VAL_FILES)
    if complete(expected, signature):
        return {"model": model_id, "revision": revision, "status": "already_complete",
                "mc": str(expected[0]), "bpb": {"en": str(expected[1]), "fr": str(expected[2])}}
    model, tok = load(model_id, torch.device(device), revision=revision,
                      max_seq_len=spec["max_seq_len"], subfolder=spec.get("subfolder"))
    mc_results = {}
    for path in BENCHMARKS:
        mc_results[Path(path).stem] = multiple_choice(model, tok, load_items(path), torch.device(device), batch_items=8)
    mc = {
        "evaluation_inputs": signature,
        "checkpoint": None, "hf_model": model_id, "hf_revision": revision,
        "model_params": model.num_params(), "max_seq_len": model.cfg.max_seq_len,
        "evaluation_harness": "SLM-Tavenel local evaluator; model context cap recorded per model; batch size 8",
        "hf_subfolder": spec.get("subfolder"),
        "benchmarks": [{"path": path, "sha256": sha256_file(path)} for path in BENCHMARKS],
        "results": mc_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "fsc-pilot-v1-mc.json", mc)
    bpb_results = {}
    for language, path in VAL_FILES.items():
        result = bits_per_byte(model, tok, read_texts(path), torch.device(device))
        bpb_results[language] = {**result, "data": path, "data_sha256": sha256_file(path)}
        write_json(output_dir / f"fsc-pilot-v1-bpb-{language}.json", {
            "evaluation_inputs": signature,
            "hf_model": model_id, "hf_revision": revision, "model_params": model.num_params(),
            "max_seq_len": model.cfg.max_seq_len, **bpb_results[language],
        })
    del model, tok
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"model": model_id, "revision": revision, "mc": str(output_dir / "fsc-pilot-v1-mc.json"),
            "bpb": bpb_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(REFERENCES), default=sorted(REFERENCES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", default="runs/reference")
    args = parser.parse_args()
    results = [evaluate(name, args.device, Path(args.out_root) / name) for name in args.models]
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
