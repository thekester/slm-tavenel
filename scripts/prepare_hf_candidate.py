"""Build a local Hugging Face candidate bundle for the SLM-Tavenel 1M checkpoint.

This command does not upload anything. The generated weights remain a local candidate
and must not be pushed publicly until the source-rights and release gates are cleared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

EOS = "<|endoftext|>"


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fsc-scaling/1m-gated-s1234"
CHECKPOINT = RUN / "checkpoints/step_00015000.pt"
TOKENIZER = ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json"
MANIFEST = RUN / "run_manifest.json"
CARD = ROOT / "docs/model-cards/slm-tavenel-1m-hf-card.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "exports/slm-tavenel-1m-candidate")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        prior_index = out / "provenance.json"
        if not prior_index.is_file():
            raise FileExistsError(f"Refusing to overwrite a directory without our candidate index: {out}")
        prior = json.loads(prior_index.read_text(encoding="utf-8"))
        if prior.get("run_id") != "slm-tavenel-1m-gated-seed-1234-fsc-pilot-v1":
            raise FileExistsError(f"Refusing to overwrite an unrelated export directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eos_id = tokenizer.token_to_id(EOS)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if eos_id is None or pad_id is None:
        raise ValueError("The pinned tokenizer must define EOS and PAD token IDs")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["step"] != manifest["train_config"]["max_steps"]:
        raise ValueError("Checkpoint step does not match the completed run manifest")
    state = checkpoint["model"]
    if manifest["model_config"].get("tie_embeddings"):
        state = {key: value for key, value in state.items() if key != "lm_head.weight"}
    state = {key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
             for key, value in state.items()}

    shutil.copyfile(ROOT / "src/slm/hf_config.py", out / "hf_config.py")
    shutil.copyfile(ROOT / "src/slm/hf_model.py", out / "hf_model.py")
    shutil.copyfile(TOKENIZER, out / "tokenizer.json")
    shutil.copyfile(CARD, out / "README.md")
    shutil.copyfile(ROOT / "LICENSE", out / "CODE-LICENSE.txt")
    record_dir = out / "training_record"
    record_dir.mkdir(exist_ok=True)
    for filename in ("run_manifest.json", "metrics.jsonl", "eval.jsonl", "observability.jsonl"):
        shutil.copyfile(RUN / filename, record_dir / filename)
    (out / "requirements.txt").write_text(
        "torch>=2.3\ntransformers==4.57.6\ntokenizers==0.22.2\nsafetensors==0.8.0\n",
        encoding="utf-8",
    )

    config = {
        "model_type": "slm_tavenel",
        "architectures": ["SlmTavenelForCausalLM"],
        "auto_map": {
            "AutoConfig": "hf_config.SlmTavenelConfig",
            "AutoModelForCausalLM": "hf_model.SlmTavenelForCausalLM",
        },
        **manifest["model_config"],
        "max_position_embeddings": manifest["model_config"]["max_seq_len"],
        "tie_word_embeddings": manifest["model_config"]["tie_embeddings"],
        "torch_dtype": "float32",
        "use_cache": False,
        "eos_token_id": eos_id,
        "bos_token_id": None,
        "pad_token_id": pad_id,
    }
    write_json(out / "config.json", config)
    write_json(out / "generation_config.json", {
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "use_cache": False,
    })
    write_json(out / "tokenizer_config.json", {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": EOS,
        "pad_token": "<|pad|>",
        "bos_token": None,
        "unk_token": None,
        "model_max_length": manifest["model_config"]["max_seq_len"],
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
        "model_input_names": ["input_ids", "attention_mask"],
    })
    write_json(out / "special_tokens_map.json", {"eos_token": EOS, "pad_token": "<|pad|>"})
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})

    provenance = {
        "status": "local candidate; not cleared for public upload or open-weight release",
        "run_id": "slm-tavenel-1m-gated-seed-1234-fsc-pilot-v1",
        "training": {
            "steps": manifest["train_config"]["max_steps"],
            "tokens": 491_520_000,
            "seed": manifest["train_config"]["seed"],
            "optimizer": manifest["train_config"]["optimizer"],
            "dtype": manifest["train_config"]["dtype"],
            "deterministic": manifest["train_config"]["deterministic"],
            "parameters": manifest["params"],
            "code_commit": manifest["code"]["commit"],
            "code_dirty": manifest["code"]["dirty"],
        },
        "data": manifest["data"]["provenance"],
        "integrity": {
            "source_checkpoint_sha256": sha256(CHECKPOINT),
            "tokenizer_sha256": sha256(TOKENIZER),
            "run_manifest_sha256": sha256(MANIFEST),
            "frozen_token_manifest_sha256": manifest["data"]["manifest_sha256"],
            "files": {},
        },
        "public_release": {
            "approved": False,
            "reason": "source rights/privacy review and native-to-Hub inference parity are unfinished",
        },
    }
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            provenance["integrity"]["files"][path.relative_to(out).as_posix()] = sha256(path)
    write_json(out / "provenance.json", provenance)
    print(f"Prepared local-only candidate bundle: {out}")
    print("Do not upload publicly until data rights/privacy review and inference parity are approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
