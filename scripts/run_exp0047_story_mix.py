"""Serial, resumable local story-mixture experiments with development-only selection.

No Hub upload, no test-set scoring, and no automatic replacement of public weights.
Put runs/EXP-0047/STOP on disk to pause before the next training/evaluation stage.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm import tokenizer
from slm.config import load_model_config, load_train_config
from slm.evaluation import bits_per_byte
from slm.hf_adapter import HFModel, HFTokenizer
from slm.model import SLM
from slm.runlog import code_revision, environment_info, sha256_file, write_json
from slm.train import load_model

BASE = ROOT / "runs/EXP-0047"
DATA = ROOT / "data/processed/exp0047-story-mix"
TOKENIZER = ROOT / "data/processed/exp0045-tokenizer-v2-s421201/tokenizer.json"
REFERENCES = {
    "published-story935k": ("data/models/fluency/story-935k-100m-v2", "8804d57668ac492454d9835841ff2a57f7867dbcbb36df4452ffedfa8813c9a8"),
    "tiny1m": ("data/models/fluency/tiny1m/hf", "f30ae85c53797b5944da7f235162756b31987233b44eac36ed449afd1b82523b"),
}


def status(stage: str, **fields) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(), "stage": stage, **fields}
    write_json(BASE / "status.json", record)
    with (BASE / "status.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def check_pause() -> None:
    if (BASE / "STOP").exists():
        status("paused", reason="STOP marker; restart same command after removing marker")
        raise SystemExit(0)


def freeze_inputs() -> dict:
    paths = sorted((ROOT / "src/slm").rglob("*.py")) + [
        ROOT / "configs/model/slm-story-935k.yaml", TOKENIZER, DATA / "manifest.json", DATA / "complete.json", DATA / "audit.json",
        ROOT / "scripts/build_exp0047_story_mix.py", ROOT / "scripts/audit_exp0047_story_mix.py", Path(__file__),
        ROOT / "configs/eval/fluency-stories-en-v1.jsonl",
    ]
    paths += sorted((ROOT / "configs/train").glob("exp0047-*.yaml"))
    for percent in (0, 30, 60):
        folder = ROOT / f"data/tokens/exp0047-story-mix/tiny{percent:02d}"
        paths += [folder / "manifest.json", folder / "provenance.json"] + sorted(folder.glob("*.bin"))
    for name in ("simple", "tiny"):
        paths.append(DATA / f"{name}-dev.jsonl")
    for folder, expected in REFERENCES.values():
        if sha256_file(ROOT / folder / "model.safetensors") != expected:
            raise ValueError(f"Reference weights changed: {folder}")
        paths += sorted(p for p in (ROOT / folder).iterdir() if p.is_file() and p.suffix in (".json", ".model", ".safetensors"))
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    target = BASE / "frozen-inputs.json"
    if target.exists():
        if json.loads(target.read_text())["hashes"] != hashes:
            raise ValueError("Frozen inputs changed; do not resume incompatible experiments")
    else:
        write_json(target, {"hashes": hashes, "code": code_revision(), "environment": environment_info(),
                            "scope": "local exploratory pilots; no rights sign-off or human evaluation"})
        for p in paths:
            if p.suffix == ".py" or (p.suffix == ".yaml" and "configs" in p.parts):
                out = BASE / "executed-source" / p.relative_to(ROOT)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, out)
    return hashes


def run_training(config: Path) -> Path:
    check_pause()
    cfg = load_train_config(config)
    checkpoint = ROOT / cfg.out_dir / "checkpoints" / f"step_{cfg.max_steps:08d}.pt"
    if checkpoint.exists():
        return checkpoint
    name = Path(cfg.out_dir).name
    status("training", run=name, config=str(config.relative_to(ROOT)), expected_steps=cfg.max_steps)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    with (BASE / f"{name}.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-u", "-m", "slm", "train", str(config)],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(BASE / "training-process.json", {"pid": child.pid, "run": name, "config": str(config)})
        if child.wait() != 0:
            raise RuntimeError(f"Training failed: {name}; see {BASE / (name+'.log')}")
    if not checkpoint.exists():
        raise RuntimeError(f"Final checkpoint missing: {checkpoint}")
    return checkpoint


def load_reference(key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

    folder = ROOT / REFERENCES[key][0]
    tok = (AutoTokenizer.from_pretrained(folder, local_files_only=True) if (folder / "tokenizer.json").exists()
           else LlamaTokenizer(vocab_file=str(folder / "tokenizer.model"), legacy=True))
    tok.model_max_length = 2**31-1
    hf = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.float32).eval()
    return HFModel(hf, 256), HFTokenizer(tok)


def score(key: str, checkpoint: Path | None = None) -> dict:
    check_pause()
    out = BASE / "dev-evaluations" / key
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = {"frozen_inputs_sha256": sha256_file(BASE / "frozen-inputs.json"),
                   "weights_sha256": sha256_file(checkpoint) if checkpoint else REFERENCES[key][1]}
    if (out / "summary.json").exists():
        saved = json.loads((out / "summary.json").read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Evaluation cache belongs to different inputs")
        return saved
    status("evaluating_dev", model=key)
    if checkpoint:
        model, tok = load_model(checkpoint, "cpu").eval(), tokenizer.load(TOKENIZER)
    else:
        model, tok = load_reference(key)
    if model.num_params() != 935040:
        raise ValueError("935,040-parameter cap violated")
    result = {"model": key, "fingerprint": fingerprint, "parameters": model.num_params(), "corpora": {},
              "protocol": "Existing bits_per_byte, CPU float32, own BOS/EOS prefix, nonoverlapping 256-token chunks; byte-weighted corpus BPB",
              "scope": "Development only. Reference exposure to source train is unknown. Native candidates vs rounded public weights. Not a narrative-quality verdict."}
    for source in ("simple", "tiny"):
        path = DATA / f"{source}-dev.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows, nll, byte_count, tokens = [], 0.0, 0, 0
        for i, record in enumerate(records):
            measured = bits_per_byte(model, tok, [record["text"]], torch.device("cpu"), max_len=256)
            item_nll = measured["nll_per_token"] * measured["tokens"]
            if not math.isfinite(item_nll):
                raise ValueError("Nonfinite evaluation loss")
            rows.append({"index": i, "id": record["id"], "text_sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
                         "group": record["group"], "bytes": measured["bytes"], "tokens": measured["tokens"], "nll": item_nll})
            nll += item_nll; byte_count += measured["bytes"]; tokens += measured["tokens"]
        item_path = out / f"{source}-items.jsonl"
        with item_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        result["corpora"][source] = {"bits_per_byte": nll / math.log(2) / byte_count, "bytes": byte_count,
                                      "tokens": tokens, "documents": len(rows), "input_sha256": sha256_file(path),
                                      "items_sha256": sha256_file(item_path)}
    result["selection_mean_bpb"] = sum(c["bits_per_byte"] for c in result["corpora"].values()) / 2
    write_json(out / "summary.json", result)
    del model, tok
    gc.collect()
    return result


def raw_examples(key: str, checkpoint: Path | None = None) -> None:
    """Fixed public development prompts, without selection or automatic quality labels."""
    check_pause()
    folder = BASE / "generations"
    folder.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        model, tok = load_model(checkpoint, "cpu").eval(), tokenizer.load(TOKENIZER)
    else:
        model, tok = load_reference(key)
    prompts = [json.loads(line) for line in (ROOT / "configs/eval/fluency-stories-en-v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt["prompt"]).ids], dtype=torch.long)
        n = min(160, 256-ids.shape[1])
        if n <= 0:
            raise ValueError("Prompt exceeds training context")
        for seed in (0, 1):
            torch.manual_seed(seed)
            with torch.inference_mode():
                if checkpoint:
                    output = model.generate(ids, n, temperature=0.8, top_k=50,
                                            eos_id=tok.token_to_id(tokenizer.EOS),
                                            generator=torch.Generator().manual_seed(seed))
                else:
                    output = model.model.generate(input_ids=ids, max_new_tokens=n, do_sample=True,
                                                  temperature=0.8, top_k=50, top_p=1.0,
                                                  eos_token_id=tok.tok.eos_token_id,
                                                  pad_token_id=tok.tok.eos_token_id)
            new_ids = output[0, ids.shape[1]:].tolist()
            completion = (tok.decode(new_ids) if checkpoint else tok.tok.decode(new_ids, skip_special_tokens=True))
            rows.append({"model": key, "prompt_id": prompt["id"], "prompt": prompt["prompt"],
                         "seed": seed, "temperature": 0.8, "top_k": 50, "max_new_tokens": n,
                         "generated_token_ids": new_ids, "completion": completion,
                         "scope": "Unselected outputs on five already public development prefixes; not blind human ratings or an independent narrative benchmark"})
    with (folder / f"{key}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    del model, tok
    gc.collect()


def run(extend: bool) -> None:
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing unexpectedly slow CPU training")
    if not (DATA / "complete.json").exists():
        raise RuntimeError("Build the corpus before starting the queue")
    audit = json.loads((DATA / "audit.json").read_text())
    if not audit["passed"] or audit["manifest_sha256"] != sha256_file(DATA / "manifest.json"):
        raise ValueError("A valid corpus audit is required")
    with torch.device("meta"):
        if SLM(load_model_config(ROOT / "configs/model/slm-story-935k.yaml")).num_params() != 935040:
            raise ValueError("Unexpected architecture size")
    freeze_inputs()
    results = {}
    # Start with a mixed run so the first completed training addresses the gap.
    for percent in (30, 0, 60):
        key = f"tiny{percent:02d}-100m-s421201"
        checkpoint = run_training(ROOT / f"configs/train/exp0047-story-tiny{percent:02d}-100m.yaml")
        results[key] = score(key, checkpoint)
        write_json(BASE / "pilot-results.json", results)
    for key in REFERENCES:
        results[key] = score(key)
    winner = min((k for k in results if k.startswith("tiny") and "100m" in k), key=lambda k: results[k]["selection_mean_bpb"])
    selection = {"rule": "minimum equal-corpus mean dev BPB across three 100M-token pilots; no test scores used",
                 "selected": winner, "results": results, "confirmation": "pending; one training seed, development selection, no human ratings",
                 "test": "reserved and unscored; unknown tiny1m exposure prevents claiming it is unseen by the reference"}
    write_json(BASE / "selection.json", selection)
    selected_checkpoint = ROOT / f"runs/EXP-0047/{winner}/checkpoints/step_00006104.pt"
    raw_examples(winner, selected_checkpoint)
    for key in REFERENCES:
        raw_examples(key)
    if extend:
        check_pause()
        # Fresh initialization and full 300M cosine schedule: not a continuation
        # from the low-learning-rate end of the 100M run.
        percent = int(winner[4:6])
        config = yaml.safe_load((ROOT / f"configs/train/exp0047-story-tiny{percent:02d}-100m.yaml").read_text())
        config.update(max_steps=18311, out_dir=f"runs/EXP-0047/tiny{percent:02d}-300m-s421201")
        config_path = BASE / "selected-300m.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        checkpoint = run_training(config_path)
        results[f"tiny{percent:02d}-300m-s421201"] = score(f"tiny{percent:02d}-300m-s421201", checkpoint)
        raw_examples(f"tiny{percent:02d}-300m-s421201", checkpoint)
    write_json(BASE / "results.json", {"results": results, "selection": winner, "public_weights_changed": False,
                                      "next": "export/check selected candidate; independent confirmation, multiple seeds and human narrative evaluation remain"})
    status("completed", results="runs/EXP-0047/results.json", public_weights_changed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-extend", action="store_true")
    args = parser.parse_args()
    try:
        run(not args.no_extend)
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
