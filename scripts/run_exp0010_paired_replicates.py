"""Run the predeclared paired 1M gate replication from an immutable clean worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "02396acd15ef9bcdc453bcd53aade40819e2f69f"
SEEDS = list(range(421101, 421111))
DATA = ROOT / "data/tokens/fsc-cc0-open-weights-v4/training"
RUNS = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22"
GATED_CONFIG = ROOT / "configs/model/slm-1m-gated.yaml"
UNGATED_CONFIG = ROOT / "configs/model/slm-1m-ungated-match.yaml"
TOK_MANIFEST_SHA = "58065825d94485cca526f93ec993d5cfc8bec25ad742a2f6d6c32213370d48b8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(code_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=code_root, text=True).strip()


def validate_initializations(gated: dict, ungated: dict, model_cfgs: dict) -> None:
    for arm, state in (("gated", gated), ("ungated", ungated)):
        if state.get("step") != 0 or state.get("model_config") != model_cfgs[arm].__dict__:
            raise ValueError(f"Cached {arm} initialization does not match the paired experiment")
    if not ungated["model"] or any(name not in gated["model"] or
            not torch.equal(value, gated["model"][name]) for name, value in ungated["model"].items()):
        raise ValueError("Cached shared initial parameters are not exactly paired")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True,
                        help="clean detached worktree at the predeclared training revision")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(SEEDS):
        parser.error("--seeds must be a nonempty, unique subset of the predeclared seeds")
    code_root = args.code_root.resolve()
    if git(code_root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise SystemExit(f"Training worktree must be at {EXPECTED_COMMIT}")
    if git(code_root, "status", "--porcelain"):
        raise SystemExit("Training worktree must be clean")
    if sha256(DATA / "manifest.json") != TOK_MANIFEST_SHA:
        raise SystemExit("Pinned token-shard manifest hash changed")

    # Import the exact model implementation from the clean detached worktree.
    sys.path.insert(0, str(code_root / "src"))
    from slm.config import ModelConfig, TrainConfig, load_model_config  # noqa: E402
    from slm.model import SLM  # noqa: E402

    model_cfgs = {
        "gated": load_model_config(GATED_CONFIG),
        "ungated": load_model_config(UNGATED_CONFIG),
    }
    param_counts = {}
    for arm, cfg in model_cfgs.items():
        with torch.device("meta"):
            param_counts[arm] = SLM(cfg).num_params()
    if abs(param_counts["gated"] - param_counts["ungated"]) / param_counts["gated"] > 0.01:
        raise SystemExit(f"Parameter counts differ by more than 1%: {param_counts}")

    RUNS.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol": "experiments/EXP-0010-1m-paired-replication-2026-10-22.md",
        "protocol_sha256": sha256(ROOT / "experiments/EXP-0010-1m-paired-replication-2026-10-22.md"),
        "protocol_amendment": "experiments/EXP-0010-amendment-2026-10-22.md",
        "protocol_amendment_sha256": sha256(ROOT / "experiments/EXP-0010-amendment-2026-10-22.md"),
        "code_commit": EXPECTED_COMMIT,
        "code_tree_clean": True,
        "driver_sha256": sha256(Path(__file__).resolve()),
        "token_manifest_sha256": sha256(DATA / "manifest.json"),
        "tokenizer_sha256": json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["tokenizer_sha256"],
        "model_config_sha256": {"gated": sha256(GATED_CONFIG), "ungated": sha256(UNGATED_CONFIG)},
        "parameters": param_counts,
        "seeds": args.seeds,
        "runs": [],
    }

    base = {
        "data_dir": str(DATA), "seq_len": 1024, "micro_batch_size": 16,
        "grad_accum_steps": 2, "max_steps": 488, "optimizer": "adamw",
        "lr": 0.003, "min_lr": 0.0003, "weight_decay": 0.1,
        "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "grad_clip": 1.0,
        "schedule": "wsd", "warmup_steps": 48, "decay_fraction": 0.2,
        "dtype": "bfloat16", "compile": False, "deterministic": True,
        "log_interval": 25, "eval_interval": 122, "eval_batches": 20,
        "observe_interval": 122, "ckpt_interval": 244, "keep_checkpoints": 2,
        "resume": True, "doc_mask": False,
    }
    (RUNS / "driver-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for seed in args.seeds:
        seed_root = RUNS / f"seed-{seed}"
        init_root = seed_root / "initializations"
        init_root.mkdir(parents=True, exist_ok=True)
        gated_init, ungated_init = init_root / "gated.pt", init_root / "ungated.pt"

        if not gated_init.exists() or not ungated_init.exists():
            torch.manual_seed(seed)
            gated_model = SLM(model_cfgs["gated"])
            torch.manual_seed(seed)
            ungated_model = SLM(model_cfgs["ungated"])
            gated_state = gated_model.state_dict()
            ungated_state = ungated_model.state_dict()
            shared = sorted(ungated_state)
            if any(name not in gated_state for name in shared):
                raise AssertionError("Model arms do not have matching shared parameter names")
            for name in shared:
                ungated_state[name].copy_(gated_state[name])
            ungated_model.load_state_dict(ungated_state)
            if any(not torch.equal(gated_state[name], ungated_model.state_dict()[name]) for name in shared):
                raise AssertionError("Shared initial parameters are not exactly paired")
            torch.save({"step": 0, "model": gated_state, "model_config": model_cfgs["gated"].__dict__}, gated_init)
            torch.save({"step": 0, "model": ungated_model.state_dict(), "model_config": model_cfgs["ungated"].__dict__}, ungated_init)

        validate_initializations(
            torch.load(gated_init, map_location="cpu", weights_only=False),
            torch.load(ungated_init, map_location="cpu", weights_only=False), model_cfgs)
        seed_record = {
            "seed": seed,
            "initialization_sha256": {"gated": sha256(gated_init), "ungated": sha256(ungated_init)},
            "shared_initial_parameters_equal": True,
            "arms": {},
        }
        for arm in ("gated", "ungated"):
            out_dir = seed_root / arm
            config_path = seed_root / f"{arm}-train.yaml"
            resolved = {
                **base,
                "model": str(GATED_CONFIG if arm == "gated" else UNGATED_CONFIG),
                "out_dir": str(out_dir), "seed": seed,
                "init_from": str(gated_init if arm == "gated" else ungated_init),
            }
            if config_path.exists():
                old = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if old != resolved:
                    raise SystemExit(f"Refusing to reuse run with different config: {config_path}")
            else:
                config_path.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")

            final_ckpt = out_dir / "checkpoints/step_00000488.pt"
            if not final_ckpt.exists():
                log_path = out_dir / "console.log"
                out_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                clean_src = str(code_root / "src")
                env["PYTHONPATH"] = clean_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                with log_path.open("a", encoding="utf-8") as log:
                    cmd = [sys.executable, "-m", "slm", "train", str(config_path)]
                    subprocess.run(cmd, cwd=code_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not final_ckpt.is_file():
                raise SystemExit(f"Training returned without final checkpoint: {final_ckpt}")
            seed_record["arms"][arm] = {
                "run_manifest_sha256": sha256(out_dir / "run_manifest.json"),
                "checkpoint_sha256": sha256(final_ckpt),
                "metrics_sha256": sha256(out_dir / "metrics.jsonl"),
                "eval_sha256": sha256(out_dir / "eval.jsonl"),
                "observability_sha256": sha256(out_dir / "observability.jsonl"),
            }
            record["runs"].append({"seed": seed, "arm": arm, **seed_record["arms"][arm]})
            (RUNS / "driver-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (seed_root / "pair-record.json").write_text(json.dumps(seed_record, indent=2) + "\n", encoding="utf-8")
        print(f"completed paired seed {seed}: {param_counts}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
