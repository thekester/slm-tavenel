"""Compare EXP-0010 and EXP-0011 checkpoints on identical fixed validation windows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from slm.config import load_train_config
from slm.data.shards import TokenShards
from slm.train import evaluate_loss, load_model


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22/seed-421101/gated/checkpoints/step_00000488.pt"
CONTINUED = ROOT / "runs/EXP-0011-1m-low-lr-continuation-2026-10-22/checkpoints/step_00000488.pt"
OUT = ROOT / "runs/EXP-0011-1m-low-lr-continuation-2026-10-22/same-windows-validation.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = replace(load_train_config(ROOT / "configs/train/fsc-cc0-open-weight-1m-exp0011-continuation.yaml"),
                  seed=421101, eval_batches=20)
    data = TokenShards(cfg.data_dir, "val")
    results = {}
    for label, checkpoint in (("exp0010", BASE), ("exp0011", CONTINUED)):
        model = load_model(checkpoint, device=device)
        results[label] = evaluate_loss(model, data, cfg, device)
        del model
    record = {
        "kind": "slm.exp0011.same_windows_validation",
        "device": str(device),
        "validation_seed": 421102,
        "validation_batches": 20,
        "baseline_checkpoint": str(BASE.relative_to(ROOT)),
        "continued_checkpoint": str(CONTINUED.relative_to(ROOT)),
        "exp0010_val_loss": results["exp0010"],
        "exp0011_val_loss": results["exp0011"],
        "delta_exp0011_minus_exp0010": results["exp0011"] - results["exp0010"],
        "lower_is_better": True,
        "scope": "development diagnostic only; not benchmark or sealed-test evidence",
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
