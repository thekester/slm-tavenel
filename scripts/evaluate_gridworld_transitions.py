#!/usr/bin/env python3
"""Next-state prediction accuracy of a gridworld transition model (EXP-0039).

For each row state(t), action, state(t+1), the model sees the true prefix and predicts every token
of state(t+1) (teacher forcing). A state counts as correct only if all 65 tokens are. Reported on the
validation split, the intervention split (all six actions from validation states, separated into
actions that change the state and actions that leave it unchanged) and the held-out layouts.
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
from slm.runlog import sha256_file  # noqa: E402
from slm.train import load_model  # noqa: E402
from slm.worlds import gridworld as gw  # noqa: E402

S = gw.STATE_TOKENS


@torch.no_grad()
def score(model, rows: np.ndarray, device: torch.device, batch: int = 256) -> dict:
    token_hits = state_hits = 0
    changed_hits = changed_n = unchanged_hits = unchanged_n = 0
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(rows[i : i + batch].astype(np.int64)).to(device)
        logits, _ = model(x[:, :-1])
        pred = logits[:, S : 2 * S].argmax(-1)  # positions predicting state(t+1) tokens
        target = x[:, S + 1 :]
        ok = pred == target
        token_hits += int(ok.sum())
        exact = ok.all(-1).cpu().numpy()
        state_hits += int(exact.sum())
        changed = (x[:, :S] != x[:, S + 1 :]).any(-1).cpu().numpy()
        changed_hits += int(exact[changed].sum())
        changed_n += int(changed.sum())
        unchanged_hits += int(exact[~changed].sum())
        unchanged_n += int((~changed).sum())
    n = len(rows)
    return {"rows": n, "token_accuracy": token_hits / (n * S), "state_exact_match": state_hits / n,
            "state_exact_match_when_state_changes": changed_hits / max(1, changed_n),
            "state_exact_match_when_action_has_no_effect": unchanged_hits / max(1, unchanged_n),
            "rows_with_state_change": changed_n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/worlds/gridworld-v1/transitions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device).eval()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["row_length"]
    results = {}
    for split in ("val", "test_intervention", "test_layouts"):
        rows = np.fromfile(args.data / f"{split}_00000.bin", dtype=np.uint8).reshape(-1, row)
        results[split] = score(model, rows, device)
        print(split, results[split], flush=True)
    out = {"kind": "slm.gridworld_transition_eval", "checkpoint": str(args.checkpoint),
           "checkpoint_sha256": sha256_file(args.checkpoint),
           "data_manifest_sha256": sha256_file(args.data / "manifest.json"), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
