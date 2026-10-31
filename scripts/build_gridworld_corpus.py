#!/usr/bin/env python3
"""Generate the key-door gridworld corpus for the second world-model pilot (EXP-0039).

Transition rows (configurations A and C): state(t) [65 tokens], action [1], state(t+1) [65] = 131
tokens, from episodes of a behaviour policy that takes the optimal action with probability 0.8.
Plan rows (configuration B): start state [65], then the optimal action sequence, end-of-plan, padded
to 131 tokens. Splits:
  train / val          20 of the 24 wall-and-door layouts, behaviour policy;
  test_intervention    the same states as val with every one of the 6 actions (uniform interventions);
  test_layouts         the 4 held-out layouts (compositional transfer), behaviour policy;
  start_worlds_*       start configurations for planning evaluation (in-distribution and held-out layouts).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import code_revision, sha256_file  # noqa: E402
from slm.worlds import gridworld as gw  # noqa: E402

ROW = 2 * gw.STATE_TOKENS + 1
END_OF_PLAN = gw.NO_KEY  # reused as a terminator inside plan rows (never ambiguous at that position)
HELD_OUT = [(2, 1), (3, 4), (4, 6), (5, 2)]


def episodes(rng, layouts, n_transitions, expert_prob=0.8, max_steps=40):
    rows, states = [], []
    while len(rows) < n_transitions:
        world = gw.random_world(rng, layouts)
        for _ in range(max_steps):
            action = gw.behaviour_action(world, rng, expert_prob)
            nxt = world.step(action)
            rows.append(world.tokens() + [gw.ACTION_BASE + action] + nxt.tokens())
            states.append(world)
            world = nxt
            if world.done or len(rows) >= n_transitions:
                break
    return np.array(rows, dtype=np.uint8), states


def plan_rows(rng, layouts, n):
    rows = []
    while len(rows) < n:
        world = gw.random_world(rng, layouts)
        plan = gw.shortest_plan(world)
        tokens = world.tokens() + [gw.ACTION_BASE + a for a in plan] + [END_OF_PLAN]
        if len(tokens) <= ROW:
            rows.append(tokens + [END_OF_PLAN] * (ROW - len(tokens)))
    return np.array(rows, dtype=np.uint8)


def write_split(out: Path, name: str, rows: np.ndarray) -> dict:
    path = out / f"{name}_00000.bin"
    rows.reshape(-1).tofile(path)
    return {"tokens": int(rows.size), "docs": int(len(rows)),
            "shards": [{"file": path.name, "tokens": int(rows.size), "sha256": sha256_file(path)}]}


def manifest(kind: str, splits: dict, notes: str) -> dict:
    return {"kind": kind, "dtype": "uint8", "eos_id": END_OF_PLAN, "vocab_size": gw.VOCAB_SIZE, "row_length": ROW,
            "splits": splits, "token_map": gw.__doc__.split("Token layout")[1].strip(), "notes": notes,
            "held_out_layouts": HELD_OUT, "code": code_revision()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/worlds/gridworld-v1")
    parser.add_argument("--train", type=int, default=1_000_000)
    parser.add_argument("--eval", type=int, default=5_000)
    parser.add_argument("--planning-worlds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    layouts = [l for l in gw.all_layouts() if l not in HELD_OUT]
    transitions, plans = args.out / "transitions", args.out / "plans"
    transitions.mkdir(parents=True, exist_ok=True)
    plans.mkdir(parents=True, exist_ok=True)

    train, _ = episodes(rng, layouts, args.train)
    val, val_states = episodes(rng, layouts, args.eval)
    test_layouts, _ = episodes(rng, HELD_OUT, args.eval)
    intervention = np.array([w.tokens() + [gw.ACTION_BASE + a] + w.step(a).tokens()
                             for w in val_states[: args.eval // 6] for a in range(len(gw.ACTIONS))], dtype=np.uint8)
    splits = {name: write_split(transitions, name, rows) for name, rows in
              [("train", train), ("val", val), ("test_intervention", intervention), ("test_layouts", test_layouts)]}
    (transitions / "manifest.json").write_text(json.dumps(manifest(
        "slm.tokens.gridworld_transitions", splits,
        "Synthetic, rights-free; behaviour policy optimal with p=0.8, else uniform."), indent=2) + "\n", encoding="utf-8")

    plan_splits = {"train": write_split(plans, "train", plan_rows(rng, layouts, args.train // 10)),
                   "val": write_split(plans, "val", plan_rows(rng, layouts, args.eval))}
    (plans / "manifest.json").write_text(json.dumps(manifest(
        "slm.tokens.gridworld_plans", plan_splits, "Start state followed by an optimal (BFS) plan."), indent=2) + "\n",
        encoding="utf-8")

    starts = {"in_distribution": [gw.random_world(rng, layouts) for _ in range(args.planning_worlds)],
              "held_out_layouts": [gw.random_world(rng, HELD_OUT) for _ in range(args.planning_worlds)]}
    with (args.out / "planning_worlds.pkl").open("wb") as f:
        pickle.dump(starts, f)
    print(json.dumps({k: v["docs"] for k, v in splits.items()}), json.dumps({k: v["docs"] for k, v in plan_splits.items()}))


if __name__ == "__main__":
    main()
