#!/usr/bin/env python3
"""Planning success in the key-door gridworld (EXP-0039).

Transition models (configurations A and C), "System 2": breadth-first search in which the model is the
simulator. Every expanded (state, action) pair is completed by greedy decoding of the 65 next-state
tokens; predicted states are deduplicated; a predicted state without a goal cell (the agent stands on
it) ends the search. The resulting action sequence is then executed in the true environment, and only
real success counts. Expansion and depth limits bound the search.

Plan models (configuration B): the model writes an action sequence after the start state. "first
attempt" is greedy decoding; "with verifier tool" lets it retry up to k times with sampling, the true
simulator reporting whether each plan reached the goal (the tool), so success means any verified plan.

The exact BFS planner is the oracle (100% by construction) and gives the optimal lengths.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_gridworld_corpus import END_OF_PLAN  # noqa: E402
from slm.model import KVCache  # noqa: E402
from slm.runlog import sha256_file  # noqa: E402
from slm.train import load_model  # noqa: E402
from slm.worlds import gridworld as gw  # noqa: E402

S = gw.STATE_TOKENS


@torch.no_grad()
def predict_next_states(model, states: list[tuple[int, ...]], actions: list[int], device) -> list[tuple[int, ...]]:
    """Greedy decoding of state(t+1) for a batch of (state, action), with a KV cache."""
    prefix = torch.tensor([list(s) + [gw.ACTION_BASE + a] for s, a in zip(states, actions)], device=device)
    cache = KVCache.empty(model.cfg.n_layers)
    logits, _ = model(prefix, cache=cache)
    out = []
    token = logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(S):
        out.append(token)
        if len(out) == S:
            break
        logits, _ = model(token, cache=cache)
        token = logits[:, -1].argmax(-1, keepdim=True)
    return [tuple(row) for row in torch.cat(out, 1).tolist()]


def goal_reached(state: tuple[int, ...]) -> bool:
    return gw.GOAL not in state[: S - 1]


def model_bfs(model, world: gw.World, device, max_depth: int, max_expansions: int, batch: int = 512):
    start = tuple(world.tokens())
    frontier, seen, expansions = [(start, [])], {start}, 0
    for _ in range(max_depth):
        pairs = [(s, p, a) for s, p in frontier for a in range(len(gw.ACTIONS))]
        if not pairs or expansions >= max_expansions:
            return None, expansions
        pairs = pairs[: max_expansions - expansions]
        expansions += len(pairs)
        nxt = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i : i + batch]
            predicted = predict_next_states(model, [s for s, _, _ in chunk], [a for _, _, a in chunk], device)
            for (_, plan, action), state in zip(chunk, predicted):
                if goal_reached(state):
                    return plan + [action], expansions
                if state not in seen:
                    seen.add(state)
                    nxt.append((state, plan + [action]))
        frontier = nxt
    return None, expansions


def execute(world: gw.World, plan: list[int]) -> bool:
    for action in plan:
        world = world.step(action)
        if world.done:
            return True
    return False


@torch.no_grad()
def write_plan(model, world: gw.World, device, temperature: float, generator, max_len: int = 40) -> list[int]:
    ids = torch.tensor([world.tokens()], device=device)
    cache = KVCache.empty(model.cfg.n_layers)
    logits, _ = model(ids, cache=cache)
    plan = []
    for _ in range(max_len):
        scores = logits[0, -1].float()
        if temperature > 0:
            token = int(torch.multinomial(torch.softmax(scores / temperature, -1), 1, generator=generator))
        else:
            token = int(scores.argmax())
        if token == END_OF_PLAN or not gw.ACTION_BASE <= token < gw.ACTION_BASE + len(gw.ACTIONS):
            break
        plan.append(token - gw.ACTION_BASE)
        logits, _ = model(torch.tensor([[token]], device=device), cache=cache)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=["transition", "plan"], required=True)
    parser.add_argument("--worlds", type=Path, default=ROOT / "data/worlds/gridworld-v1/planning_worlds.pkl")
    parser.add_argument("--n", type=int, default=200, help="start worlds per split")
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--max-expansions", type=int, default=4000)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device).eval()
    starts = pickle.loads(args.worlds.read_bytes())
    gen = torch.Generator(device=device).manual_seed(0)
    results = {}
    for split, worlds in starts.items():
        worlds = worlds[: args.n]
        optimal = [len(gw.shortest_plan(w)) for w in worlds]
        rows = []
        for world, best in zip(worlds, optimal):
            if args.kind == "transition":
                plan, expansions = model_bfs(model, world, device, args.max_depth, args.max_expansions)
                ok = plan is not None and execute(world, plan)
                rows.append({"success": ok, "plan_length": len(plan) if plan else None, "optimal": best,
                             "expansions": expansions})
            else:
                first = write_plan(model, world, device, 0.0, gen)
                attempts, ok_first = [first], execute(world, first)
                ok_any = ok_first
                while not ok_any and len(attempts) < args.retries:
                    attempts.append(write_plan(model, world, device, 1.0, gen))
                    ok_any = execute(world, attempts[-1])
                rows.append({"success_first_attempt": ok_first, "success_with_verifier_tool": ok_any,
                             "attempts": len(attempts), "plan_length": len(first), "optimal": best})
        if args.kind == "transition":
            solved = [r for r in rows if r["success"]]
            summary = {"success_rate": len(solved) / len(rows),
                       "mean_length_over_optimal": (sum(r["plan_length"] / r["optimal"] for r in solved) / len(solved))
                       if solved else None,
                       "mean_expansions": sum(r["expansions"] for r in rows) / len(rows)}
        else:
            summary = {"success_first_attempt": sum(r["success_first_attempt"] for r in rows) / len(rows),
                       f"success_with_verifier_tool_k{args.retries}": sum(r["success_with_verifier_tool"] for r in rows) / len(rows),
                       "mean_attempts": sum(r["attempts"] for r in rows) / len(rows)}
        results[split] = {"worlds": len(rows), "summary": summary, "per_world": rows}
        print(split, summary, flush=True)
    out = {"kind": "slm.gridworld_planning_eval", "planner": args.kind, "checkpoint": str(args.checkpoint),
           "checkpoint_sha256": sha256_file(args.checkpoint), "worlds_sha256": sha256_file(args.worlds),
           "limits": {"max_depth": args.max_depth, "max_expansions": args.max_expansions, "retries": args.retries},
           "oracle_success_rate": 1.0, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
