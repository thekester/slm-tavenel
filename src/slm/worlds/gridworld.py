"""Key-door gridworld: a small, deterministic, fully verifiable world for world-model experiments.

A 6 x 6 room surrounded by walls is split by an internal wall column containing one locked door. The
agent starts on one side with a key somewhere on that side; the goal is on the other side. To succeed
it must reach the key, pick it up, open the door from an adjacent cell and walk to the goal.

Actions: 0 north, 1 south, 2 west, 3 east, 4 pick up (on the key cell), 5 open (door adjacent, key held).
Invalid actions leave the state unchanged, which makes interventions informative: the effect of
"open" without the key, or "pick" off the key, must be predicted as "nothing happens".

Token layout for sequence models (vocabulary of 16):
  cells 0 empty, 1 wall, 2 key, 3 locked door, 4 open door, 5 goal, 6 agent, 7 agent on open door;
  inventory 8 no key, 9 key held; actions 10..15.
A state is 64 cell tokens (8 x 8 including the outer wall) followed by one inventory token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMPTY, WALL, KEY, LOCKED, OPEN, GOAL, AGENT, AGENT_ON_DOOR = range(8)
NO_KEY, HAS_KEY = 8, 9
ACTION_BASE = 10
ACTIONS = ["north", "south", "west", "east", "pick", "open"]
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE = 8  # including the outer wall
STATE_TOKENS = SIZE * SIZE + 1
VOCAB_SIZE = 16


@dataclass
class World:
    wall_col: int
    door_row: int
    key: tuple[int, int] | None
    goal: tuple[int, int]
    agent: tuple[int, int]
    has_key: bool = False
    door_open: bool = False
    done: bool = field(default=False)

    def layout(self) -> tuple[int, int]:
        return self.wall_col, self.door_row

    def copy(self) -> "World":
        return World(self.wall_col, self.door_row, self.key, self.goal, self.agent, self.has_key, self.door_open, self.done)

    def cell(self, r: int, c: int) -> int:
        if r in (0, SIZE - 1) or c in (0, SIZE - 1):
            return WALL
        if c == self.wall_col:
            if r == self.door_row:
                return OPEN if self.door_open else LOCKED
            return WALL
        if self.key == (r, c):
            return KEY
        if self.goal == (r, c):
            return GOAL
        return EMPTY

    def tokens(self) -> list[int]:
        out = []
        for r in range(SIZE):
            for c in range(SIZE):
                base = self.cell(r, c)
                if (r, c) == self.agent:
                    base = AGENT_ON_DOOR if base == OPEN else AGENT
                out.append(base)
        out.append(HAS_KEY if self.has_key else NO_KEY)
        return out

    def step(self, action: int) -> "World":
        new = self.copy()
        r, c = self.agent
        if action in MOVES:
            dr, dc = MOVES[action]
            target = self.cell(r + dr, c + dc)
            if target not in (WALL, LOCKED):
                new.agent = (r + dr, c + dc)
                new.done = new.agent == self.goal
        elif action == 4:
            if self.key == self.agent:
                new.key, new.has_key = None, True
        elif action == 5:
            adjacent = abs(r - self.door_row) + abs(c - self.wall_col) == 1
            if adjacent and self.has_key and not self.door_open:
                new.door_open = True
        else:
            raise ValueError(f"unknown action {action}")
        return new


def random_world(rng: np.random.Generator, layouts: list[tuple[int, int]]) -> World:
    wall_col, door_row = layouts[int(rng.integers(len(layouts)))]
    left = [(r, c) for r in range(1, SIZE - 1) for c in range(1, wall_col)]
    right = [(r, c) for r in range(1, SIZE - 1) for c in range(wall_col + 1, SIZE - 1)]
    agent, key = (left[i] for i in rng.choice(len(left), 2, replace=False))
    goal = right[int(rng.integers(len(right)))]
    return World(wall_col, door_row, key, goal, agent)


def all_layouts() -> list[tuple[int, int]]:
    """Internal wall in columns 2..5 (both sides non-empty), door in rows 1..6: 24 layouts."""
    return [(col, row) for col in range(2, 6) for row in range(1, SIZE - 1)]


def shortest_plan(world: World, max_depth: int = 60) -> list[int] | None:
    """Breadth-first search with the exact simulator (the 'agentic' tool and the planning oracle)."""
    def key(w: World):
        return w.agent, w.has_key, w.door_open, w.key

    frontier, seen = [(world, [])], {key(world)}
    for _ in range(max_depth):
        nxt = []
        for state, plan in frontier:
            for action in range(len(ACTIONS)):
                child = state.step(action)
                if child.done:
                    return plan + [action]
                k = key(child)
                if k not in seen:
                    seen.add(k)
                    nxt.append((child, plan + [action]))
        frontier = nxt
    return None


def behaviour_action(world: World, rng: np.random.Generator, expert_prob: float) -> int:
    """Training-data policy: the next optimal action with probability ``expert_prob``, else uniform.
    Its bias is what makes the uniform-action evaluation an intervention test."""
    if rng.random() < expert_prob:
        plan = shortest_plan(world)
        if plan:
            return plan[0]
    return int(rng.integers(len(ACTIONS)))
