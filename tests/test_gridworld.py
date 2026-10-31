import numpy as np

from slm.worlds import gridworld as gw


def _world():
    return gw.World(wall_col=4, door_row=3, key=(2, 2), goal=(5, 6), agent=(2, 1))


def test_tokens_layout_and_inventory():
    tokens = _world().tokens()
    assert len(tokens) == gw.STATE_TOKENS
    assert tokens[2 * gw.SIZE + 1] == gw.AGENT and tokens[2 * gw.SIZE + 2] == gw.KEY
    assert tokens[3 * gw.SIZE + 4] == gw.LOCKED and tokens[-1] == gw.NO_KEY


def test_invalid_actions_change_nothing():
    world = _world()
    assert world.step(4).tokens() == world.tokens()  # pick off the key cell
    assert world.step(5).tokens() == world.tokens()
    assert world.step(2).tokens() == world.tokens()  # west into the outer wall


def test_key_door_goal_sequence_and_planner():
    world = _world()
    plan = gw.shortest_plan(world)
    assert plan is not None
    state = world
    for action in plan:
        state = state.step(action)
    assert state.done and state.has_key and state.door_open
    assert 4 in plan and 5 in plan


def test_model_bfs_with_a_perfect_simulator_always_succeeds(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("grid_plan", "scripts/evaluate_gridworld_planning.py")
    planning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planning)
    states = {}

    def oracle(model, batch_states, actions, device):
        return [tuple(states[s].step(a).tokens()) for s, a in zip(batch_states, actions)]

    monkeypatch.setattr(planning, "predict_next_states", oracle)
    rng = np.random.default_rng(1)
    for _ in range(10):
        world = gw.random_world(rng, gw.all_layouts())
        # Register every reachable state so the oracle can map tokens back to worlds.
        frontier = [world]
        while frontier:
            w = frontier.pop()
            key = tuple(w.tokens())
            if key in states:
                continue
            states[key] = w
            frontier.extend(w.step(a) for a in range(len(gw.ACTIONS)) if not w.step(a).done)
        plan, _ = planning.model_bfs(None, world, None, max_depth=60, max_expansions=100_000)
        assert plan is not None and planning.execute(world, plan)
        assert len(plan) == len(gw.shortest_plan(world))


def test_random_worlds_are_solvable():
    rng = np.random.default_rng(0)
    for _ in range(50):
        world = gw.random_world(rng, gw.all_layouts())
        assert gw.shortest_plan(world) is not None
