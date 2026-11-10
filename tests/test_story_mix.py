"""Leakage-sensitive invariants of the EXP-0047 corpus construction."""
import hashlib
import importlib.util
from pathlib import Path

import yaml


def builder():
    path = Path(__file__).parents[1] / "scripts/build_exp0047_story_mix.py"
    spec = importlib.util.spec_from_file_location("story_mix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_old_train_completion_groups_cannot_become_new_holdouts():
    mod = builder()
    old_train, holdout = 0, 0
    for i in range(1000):
        group = f"completion-{i}"
        old_bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big") % 100
        if old_bucket >= 5:
            old_train += 1
            assert mod.assigned_split(group) == "train"
        else:
            holdout += 1
            assert mod.assigned_split(group) in {"dev", "test"}
        assert mod.completion_group(f"{group}-0") == mod.completion_group(f"{group}-7") == group
    assert old_train and holdout


def test_duplicate_key_catches_unicode_case_and_spacing():
    mod = builder()
    assert mod.normalized_hash("CAFÉ\n  Fox") == mod.normalized_hash("cafe\u0301 fox")
    assert mod.normalized_hash("fox finds key") != mod.normalized_hash("fox loses key")


def test_reservoir_is_seeded_and_not_just_source_prefix():
    mod = builder()
    a, b = mod.Reservoir(10, 42), mod.Reservoir(10, 42)
    for i in range(1000):
        a.add({"row": i})
        b.add({"row": i})
    assert a.items == b.items
    assert len(a.items) == 10
    assert len({x["row"] for x in a.items}) == 10
    assert max(x["row"] for x in a.items) > 100


def test_mixture_comparison_preserves_training_recipe_and_budget():
    root = Path(__file__).parents[1]
    recipes = []
    for percent in (0, 30, 60):
        recipe = yaml.safe_load((root / f"configs/train/exp0047-story-tiny{percent:02d}-100m.yaml").read_text())
        assert recipe.pop("data_dir").endswith(f"tiny{percent:02d}")
        assert recipe.pop("out_dir").endswith(f"tiny{percent:02d}-100m-s421201")
        positions = recipe["seq_len"] * recipe["micro_batch_size"] * recipe["grad_accum_steps"] * recipe["max_steps"]
        assert positions == 100007936
        assert not recipe.get("init_from")
        recipes.append(recipe)
    assert recipes[0] == recipes[1] == recipes[2]
