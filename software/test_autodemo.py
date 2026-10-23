import math
import random

import pytest

import autodemo as A
from recorder import TASKS, Recorder, target_of
from run_policy import instruction


@pytest.mark.parametrize("target", sorted(A.OBJECTS))
@pytest.mark.parametrize("spot", sorted(A.SPOTS))
def test_every_spot_plans_for_both_objects(target, spot):
    wps = A.plan(spot, target, jitter=8.0)
    steps = list(A.trajectory(wps, 30))
    assert len(steps) > 200
    grip_z = min(wp[2] for wp in wps)
    assert grip_z == pytest.approx(A.OBJECTS[target])


def test_pairs_are_far_apart_and_use_every_spot_but_the_middle():
    pairs = A.pairs()
    assert all(math.dist(A.SPOTS[a], A.SPOTS[b]) >= A.MIN_GAP for a, b in pairs)
    assert ("C3", "A2") not in pairs and ("A2", "C3") not in pairs
    assert {a for a, _ in pairs} == set(A.SPOTS) - {"B2"}


def test_layout_picks_both_targets_about_half_each():
    random.seed(0)
    picks = [A.layout()[0] for _ in range(2000)]
    assert 0.45 < picks.count("ball") / len(picks) < 0.55


@pytest.mark.parametrize("text, want", [
    ("pick up the ball", "ball"), ("Pick up the BLOCK", "block"),
    ("ball", "ball"), ("pick up the cup", None), ("ball and block", None),
])
def test_target_of(text, want):
    assert target_of(text) == want


def test_instruction_expands_short_words():
    assert instruction(" block ") == TASKS["block"]
    assert instruction("pick up the ball") == TASKS["ball"]
    assert instruction("grab the block") == "grab the block"
    assert instruction("hello") is None


def test_recorded_episode_saves_its_instruction(tmp_path, monkeypatch):
    import json
    import numpy as np
    import recorder
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    rec = Recorder("t", fps=30, source="autodemo")
    for task in (TASKS["block"], TASKS["ball"]):
        rec.start(task=task)
        for _ in range(12):
            rec.add(np.zeros((8, 8, 3), np.uint8), [90] * 5, [90] * 5)
        rec.stop()
    tasks = [json.loads((ep / "meta.json").read_text())["task"]
             for ep in sorted((tmp_path / "t").glob("episode_*"))]
    assert tasks == [TASKS["block"], TASKS["ball"]]
