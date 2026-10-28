import math
import random

import pytest

import autodemo as A
import recorder
from arm_io import FakeCamera, SimArm
from conftest import RecordingSimArm
from recorder import TASKS, Recorder, target_of
from run_policy import instruction


@pytest.mark.parametrize("target", sorted(A.OBJECTS))
@pytest.mark.parametrize("spot", sorted(A.SPOTS))
def test_every_spot_plans_for_both_objects(target, spot):
    wps = A.plan(spot, target, jitter=8.0)
    steps = list(A.trajectory(wps, 30))
    assert len(steps) > 200
    grip_z = min(wp[2] for wp in wps)
    assert grip_z == pytest.approx(A.OBJECTS[target] + A.SPOT_FIX.get(spot, (0, 0, 0))[2])


def test_spot_fix_moves_only_the_grip_at_that_spot(monkeypatch):
    monkeypatch.setattr(A, "SPOT_FIX", {"A2": (-3.0, -2.0, 4.0)})
    grip = lambda spot: min(A.plan(spot, "ball"), key=lambda wp: wp[2])[:3]
    assert grip("A2") == pytest.approx((A.SPOTS["A2"][0] - 3, A.SPOTS["A2"][1] - 2,
                                        A.OBJECTS["ball"] + 4))
    assert grip("A1") == pytest.approx((*A.SPOTS["A1"], A.OBJECTS["ball"]))


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



def test_answering_no_discards_the_episode(tmp_path, monkeypatch, no_sleep):
    import recorder
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    rec = Recorder("t", fps=30, source="autodemo")
    arm, cam = SimArm(), FakeCamera()
    for answer in (False, True):
        A.run(arm, cam, "B2", "ball", rec, keep=lambda: answer)
        assert rec.last_kept is answer
    assert len(list((tmp_path / "t").glob("episode_*"))) == 1

MAX_STEP = 6.0   # deg per 1/30 s step on any arm servo (the firmware's slew is 180 deg/s)


class CrashingCamera(FakeCamera):
    """FakeCamera whose read() raises KeyboardInterrupt after `after` frames."""

    def __init__(self, after):
        super().__init__()
        self.left = after

    def read(self, fresh=False):
        if self.left == 0:
            raise KeyboardInterrupt
        self.left -= 1
        return super().read(fresh)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(A.time, "sleep", lambda s: None)


@pytest.mark.parametrize("seed", range(5))
def test_no_step_jumps(seed):
    random.seed(seed)
    for target in A.OBJECTS:
        for spot in A.SPOTS:
            c = list(A.trajectory(A.plan(spot, target, jitter=8.0), 30))
            assert max(max(abs(a - b) for a, b in zip(p[:4], q[:4]))
                       for p, q in zip(c, c[1:])) <= MAX_STEP, (spot, target)


def test_unreachable_plan_fails_before_the_arm_moves(monkeypatch, no_sleep):
    arm = RecordingSimArm()
    monkeypatch.setattr(A, "solve_near", lambda *a: None)
    with pytest.raises(ValueError):
        A.run(arm, FakeCamera(), "B2", "ball")
    assert arm.sent == []


def test_interrupted_recording_leaves_no_episode(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    rec = Recorder("t", fps=30, source="autodemo")
    with pytest.raises(KeyboardInterrupt):
        A.run(SimArm(), CrashingCamera(after=20), "B2", "ball", rec)
    assert list((tmp_path / "t").glob("episode_*")) == []
