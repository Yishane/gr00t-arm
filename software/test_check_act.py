"""act/check_act.py: comparing a model with the recorded demos (fake models,
no PyTorch needed)."""
import sys
from pathlib import Path

import numpy as np

import autodemo as A
import recorder
from conftest import RecordingSimArm
from arm_io import FakeCamera

sys.path.insert(0, str(Path(__file__).parent / "act"))
import check_act  # noqa: E402


class Script:
    """Answers with exactly what the script sent next (a perfect model). Its
    "frame" is (episode, index): see record()."""

    def chunk(self, frame, servo, task):
        ep, i = frame
        return list(check_act.load(ep)[2][i:i + 100])


class StandStill:
    def chunk(self, frame, servo, task):
        return [list(servo)] * 100


def record(tmp_path, monkeypatch, n=2):
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    monkeypatch.setattr(check_act, "DATA", tmp_path)
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    rec = recorder.Recorder("demos", source="autodemo")
    for spot, target in (("C1", "ball"), ("A3", "block"))[:n]:
        A.run(RecordingSimArm(), FakeCamera(), spot, target, rec)
    # the fake camera's pictures are all alike: hand the models (episode, index) instead
    monkeypatch.setattr(check_act, "picture", lambda ep, i: (ep, i))
    return recorder.episodes_in(tmp_path / "demos")


def test_a_model_that_copies_the_script_learned_the_demos(tmp_path, monkeypatch, capsys):
    eps = record(tmp_path, monkeypatch)
    start = check_act.report(Script(), eps)
    out = capsys.readouterr().out
    assert "LEARNED the demos" in out
    assert "0 mm from the script's" in out
    # the start check measures the model's first moves: here, the script's own
    script_start = np.mean([check_act.first_move(check_act.load(ep)[2], check_act.load(ep)[1][0])
                            for ep in eps])
    assert abs(start - script_start) < 1e-6


def test_a_model_that_stands_still_did_not_learn(tmp_path, monkeypatch, capsys):
    eps = record(tmp_path, monkeypatch)
    check_act.report(StandStill(), eps)
    assert "did NOT learn" in capsys.readouterr().out


def test_spread_picks_evenly():
    assert check_act.spread(list(range(10)), 3) == [0, 3, 6]
    assert check_act.spread([1, 2], 5) == [1, 2]


def test_place_name_finds_the_nearest_spot_or_the_cup():
    assert check_act.place_name(np.array([*A.SPOTS["B2"], 20.0])).startswith("B2 (0 mm")
    assert check_act.place_name(np.array([*A.CUP, 80.0])).startswith("the cup")


def test_live_check_reads_one_frame_and_says_where_the_model_heads(tmp_path, monkeypatch, capsys):
    import arm_io
    monkeypatch.setattr(check_act, "DATA", tmp_path / "raw")
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(arm_io, "Camera", lambda index: FakeCamera())
    check_act.live_check(StandStill(), 1, "ball", recorded_start=20.0)
    out = capsys.readouterr().out
    assert "up to 0°" in out and "recorded start pictures: 20°" in out
    assert (tmp_path / "check_live.jpg").exists()


class BoltsOnLivePictures:
    """Calm on recorded pictures, swings 90 degrees on a live camera picture."""

    def chunk(self, frame, servo, task):
        if isinstance(frame, np.ndarray):                   # the (fake) live camera
            return [[servo[0] + 90, *servo[1:]]] * 100
        return [list(servo)] * 100


def test_swapping_inputs_blames_the_live_picture(tmp_path, monkeypatch, capsys):
    import arm_io
    eps = record(tmp_path, monkeypatch, n=1)
    monkeypatch.setattr(arm_io, "Camera", lambda index: FakeCamera())
    check_act.live_check(BoltsOnLivePictures(), 1, "ball", recorded_start=20.0, demo=eps[0])
    out = capsys.readouterr().out
    assert "THE LIVE PICTURE throws it" in out
    assert "home now (servo deg)" in out and "that demo started at" in out


def test_swapping_inputs_blames_the_home_angles(tmp_path, monkeypatch, capsys):
    import arm_io
    eps = record(tmp_path, monkeypatch, n=1)
    start = list(check_act.load(eps[0])[1][0])
    monkeypatch.setattr(check_act, "HOME", [a + 20 for a in start[:4]])   # home moved since
    monkeypatch.setattr(arm_io, "Camera", lambda index: FakeCamera())

    class BoltsOffTheDemoAngles:
        def chunk(self, frame, servo, task):
            off = max(abs(a - b) for a, b in zip(servo[:4], start[:4]))
            return [[servo[0] + (90 if off > 5 else 0), *servo[1:]]] * 100

    check_act.live_check(BoltsOffTheDemoAngles(), 1, "ball", recorded_start=20.0, demo=eps[0])
    assert "TODAY'S HOME ANGLES throw it" in capsys.readouterr().out


def test_a_calm_model_is_called_calm(tmp_path, monkeypatch, capsys):
    import arm_io
    eps = record(tmp_path, monkeypatch, n=1)
    monkeypatch.setattr(arm_io, "Camera", lambda index: FakeCamera())
    check_act.live_check(StandStill(), 1, "ball", recorded_start=20.0, demo=eps[0])
    assert "CALM on all four" in capsys.readouterr().out
