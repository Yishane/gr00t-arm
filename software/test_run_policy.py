import pytest

import arm_io
import autodemo as A
import config as C
import kinematics as K
import run_policy
from arm_io import FakeCamera, SimArm
from conftest import RecordingSimArm
from policy_client import PolicyClient

HOME5 = arm_io.HOME + [C.GRIPPER_OPEN]


class ScriptedPolicy:
    """Policy whose chunk() replays a list: an exception is raised, anything
    else (a list of steps, maybe empty) is returned."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def chunk(self, frame, servo, task):
        self.calls += 1
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(run_policy.time, "sleep", lambda s: None)


def test_timeouts_and_empty_chunks_stop_the_run_not_the_program():
    pol = ScriptedPolicy([TimeoutError(), [], RuntimeError("server error")])
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the ball", 8, 60)
    assert pol.calls == 3                             # MAX_FAILS, then returned


def test_a_success_resets_the_failure_count():
    pol = ScriptedPolicy([TimeoutError(), TimeoutError(), [HOME5], TimeoutError(),
                          TimeoutError(), TimeoutError()])
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the ball", 8, 60)
    assert pol.calls == 6


def test_unsafe_commands_are_never_sent():
    arm = RecordingSimArm()
    fold = [90, 90, 120, 150, 125]                    # gripper folded into the upper arm
    pol = ScriptedPolicy([[fold, fold]] + [TimeoutError()] * 3)
    run_policy.run(pol, arm, FakeCamera(), "pick up the ball", 8, 60)
    assert fold not in arm.sent
    assert all(K.servo_ok(c[:4]) for c in arm.sent)


def test_ping_is_false_on_a_server_error(monkeypatch):
    c = PolicyClient("localhost", 5999, timeout_ms=10)
    monkeypatch.setattr(c, "call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    try:
        assert c.ping() is False
    finally:
        c.close()                                     # an open ZMQ socket blocks exit


def test_docstring_act_path_matches_train_act():
    assert "outputs/act/pick_ball/checkpoints/" in run_policy.__doc__


def test_nan_or_short_commands_are_never_sent():
    arm = RecordingSimArm()
    bad = [[90, 90, 90, 90, float("nan")], [90, 90, 90, 90]]
    pol = ScriptedPolicy([bad] + [TimeoutError()] * 3)
    run_policy.run(pol, arm, FakeCamera(), "pick up the ball", 8, 60)
    assert all(len(c) == 5 and all(a == a for a in c) for c in arm.sent)   # a != a only for NaN


def test_an_unsafe_move_is_cut_short_and_the_gripper_still_closes():
    x, y = 110, 40
    pitch = A.best_pitch(x, y, 18)
    above = K.solve(x, y, 40, A.best_pitch(x, y, 40))     # coming down on the object...
    under = K.to_servo(K.ik(x, y, 6, pitch))              # ...asked to go into the plate
    assert under and not K.servo_ok(under)
    cmd, ok = run_policy.make_safe(under + [C.GRIPPER_CLOSED], above + [C.GRIPPER_OPEN])
    assert not ok and K.servo_ok(cmd[:4])
    assert cmd[4] == C.GRIPPER_CLOSED                     # the grip isn't thrown away
    z = K.fk(*K.from_servo(cmd[:4]))[2]
    assert z < 22                                         # went most of the way down, not frozen at 40
    ok_cmd = above + [C.GRIPPER_OPEN]
    assert run_policy.make_safe(ok_cmd, ok_cmd) == (ok_cmd, True)


def test_gripper_goes_fully_open_or_fully_closed():
    o, c = C.GRIPPER_OPEN, C.GRIPPER_CLOSED
    near_closed = c + 0.2 * (o - c)                       # e.g. 82 when closed is 90, open 50
    assert run_policy.snap_gripper(near_closed) == c
    assert run_policy.snap_gripper(o + 0.2 * (c - o)) == o
    pose = list(arm_io.HOME)
    assert run_policy.make_safe(pose + [near_closed], pose + [o])[0][4] == c


def _policy_run(tmp_path, monkeypatch, keep):
    """Two safe steps, one unsafe one, then the run gives up."""
    import recorder
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    rec = recorder.Recorder("p", fps=30, source="policy")
    turn = [HOME5[0] + 5] + HOME5[1:]
    fold = [90, 90, 120, 150, 125]
    steps = [turn] * 6 + [fold] + [HOME5] * 6
    pol = ScriptedPolicy([steps] + [TimeoutError()] * 3)
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the block", 16, 60,
                   pipeline=False, rec=rec, keep=keep)
    return sorted((tmp_path / "p").glob("episode_*")), turn


def test_recorded_run_saves_every_played_step(tmp_path, monkeypatch):
    import json
    import pandas as pd
    eps, turn = _policy_run(tmp_path, monkeypatch, keep=lambda: True)
    assert len(eps) == 1
    meta = json.loads((eps[0] / "meta.json").read_text())
    assert meta["task"] == "pick up the block" and meta["source"] == "policy"
    df = pd.read_csv(eps[0] / "steps.csv")
    state = df[[f"state_{i}" for i in range(5)]].to_numpy().tolist()
    action = df[[f"action_{i}" for i in range(5)]].to_numpy().tolist()
    assert len(df) == 13 == len(list((eps[0] / "frames").glob("*.jpg")))
    assert state[0] == HOME5                          # starts from home
    assert state[1:] == action[:-1]                   # state = the last command sent
    # the unsafe step is cut short: a safe pose between turn and the fold
    assert K.servo_ok(action[6][:4]) and action[6] != turn


def test_answering_no_or_ctrl_c_leaves_no_episode(tmp_path, monkeypatch):
    assert _policy_run(tmp_path, monkeypatch, keep=lambda: False)[0] == []

    def ctrl_c():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        _policy_run(tmp_path, monkeypatch, keep=ctrl_c)
    assert list((tmp_path / "p").glob("episode_*")) == []


def test_latest_checkpoint_is_the_newest_step_of_a_model(tmp_path, monkeypatch):
    for step in ("004000", "016000", "008000"):
        (tmp_path / "act" / "pick_ball" / "checkpoints" / step / "pretrained_model").mkdir(parents=True)
    monkeypatch.setattr(run_policy, "OUTPUTS", tmp_path)
    assert run_policy.latest_checkpoint("pick_ball") == (
        tmp_path / "act" / "pick_ball" / "checkpoints" / "016000" / "pretrained_model")
    with pytest.raises(SystemExit):
        run_policy.latest_checkpoint("not_trained")


def test_the_view_shows_a_frame_with_the_instruction_every_step():
    shown = []
    steps = [[90, 90, 90, 90, 50]] * 5
    pol = ScriptedPolicy([steps] + [TimeoutError()] * 3)
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the ball", 8, 60,
                   show=lambda frame, task: shown.append((frame.shape, task)))
    assert len(shown) >= len(steps)
    assert all(task == "pick up the ball" for _, task in shown)


def test_viewer_draws_a_frame_windowed_and_full_screen(monkeypatch):
    import numpy as np
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")           # no real screen in tests
    for full in (False, True):
        show = run_policy.viewer(full)
        show(np.zeros((480, 640, 3), np.uint8), "pick up the ball")
