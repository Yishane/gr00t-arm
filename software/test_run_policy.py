import pytest

import arm_io
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
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the red ball", 8, 60)
    assert pol.calls == 3                             # MAX_FAILS, then returned


def test_a_success_resets_the_failure_count():
    pol = ScriptedPolicy([TimeoutError(), TimeoutError(), [HOME5], TimeoutError(),
                          TimeoutError(), TimeoutError()])
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the red ball", 8, 60)
    assert pol.calls == 6


def test_unsafe_commands_are_never_sent():
    arm = RecordingSimArm()
    fold = [90, 90, 120, 150, 125]                    # gripper folded into the upper arm
    pol = ScriptedPolicy([[fold, fold]] + [TimeoutError()] * 3)
    run_policy.run(pol, arm, FakeCamera(), "pick up the red ball", 8, 60)
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
    run_policy.run(pol, arm, FakeCamera(), "pick up the red ball", 8, 60)
    assert all(len(c) == 5 and all(a == a for a in c) for c in arm.sent)   # a != a only for NaN


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
    assert action[6] == turn                          # the unsafe step holds still


def test_answering_no_or_ctrl_c_leaves_no_episode(tmp_path, monkeypatch):
    assert _policy_run(tmp_path, monkeypatch, keep=lambda: False)[0] == []

    def ctrl_c():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        _policy_run(tmp_path, monkeypatch, keep=ctrl_c)
    assert list((tmp_path / "p").glob("episode_*")) == []
