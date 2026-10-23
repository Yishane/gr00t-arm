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
