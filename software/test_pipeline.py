"""Pipelined policy calls in run_policy.run(). These tests use real sleeps
(test_run_policy.py turns time.sleep off, which would hide the timing)."""
import sys
import threading
import time

import pytest

import arm_io
import config as C
import kinematics as K
import run_policy
from arm_io import FakeCamera, SimArm

HOME5 = arm_io.HOME + [C.GRIPPER_OPEN]


class SlowPolicy:
    """chunk() takes `delay` s and returns 16 steps: base = 80 + step index,
    gripper = 100 + call number (all safe)."""

    def __init__(self, delay):
        self.delay = delay
        self.calls = 0

    def chunk(self, frame, servo, task):
        time.sleep(self.delay)
        self.calls += 1
        return [[80 + k, 90, 90, 90, 100 + self.calls] for k in range(16)]


class TimedArm(SimArm):
    """SimArm that logs (time, angles) for every command."""

    def __init__(self, port=None):
        super().__init__(port)
        self.log = []

    def send(self, angles):
        super().send(angles)
        self.log.append((time.perf_counter(), list(angles)))


class CrashArm(TimedArm):
    """Its 5th send() raises KeyboardInterrupt (Ctrl+C) once."""

    def send(self, angles):
        super().send(angles)
        if len(self.log) == 5:
            raise KeyboardInterrupt


def call_of(a):
    """Which policy call a command came from (None for home commands)."""
    return int(a[4]) - 100 if 101 <= a[4] < 120 else None


@pytest.fixture(autouse=True)
def quick_home(monkeypatch):
    monkeypatch.setattr(run_policy, "go_home",
                        lambda arm: (arm.send(list(HOME5)), list(HOME5))[1])


def test_no_pause_between_chunks():
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.28), arm, FakeCamera(), "pick up the red ball", 8, 2.0)
    times = [t for t, a in arm.log if (call_of(a) or 0) >= 2]   # chunks 2+ only
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert max(gaps) < 0.1                                       # was ~0.3 s


def test_sync_mode_still_pauses():                               # proves the test above measures something
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.28), arm, FakeCamera(), "pick up the red ball", 8, 2.0,
                   pipeline=False)
    times = [t for t, a in arm.log if call_of(a)]
    assert max(b - a for a, b in zip(times, times[1:])) > 0.25


def test_steps_planned_during_the_call_are_skipped():
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.28), arm, FakeCamera(), "pick up the red ball", 8, 2.0)
    first_k = {}                                                 # call number -> first k played
    for t, a in arm.log:
        if call_of(a):
            first_k.setdefault(call_of(a), a[0] - 80)
    assert all(k >= 5 for call, k in first_k.items() if call >= 2)


def test_call_slower_than_a_chunk_still_plays():
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.6), arm, FakeCamera(), "pick up the red ball", 8, 2.5)
    assert any((call_of(a) or 0) >= 2 for t, a in arm.log)       # later chunks were played


def test_ctrl_c_with_a_slow_call_pending_returns_quickly():
    arm = CrashArm()                                             # Ctrl+C mid-chunk, prefetch already running
    t0 = time.perf_counter()
    run_policy.run(SlowPolicy(2.0), arm, FakeCamera(), "pick up the red ball", 8, 30)
    assert time.perf_counter() - t0 < 3.0                       # 2 s first call + a few steps; waiting would be ~4 s
    assert arm.log[-1][1] == HOME5                               # went home


class OneAtATimePolicy(SlowPolicy):
    """Fails the test if two chunk() calls ever overlap (one ZMQ socket can't
    take two requests at once)."""

    def __init__(self, delay):
        super().__init__(delay)
        self.busy = self.overlapped = False

    def chunk(self, frame, servo, task):
        if self.busy:
            self.overlapped = True
        self.busy = True
        try:
            return super().chunk(frame, servo, task)
        finally:
            self.busy = False


def test_a_call_left_running_by_ctrl_c_never_overlaps_the_next_run():
    pol = OneAtATimePolicy(0.5)
    run_policy.run(pol, CrashArm(), FakeCamera(), "pick up the red ball", 8, 30)   # leaves a prefetch running
    run_policy.run(pol, TimedArm(), FakeCamera(), "pick up the red ball", 8, 0.1)  # next instruction right away
    assert not pol.overlapped


def test_no_pipeline_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_policy, "run", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(run_policy, "Gr00tBackend", lambda *a: object())
    monkeypatch.setattr(sys, "argv", ["run_policy.py", "gr00t", "--sim", "--task",
                                      "pick up the red ball", "--no-pipeline"])
    run_policy.main()
    assert seen["pipeline"] is False


class SlowCamera(FakeCamera):
    """A webcam read with fresh=True takes `delay` s (it flushes old frames)."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def read(self, fresh=False):
        if fresh:
            time.sleep(self.delay)
        return super().read(fresh)


def test_a_slow_frame_read_does_not_stall_the_chunk():
    arm = TimedArm()
    # call + read (0.25 s) must fit in half a 16-step chunk (0.27 s), or no
    # amount of prefetching can hide it
    run_policy.run(SlowPolicy(0.2), arm, SlowCamera(0.05), "pick up the red ball", 8, 2.0)
    times = [t for t, a in arm.log if (call_of(a) or 0) >= 2]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert max(gaps) < 0.07             # the read overlaps a step's wait (was ~0.083 s)


class ScriptedSlowPolicy:
    """chunk() takes `delay` s, then raises or returns the next scripted result;
    it notes whether each call ran on the main thread."""

    def __init__(self, delay, results):
        self.delay, self.results = delay, list(results)
        self.on_main = []

    def chunk(self, frame, servo, task):
        time.sleep(self.delay)
        self.on_main.append(threading.current_thread() is threading.main_thread())
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


GOOD = [[80 + k, 90, 90, 90, 101] for k in range(16)]


def test_a_failed_prefetch_counts_once_then_calls_block():
    pol = ScriptedSlowPolicy(0.28, [GOOD, TimeoutError(), TimeoutError(), TimeoutError()])
    run_policy.run(pol, TimedArm(), FakeCamera(), "pick up the red ball", 8, 30)
    assert pol.on_main == [True, False, True, True]   # call 2 prefetched, 3 and 4 blocking, then MAX_FAILS


def test_unsafe_steps_in_a_prefetched_chunk_are_blocked():
    fold = [90, 90, 120, 150, 125]                    # gripper folded into the upper arm
    pol = ScriptedSlowPolicy(0.28, [GOOD, [fold] * 16] + [TimeoutError()] * 3)
    arm = TimedArm()
    run_policy.run(pol, arm, FakeCamera(), "pick up the red ball", 8, 30)
    assert pol.on_main[1] is False                    # the unsafe chunk was prefetched
    assert fold not in [a for t, a in arm.log]
    assert all(K.servo_ok(a[:4]) for t, a in arm.log)
