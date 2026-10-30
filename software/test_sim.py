"""The MuJoCo simulation (sim.py): the simulated arm goes where the
kinematics say, scripted picks work in it, and autodemo / run_policy record
and score runs in it."""
import json
import math

import numpy as np
import pytest

import sim  # noqa: E402,I001  (first: it picks off-screen rendering before mujoco loads)

pytest.importorskip("mujoco")

import autodemo as A  # noqa: E402
import config as C  # noqa: E402
import kinematics as K  # noqa: E402
import recorder  # noqa: E402
import run_policy  # noqa: E402
from arm_io import HOME, FakeCamera  # noqa: E402


@pytest.fixture(scope="module")
def world():
    w = sim.World(realtime=False)
    yield w
    w.close()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    monkeypatch.setattr(run_policy.time, "sleep", lambda s: None)


def other(target):
    return "block" if target == "ball" else "ball"


@pytest.mark.parametrize("xyz", [(110, 40, 40), (135, 0, 20), (85, 80, 60), (20, 110, 80)])
def test_simulated_arm_goes_where_the_kinematics_say(world, xyz):
    x, y, z = xyz
    servo = K.solve(x, y, z, A.best_pitch(x, y, z))
    world.place(A.SPOTS["A1"], A.SPOTS["C3"])            # out of the way
    world.arm.send(servo + [C.GRIPPER_OPEN])
    world.arm.wait(1.5)
    # a few mm of sag under gravity, like the real SG90s; a wrong joint
    # direction or offset would be off by centimetres
    assert math.dist(world.tool(), (x, y, z)) < 8


@pytest.mark.parametrize("target, spot, other_spot", [
    ("ball", "C1", "A3"), ("block", "A3", "C1"), ("ball", "A1", "C3"),
    ("block", "C2", "A1"), ("block", "C3", "A1"), ("ball", "B3", "C1")])
def test_scripted_picks_work_in_the_simulator(world, no_sleep, monkeypatch, target, spot,
                                             other_spot):
    monkeypatch.setattr(A, "SPOT_FIX", {})          # those correct the real arm's sag
    monkeypatch.setattr(A, "GRIP_FIX", {})
    world.place_for(target, A.SPOTS[spot], A.SPOTS[other_spot])
    A.run(world.arm, FakeCamera(), spot, target)
    assert world.result(target) == "ok"


def test_picks_off_the_spots(world, no_sleep):
    world.rng.seed(1)
    A.random.seed(1)
    for _ in range(4):
        target, xy, other_xy = world.random_layout(offset=15)
        assert math.dist(xy, other_xy) >= A.MIN_GAP
        world.place_for(target, xy, other_xy)
        A.run(world.arm, FakeCamera(), xy, target)
        assert world.result(target) == "ok", (target, xy, other_xy)


def test_anywhere_layouts_cover_the_plate_and_can_be_picked(world, no_sleep):
    world.rng.seed(2)
    A.random.seed(2)
    off_spot = 0
    for i in range(24):
        target, xy, other_xy = world.random_layout(anywhere=True)
        assert math.dist(xy, other_xy) >= A.MIN_GAP
        assert sim.clear_of_fixed_parts(xy) and sim.clear_of_fixed_parts(other_xy)
        assert world.in_view((*xy, 15)) and world.in_view((*other_xy, 15))
        off_spot += min(math.dist(xy, s) for s in A.SPOTS.values()) > 15
        if i < 3:                                 # the pick it promised works here too
            world.place_for(target, xy, other_xy)
            A.run(world.arm, FakeCamera(), xy, target)
            assert world.result(target) == "ok", (target, xy, other_xy)
    assert off_spot >= 8                          # not just round the spots (~60% are)


def test_camera_view_check_matches_the_picture(world):
    assert all(world.in_view((*s, 0)) for s in A.SPOTS.values())
    assert world.in_view((*A.CUP, 35))
    assert not world.in_view((175, 0, 15))        # under the camera
    assert not world.in_view((0, 0, 500))         # above it


def test_objects_never_overlap_the_fixed_parts():
    assert sim.clear_of_fixed_parts(A.SPOTS["B2"])
    assert not sim.clear_of_fixed_parts((40, 0))          # in the base drum
    assert not sim.clear_of_fixed_parts(A.CUP)
    assert not sim.clear_of_fixed_parts((155, 55))        # against the mast's socket
    assert not sim.clear_of_fixed_parts((100, 150))       # off the back of the plate


def test_result_tells_ok_wrong_object_and_missed(world):
    world.place(A.SPOTS["C1"], A.SPOTS["A3"])
    assert world.result("ball") == "missed"
    q = world.model.joint("ball").qposadr[0]
    world.data.qpos[q:q + 3] = [A.CUP[0] / 1000, A.CUP[1] / 1000, 0.03]   # drop it in the cup
    world.advance(0.5)
    assert world.in_cup("ball")
    assert world.result("ball") == "ok"
    assert world.result("block") == "wrong object"


def test_camera_sees_the_orange_ball(world):
    world.place(A.SPOTS["C1"], A.SPOTS["A3"])
    img = world.camera.read()
    assert img.shape == (sim.HEIGHT, sim.WIDTH, 3) and img.dtype == np.uint8
    r, g, b = (img[..., i].astype(int) for i in range(3))
    assert ((r > 180) & (g > 50) & (g < 150) & (b < 80)).sum() > 500


def test_realtime_world_moves_on_its_own():
    w = sim.World(realtime=True)
    try:
        start = w.tool()
        w.arm.send(K.solve(110, 40, 60, math.radians(-45)) + [C.GRIPPER_OPEN])
        w.arm.wait(1.0)                       # a real sleep: the thread steps the physics
        assert math.dist(w.tool(), start) > 30
    finally:
        w.close()


def test_autodemo_records_only_successes_in_the_simulator(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    monkeypatch.setattr(sim.MujocoCamera, "read",
                        lambda self, fresh=False: np.zeros((48, 64, 3), np.uint8))
    monkeypatch.setattr("sys.argv", ["autodemo.py", "--mujoco", "--record", "2", "--anywhere"])
    A.main()
    eps = sorted((tmp_path / "pick_ball_mujoco").glob("episode_*"))
    assert len(eps) == 2
    meta = json.loads((eps[0] / "meta.json").read_text())
    assert meta["sim"] is True and meta["source"] == "autodemo"


class ReplayPolicy:
    """Plays a scripted pick back as a policy would: 16-step chunks,
    starting where the last 8 played steps ended."""

    def __init__(self, cmds, horizon=8):
        self.cmds, self.i, self.h = cmds, 0, horizon

    def chunk(self, frame, servo, task):
        out = self.cmds[self.i:self.i + 16] or [self.cmds[-1]]
        self.i += self.h
        return out


def test_run_policy_in_the_simulator_stops_at_the_cup_and_keeps_successes(
        world, tmp_path, monkeypatch, no_sleep):
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    monkeypatch.setattr(A, "SPOT_FIX", {})            # real-arm corrections, not the sim's
    monkeypatch.setattr(A, "GRIP_FIX", {})
    rec = recorder.Recorder("p", fps=30, source="policy", sim=True)
    target, spot = "block", "C2"
    world.place_for(target, A.SPOTS[spot], A.SPOTS["A1"])
    cmds = list(A.trajectory(A.plan(spot, target), C.FPS))
    run_policy.run(ReplayPolicy(cmds), world.arm, FakeCamera(), run_policy.TASKS[target], 8, 60,
                   pipeline=False, rec=rec, keep=lambda: world.success(target),
                   done=lambda: world.in_cup("ball") or world.in_cup("block"))
    assert world.result(target) == "ok"
    eps = sorted((tmp_path / "p").glob("episode_*"))
    assert len(eps) == 1
    rows = (eps[0] / "steps.csv").read_text().count("\n") - 1
    assert rows < len(cmds) - 20                # stopped at the drop, not after going home


def test_finger_angle_follows_the_calibration():
    assert sim.finger_angle(C.GRIPPER_OPEN) == pytest.approx(sim.FINGER_OPEN_DEG)
    assert sim.finger_angle(C.GRIPPER_CLOSED) == pytest.approx(sim.FINGER_CLOSED_DEG)


def test_episodes_of_several_datasets_merge_in_order(tmp_path):
    for name, n in (("real", 2), ("sim", 3)):
        for i in range(n):
            ep = tmp_path / name / f"episode_{i:06d}"
            ep.mkdir(parents=True)
            (ep / "steps.csv").write_text("t\n")
    eps = recorder.episodes_in([tmp_path / "real", tmp_path / "sim"])
    assert [p.parent.name for p in eps] == ["real"] * 2 + ["sim"] * 3


def test_home_pose_is_where_the_arm_starts(world):
    world.place(A.SPOTS["C1"], A.SPOTS["A3"])
    assert world.arm.world.out == HOME + [C.GRIPPER_OPEN]


class OraclePolicy:
    """Stands in for a trained model: plays autodemo's pick for wherever the
    simulator put the object it was asked for."""

    def __init__(self, world):
        self.world, self.seen, self.replay = world, None, None

    def chunk(self, frame, servo, task):
        if self.world.start is not self.seen:            # a new layout: a new trial
            self.seen = self.world.start
            target = recorder.target_of(task)
            spot = self.world.start[target][:2]
            self.replay = ReplayPolicy(list(A.trajectory(A.plan(spot, target), C.FPS)))
        return self.replay.chunk(frame, servo, task)


def test_run_policy_trials_score_and_record_in_the_simulator(tmp_path, monkeypatch, no_sleep,
                                                               capsys):
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    w = sim.World(realtime=False, randomize=True, seed=3)
    monkeypatch.setattr(run_policy, "open_hardware", lambda args: (w.arm, FakeCamera()))
    monkeypatch.setattr(run_policy, "ActBackend", lambda checkpoint: OraclePolicy(w))
    monkeypatch.setattr("sys.argv", ["run_policy.py", "act", "--mujoco", "--trials", "2",
                                     "--offset", "10", "--record", "--no-pipeline"])
    run_policy.main()
    out = capsys.readouterr().out
    assert "ball: 1/1 ok" in out and "block: 1/1 ok" in out
    eps = sorted((tmp_path / "pick_ball_mujoco").glob("episode_*"))
    assert len(eps) == 2
    assert json.loads((eps[0] / "meta.json").read_text())["source"] == "policy"


def test_watch_shows_the_simulated_camera_while_recording(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    monkeypatch.setattr(sim.MujocoCamera, "read",
                        lambda self, fresh=False: np.zeros((48, 64, 3), np.uint8))
    shown = []
    monkeypatch.setattr(A, "preview", lambda frame: shown.append(frame.shape))
    monkeypatch.setattr("sys.argv", ["autodemo.py", "--mujoco", "--record", "1", "--anywhere",
                                     "--watch"])
    A.main()
    assert len(shown) > 200                   # every recorded frame was shown


def test_every_layout_gets_a_new_ball_colour_never_green_or_grey(world):
    import colorsys
    hues = []
    for _ in range(30):
        world._randomize()
        r, g, b = world.model.mat("ball").rgba[:3]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        assert s > 0.5 and v > 0.5                       # strong colour, not white/grey/dark
        assert not 80 / 360 <= h <= 170 / 360           # not the block's green
        hues.append(h)
    assert max(hues) - min(hues) > 0.4                   # really different colours


def test_no_object_is_placed_closer_to_the_arm_than_a1(world):
    world.rng.seed(3)
    A.random.seed(3)
    nearest = math.hypot(*A.SPOTS["A1"])
    for _ in range(25):
        for anywhere, offset in ((True, 0.0), (False, 20.0)):
            t, xy, other = world.random_layout(offset, anywhere=anywhere)
            assert math.hypot(*xy) >= nearest - 1e-9
            assert math.hypot(*other) >= nearest - 1e-9
