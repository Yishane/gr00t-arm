import math

import pytest

import config as C
from kinematics import fk, from_servo, ik, servo_ok, solve, to_servo


@pytest.mark.parametrize("target", [
    (120, 60, 60, 0),
    (90, 80, 30, -math.pi / 4),
    (40, 20, 60, -math.pi / 2),
    (150, 20, 80, 0.2),
    (-30, 140, 40, -0.6),
])
def test_ik_then_fk_round_trips(target):
    joints = ik(*target)
    assert joints is not None
    for got, want in zip(fk(*joints), target):
        assert got == pytest.approx(want, abs=1e-6)


def test_unreachable_returns_none():
    assert ik(500, 0, 0, 0) is None
    assert ik(1, 1, 50, 0) is None  # on the base axis, inside the lateral offset


def test_home_pose_matches_cad():
    # upper arm up, forearm forward, gripper in line: all servos at 90
    home = [0, math.pi / 2, -math.pi / 2, 0]
    assert to_servo(home) == pytest.approx([90, 90, 90, 90])
    x, y, z, pitch = fk(*home)
    assert (x, y, z) == pytest.approx(
        (C.L_FORE + C.L_TOOL, C.LATERAL, C.SHOULDER_HEIGHT + C.L_UPPER + C.TOOL_UP))
    assert pitch == pytest.approx(0)


def test_cad_tool_point_in_home_pose():
    # cad/assembly.py home pose (from the author's STEP, Arm60 x2, no legs):
    # the point between the finger pads, in the plane of the fingers
    x, y, z, _ = fk(0, math.pi / 2, -math.pi / 2, 0)
    assert (x, y, z) == pytest.approx((146.5, -1.8, 89.3))


def test_servo_round_trip():
    joints = [0.3, 1.2, -1.0, 0.4]
    assert from_servo(to_servo(joints)) == pytest.approx(joints)


def test_limits_reject_elbow_up_bend():
    assert to_servo([0, 1.0, 0.3, 0]) is None


def test_can_reach_a_ball_on_the_desk():
    # 40 mm ball resting on the plate in front of the arm, gripper ~down
    for x in (85, 110, 135):
        assert any(solve(x, C.LATERAL, 20, math.radians(p)) is not None
                   for p in range(-90, -35, 5))


def test_floor_check():
    # gripper pointing nearly straight down at a resting ball: clear
    assert solve(70, C.LATERAL, 20, math.radians(-80)) is not None
    # 5 mm lower would drive the fingertips into the plate
    assert solve(70, C.LATERAL, 15, math.radians(-80)) is None


def test_gripper_cant_fold_into_upper_arm():
    # forearm folded down, gripper curled back further: blocked by FOLD_LIMIT
    assert to_servo([0, math.pi / 2, math.radians(-110), math.radians(-70)]) is None
    assert to_servo([0, math.pi / 2, math.radians(-100), math.radians(-70)]) is not None


def to_servo_unchecked(j):
    """Servo degrees for joint angles, without the limit checks."""
    return [C.OFFSET[n] + C.SIGN[n] * math.degrees(a) for n, a in zip(C.JOINTS, j)]


def test_servo_ok_accepts_home_and_rejects_unsafe():
    assert servo_ok([90, 90, 90, 90])
    assert servo_ok([90, 90, 90, 90, 125])          # gripper value ignored
    assert not servo_ok([90, 90, 120, 150])         # elbow -120 + wrist -60 < FOLD_LIMIT
    assert not servo_ok([90, 90, float("nan"), 90])
    assert not servo_ok([90, 90, 90, 190])          # outside servo range


def test_servo_ok_rejects_gripper_in_the_plate():
    j = ik(80, 0, 0, -math.pi / 2)     # in limits, but the fingertips go through the plate
    assert j is not None and not servo_ok(to_servo_unchecked(j))


def test_wrist_limit_matches_servo_range():
    lo, hi = C.LIMITS["wrist"]
    assert hi == 90
    assert to_servo([0, math.pi / 2, -math.pi / 2, math.radians(hi)]) is not None
