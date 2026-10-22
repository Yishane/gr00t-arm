import math

import pytest

import config as C
from kinematics import fk, from_servo, ik, solve, to_servo


@pytest.mark.parametrize("target", [
    (120, 60, 60, 0),
    (90, 80, 30, -math.pi / 4),
    (40, 120, 20, -math.pi / 2),
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
    assert ik(10, 10, 50, 0) is None  # inside the lateral offset


def test_home_pose_matches_cad():
    # upper arm up, forearm forward, gripper in line: all servos at 90
    home = [0, math.pi / 2, -math.pi / 2, 0]
    assert to_servo(home) == pytest.approx([90, 90, 90, 90])
    x, y, z, pitch = fk(*home)
    assert (x, y, z) == pytest.approx(
        (C.L_FORE + C.L_TOOL, C.LATERAL, C.SHOULDER_HEIGHT + C.L_UPPER + C.TOOL_UP))
    assert pitch == pytest.approx(0)


def test_cad_tool_point_with_arm_horizontal():
    # cad/assembly.py with every horn at 0 puts the ball centre at
    # (200, 52.8, 65.75): arm straight out along +x
    x, y, z, _ = fk(0, 0, 0, 0)
    assert (x, y, z) == pytest.approx((200, 52.8, 65.75))


def test_servo_round_trip():
    joints = [0.3, 1.2, -1.0, 0.4]
    assert from_servo(to_servo(joints)) == pytest.approx(joints)


def test_limits_reject_elbow_up_bend():
    assert to_servo([0, 1.0, 0.3, 0]) is None


def test_can_reach_a_ball_on_the_desk():
    # 40 mm ball resting on the plate in front of the arm, gripper ~down
    for x in (80, 100, 120, 140):
        assert any(solve(x, C.LATERAL, 20, math.radians(p)) is not None
                   for p in range(-90, -70, 5))


def test_floor_check():
    # gripper pointing straight down at a resting ball: clear
    assert solve(115, 40, 20, -math.pi / 2) is not None
    # same spot but 15 mm lower would drive the fingertips into the plate
    assert solve(115, 40, 5, -math.pi / 2) is None
