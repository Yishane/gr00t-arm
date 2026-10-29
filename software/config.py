"""Arm geometry and servo calibration. Geometry is measured from the Easy
Robot Arm's assembled model (mobiobi, CC BY-NC-SA; see cad/assembly.py),
built with two 60 mm links and no base legs. Calibration (OFFSET/SIGN/
gripper) is checked after assembly with calibrate.py and saved to
calibration.json, which overrides the defaults below.

Joint angle conventions (degrees here, radians in kinematics.py):
  base      yaw about the vertical axis, 0 = arm plane along +x,
            positive = counter-clockwise seen from above
  shoulder  upper arm elevation above horizontal, 90 = straight up
  elbow     forearm angle relative to the upper arm, 0 = straight,
            negative = bent down (the arm always works elbow-up)
  wrist     gripper angle relative to the forearm, 0 = straight
"""
import json
import os
from pathlib import Path

# ------------------------------------------------------------- geometry (mm)
SHOULDER_HEIGHT = 54.05  # baseplate top -> shoulder axis (drum stands on the plate)
L_UPPER = 60.0           # shoulder axis -> elbow axis (Arm60mm)
L_FORE = 60.0            # elbow axis -> wrist axis (Arm60mm)
L_TOOL = 86.5            # wrist axis -> ball centre, along the gripper
TOOL_UP = -24.75         # the off-axis trunk hangs the fingers below the wrist axis
LATERAL = -1.8           # ball centre is this far to +y of the base axis

# outline of the gripper in its own frame (mm): x along the gripper from the
# wrist axis, y = "down" when the gripper is level. Corners of the trunk,
# gripper mount + cap, and the fingers (they open sideways, so their outline
# in the arm plane doesn't change with the gripper).
GRIPPER_POINTS = [(-12, -10), (31.5, -10), (52.8, -0.5), (52.8, 44.5),
                  (22.2, 44.5), (-12, 42.3), (102.8, 21.5), (102.8, 28.0),
                  (34.4, 28.0)]
FLOOR_CLEARANCE = 2.0   # keep every gripper point this far above the plate

# ---------------------------------------------------------- calibration
# servo_deg = OFFSET + SIGN * joint_deg. Defaults assume each horn was glued
# with the servo at 90 deg in the HOME pose:
#   arm plane along +x, upper arm straight up, forearm horizontal pointing
#   forward, gripper in line with the forearm, jaw closed (parallel).
# SIGN assumes a bigger pulse turns the horn counter-clockwise seen from the
# horn side, which is what most SG90s do. calibrate.py checks it.
OFFSET = {"base": 90.0, "shoulder": 180.0, "elbow": 0.0, "wrist": 90.0}
SIGN = {"base": 1, "shoulder": -1, "elbow": -1, "wrist": -1}

GRIPPER_OPEN = 125.0    # servo deg
GRIPPER_CLOSED = 85.0   # a little past parallel so it squeezes the ball

# soft joint limits (degrees) from the collision sweep in cad/assembly.py,
# with 5 deg of margin
LIMITS = {
    "base": (-90, 90),
    "shoulder": (0, 180),
    "elbow": (-125, 0),
    "wrist": (-85, 90),     # the sweep allows 125, but at +90 the servo (OFFSET 90, SIGN -1) is at 0
}
FOLD_LIMIT = -175       # elbow + wrist: below this the gripper hits the upper arm
SERVO_MIN = 0.0
SERVO_MAX = 180.0

JOINTS = ("base", "shoulder", "elbow", "wrist")

# The home pose every script starts and ends in is the calibration pose above
# except the elbow: -83 instead of -90 (forearm 7 deg off level, servo 83 with a
# true calibration) so the forearm sits on the elbow horn and the servo idles.
HOME_ELBOW = -83.0      # deg

# ---------------------------------------------------------- serial / camera
BAUD = 115200
FPS = 30                # control + recording rate

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"


def load_calibration():
    """Override the defaults above with calibration.json if it exists."""
    global GRIPPER_OPEN, GRIPPER_CLOSED
    if not CALIBRATION_FILE.exists():
        return
    data = json.loads(CALIBRATION_FILE.read_text())
    OFFSET.update(data.get("offset", {}))
    SIGN.update(data.get("sign", {}))
    GRIPPER_OPEN = data.get("gripper_open", GRIPPER_OPEN)
    GRIPPER_CLOSED = data.get("gripper_closed", GRIPPER_CLOSED)


def save_calibration():
    CALIBRATION_FILE.write_text(json.dumps({
        "offset": OFFSET, "sign": SIGN,
        "gripper_open": GRIPPER_OPEN, "gripper_closed": GRIPPER_CLOSED,
    }, indent=2))


# the tests set this so a calibration.json from the real arm doesn't change them
if not os.environ.get("GR00T_ARM_DEFAULT_CALIBRATION"):
    load_calibration()
