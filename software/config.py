"""Arm geometry and servo calibration. Geometry comes from the CAD
(cad/assembly.py); calibration (OFFSET/SIGN/gripper) is checked after
assembly with calibrate.py and saved to calibration.json, which overrides
the defaults below.

Joint angle conventions (degrees here, radians in kinematics.py):
  base      yaw about the vertical axis, 0 = arm plane along +x,
            positive = counter-clockwise seen from above
  shoulder  upper arm elevation above horizontal, 90 = straight up
  elbow     forearm angle relative to the upper arm, 0 = straight,
            negative = bent down (the arm always works elbow-up)
  wrist     gripper angle relative to the forearm, 0 = straight
"""
import json
from pathlib import Path

# ------------------------------------------------------------- geometry (mm)
SHOULDER_HEIGHT = 55.25  # desk -> shoulder axis
L_UPPER = 65.0           # shoulder axis -> elbow axis
L_FORE = 60.0            # elbow axis -> wrist axis
L_TOOL = 75.0            # wrist axis -> ball centre, along the gripper
TOOL_UP = 10.5           # ball centre sits this far "above" the gripper axis
LATERAL = 52.8           # ball centre is offset sideways from the base axis
                         # (each joint stacks its link 13.7 mm further out)

# outline of the gripper in its own frame (mm): x along the gripper from the
# wrist axis, y = "down" when the gripper is level (cad gripper_body frame).
# Fingertips (jaw closed and open) plus the body corners.
GRIPPER_POINTS = [(85, -37), (85, 16), (77, 38), (56, 26), (66, -37),
                  (40, -37), (-14, -12), (-14, 14)] + [
    (14 * c, 14 * s) for c, s in ((1, 0), (0.71, 0.71), (0, 1), (-0.71, 0.71),
                                  (-1, 0), (-0.71, -0.71), (0, -1), (0.71, -0.71))]
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

# soft joint limits (degrees) from the CAD collision sweep
LIMITS = {
    "base": (-90, 90),
    "shoulder": (0, 180),
    "elbow": (-150, 0),
    "wrist": (-90, 90),
}
SERVO_MIN = 0.0
SERVO_MAX = 180.0

JOINTS = ("base", "shoulder", "elbow", "wrist")

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


load_calibration()
