"""Forward / inverse kinematics for the 4-DOF arm, plus conversion between
joint angles and servo degrees.

The gripper target is the ball centre (x, y, z) in mm, origin on the desk
under the base axis, z up, plus a pitch: the gripper's angle above
horizontal (radians, 0 = level, -pi/2 = pointing straight down).

The ball centre can sit LATERAL mm off the plane the arm swings in; the
base yaw compensates for that.
"""
import math

import config as C


def fk(base, shoulder, elbow, wrist):
    """Joint angles (radians) -> (x, y, z, pitch) of the ball centre."""
    a1 = shoulder
    a2 = a1 + elbow
    a3 = a2 + wrist
    r = (C.L_UPPER * math.cos(a1) + C.L_FORE * math.cos(a2)
         + C.L_TOOL * math.cos(a3) - C.TOOL_UP * math.sin(a3))
    z = (C.SHOULDER_HEIGHT + C.L_UPPER * math.sin(a1) + C.L_FORE * math.sin(a2)
         + C.L_TOOL * math.sin(a3) + C.TOOL_UP * math.cos(a3))
    x = r * math.cos(base) - C.LATERAL * math.sin(base)
    y = r * math.sin(base) + C.LATERAL * math.cos(base)
    return x, y, z, a3


def ik(x, y, z, pitch):
    """(x, y, z, pitch) -> joint angles (radians), elbow-up solution.
    Returns None if the point is out of reach."""
    rho2 = x * x + y * y - C.LATERAL ** 2
    if rho2 <= 0:
        return None
    r = math.sqrt(rho2)
    base = math.atan2(y, x) - math.atan2(C.LATERAL, r)
    # wrist axis position in the arm plane
    rw = r - C.L_TOOL * math.cos(pitch) + C.TOOL_UP * math.sin(pitch)
    zw = z - C.SHOULDER_HEIGHT - C.L_TOOL * math.sin(pitch) - C.TOOL_UP * math.cos(pitch)
    d = (rw * rw + zw * zw - C.L_UPPER ** 2 - C.L_FORE ** 2) / (2 * C.L_UPPER * C.L_FORE)
    if abs(d) > 1:
        return None
    elbow = -math.acos(d)  # negative = elbow above the shoulder-wrist line
    shoulder = math.atan2(zw, rw) - math.atan2(C.L_FORE * math.sin(elbow),
                                               C.L_UPPER + C.L_FORE * math.cos(elbow))
    wrist = pitch - shoulder - elbow
    base = math.atan2(math.sin(base), math.cos(base))
    return base, shoulder, elbow, wrist


def clear_of_floor(joints_rad):
    """True if no part of the gripper dips below the baseplate."""
    _, a1, el, wr = joints_rad
    a2, a3 = a1 + el, a1 + el + wr
    zw = C.SHOULDER_HEIGHT + C.L_UPPER * math.sin(a1) + C.L_FORE * math.sin(a2)
    return all(zw + px * math.sin(a3) - py * math.cos(a3) >= C.FLOOR_CLEARANCE
               for px, py in C.GRIPPER_POINTS)


def within_limits(joints_rad):
    for name, a in zip(C.JOINTS, joints_rad):
        lo, hi = C.LIMITS[name]
        if not lo - 1e-6 <= math.degrees(a) <= hi + 1e-6:
            return False
    return math.degrees(joints_rad[2] + joints_rad[3]) >= C.FOLD_LIMIT - 1e-6


def to_servo(joints_rad):
    """Joint angles (radians) -> servo degrees. Returns None if a joint is
    outside its soft limit or a servo would leave its range."""
    if not within_limits(joints_rad):
        return None
    out = []
    for name, a in zip(C.JOINTS, joints_rad):
        s = C.OFFSET[name] + C.SIGN[name] * math.degrees(a)
        if not C.SERVO_MIN - 1e-6 <= s <= C.SERVO_MAX + 1e-6:
            return None
        out.append(s)
    return out


def from_servo(servo_deg):
    """Servo degrees (4 arm joints) -> joint angles (radians)."""
    return [math.radians((s - C.OFFSET[n]) / C.SIGN[n])
            for n, s in zip(C.JOINTS, servo_deg)]


def solve(x, y, z, pitch):
    """IK straight to servo degrees, or None if unreachable or the gripper
    would hit the plate."""
    j = ik(x, y, z, pitch)
    if j is None or not clear_of_floor(j):
        return None
    return to_servo(j)
