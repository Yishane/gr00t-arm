"""Make Timothy freak out (for the talk): twitches, whips, chomps, head-bangs,
dead stops and lunges at the plate, in a random order, at full servo speed.

It LOOKS out of control but isn't: every pose goes through the same checks as
the policies (each joint inside its soft limit, the gripper above the plate),
and the gripper never goes below MIN_Z except in a lunge, which stops at
LUNGE_Z over the plate. Clear the plate first so a lunge can't hit anything.

    uv run freakout.py                  # 20 s of chaos, then home (software folder)
    uv run freakout.py --seconds 40
    uv run freakout.py --speed 120      # calmer, if the whole rig starts walking
    uv run freakout.py --sim            # dry run, no arm
Ctrl+C stops it and sends Timothy home.
"""
import argparse
import math
import random
import time

import config as C
import kinematics as K
from arm_io import HOME, add_hw_args, open_hardware
from run_policy import go_home, make_safe, servo_limits, _height

MIN_Z = 80.0     # mm: normal moves keep the gripper this high (clears the cup and objects)
LUNGE_Z = 45.0   # mm: a lunge dives to here, then snaps back up
O, X = C.GRIPPER_OPEN, C.GRIPPER_CLOSED


def ok(pose, min_z=MIN_Z):
    return K.servo_ok(pose[:4]) and _height(pose) >= min_z


def random_pose(near=None, spread=None, min_z=MIN_Z, tries=300):
    """A random safe arm pose (4 servo degrees): anywhere, or within
    spread degrees of near."""
    for _ in range(tries):
        if near is None:
            p = [random.uniform(*servo_limits(n)) for n in C.JOINTS]
        else:
            p = [a + random.uniform(-spread, spread) for a in near[:4]]
        if ok(p, min_z):
            return p
    return list(near[:4]) if near is not None else list(HOME)


class Freak:
    def __init__(self, arm):
        self.arm = arm
        self.pose = list(HOME)
        self.grip = O

    def go(self, pose, grip=None, wait=0.0, min_z=MIN_Z):
        grip = self.grip if grip is None else grip
        if not ok(pose, min_z):
            return
        cmd, _ = make_safe(list(pose) + [grip], self.pose + [self.grip])
        self.arm.send(cmd)
        self.pose, self.grip = cmd[:4], cmd[4]
        if wait:
            getattr(self.arm, "wait", time.sleep)(wait)

    # --- the moves -----------------------------------------------------------
    def twitch(self):
        """Tiny violent jitters around where it is."""
        here = list(self.pose)
        for _ in range(random.randint(10, 22)):
            self.go(random_pose(here, 7), wait=random.uniform(0.04, 0.09))

    def whip(self):
        """Snap to somewhere completely different."""
        self.go(random_pose(), wait=random.uniform(0.5, 0.9))

    def chomp(self):
        """Gripper snapping open and shut."""
        for _ in range(random.randint(5, 10)):
            self.go(self.pose, X if self.grip == O else O, wait=random.uniform(0.12, 0.2))
        self.go(self.pose, O)

    def headbang(self):
        """Wrist and elbow thrashing back and forth."""
        i = C.JOINTS.index("wrist")
        base = list(self.pose)
        for k in range(random.randint(6, 12)):
            p = list(base)
            p[i] += 28 if k % 2 else -28
            p[i - 1] += 10 if k % 2 else -10
            self.go(p, wait=0.16)

    def freeze(self):
        """Dead still. Very suspicious."""
        getattr(self.arm, "wait", time.sleep)(random.uniform(0.6, 1.6))

    def lunge(self):
        """Dive at the plate, stop just above it, chomp, yank back up."""
        spot = random.choice(["A1", "B1", "C1", "A2", "B2", "C2", "B3"])
        from autodemo import SPOTS, solve_near
        x, y = SPOTS[spot]
        down = solve_near(x, y, LUNGE_Z, math.radians(-70))
        up = solve_near(x, y, 110.0, math.radians(-40))
        if not down or not up:
            return
        self.go(up, O, wait=0.6)
        self.go(down, O, wait=0.45, min_z=LUNGE_Z - 1)
        self.go(down, X, wait=0.15, min_z=LUNGE_Z - 1)
        self.go(down, O, wait=0.15, min_z=LUNGE_Z - 1)
        self.go(up, X, wait=0.5)

    def spin(self):
        """Look left, right, left, right, faster each time."""
        b = C.JOINTS.index("base")
        lo, hi = servo_limits("base")
        p = list(self.pose)
        for k, w in enumerate((0.6, 0.45, 0.35, 0.25, 0.2)):
            p[b] = max(lo + 20, min(hi - 20, HOME[b] + (60 if k % 2 else -60)))
            self.go(p, wait=w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hw_args(ap)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=180.0,
                    help="fastest any servo turns, deg/s (firmware max 180)")
    ap.add_argument("--seed", type=int, help="same chaos every time")
    args = ap.parse_args()
    if not 10 <= args.speed <= 180:
        raise SystemExit("--speed: between 10 and 180 deg/s")
    random.seed(args.seed)
    arm, cam = open_hardware(args)
    cam.close()
    f = Freak(arm)
    moves = [f.twitch, f.whip, f.whip, f.chomp, f.headbang, f.freeze, f.lunge, f.spin]
    try:
        arm.speed(args.speed)
        go_home(arm)
        t_end = time.time() + args.seconds
        last = None
        while time.time() < t_end:
            m = random.choice([m for m in moves if m is not last])
            print(m.__name__)
            m()
            last = m
    except KeyboardInterrupt:
        print("stopped")
    finally:
        arm.speed(60)
        go_home(arm)
        arm.close()
        print("Timothy is calm now.")


if __name__ == "__main__":
    main()
