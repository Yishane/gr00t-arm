"""Scripted pick-and-place: picks the ball from a marked spot (A1..C3 on the
baseplate) and drops it in the cup. Two uses:

  1. Check the arm works / show it off:
         uv run autodemo.py --spot B2
  2. Collect training demos automatically (the camera records while the
     script moves the arm; you only place the ball where it tells you):
         uv run autodemo.py --record 60

The learned policy never sees the spot name, only the camera image, so it
has to learn to find the ball from what it sees.
"""
import argparse
import math
import random
import time

import config as C
import kinematics as K
from arm_io import HOME, add_hw_args, open_hardware
from recorder import TASK, Recorder

# keep in sync with cad/arm.py SPOT_X / SPOT_Y / CUP_XY
SPOTS = {f"{c}{r + 1}": (x, y) for c, x in zip("ABC", (85.0, 110.0, 135.0))
         for r, y in enumerate((0.0, 40.0, 80.0))}
CUP = (0.0, 110.0)
BALL_Z = 20.0        # ball centre when resting on the plate
SAFE_Z = 75.0
DROP_Z = 80.0


def best_pitch(x, y, z):
    """Gripper pitch closest to straight down that reaches (x, y, z)."""
    for p in range(-90, 1, 5):
        if K.solve(x, y, z, math.radians(p)):
            return math.radians(p)
    return None


def plan(spot, jitter=0.0):
    """Waypoints (x, y, z, pitch, gripper_closed, seconds) for one pick."""
    sx, sy = SPOTS[spot]
    j = lambda: random.uniform(-jitter, jitter)
    home = K.fk(*K.from_servo(HOME))
    pts = [
        (*home, False, 0.8),
        (sx, sy, SAFE_Z + j(), None, False, 1.6),
        (sx, sy, BALL_Z, None, False, 1.2),
        (sx, sy, BALL_Z, None, True, 0.7),          # close
        (sx, sy, SAFE_Z + 10 + j(), None, True, 1.0),
        (CUP[0], CUP[1], DROP_Z + j(), None, True, 2.0),
        (CUP[0], CUP[1], DROP_Z, None, False, 0.6),  # open
        (*home, False, 1.6),
    ]
    out = []
    for x, y, z, p, closed, secs in pts:
        if p is None:
            p = best_pitch(x, y, z)
            if p is None:
                raise ValueError(f"can't reach {(x, y, z)}")
        out.append((x, y, z, p, closed, secs * random.uniform(1 - jitter / 40, 1 + jitter / 40)))
    return out


def trajectory(waypoints, fps):
    """Servo commands at fps, moving in straight lines between waypoints."""
    cur = waypoints[0]
    for wp in waypoints[1:]:
        n = max(1, int(wp[5] * fps))
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)                      # ease in/out
            pose = [a + (b - a) * s for a, b in zip(cur[:4], wp[:4])]
            servo = K.solve(*pose)
            if servo is None:                             # fall back to the end pose
                servo = K.solve(*wp[:4])
            grip = C.GRIPPER_CLOSED if wp[4] else C.GRIPPER_OPEN
            yield servo + [grip]
        cur = wp


def run(arm, cam, spot, rec=None, jitter=0.0):
    dt = 1.0 / C.FPS
    last = arm.last
    if rec:
        rec.start()
    t_next = time.perf_counter()
    for cmd in trajectory(plan(spot, jitter), C.FPS):
        frame = cam.read()
        if rec:
            rec.add(frame, last, cmd)
        arm.send(cmd)
        last = cmd
        t_next += dt
        time.sleep(max(0.0, t_next - time.perf_counter()))
    if rec:
        rec.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hw_args(ap)
    ap.add_argument("--spot", default="B2", choices=sorted(SPOTS))
    ap.add_argument("--record", type=int, default=0, metavar="N",
                    help="record N demos at random spots")
    ap.add_argument("--dataset", default="pick_ball")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--yes", action="store_true", help="don't wait for Enter (testing)")
    args = ap.parse_args()

    arm, cam = open_hardware(args)
    arm.send(HOME + [C.GRIPPER_OPEN])
    time.sleep(1.0)
    try:
        if not args.record:
            run(arm, cam, args.spot)
            return
        rec = Recorder(args.dataset, args.task, C.FPS, "autodemo")
        spots = sorted(SPOTS)
        for i in range(args.record):
            spot = random.choice(spots)
            if not args.yes:
                input(f"[{i + 1}/{args.record}] put the ball on {spot}, then press Enter ")
            run(arm, cam, spot, rec, jitter=8.0)
    finally:
        arm.close()
        cam.close()


if __name__ == "__main__":
    main()
