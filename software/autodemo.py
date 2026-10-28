"""Scripted pick-and-place with two objects on the plate, a ball and a
block. The script picks the one it's told to from its marked spot (A1..C3 on
the baseplate) and drops it in the cup. Two uses:

  1. Check the arm works / show it off:
         uv run autodemo.py --target block --spot C1 --other-spot A3
  2. Collect training demos automatically (the camera records while the
     script moves the arm; you only place the two objects where it tells you):
         uv run autodemo.py --record 150

Each recorded episode puts both objects on random spots, picks the ball or
the block (about 50/50) and saves the matching instruction with the episode
("pick up the ball" / "pick up the block"). The learned policy never sees the
spot names, only the camera image and the instruction, so it has to learn
which object the words mean and where that object is.
"""
import argparse
import math
import random
import time

import config as C
import kinematics as K
from arm_io import HOME, add_hw_args, open_hardware
from recorder import TASKS, Recorder, ask_keep

# keep in sync with cad/arm.py SPOT_X / SPOT_Y / CUP_XY
SPOTS = {f"{c}{r + 1}": (x, y) for c, x in zip("ABC", (85.0, 110.0, 135.0))
         for r, y in enumerate((0.0, 40.0, 80.0))}
CUP = (20.0, 110.0)
# grip height (tool point z) for each object resting on the plate. The ball
# is gripped at its centre; the 25 mm block (cad/arm.py BLOCK) above its
# centre: lower down the fingertips hit the plate at most spots and the
# gripper tilts so far that it hits the base drum (A1) or the other object.
OBJECTS = {"ball": 20.0, "block": 18.0}
# Per-spot corrections (mm) for spots the real arm misses while the rest are
# fine: (toward column C, toward row 3, up). Tune these by watching picks.
#   gripper lands too far toward C3 at A2 -> make the first two negative
#   grips too low at C3                    -> make the third positive
SPOT_FIX = {
    "C3": (0.0, 0.0, 4.0),
}
# Both objects are on the plate. Spot centres at least MIN_GAP apart, and not
# a pair in TOO_CLOSE, so the gripper can't touch the object it isn't picking
# (cad/assembly.py --cell checks every pair). That leaves B2 (the middle) out.
MIN_GAP = 60.0
TOO_CLOSE = {frozenset(("C3", "A2"))}   # at C3 the tilted gripper reaches back over A2
SAFE_Z = 75.0
DROP_Z = 80.0
PITCH_SEARCH = 30   # deg: how far solve_near tilts the gripper to find a solution


def best_pitch(x, y, z):
    """Gripper pitch closest to straight down that reaches (x, y, z)."""
    for p in range(-90, 1, 5):
        if K.solve(x, y, z, math.radians(p)):
            return math.radians(p)
    return None


def pairs():
    """(target spot, other spot) layouts far enough apart to pick either object."""
    return [(a, b) for a in sorted(SPOTS) for b in sorted(SPOTS)
            if math.dist(SPOTS[a], SPOTS[b]) >= MIN_GAP
            and frozenset((a, b)) not in TOO_CLOSE]


def layout():
    """Random target object and spots for one episode."""
    target = random.choice(sorted(OBJECTS))
    spot, other_spot = random.choice(pairs())
    return target, spot, other_spot


def plan(spot, target="ball", jitter=0.0):
    """Waypoints (x, y, z, pitch, gripper_closed, seconds) for one pick."""
    dx, dy, dz = SPOT_FIX.get(spot, (0.0, 0.0, 0.0))
    sx, sy = SPOTS[spot][0] + dx, SPOTS[spot][1] + dy
    grip_z = OBJECTS[target] + dz
    j = lambda: random.uniform(-jitter, jitter)
    home = K.fk(*K.from_servo(HOME))
    pts = [
        (*home, False, 0.8),
        (sx, sy, SAFE_Z + j(), None, False, 1.6),
        (sx, sy, grip_z, None, False, 1.2),
        (sx, sy, grip_z, None, True, 0.7),          # close
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


def solve_near(x, y, z, pitch):
    """K.solve at this pitch, or at the nearest pitch (1 deg steps, up to
    PITCH_SEARCH either way) that reaches (x, y, z). None if none does."""
    for d in range(PITCH_SEARCH + 1):
        for sgn in ((1,) if d == 0 else (1, -1)):
            servo = K.solve(x, y, z, pitch + math.radians(sgn * d))
            if servo:
                return servo
    return None


def trajectory(waypoints, fps):
    """Servo commands at fps, moving in straight lines between waypoints.
    Where the exact pitch can't reach a point, the gripper tilts a little
    instead, so the arm never jumps."""
    cur = waypoints[0]
    for wp in waypoints[1:]:
        n = max(1, int(wp[5] * fps))
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)                      # ease in/out
            pose = [a + (b - a) * s for a, b in zip(cur[:4], wp[:4])]
            servo = solve_near(*pose)
            if servo is None:
                raise ValueError(f"can't reach {pose}")
            grip = C.GRIPPER_CLOSED if wp[4] else C.GRIPPER_OPEN
            yield servo + [grip]
        cur = wp


def run(arm, cam, spot, target="ball", rec=None, jitter=0.0, keep=None):
    """keep: asked after the moves whether to save the episode (default: save)."""
    dt = 1.0 / C.FPS
    last = arm.last
    cmds = list(trajectory(plan(spot, target, jitter), C.FPS))   # fails here, before moving
    if rec:
        rec.start(task=TASKS[target])
    t_next = time.perf_counter()
    try:
        for cmd in cmds:
            frame = cam.read()
            if rec:
                rec.add(frame, last, cmd)
            arm.send(cmd)
            last = cmd
            t_next += dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
    except BaseException:
        if rec:
            rec.stop(keep=False)                          # don't leave half an episode
        raise
    if rec:
        rec.stop(keep=keep() if keep else True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hw_args(ap)
    ap.add_argument("--target", default="ball", choices=sorted(OBJECTS),
                    help="object to pick when not recording")
    ap.add_argument("--spot", default="B2", choices=sorted(SPOTS),
                    help="where the target is when not recording")
    ap.add_argument("--other-spot", choices=sorted(SPOTS),
                    help="where the other object is (only checked, it isn't moved)")
    ap.add_argument("--record", type=int, default=0, metavar="N",
                    help="record N demos (that you keep), random layout and target each time")
    ap.add_argument("--dataset", default="pick_ball")
    ap.add_argument("--yes", action="store_true", help="don't wait for Enter (testing)")
    args = ap.parse_args()

    arm, cam = open_hardware(args)
    arm.send(HOME + [C.GRIPPER_OPEN])
    time.sleep(1.0)
    try:
        if not args.record:
            if args.other_spot and (args.spot, args.other_spot) not in pairs():
                raise SystemExit(f"{args.spot} and {args.other_spot} are too close: "
                                 "the gripper could hit the other object")
            run(arm, cam, args.spot, args.target)
            return
        rec = Recorder(args.dataset, fps=C.FPS, source="autodemo")
        counts = {name: 0 for name in OBJECTS}
        while sum(counts.values()) < args.record:           # N kept demos, not N tries
            target, spot, other_spot = layout()
            other = next(name for name in OBJECTS if name != target)
            if not args.yes:
                done = sum(counts.values())
                input(f"[{done + 1}/{args.record}] {target} on {spot}, {other} on {other_spot}"
                      f' -> "{TASKS[target]}". Place them, then press Enter ')
            run(arm, cam, spot, target, rec, jitter=8.0,
                keep=None if args.yes else ask_keep)
            if rec.last_kept:
                counts[target] += 1
        print("recorded:", ", ".join(f"{n} {name}" for name, n in counts.items()))
    finally:
        arm.close()
        cam.close()


if __name__ == "__main__":
    main()
