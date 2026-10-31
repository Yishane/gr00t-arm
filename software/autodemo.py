"""Scripted pick-and-place with two objects on the plate, a ball and a
block. The script picks the one it's told to from its marked spot (A1..C3 on
the baseplate) and drops it in the cup. Two uses:

  1. Check the arm works / show it off:
         uv run autodemo.py --target block --spot C1 --other-spot A3
  2. Collect training demos automatically (the camera records while the
     script moves the arm; you only place the two objects where it tells you):
         uv run autodemo.py --record 150
  3. The same in the MuJoCo simulation (sim.py): it places the objects
     itself (--offset 15 moves them up to 15 mm off the spots; --anywhere puts
     them anywhere the arm can pick and the camera can see), keeps only the
     picks that land the right object in the cup, and saves them in
     data/raw/pick_ball_mujoco:
         uv run autodemo.py --mujoco --record 300 --anywhere

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

import cv2

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
# Real arm only (the simulator doesn't sag): grip this much higher (mm) for
# each object at every spot, on top of SPOT_FIX.
GRIP_FIX = {"ball": 4.0}
# Per-spot corrections (mm) for spots the real arm misses while the rest are
# fine: (toward column C, toward row 3, up). Tune these by watching picks.
#   gripper lands too far toward C3 at A2 -> make the first two negative
#   grips too low at C3                    -> make the third positive


SPOT_FIX = {}


# Extra corrections for ONE object at a spot, added on top of SPOT_FIX (same
# order: toward column C, toward row 3, up). For a spot where the block misses
# but the ball is fine, e.g. the block grips 3 mm too low and 2 mm short of C3:
#   "block": {"C3": (0.0, 2.0, 3.0)},
OBJECT_SPOT_FIX = {
    "block": {
        
        "C3": (0,0,0),
        "B3": (0,0,0),
        "A3": (0,0,0),
        "A2": (0,0,0),
        "C2": (0,0,0),  
        "A1": (4.0, 0.0, 0.0),
        "B1": (0.0, 0.0, 0.0),
        "C1": (0.0, 0.0, 0.0),
        "B2": (0.0, 0.0, 0.0),
                
                },
    "ball": { 
        
        "C3": (8.0, 5.0, 15.0),
        "B3": (9.0, 9.0, 12.0),
        "A3": (2.0, -5.0, 12.0),
        "A2": (-3.0, -3.0, 0.0),
        "C2": (7.0, 7.0, 16.0),  
        "A1": (3.0, 0.0, 0.0),
        "B1": (7.0, 0.0, 0.0),
        "C1": (12.0, -2.0, 9.0),
        "B2": (3.0, 12.0, 4.0),
    }
}
# Both objects are on the plate. Spot centres at least MIN_GAP apart, and not
# a pair in TOO_CLOSE, so the gripper can't touch the object it isn't picking
# (cad/assembly.py --cell checks every pair). That leaves B2 (the middle) out.
MIN_GAP = 60.0
TOO_CLOSE = {frozenset(("C3", "A2"))}   # at C3 the tilted gripper reaches back over A2
SAFE_Z = 75.0
DROP_Z = 80.0
MAX_SPEED = 60.0    # deg/s: the fastest any servo is commanded to move
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
    """Waypoints (x, y, z, pitch, gripper_closed, seconds) for one pick.
    spot: a marked spot ("C1"), or an (x, y) position in mm (the simulator
    puts objects anywhere; SPOT_FIX is only for the real arm's spots)."""
    if isinstance(spot, str):
        fixes = (SPOT_FIX.get(spot, (0.0, 0.0, 0.0)),
                 OBJECT_SPOT_FIX.get(target, {}).get(spot, (0.0, 0.0, 0.0)))
        dx, dy, dz = (sum(f[i] for f in fixes) for i in range(3))
        dz += GRIP_FIX.get(target, 0.0)
        sx, sy = SPOTS[spot][0] + dx, SPOTS[spot][1] + dy
    else:
        (sx, sy), dz = spot, 0.0
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
    instead, so the arm never jumps. A move that would turn any servo faster
    than MAX_SPEED takes longer instead."""
    cur = waypoints[0]
    last = solve_near(*cur[:4])
    limit = MAX_SPEED / fps                               # deg per step
    for wp in waypoints[1:]:
        n = max(1, int(wp[5] * fps))
        leg, peak = _leg(cur, wp, n), math.inf
        while True:
            new_peak = max(max(abs(a - b) for a, b in zip(p, q))
                           for p, q in zip([last] + leg, leg))
            # a small jump where the gripper tilt changes solution doesn't get
            # smaller when the move is slower: stop stretching then
            if new_peak <= limit or new_peak > 0.9 * peak:
                break
            peak = new_peak
            n = math.ceil(n * peak / limit) + 1
            leg = _leg(cur, wp, n)
        grip = C.GRIPPER_CLOSED if wp[4] else C.GRIPPER_OPEN
        for servo in leg:
            yield servo + [grip]
        last, cur = leg[-1], wp


def _leg(cur, wp, n):
    """n servo commands easing from waypoint cur to waypoint wp."""
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)                          # ease in/out
        pose = [a + (b - a) * s for a, b in zip(cur[:4], wp[:4])]
        servo = solve_near(*pose)
        if servo is None:
            raise ValueError(f"can't reach {pose}")
        out.append(servo)
    return out


def preview(frame):
    """Show a camera frame (RGB) in a window; the arm keeps moving meanwhile."""
    cv2.imshow("autodemo camera", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def run(arm, cam, spot, target="ball", rec=None, jitter=0.0, keep=None, show=None):
    """keep: asked after the moves whether to save the episode (default: save).
    show: called with every camera frame (the live preview)."""
    dt = 1.0 / C.FPS
    last = arm.last
    cmds = list(trajectory(plan(spot, target, jitter), C.FPS))   # fails here, before moving
    if rec:
        rec.start(task=TASKS[target])
    t_next = time.perf_counter()
    try:
        for cmd in cmds:
            frame = cam.read()
            if show:
                show(frame)
            if rec:
                rec.add(frame, last, cmd)
            arm.send(cmd)
            last = cmd
            if not getattr(arm, "lockstep", False):     # the simulator steps in send()
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
    ap.add_argument("--dataset", help="where to save: data/raw/DATASET "
                    "(default pick_ball, or pick_ball_mujoco with --mujoco)")
    ap.add_argument("--offset", type=float, default=0.0, metavar="MM",
                    help="with --mujoco: put each object up to MM away from its spot")
    ap.add_argument("--anywhere", action="store_true",
                    help="with --mujoco: put the objects anywhere the arm can pick "
                         "and the camera can see, not just near the spots")
    ap.add_argument("--yes", action="store_true", help="don't wait for Enter (testing)")
    ap.add_argument("--no-preview", action="store_true",
                    help="don't show the camera while the arm moves")
    ap.add_argument("--watch", type=float, nargs="?", const=15.0, metavar="SPEED",
                    help="with --mujoco: show the simulated camera, SPEED times real "
                         "speed (default 15; recording is fastest without --watch)")
    args = ap.parse_args()
    if (args.offset or args.anywhere) and not args.mujoco:
        raise SystemExit("--offset and --anywhere are for the simulator (--mujoco): on the "
                         "real arm the objects go on the marked spots")
    if args.offset and args.anywhere:
        raise SystemExit("use --offset or --anywhere, not both")
    dataset = args.dataset or ("pick_ball_mujoco" if args.mujoco else "pick_ball")

    # the simulator only moves when commands arrive (no waiting for the clock)
    arm, cam = open_hardware(args, realtime=False)
    world = getattr(arm, "world", None)
    show = None if (world or args.yes or args.no_preview) else preview
    if world and args.watch:
        def show(frame):
            preview(frame)
            time.sleep(1.0 / C.FPS / args.watch)   # the simulator would otherwise run flat out
    arm.send(HOME + [C.GRIPPER_OPEN])
    getattr(arm, "wait", time.sleep)(1.0)
    try:
        if not args.record:
            if args.other_spot and (args.spot, args.other_spot) not in pairs():
                raise SystemExit(f"{args.spot} and {args.other_spot} are too close: "
                                 "the gripper could hit the other object")
            spot = args.spot
            if world:
                # both objects are always in the simulator: without --other-spot
                # the other one waits at the back of the plate, out of the way
                other_xy = SPOTS[args.other_spot] if args.other_spot else (165.0, 135.0)
                world.place_for(args.target, SPOTS[args.spot], other_xy)
                # by coordinates: SPOT_FIX / GRIP_FIX correct the REAL arm's errors,
                # and would make the simulated arm grip beside or above the object
                spot = SPOTS[args.spot]
            run(arm, cam, spot, args.target, show=show)
            if world:
                print(f"{args.target}: {world.result(args.target)}")
            return
        rec = Recorder(dataset, fps=C.FPS, source="autodemo", sim=world is not None)
        counts = {name: 0 for name in OBJECTS}
        fails = 0
        while sum(counts.values()) < args.record:           # N kept demos, not N tries
            done = sum(counts.values())
            if world:
                # the simulator places the objects itself and checks the result
                target, xy, other_xy = world.random_layout(args.offset, anywhere=args.anywhere)
                world.place_for(target, xy, other_xy)
                run(arm, cam, xy, target, rec, jitter=8.0, keep=lambda: world.success(target),
                    show=show)
                print(f"[{done + 1}/{args.record}] {target} at ({xy[0]:.0f}, {xy[1]:.0f}): "
                      f"{world.result(target)}")
                fails = 0 if rec.last_kept else fails + 1
                if fails >= 20:
                    raise SystemExit("20 simulated picks in a row failed: check the "
                                     "calibration and the gripper angles in sim.py")
            else:
                target, spot, other_spot = layout()
                other = next(name for name in OBJECTS if name != target)
                if not args.yes:
                    input(f"[{done + 1}/{args.record}] {target} on {spot}, {other} on {other_spot}"
                          f' -> "{TASKS[target]}". Place them, then press Enter ')
                run(arm, cam, spot, target, rec, jitter=8.0,
                    keep=None if args.yes else ask_keep, show=show)
            if rec.last_kept:
                counts[target] += 1
        print("recorded:", ", ".join(f"{n} {name}" for name, n in counts.items()))
    finally:
        arm.close()
        cam.close()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
