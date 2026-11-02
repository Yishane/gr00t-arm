"""Did a trained ACT model learn the demos? No arm needed.

Runs the model on frames of recorded demos (the same picture, servo angles and
ball/block number it trained on) and compares its next moves with what the
script really sent. With --live it also looks at one live camera frame (arm at
home, objects on their spots), to see whether the live picture makes it jump.

    uv run check_act.py --model pick_ball_noaug --step 60k
    uv run check_act.py --model pick_ball_noaug --step 60k --live --target ball

How to read it:
- "model" vs "stand still": the average error (degrees) of the model's next
  25 moves, next to the error of not moving at all. A model that learned the
  demos is far below "stand still"; one that didn't is near or above it.
- "gripper ends": how far (mm) the model would put the gripper after those
  25 moves from where the script put it.
- --live: how far the model wants to move in its first 25 steps from home, on
  the live picture vs on the recorded start pictures. Much more on the live
  picture = the live picture looks different to it (light, layout, view).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import kinematics as K  # noqa: E402
from arm_io import HOME  # noqa: E402
from autodemo import CUP, SPOTS  # noqa: E402
from recorder import DATA, TASKS, episodes_in  # noqa: E402
import run_policy  # noqa: E402

HORIZON = 25          # moves compared per policy call (what run_act plays with --horizon 25)
JOINTS = ["base", "shoulder", "elbow", "wrist"]


def load(ep):
    """(task, state, action) of one recorded episode; state/action: servo degrees."""
    task = json.loads((ep / "meta.json").read_text())["task"]
    steps = pd.read_csv(ep / "steps.csv")
    state = steps[[f"state_{i}" for i in range(5)]].to_numpy(float)
    action = steps[[f"action_{i}" for i in range(5)]].to_numpy(float)
    return task, state, action


def picture(ep, i):
    img = cv2.imread(str(ep / "frames" / f"{i:06d}.jpg"))
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gripper_xyz(servo):
    return np.array(K.fk(*K.from_servo(list(servo[:4])))[:3])


def place_name(xyz):
    """Nearest marked spot or the cup, for a gripper position (mm)."""
    places = {**SPOTS, "the cup": CUP}
    name = min(places, key=lambda n: math.dist(places[n], xyz[:2]))
    return f"{name} ({math.dist(places[name], xyz[:2]):.0f} mm off)"


def first_move(chunk, servo, horizon=HORIZON):
    """Biggest change (deg) of any arm servo from `servo` in the first moves."""
    c = np.asarray(chunk, float)[:horizon, :4]
    return float(np.abs(c - np.asarray(servo[:4], float)).max())


def check_episode(policy, ep, every=15, horizon=HORIZON):
    """Compare the model with the script every `every` frames of one episode."""
    task, state, action = load(ep)
    model, still, grip, end_mm = [], [], [], []
    start = None
    for i in range(0, len(state), every):
        img = picture(ep, i)
        if img is None:
            continue
        pred = np.asarray(policy.chunk(img, list(state[i]), task), float)
        n = min(horizon, len(pred), len(action) - i)
        truth = action[i:i + n]
        model.append(np.abs(pred[:n, :4] - truth[:, :4]).mean(axis=0))
        still.append(np.abs(state[i, :4] - truth[:, :4]).mean(axis=0))
        grip.append(np.mean([run_policy.snap_gripper(p) == run_policy.snap_gripper(t)
                             for p, t in zip(pred[:n, 4], truth[:, 4])]))
        end_mm.append(float(np.linalg.norm(gripper_xyz(pred[n - 1]) - gripper_xyz(truth[-1]))))
        if i == 0:
            start = first_move(pred, state[0], horizon)
    return {"task": task, "model": np.mean(model, axis=0), "still": np.mean(still, axis=0),
            "grip": float(np.mean(grip)), "end_mm": float(np.mean(end_mm)), "start": start}


def spread(items, n):
    """n items spread evenly over the list (all if it's shorter)."""
    if len(items) <= n:
        return items
    return [items[k * len(items) // n] for k in range(n)]


def report(policy, episodes, every=15):
    rows = []
    print(f"{'episode':16s} {'instruction':18s} {'model':>6s} {'still':>6s} "
          f"{'gripper ends':>12s} {'open/close':>10s}")
    for ep in episodes:
        r = check_episode(policy, ep, every)
        rows.append(r)
        print(f"{ep.name[-16:]:16s} {r['task']:18s} {r['model'].mean():5.1f}° {r['still'].mean():5.1f}° "
              f"{r['end_mm']:9.0f} mm {100 * r['grip']:9.0f}%")
    model = np.mean([r["model"] for r in rows], axis=0)
    still = np.mean([r["still"] for r in rows], axis=0)
    print("\naverage error of the next", HORIZON, "moves (degrees):")
    for j, name in enumerate(JOINTS):
        print(f"  {name:8s} model {model[j]:5.1f}   stand still {still[j]:5.1f}")
    ratio = model.mean() / max(still.mean(), 1e-6)
    end = np.mean([r["end_mm"] for r in rows])
    grip = np.mean([r["grip"] for r in rows])
    print(f"gripper ends {end:.0f} mm from the script's on average; open/close agrees {100 * grip:.0f}%")
    if ratio < 0.35:
        verdict = "LEARNED the demos: it does what the script did on these pictures."
    elif ratio < 0.7:
        verdict = "PARTLY learned: closer than standing still, but sloppy."
    else:
        verdict = "did NOT learn the demos: no better than standing still."
    print(f"model error = {100 * ratio:.0f}% of stand-still -> {verdict}")
    starts = [r["start"] for r in rows if r["start"] is not None]
    return float(np.mean(starts)) if starts else None


def live_check(policy, camera, target, recorded_start):
    from arm_io import Camera
    cam = Camera(camera)
    try:
        img = cam.read(fresh=True)
    finally:
        cam.close()
    out = DATA.parent / "check_live.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    servo = HOME + [C.GRIPPER_OPEN]
    pred = np.asarray(policy.chunk(img, servo, TASKS[target]), float)
    move = first_move(pred, servo)
    print(f'\nLIVE picture (saved to {out}), "{TASKS[target]}", arm at home:')
    print(f"  first {HORIZON} moves change a servo by up to {move:.0f}°"
          + (f" (recorded start pictures: {recorded_start:.0f}° on average)"
             if recorded_start is not None else ""))
    for k in (HORIZON - 1, len(pred) - 1):
        xyz = gripper_xyz(pred[k])
        print(f"  after {k + 1:3d} moves the gripper would be at ({xyz[0]:.0f}, {xyz[1]:.0f}, "
              f"{xyz[2]:.0f}) mm, nearest {place_name(xyz)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="pick_ball_noaug", help="outputs/act/<name>")
    ap.add_argument("--step", help="checkpoint, e.g. 60k (default: the newest)")
    ap.add_argument("--dataset", default="pick_ball", help="recorded demos: data/raw/<name>")
    ap.add_argument("--episodes", type=int, default=12, help="how many demos to check (spread out)")
    ap.add_argument("--live", action="store_true", help="also check one live camera frame")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--target", default="ball", choices=sorted(TASKS), help="for --live")
    a = ap.parse_args()
    (step, folder), = run_policy.checkpoints(a.model, a.step)[-1:]
    print(f"{a.model}, checkpoint {step}")
    policy = run_policy.ActBackend(folder)
    episodes = spread(episodes_in(DATA / a.dataset), a.episodes)
    recorded_start = report(policy, episodes)
    if a.live:
        live_check(policy, a.camera, a.target, recorded_start)


if __name__ == "__main__":
    main()
