"""Build a LeRobot v3 dataset (what `lerobot-train` wants) from the raw
recordings, for training ACT on the laptop.

    uv run export_act.py                    # data/raw/pick_ball -> data/act/pick_ball
    uv run export_act.py --with pick_ball_mujoco --out pick_ball_mix
                                            # real + simulated demos -> data/act/pick_ball_mix

ACT has no language input, so the instruction ("pick up the red ball", ...,
"pick up the block") goes in as a 6th state number: the target's place in
sorted(TASKS), 0 = block, 1 = blue ball, 2 = pink, 3 = red, 4 = yellow ball. run_policy.py
adds the same number from what you type. GR00T reads the words themselves
(export_dataset.py).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from recorder import TASKS, episodes_in, target_of  # noqa: E402

NAMES = ["base.pos", "shoulder.pos", "elbow.pos", "wrist.pos", "gripper.pos"]
TARGETS = sorted(TASKS)                 # index = the 6th state number


def export(raw, out: Path):
    """raw: a data/raw/<name> folder, or a list of them to merge."""
    episodes = episodes_in(raw)
    first = cv2.imread(str(next((episodes[0] / "frames").glob("*.jpg"))))
    h, w = first.shape[:2]
    fps = json.loads((episodes[0] / "meta.json").read_text())["fps"]
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=f"local/{out.name}", fps=fps, root=out, robot_type="gr00t_arm_sg90",
        use_videos=True,
        features={
            "observation.state": {"dtype": "float32", "shape": (6,),
                                  "names": NAMES + ["target"]},
            "action": {"dtype": "float32", "shape": (5,), "names": NAMES},
            "observation.images.front": {"dtype": "video", "shape": (h, w, 3),
                                         "names": ["height", "width", "channels"]},
        },
    )
    for ep in episodes:
        task = json.loads((ep / "meta.json").read_text())["task"]
        target = target_of(task)
        if target is None:
            raise SystemExit(f"{ep.name}: can't tell which object task {task!r} means "
                             f"(one of {', '.join(TARGETS)}; old ball/block-only "
                             "demos say just \"ball\")")
        steps = pd.read_csv(ep / "steps.csv")
        state = steps[[f"state_{i}" for i in range(5)]].to_numpy(np.float32)
        state = np.hstack([state, np.full((len(state), 1), TARGETS.index(target), np.float32)])
        action = steps[[f"action_{i}" for i in range(5)]].to_numpy(np.float32)
        for i in range(len(steps)):
            img = cv2.cvtColor(cv2.imread(str(ep / "frames" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
            ds.add_frame({"observation.state": state[i], "action": action[i],
                          "observation.images.front": img, "task": task})
        ds.save_episode()
        print(f"  {ep.name}: {len(steps)} frames, {task!r}")
    ds.finalize()
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pick_ball")
    ap.add_argument("--with", dest="also", action="append", default=[], metavar="NAME",
                    help="also take the episodes of data/raw/NAME (repeatable)")
    ap.add_argument("--out", metavar="NAME", help="write data/act/NAME (default: --name)")
    a = ap.parse_args()
    raws = [ROOT / "data" / "raw" / n for n in [a.name, *a.also]]
    export(raws, ROOT / "data" / "act" / (a.out or a.name))


if __name__ == "__main__":
    main()
