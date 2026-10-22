"""Build a LeRobot v3 dataset (what `lerobot-train` wants) from the raw
recordings, for training ACT on the laptop.

    uv run export_act.py                    # data/raw/pick_ball -> data/act/pick_ball
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["base.pos", "shoulder.pos", "elbow.pos", "wrist.pos", "gripper.pos"]


def export(raw: Path, out: Path):
    episodes = sorted(p for p in raw.glob("episode_*") if (p / "steps.csv").exists())
    if not episodes:
        raise SystemExit(f"no recorded episodes in {raw}")
    first = cv2.imread(str(next((episodes[0] / "frames").glob("*.jpg"))))
    h, w = first.shape[:2]
    fps = json.loads((episodes[0] / "meta.json").read_text())["fps"]
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=f"local/{out.name}", fps=fps, root=out, robot_type="gr00t_arm_sg90",
        use_videos=True,
        features={
            "observation.state": {"dtype": "float32", "shape": (5,), "names": NAMES},
            "action": {"dtype": "float32", "shape": (5,), "names": NAMES},
            "observation.images.front": {"dtype": "video", "shape": (h, w, 3),
                                         "names": ["height", "width", "channels"]},
        },
    )
    for ep in episodes:
        task = json.loads((ep / "meta.json").read_text())["task"]
        steps = pd.read_csv(ep / "steps.csv")
        state = steps[[f"state_{i}" for i in range(5)]].to_numpy(np.float32)
        action = steps[[f"action_{i}" for i in range(5)]].to_numpy(np.float32)
        for i in range(len(steps)):
            img = cv2.cvtColor(cv2.imread(str(ep / "frames" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
            ds.add_frame({"observation.state": state[i], "action": action[i],
                          "observation.images.front": img, "task": task})
        ds.save_episode()
        print(f"  {ep.name}: {len(steps)} frames")
    ds.finalize()
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pick_ball")
    a = ap.parse_args()
    export(ROOT / "data" / "raw" / a.name, ROOT / "data" / "act" / a.name)


if __name__ == "__main__":
    main()
