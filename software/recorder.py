"""Saves demonstration episodes in a simple raw format:

data/raw/<dataset>/episode_000000/
    frames/000000.jpg ...   camera frames (RGB)
    steps.csv               t, state_0..4, action_0..4 (servo degrees)
    meta.json               task (the instruction), fps, source (teleop / autodemo)

export_dataset.py turns these into a GR00T / LeRobot dataset.
"""
import csv
import json
import shutil
import time
from pathlib import Path

import cv2

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
# The instruction typed at run time (run_policy.py) is one of these. Both
# objects are on the plate in every episode; the instruction says which one
# goes in the cup.
TASKS = {"ball": "pick up the ball", "block": "pick up the block"}
TASK = TASKS["ball"]


def target_of(task):
    """Which object an instruction means ("ball" / "block"), or None."""
    words = task.lower()
    found = [name for name in TASKS if name in words]
    return found[0] if len(found) == 1 else None


class Recorder:
    def __init__(self, dataset="pick_ball", task=TASK, fps=30, source="teleop"):
        self.dir = DATA / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task, self.fps, self.source = task, fps, source
        self.ep = None

    @property
    def recording(self):
        return self.ep is not None

    def next_index(self):
        done = sorted(self.dir.glob("episode_*"))
        return int(done[-1].name.split("_")[1]) + 1 if done else 0

    def start(self, task=None):
        """task: this episode's instruction (default: the one given to __init__)."""
        self.ep_task = task or self.task
        self.ep = self.dir / f"episode_{self.next_index():06d}"
        (self.ep / "frames").mkdir(parents=True)
        self.rows = []
        self.t0 = time.perf_counter()
        print(f"recording {self.ep.name}")

    def add(self, frame_rgb, state, action):
        i = len(self.rows)
        cv2.imwrite(str(self.ep / "frames" / f"{i:06d}.jpg"),
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.rows.append([i / self.fps, *state, *action])

    def stop(self, keep=True):
        if self.ep is None:
            return
        if keep and len(self.rows) > 10:
            with open(self.ep / "steps.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t"] + [f"state_{i}" for i in range(5)]
                           + [f"action_{i}" for i in range(5)])
                w.writerows(self.rows)
            (self.ep / "meta.json").write_text(json.dumps(
                {"task": self.ep_task, "fps": self.fps, "source": self.source}, indent=2))
            print(f'saved {self.ep.name}: {len(self.rows)} frames, "{self.ep_task}"')
        else:
            shutil.rmtree(self.ep)
            print("discarded episode")
        self.ep = None
