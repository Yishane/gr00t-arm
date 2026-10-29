"""Saves demonstration episodes in a simple raw format:

data/raw/<dataset>/episode_000000/
    frames/000000.jpg ...   camera frames (RGB)
    steps.csv               t, state_0..4, action_0..4 (servo degrees)
    meta.json               task (the instruction), fps, source (teleop / autodemo / policy),
                            and "sim": true for episodes recorded in the simulator (sim.py)

export_dataset.py turns these into a GR00T / LeRobot dataset.
"""
import csv
import json
import shutil
import time
from pathlib import Path

import cv2

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
# The instruction typed at run time (run_policy.py) is one of these. Two
# objects are on the plate in every episode (a ball and the block, or two
# balls of different colours); the instruction says which one goes in the cup.
BALL_COLOURS = ("red", "pink", "yellow", "blue")
TARGETS = [f"{c} ball" for c in BALL_COLOURS] + ["block"]
TASKS = {t: f"pick up the {t}" for t in TARGETS}
TASK = TASKS["red ball"]


def kind_of(target):
    """"red ball" -> "ball", "block" -> "block"."""
    return target.split()[-1]


def target_of(task):
    """Which object an instruction means ("red ball", ..., "block"), or None
    (no object, two objects, or a ball without its colour)."""
    words = task.lower()
    found = [name for name in TASKS if name in words]
    return found[0] if len(found) == 1 else None


def ask_keep():
    """y / Enter keeps the episode, n throws it away (missed grip, dropped object)."""
    return not input("keep it? [Y/n] ").strip().lower().startswith("n")


def episodes_in(raws):
    """Saved episodes of one or more raw dataset folders, in order."""
    raws = [raws] if isinstance(raws, Path) else raws
    episodes = [p for raw in raws for p in sorted(raw.glob("episode_*"))
                if (p / "steps.csv").exists()]
    if not episodes:
        raise SystemExit(f"no recorded episodes in {', '.join(map(str, raws))}")
    return episodes


class Recorder:
    def __init__(self, dataset="pick_ball", task=TASK, fps=30, source="teleop", sim=False):
        self.dir = DATA / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task, self.fps, self.source, self.sim = task, fps, source, sim
        self.ep = None
        self.last_kept = False   # whether the last stop() saved its episode

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
        self.last_kept = keep and len(self.rows) > 10
        if self.last_kept:
            with open(self.ep / "steps.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t"] + [f"state_{i}" for i in range(5)]
                           + [f"action_{i}" for i in range(5)])
                w.writerows(self.rows)
            meta = {"task": self.ep_task, "fps": self.fps, "source": self.source}
            if self.sim:
                meta["sim"] = True
            (self.ep / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f'saved {self.ep.name}: {len(self.rows)} frames, "{self.ep_task}"')
        else:
            shutil.rmtree(self.ep)
            print("discarded episode")
        self.ep = None
