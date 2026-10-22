"""Turn recorded episodes (data/raw/<name>) into a GR00T-ready dataset:
LeRobot v2.1 layout + meta/modality.json, the format Isaac-GR00T fine-tunes on.

    uv run export_dataset.py                 # data/raw/pick_ball -> data/lerobot/pick_ball
    uv run export_dataset.py --name pick_ball --out ../data/lerobot/pick_ball

Layout written:
    meta/info.json  meta/episodes.jsonl  meta/tasks.jsonl  meta/modality.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.front/episode_000000.mp4   (H.264)

State and action are 5 servo angles in degrees: base, shoulder, elbow, wrist,
gripper. modality.json splits them into "single_arm" (0:4) and "gripper" (4:5).
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JOINT_NAMES = ["base.pos", "shoulder.pos", "elbow.pos", "wrist.pos", "gripper.pos"]
VIDEO_KEY = "observation.images.front"
CHUNK = 1000

MODALITY = {
    "state": {"single_arm": {"start": 0, "end": 4}, "gripper": {"start": 4, "end": 5}},
    "action": {"single_arm": {"start": 0, "end": 4}, "gripper": {"start": 4, "end": 5}},
    "video": {"front": {"original_key": VIDEO_KEY}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def encode_video(frames_dir: Path, out: Path, fps: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-framerate", str(fps), "-i", str(frames_dir / "%06d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "20", str(out)]
    subprocess.run(cmd, check=True)


def frame_size(frames_dir: Path):
    import cv2
    h, w = cv2.imread(str(next(frames_dir.glob("*.jpg")))).shape[:2]
    return h, w


def export(raw: Path, out: Path):
    episodes = sorted(p for p in raw.glob("episode_*") if (p / "steps.csv").exists())
    if not episodes:
        raise SystemExit(f"no recorded episodes in {raw}")
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    tasks, ep_meta, total, fps = {}, [], 0, None
    for ei, ep in enumerate(episodes):
        meta = json.loads((ep / "meta.json").read_text())
        fps = fps or meta["fps"]
        task_index = tasks.setdefault(meta["task"], len(tasks))
        steps = pd.read_csv(ep / "steps.csv")
        n = len(steps)
        state = steps[[f"state_{i}" for i in range(5)]].to_numpy(np.float32)
        action = steps[[f"action_{i}" for i in range(5)]].to_numpy(np.float32)
        chunk = ei // CHUNK
        df = pd.DataFrame({
            "observation.state": list(state),
            "action": list(action),
            "timestamp": (np.arange(n) / fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, np.int64),
            "index": np.arange(total, total + n, dtype=np.int64),
            "task_index": np.full(n, task_index, np.int64),
            "annotation.human.task_description": np.full(n, task_index, np.int64),
        })
        pq = out / "data" / f"chunk-{chunk:03d}" / f"episode_{ei:06d}.parquet"
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
        encode_video(ep / "frames",
                     out / "videos" / f"chunk-{chunk:03d}" / VIDEO_KEY / f"episode_{ei:06d}.mp4",
                     fps)
        ep_meta.append({"episode_index": ei, "tasks": [meta["task"]], "length": n})
        total += n
        print(f"  {ep.name} -> episode {ei} ({n} frames)")

    h, w = frame_size(episodes[0] / "frames")
    vec = lambda: {"dtype": "float32", "shape": [5], "names": JOINT_NAMES}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.1",
        "robot_type": "gr00t_arm_sg90",
        "total_episodes": len(episodes),
        "total_frames": total,
        "total_tasks": len(tasks),
        "total_videos": len(episodes),
        "total_chunks": (len(episodes) - 1) // CHUNK + 1,
        "chunks_size": CHUNK,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": vec(),
            "observation.state": vec(),
            VIDEO_KEY: {
                "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
                "info": {"video.height": h, "video.width": w, "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                         "video.fps": fps, "video.channels": 3, "has_audio": False},
            },
            "timestamp": scalar("float32"),
            "frame_index": scalar("int64"),
            "episode_index": scalar("int64"),
            "index": scalar("int64"),
            "task_index": scalar("int64"),
            "annotation.human.task_description": scalar("int64"),
        },
    }
    m = out / "meta"
    (m / "info.json").write_text(json.dumps(info, indent=4))
    (m / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_meta))
    (m / "tasks.jsonl").write_text("".join(
        json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
    (m / "modality.json").write_text(json.dumps(MODALITY, indent=4))
    print(f"wrote {out}: {len(episodes)} episodes, {total} frames, {len(tasks)} task(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="pick_ball")
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    export(a.raw or ROOT / "data" / "raw" / a.name, a.out or ROOT / "data" / "lerobot" / a.name)


if __name__ == "__main__":
    main()
