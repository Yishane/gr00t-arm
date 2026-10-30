"""Train ACT on the laptop GPU.

    uv run train_act.py                          # data/act/pick_ball, 20k steps
    uv run train_act.py --name pick_ball --steps 30000
    uv run train_act.py --resume                 # continue from the newest checkpoint

~2.3 steps/s on an RTX 5050 at batch 8, so 20k steps is about 2.5 hours.
Checkpoints: outputs/act/<name>/checkpoints/<step>/pretrained_model
Extra lerobot-train options can be appended, e.g. --batch_size=16.
"""
import argparse
import sys
from pathlib import Path

from lerobot.scripts import lerobot_train

ROOT = Path(__file__).resolve().parents[2]


def _no_symlink(checkpoint_dir):
    # lerobot links checkpoints/last -> newest, which Windows refuses without
    # Developer Mode and which would crash training at the first save. Write
    # the newest step to a text file instead.
    (Path(checkpoint_dir).parent / "last.txt").write_text(Path(checkpoint_dir).name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="pick_ball")
    ap.add_argument("--steps", type=int, default=20000,
                    help="total steps (with --resume: where to stop, counted from 0)")
    ap.add_argument("--resume", action="store_true",
                    help="continue outputs/act/<name> from its newest checkpoint")
    a, extra = ap.parse_known_args()
    if a.resume:
        saved = sorted((ROOT / "outputs" / "act" / a.name / "checkpoints").glob(
            "*/pretrained_model/train_config.json"))
        if not saved:
            raise SystemExit(f"nothing to resume: no checkpoint in outputs/act/{a.name}")
        print(f"resuming from {saved[-1].parent.parent.name}")
        extra = ["--resume=true", f"--config_path={saved[-1]}", *extra]
    lerobot_train.update_last_checkpoint = _no_symlink
    sys.argv = [
        "lerobot-train",
        f"--dataset.repo_id=local/{a.name}",
        f"--dataset.root={ROOT / 'data' / 'act' / a.name}",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--output_dir={ROOT / 'outputs' / 'act' / a.name}",
        f"--steps={a.steps}",
        "--batch_size=8",
        f"--save_freq={max(1000, a.steps // 5)}",
        "--log_freq=200",
        "--num_workers=0",
        "--wandb.enable=false",
        *extra,
    ]
    lerobot_train.main()


if __name__ == "__main__":
    main()
