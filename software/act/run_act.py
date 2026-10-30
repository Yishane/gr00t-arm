"""Let a trained ACT model drive the arm (run_policy.py act, in this folder's
environment, which has PyTorch and LeRobot).

    uv run run_act.py                          # model A: newest checkpoint of pick_ball
    uv run run_act.py --model pick_ball_mixed  # model B
    uv run run_act.py --mujoco --trials 10     # in the simulator, scored

Every run_policy.py option works: uv run run_act.py --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_policy  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "act", *sys.argv[1:]]
    run_policy.main()
