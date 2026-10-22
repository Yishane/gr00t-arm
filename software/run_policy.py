"""Let a trained policy drive the arm.

GR00T (model running on the cloud GPU, see gr00t/README.md):
    uv run run_policy.py gr00t --host <cloud-ip> --server-port 5555
ACT (trained locally, run from software/act):
    cd act && uv run ../run_policy.py act --checkpoint ../../outputs/act/checkpoints/last/pretrained_model

Each step: grab a camera frame + the current servo angles, ask the policy for
the next chunk of actions, play the first --horizon of them at 30 Hz, repeat.
Press Ctrl+C to stop (the arm goes home).
"""
import argparse
import time

import numpy as np

import config as C
from arm_io import HOME, add_hw_args, open_hardware
from recorder import TASK


class Gr00tBackend:
    def __init__(self, host, port, task):
        from policy_client import PolicyClient
        self.client = PolicyClient(host, port)
        self.task = task
        if not self.client.ping():
            raise SystemExit(f"GR00T server not reachable at {host}:{port}")
        print(f"connected to GR00T at {host}:{port}")

    def chunk(self, frame, servo):
        from policy_client import action_steps, make_observation
        action, _ = self.client.get_action(make_observation(frame, servo, self.task))
        return action_steps(action)


class ActBackend:
    """LeRobot ACT checkpoint, run on the laptop GPU."""

    def __init__(self, checkpoint, task):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config,
                                                       pretrained_path=checkpoint)
        self.task = task
        print(f"ACT loaded on {self.device}")

    def chunk(self, frame, servo):
        t = self.torch
        obs = {
            "observation.state": t.tensor(np.asarray(servo, np.float32))[None],
            "observation.images.front": t.from_numpy(frame).permute(2, 0, 1)[None].float() / 255.0,
            "task": [self.task],
        }
        with t.inference_mode():
            batch = self.pre(obs)
            # predict_action_chunk returns the whole chunk (B, T, 5)
            actions = self.policy.predict_action_chunk(batch)
            actions = self.post(actions)
        return [list(map(float, a)) for a in actions[0].cpu().numpy()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["gr00t", "act"])
    add_hw_args(ap)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--server-port", type=int, default=5555)
    ap.add_argument("--checkpoint", help="ACT pretrained_model folder")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--horizon", type=int, default=8,
                    help="actions played per policy call (GR00T predicts 16)")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    policy = (Gr00tBackend(args.host, args.server_port, args.task) if args.backend == "gr00t"
              else ActBackend(args.checkpoint, args.task))
    arm, cam = open_hardware(args)
    servo = HOME + [C.GRIPPER_OPEN]
    arm.send(servo)
    time.sleep(1.0)
    dt = 1.0 / C.FPS
    t_end = time.time() + args.seconds
    try:
        while time.time() < t_end:
            frame = cam.read()
            t0 = time.perf_counter()
            steps = policy.chunk(frame, servo)[:args.horizon]
            print(f"policy call {1000 * (time.perf_counter() - t0):.0f} ms, "
                  f"{len(steps)} steps, grip {steps[0][4]:.0f}")
            t_next = time.perf_counter()
            for cmd in steps:
                cmd = [float(np.clip(a, C.SERVO_MIN, C.SERVO_MAX)) for a in cmd]
                arm.send(cmd)
                servo = cmd
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        arm.send(HOME + [C.GRIPPER_OPEN])
        time.sleep(0.5)
        arm.close()
        cam.close()


if __name__ == "__main__":
    main()
