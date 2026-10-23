"""Let a trained policy drive the arm. Put the ball and the block on the
plate, then type what to do:

    instruction> pick up the block

The arm runs the policy for --seconds (or until Ctrl+C), goes home, and asks
again. Type just "ball" or "block" for short, or q / an empty line to quit.

GR00T (model running on the cloud GPU, see gr00t/README.md):
    uv run run_policy.py gr00t --host <cloud-ip> --server-port 5555
ACT (trained locally, run from software/act):
    cd act && uv run ../run_policy.py act --checkpoint ../../outputs/act/pick_ball/checkpoints/020000/pretrained_model
--task "pick up the ball" runs one instruction without asking.

Each step: grab a camera frame + the current servo angles, ask the policy for
the next chunk of actions, play the first --horizon of them at 30 Hz, repeat.
A command that would break a joint limit or put the gripper in the plate is
not sent (the arm holds still instead). If the policy fails 3 times in a row
(timeout, server error, empty answer) the run stops and asks again.
GR00T reads the instruction as words. ACT has no language input, so it gets
the target as one extra state number (see act/export_act.py).
"""
import argparse
import time

import numpy as np

import config as C
import kinematics as K
from arm_io import HOME, add_hw_args, open_hardware
from recorder import TASKS, target_of

MAX_FAILS = 3   # policy failures in a row before a run gives up


class Gr00tBackend:
    def __init__(self, host, port):
        from policy_client import PolicyClient
        self.client = PolicyClient(host, port)
        if not self.client.ping():
            raise SystemExit(f"GR00T server not reachable at {host}:{port}")
        print(f"connected to GR00T at {host}:{port}")

    def chunk(self, frame, servo, task):
        from policy_client import action_steps, make_observation
        action, _ = self.client.get_action(make_observation(frame, servo, task))
        return action_steps(action)


class ActBackend:
    """LeRobot ACT checkpoint, run on the laptop GPU."""

    def __init__(self, checkpoint):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config,
                                                       pretrained_path=checkpoint)
        # 6 = servo angles + target number; 5 = an older ball-only checkpoint
        self.with_target = self.policy.config.input_features["observation.state"].shape[0] == 6
        print(f"ACT loaded on {self.device}")

    def chunk(self, frame, servo, task):
        t = self.torch
        state = list(servo)
        if self.with_target:
            state.append(float(sorted(TASKS).index(target_of(task))))
        obs = {
            "observation.state": t.tensor(np.asarray(state, np.float32))[None],
            "observation.images.front": t.from_numpy(frame).permute(2, 0, 1)[None].float() / 255.0,
            "task": [task],
        }
        with t.inference_mode():
            batch = self.pre(obs)
            # predict_action_chunk returns the whole chunk (B, T, 5)
            actions = self.policy.predict_action_chunk(batch)
            actions = self.post(actions)
        return [list(map(float, a)) for a in actions[0].cpu().numpy()]


def instruction(text):
    """What was typed -> the instruction to send, or None if it isn't usable.
    "ball" / "block" expand to the exact wording the demos were recorded with."""
    text = text.strip()
    target = target_of(text)
    if target is None:
        return None
    return TASKS[target] if text.lower() in (target, TASKS[target]) else text


def ask():
    while True:
        try:
            text = input('\ninstruction> ').strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if text.lower() in ("", "q", "quit", "exit"):
            return None
        task = instruction(text)
        if task:
            return task
        print(f'say which object: "{TASKS["ball"]}" or "{TASKS["block"]}"')


def go_home(arm):
    servo = HOME + [C.GRIPPER_OPEN]
    arm.send(servo)
    time.sleep(1.0)
    return servo


def run(policy, arm, cam, task, horizon, seconds):
    """Run one instruction until the time is up or Ctrl+C."""
    print(f'running "{task}" for {seconds:.0f} s (Ctrl+C stops it)')
    servo = go_home(arm)
    dt = 1.0 / C.FPS
    t_end = time.time() + seconds
    fails = 0
    try:
        while time.time() < t_end:
            frame = cam.read(fresh=True)
            t0 = time.perf_counter()
            try:
                steps = policy.chunk(frame, servo, task)[:horizon]
                if not steps:
                    raise RuntimeError("empty action chunk")
            except (TimeoutError, RuntimeError) as e:
                fails += 1
                print(f"policy call failed ({e})")
                if fails >= MAX_FAILS:
                    print(f"policy failed {MAX_FAILS} times, stopping this run")
                    break
                continue
            fails = 0
            ms = 1000 * (time.perf_counter() - t0)
            sent, blocked = [], 0
            t_next = time.perf_counter()
            for cmd in steps:
                cmd = [float(np.clip(a, C.SERVO_MIN, C.SERVO_MAX)) for a in cmd]
                if K.servo_ok(cmd[:4]):
                    arm.send(cmd)
                    servo = cmd
                    sent.append(cmd)
                else:
                    blocked += 1                          # hold the last safe command
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
            grip = f", grip {sent[0][4]:.0f}" if sent else ""
            print(f"policy call {ms:.0f} ms, {len(steps)} steps{grip}"
                  + (f", {blocked} blocked" if blocked else ""))
    except KeyboardInterrupt:
        print("stopped")
    go_home(arm)


def positive_int(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["gr00t", "act"])
    add_hw_args(ap)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--server-port", type=int, default=5555)
    ap.add_argument("--checkpoint", help="ACT pretrained_model folder")
    ap.add_argument("--task", help="run this one instruction and exit (no prompt)")
    ap.add_argument("--horizon", type=positive_int, default=8,
                    help="actions played per policy call (GR00T predicts 16)")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="how long each instruction runs")
    args = ap.parse_args()

    task = None
    if args.task:
        task = instruction(args.task)
        if task is None:
            raise SystemExit(f"--task must name the ball or the block: {args.task!r}")
    policy = (Gr00tBackend(args.host, args.server_port) if args.backend == "gr00t"
              else ActBackend(args.checkpoint))
    arm, cam = open_hardware(args)
    go_home(arm)
    try:
        if task:
            run(policy, arm, cam, task, args.horizon, args.seconds)
            return
        print(f'type an instruction, e.g. "{TASKS["ball"]}" or "{TASKS["block"]}" '
              f"(q to quit)")
        while (task := ask()) is not None:
            run(policy, arm, cam, task, args.horizon, args.seconds)
    finally:
        go_home(arm)
        arm.close()
        cam.close()


if __name__ == "__main__":
    main()
