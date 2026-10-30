"""Let a trained policy drive the arm. Put the ball and the block on the
plate, then type what to do:

    instruction> pick up the block

The arm runs the policy for --seconds (or until Ctrl+C), goes home, and asks
again. Type just "ball" or "block" for short, or q / an empty line to quit.

GR00T (model running on the cloud GPU, see gr00t/README.md):
    uv run run_policy.py gr00t --host <cloud-ip> --server-port 5555
ACT (trained locally, run from software/act):
    cd act && uv run run_act.py                       # newest checkpoint of model pick_ball
    cd act && uv run run_act.py --model pick_ball_mixed
    (--checkpoint ../../outputs/act/pick_ball/checkpoints/016000/pretrained_model picks one)
--task "pick up the ball" runs one instruction without asking.
--view shows the camera live with the instruction on top; --fullscreen fills
the screen with it (for a projector).

Each step: grab a camera frame + the current servo angles, ask the policy for
the next chunk of actions, play the first --horizon of them at 30 Hz, repeat.
The next call goes out while the current chunk is still playing, so the arm
doesn't stop and wait for it; the steps that played during the call are
skipped in the new chunk. --no-pipeline waits for each call instead.
A command that would break a joint limit or put the gripper in the plate is
cut short: the arm goes as far toward it as is safe (printed as "N limited"),
and the gripper still opens or closes as asked. (Holding still instead left
the model asking for the same move forever, gripper included.) Garbage
commands (NaN, wrong length) are not sent at all. The gripper goes fully open
or fully closed, whichever the model's value is nearer: every demo does that,
and a model stopping half-way (e.g. 82 of 90) grips looser than anything it
learned from. If the policy fails 3 times in a row
(timeout, server error, empty answer) the run stops and asks again.
GR00T reads the instruction as words. ACT has no language input, so it gets
the target as one extra state number (see act/export_act.py).

--record saves each run as an episode (source "policy", same format as the
teleop / autodemo demos) in data/raw/pick_ball_policy, or --record <dataset>.
Press Ctrl+C once the object is in the cup so the idle rest isn't recorded,
then answer keep it? [Y/n]: keep only the runs that picked the right object
cleanly, so retraining on them doesn't teach the policy its own mistakes.

--mujoco runs it in the simulation (sim.py) instead: each run puts the ball
and the block on a random layout (--offset MM moves them off the spots,
--anywhere puts them anywhere the arm can pick and the camera can see), stops
as soon as an object lands in the cup, and prints ok / knocked / wrong object
/ missed. --trials N runs N of them on their own and prints the score; with
--record, only the "ok" runs are kept (data/raw/pick_ball_mujoco).
"""
import argparse
import math
import random
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

import config as C
import kinematics as K
from arm_io import HOME, add_hw_args, open_hardware
from recorder import TASKS, Recorder, ask_keep, target_of

MAX_FAILS = 3   # policy failures in a row before a run gives up
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"   # trained models: OUTPUTS/act/<name>
# one policy call at a time: the GR00T client has one socket, and a call left
# running by Ctrl+C must finish before the next instruction's first call
_CALL_LOCK = threading.Lock()


def checkpoints(name, which=None):
    """[(step, folder)] of ACT model `name` (outputs/act/<name>): its newest
    checkpoint, or `which`: one step ("60000" or "60k"), several ("40k,60k"),
    or "all"."""
    saved = {int(p.parent.name): p for p in
             (OUTPUTS / "act" / name / "checkpoints").glob("*/pretrained_model")
             if p.parent.name.isdigit()}
    if not saved:
        raise SystemExit(f"no trained model '{name}' in {OUTPUTS / 'act'}: train it first "
                         "(software/act: uv run train_act.py --name ...)")
    if which is None:
        return [max(saved.items())]
    if which.strip().lower() == "all":
        return sorted(saved.items())
    out = []
    for item in which.split(","):
        item = item.strip().lower()
        step = int(float(item[:-1]) * 1000) if item.endswith("k") else int(item)
        if step not in saved:
            raise SystemExit(f"{name} has no checkpoint at step {step}; it has: "
                             + ", ".join(str(s) for s in sorted(saved)))
        out.append((step, saved[step]))
    return out


def latest_checkpoint(name):
    """The newest saved checkpoint of ACT model `name` (outputs/act/<name>)."""
    return checkpoints(name)[0][1]


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
    getattr(arm, "wait", time.sleep)(1.0)     # the simulator's arm has its own clock
    return servo


def _timed_chunk(policy, frame, servo, task):
    """One policy call -> (steps, ms). Runs on the worker thread when prefetching."""
    while not _CALL_LOCK.acquire(timeout=0.1):   # short waits so Ctrl+C gets through
        pass
    try:
        t0 = time.perf_counter()
        steps = policy.chunk(frame, servo, task)
        return steps, 1000 * (time.perf_counter() - t0)
    finally:
        _CALL_LOCK.release()


def _result(future):
    """future.result(), but Ctrl+C still gets through while waiting (a plain
    blocking wait ignores it on Windows)."""
    while not wait([future], timeout=0.1).done:
        pass
    return future.result()


def run(policy, arm, cam, task, horizon, seconds, pipeline=True, rec=None, keep=None,
        done=None, show=None):
    """Run one instruction until the time is up or Ctrl+C.
    pipeline: ask for the next chunk while the current one plays, so the arm
    doesn't stop for the length of each policy call.
    rec: a Recorder; every step played is saved (frame, last command, command).
    keep: asked once the arm is home whether to save the episode (default: save).
    done: checked after every step; the run ends when it returns True.
    show: called as show(frame, task) with a camera frame every step (the live view)."""
    print(f'running "{task}" for {seconds:.0f} s (Ctrl+C stops it)')
    servo = go_home(arm)
    if rec:
        rec.start(task=task)
    try:
        _drive(policy, arm, cam, task, horizon, seconds, pipeline, rec, servo, done, show)
        go_home(arm)
        if rec:
            rec.stop(keep=keep() if keep else True)
    finally:
        if rec:
            rec.stop(keep=False)        # no-op once saved; drops a half episode on errors


def _drive(policy, arm, cam, task, horizon, seconds, pipeline, rec, servo, done=None,
           show=None):
    """The policy loop of run(); servo = the command the arm holds now."""
    dt = 1.0 / C.FPS
    t_end = time.time() + seconds
    fails = 0
    lat = 0.0                # how long the last policy call took, s
    rd = 0.0                 # how long the last fresh camera read took, s
    pending = t_req = None   # the next chunk, requested while this one plays
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while time.time() < t_end:
            prefetched = pending is not None
            try:
                if prefetched:
                    future, pending = pending, None
                    steps, ms = _result(future)
                    # drop the steps the model planned for while the call was out
                    skip = round((time.perf_counter() - t_req) / dt)
                    steps = steps[min(skip, max(len(steps) - 1, 0)):]
                else:
                    t0 = time.perf_counter()
                    frame = cam.read(fresh=True)
                    rd = time.perf_counter() - t0
                    steps, ms = _timed_chunk(policy, frame, servo, task)
                steps = steps[:horizon]
                if not steps:
                    raise RuntimeError("empty action chunk")
            except (TimeoutError, RuntimeError) as e:
                fails += 1
                print(f"policy call failed ({type(e).__name__}: {e})")
                if fails >= MAX_FAILS:
                    print(f"policy failed {MAX_FAILS} times, stopping this run")
                    break
                continue
            fails = 0
            lat = ms / 1000
            sent, limited = [], 0
            t_next = time.perf_counter()
            for i, cmd in enumerate(steps):
                cmd, ok = make_safe(cmd, servo)
                if rec or show:                           # what the camera sees at this step
                    seen = cam.read()
                    if rec:                               # ...then what was sent
                        rec.add(seen, servo, cmd)
                    if show:
                        show(seen, task)
                arm.send(cmd)
                servo = cmd
                sent.append(cmd)
                limited += not ok
                if pipeline and pending is None and (len(steps) - i) * dt <= lat + rd:
                    # read + call take about as long as what's left to play: send it
                    # now with a fresh frame + the command just sent (the read uses
                    # this step's wait; the skip counts from when the frame was taken)
                    t0 = time.perf_counter()
                    frame = cam.read(fresh=True)
                    t_req = time.perf_counter()
                    rd = t_req - t0
                    pending = pool.submit(_timed_chunk, policy, frame, servo, task)
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
                if done and done():
                    print("an object is in the cup")
                    return
            grip = f", grip {sent[0][4]:.0f}" if sent else ""
            print(f"policy call {ms:.0f} ms, {len(steps)} steps{grip}"
                  + (f", {limited} limited" if limited else "")
                  + (", prefetched" if prefetched else ""))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)   # never wait for a slow call


def snap_gripper(value):
    """Fully open or fully closed, whichever the value is nearer."""
    return min((C.GRIPPER_OPEN, C.GRIPPER_CLOSED), key=lambda g: abs(g - float(value)))


def make_safe(cmd, last):
    """(command to send, whether it is the one asked for). last is the
    command sent before, which passed this check. An arm move that breaks a
    joint limit or puts the gripper in the plate is cut short: the farthest
    point on the way from last to it that is safe. The gripper value is used
    as asked (opening or closing can't hit anything). NaN or a wrong number
    of values: hold still."""
    if len(cmd) != 5 or not all(math.isfinite(float(a)) for a in cmd):
        return list(last), False
    cmd = [float(np.clip(a, C.SERVO_MIN, C.SERVO_MAX)) for a in cmd[:4]] + [snap_gripper(cmd[4])]
    if K.servo_ok(cmd[:4]):
        return cmd, True
    lo, hi = 0.0, 1.0                                     # last is safe, cmd isn't
    for _ in range(10):
        mid = (lo + hi) / 2
        if K.servo_ok([a + mid * (b - a) for a, b in zip(last[:4], cmd[:4])]):
            lo = mid
        else:
            hi = mid
    return [a + lo * (b - a) for a, b in zip(last[:4], cmd[:4])] + [cmd[4]], False


def viewer(fullscreen=False):
    """show(frame, task) for run(): the camera picture with the instruction on
    top, in a window (full screen for a projector; F11 switches). pygame, not
    OpenCV: the ACT environment's OpenCV can't open windows."""
    import pygame
    pygame.init()
    pygame.display.set_caption("gr00t-arm camera")
    state = {"full": fullscreen, "screen": None, "fonts": {}}

    def set_screen():
        state["screen"] = (pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if state["full"]
                           else pygame.display.set_mode((960, 720), pygame.RESIZABLE))
    set_screen()

    def show(frame, task):
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                state["full"] = not state["full"]
                set_screen()
        screen = state["screen"]
        w, h = screen.get_size()
        s = min(w / frame.shape[1], h / frame.shape[0])
        pic = pygame.transform.smoothscale(
            pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))),
            (max(1, int(frame.shape[1] * s)), max(1, int(frame.shape[0] * s))))
        screen.fill((0, 0, 0))
        screen.blit(pic, ((w - pic.get_width()) // 2, (h - pic.get_height()) // 2))
        size = max(24, h // 18)
        font = state["fonts"].get(size) or state["fonts"].setdefault(
            size, pygame.font.SysFont("arial", size, bold=True))
        text = font.render(f'"{task}"', True, (255, 255, 255))
        bar = pygame.Surface((w, text.get_height() + 20), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        screen.blit(bar, (0, 0))
        screen.blit(text, ((w - text.get_width()) // 2, 10))
        pygame.display.flip()
    return show


def free_gpu_memory():
    """Let the next checkpoint load into the GPU memory the last one used."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
    ap.add_argument("--model", default="pick_ball",
                    help="ACT model name (outputs/act/<name>); its newest checkpoint is used")
    ap.add_argument("--checkpoint", help="ACT pretrained_model folder (instead of --model)")
    ap.add_argument("--step", metavar="N",
                    help="ACT: which saved checkpoint of --model: 60000 or 60k; several "
                         "(40k,60k,80k) or all are scored one after another with --mujoco "
                         "--trials, on the same layouts, and compared in a table")
    ap.add_argument("--task", help="run this one instruction and exit (no prompt)")
    ap.add_argument("--horizon", type=positive_int, default=8,
                    help="actions played per policy call (GR00T predicts 16)")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="how long each instruction runs")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="wait for each policy call instead of asking for the next "
                         "chunk while the current one plays")
    ap.add_argument("--record", nargs="?", const="", metavar="DATASET",
                    help="save each run as an episode in data/raw/DATASET (default "
                         "pick_ball_policy, or pick_ball_mujoco with --mujoco)")
    ap.add_argument("--trials", type=positive_int, metavar="N",
                    help="with --mujoco: N runs on random layouts, ball and block "
                         "taking turns, then the score")
    ap.add_argument("--offset", type=float, default=0.0, metavar="MM",
                    help="with --mujoco: put each object up to MM away from its spot")
    ap.add_argument("--anywhere", action="store_true",
                    help="with --mujoco: put the objects anywhere the arm can pick "
                         "and the camera can see")
    ap.add_argument("--view", action="store_true",
                    help="show the camera live, with the instruction on top")
    ap.add_argument("--fullscreen", action="store_true",
                    help="--view, full screen (for a projector; F11 switches, Ctrl+C stops)")
    args = ap.parse_args()
    if (args.trials or args.offset or args.anywhere) and not args.mujoco:
        raise SystemExit("--trials, --offset and --anywhere are for the simulator (--mujoco)")
    if args.offset and args.anywhere:
        raise SystemExit("use --offset or --anywhere, not both")
    if args.step and (args.backend != "act" or args.checkpoint):
        raise SystemExit("--step picks a checkpoint of an ACT --model (not with --checkpoint)")
    chosen = ([(None, Path(args.checkpoint))] if args.checkpoint
              else checkpoints(args.model, args.step) if args.backend == "act" else [None])
    if len(chosen) > 1 and not args.trials:
        raise SystemExit("several checkpoints are compared with --mujoco --trials N")

    task = None
    if args.task:
        task = instruction(args.task)
        if task is None:
            raise SystemExit(f"--task must name the ball or the block: {args.task!r}")
    rec = None
    if args.record is not None:
        dataset = args.record or ("pick_ball_mujoco" if args.mujoco else "pick_ball_policy")
        rec = Recorder(dataset, fps=C.FPS, source="policy", sim=args.mujoco)
    arm, cam = open_hardware(args)
    world = getattr(arm, "world", None)
    show = viewer(args.fullscreen) if (args.view or args.fullscreen) else None

    def load(entry):
        if args.backend == "gr00t":
            return Gr00tBackend(args.host, args.server_port)
        step, folder = entry
        if step is not None:
            print(f"\n=== {args.model}, checkpoint {step} ===")
        return ActBackend(folder)

    def one(policy, task):
        """One instruction. In the simulator: lay the objects out, keep only
        clean picks, and return the result."""
        opts = dict(pipeline=not args.no_pipeline, rec=rec, show=show)
        if not world:
            run(policy, arm, cam, task, args.horizon, args.seconds,
                keep=ask_keep if rec else None, **opts)
            return None
        target = target_of(task)
        _, xy, other_xy = world.random_layout(args.offset, target=target, anywhere=args.anywhere)
        world.place_for(target, xy, other_xy)
        run(policy, arm, cam, task, args.horizon, args.seconds,
            keep=lambda: world.success(target),
            done=lambda: world.in_cup("ball") or world.in_cup("block"), **opts)
        result = world.result(target)
        print(f"{target} at ({xy[0]:.0f}, {xy[1]:.0f}): {result}")
        return result

    def trials(policy):
        """--trials N runs, ball and block taking turns. Trial i always gets
        the same layout and look, so runs (and checkpoints) compare fairly."""
        score = {name: [] for name in sorted(TASKS)}
        for i in range(args.trials):
            random.seed(i)
            world.rng.seed(i)
            target = sorted(TASKS)[i % len(TASKS)]
            print(f"\ntrial {i + 1}/{args.trials}")
            score[target].append(one(policy, TASKS[target]))
        for target, results in score.items():
            others = ", ".join(f"{results.count(r)} {r}" for r in sorted(set(results)) if r != "ok")
            print(f"{target}: {results.count('ok')}/{len(results)} ok"
                  + (f" ({others})" if others else ""))
        return score

    go_home(arm)
    try:
        if args.trials:
            table = []
            for entry in chosen:
                policy = load(entry)
                score = trials(policy)
                table.append((entry and entry[0], score))
                del policy
                free_gpu_memory()
            if len(table) > 1:
                print(f"\n{args.model}: ok out of {args.trials} trials per checkpoint")
                for step, score in table:
                    oks = {t: r.count("ok") for t, r in score.items()}
                    print(f"  {step:>7}: " + ", ".join(f"{t} {n}/{len(score[t])}" for t, n in oks.items())
                          + f"  -> {sum(oks.values())}/{args.trials}")
            return
        policy = load(chosen[0])
        if task:
            one(policy, task)
            return
        print(f'type an instruction, e.g. "{TASKS["ball"]}" or "{TASKS["block"]}" '
              f"(q to quit)")
        while (task := ask()) is not None:
            one(policy, task)
    finally:
        go_home(arm)
        arm.close()
        cam.close()

if __name__ == "__main__":
    main()
