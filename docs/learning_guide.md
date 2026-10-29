# Learning guide: what this project teaches

Five topics, each tied to the files in this repo where you can see it working.
Suggested order: **2 → 3 → 1 → 4**. Start with the simplest layer (the board), then the
script that runs the arm on its own, then GR00T, which sits on top of both, and finish
with the simulation, which copies all of it into a virtual world.
IK is at the end as an optional bonus: you don't need it to understand the rest.

| # | Topic | Main files |
|---|---|---|
| 1 | GR00T | `software/gr00t/`, `software/export_dataset.py`, `software/policy_client.py`, `software/run_policy.py` |
| 2 | How the computer talks to the board | `firmware/arm/arm.ino`, `software/arm_io.py`, `docs/computer_to_nano.md` |
| 3 | The automatic pick demo | `software/autodemo.py`, `software/recorder.py`, `software/test_autodemo.py` |
| 4 | Simulation (MuJoCo) | `software/sim.py`, `software/test_sim.py`, `cad/sim_meshes.py` |
| Bonus | Inverse kinematics (IK) | `software/kinematics.py`, `software/config.py`, `software/test_kinematics.py` |

---

## 1. GR00T

### What it is
GR00T N1.7 is NVIDIA's **robot foundation model**: one neural network (3 billion
parameters) that takes in

- **what the robot sees** (camera image),
- **what it's told** (a sentence, e.g. "pick up the block"),
- **where its joints are** (the 5 servo angles),

and puts out **the next motions** (servo angles for the next ~half second).

"Foundation model" means the same idea as ChatGPT: it's pretrained on a huge pile of data
(real robots, simulated robots, videos of humans), then **fine-tuned** on a small set of
examples for one robot and one job. Our arm is a robot it has never seen, and 100–200
demos are the fine-tuning data.

### How it works inside
It has two parts, nicknamed "slow thinking" and "fast thinking":

1. **System 2, the vision-language model** (Cosmos-Reason2-2B). It looks at the
   image and reads the sentence and works out *what's going on*: there's a pink ball
   here, a blue ball there, and I've been asked for the blue one.
2. **System 1, the action head** (a diffusion transformer). It takes System 2's
   understanding plus the current joint angles and works out *how the joints should move*.

The action head doesn't output one move. It outputs an **action chunk**: the next 16
commands at once. It makes them by **denoising**: it starts from random numbers and cleans them
up over a few steps into a smooth motion. That's the same trick image generators use, applied
to joint angles instead of pixels.

### What it does in this project
GR00T **replaces the hand-written script**. `autodemo.py` knows exactly where the ball is
because you put it on a marked spot. GR00T has to *see* where it is and *read* which object
you asked for. The choice is there on purpose: the only way to know which one to pick is
the sentence, so it shows the language input really being used. There are five ball colours
(red, green, pink, yellow, blue) plus the block, and half the ball demos have a second ball
of another colour next to the target. Without that, every scene would be "a ball and a
block", and the action head could learn "go to the round thing" while ignoring the colour
word; with two balls side by side, only the colour word says which one to take.

### How it's wired in, step by step
1. **Collect demos.** `autodemo.py` (or `teleop.py`) moves the arm while `recorder.py` saves,
   30 times a second: the camera frame, the angles last sent (the *state*), the angles sent
   now (the *action*), plus the sentence for that episode. These go in `data/raw/pick_ball/`.
   Later, the trained model's own good runs can be added the same way
   (`run_policy.py --record`, guide section "Learning from its own runs").
2. **Convert to GR00T's format.** `export_dataset.py` writes the LeRobot v2.1 format GR00T
   reads: tables of numbers (parquet), the video (mp4), and `meta/modality.json`, which says
   which numbers are the arm and which is the gripper.
3. **Describe the robot to GR00T.** `software/gr00t/gr00t_arm_config.py` registers our arm in
   GR00T's slot for new robots (`NEW_EMBODIMENT`): 1 camera called `front`, 4 arm joints
   (`single_arm`), 1 gripper, predict 16 steps. Arm actions are *relative* (changes from the
   current angles, which generalises better), and the gripper is *absolute* (open or closed).
4. **Fine-tune on a cloud GPU** (it needs ≥ 40 GB of GPU memory, and the laptop has 8 GB).
   `setup_cloud.sh` installs GR00T, `finetune.sh` trains about 3000 steps (1–2 h), and
   `serve.sh` runs the trained model as a server on port 5555.
5. **Drive the arm with it.** On the laptop, `run_policy.py gr00t` loops: photo + angles +
   your sentence → `policy_client.py` sends them to the server → 16 steps come back → play
   the first 8 on the arm (every command passes the safety check first) → repeat.

**ACT** (`software/act/`) is the small, free version of the same pipeline. It's a much
smaller model that trains on the laptop. It has no language input, so the target goes in as
an extra number (0–5: block, blue, green, pink, red, yellow ball, in alphabetical order). Use it to check your demos are good before paying for
a GPU.

### Try it / read more
- Guide Part I (`docs/guide.pdf`, sections 1–5): diagrams and a talk script.
- Read `software/gr00t/gr00t_arm_config.py` first. It's about 40 lines and shows exactly
  what GR00T is told about this arm.
- Questions to test yourself: why predict 16 steps but only play 8? Why are arm actions
  relative but the gripper absolute? What would break if the camera moved between recording
  and running?

---

## 2. How the computer talks to the board

This one already has its own write-up: **`docs/computer_to_nano.md`** (also guide
section 10). In short:

- The laptop sends the Nano one line of text over USB, 30 times a second: five servo
  angles, e.g. `90.0,120.5,45.0,90.0,125.0`. Speed is 115200 baud.
- The Nano (`firmware/arm/arm.ino`) splits the line, clamps each angle to 0–180°, and
  moves each servo toward it at most 180°/s, so the arm never slams. It turns each angle
  into a pulse width (500–2400 µs) on pins D3, D5, D6, D9, D10.
- The Nano knows nothing about IK or GR00T. All the thinking happens on the laptop.
- Servo power comes from the bench supply, not USB, with a shared ground.

### Try it
Without Python: Arduino IDE → Serial Monitor at 115200, line ending "Newline", type
`90,90,90,90,125`, then `?` to see where the Nano thinks the servos are.
In Python, `arm_io.Arm.send()` is the whole sending side, in 4 lines.

---

## 3. The automatic pick demo (`autodemo.py`)

This is the "runs on its own" part: it does a full pick-and-place with no one driving,
and it can record itself as training data. It uses IK, the maths that turns "put the
gripper here" into joint angles, but you can treat that as a black box (the bonus section
explains it if you're curious).

### How to run it
From `software/` on the laptop:

| What | Command |
|---|---|
| Try it with no hardware | `uv run autodemo.py --sim --target block --spot C1 --other-spot A3` |
| One real pick (red ball on B2) | `uv run autodemo.py --spot B2 --camera 1` |
| One real pick of the block | `uv run autodemo.py --target block --spot C1 --other-spot A3 --camera 1` |
| Record 250 demos | `uv run autodemo.py --record 250 --camera 1` |
| Record without pressing Enter (sim testing only) | `uv run autodemo.py --sim --record 5 --yes` |

When recording, each episode prints something like
`pink ball on C1, blue ball on A3 -> "pick up the pink ball". Place them (nothing else on the plate), then press Enter`.
You place the two objects on those marked spots, press Enter, and the arm does the rest
(about 9 s per pick). Then it asks `keep it? [Y/n]`: Enter keeps the episode, `n` throws it
away (missed grip, dropped or knocked object), and `--record 250` keeps going until 250 are
kept. Stop any time with Ctrl+C: the half-finished episode is thrown away, and the next run
carries on numbering from where you stopped.

Its automatic tests run with `uv run pytest -q` (103 tests in the whole project,
including every spot × both objects, checks that the arm never jumps more than 6° between
frames, and scripted picks inside the simulation).

### How it works
1. **Spots.** The baseplate has a 3×3 grid of marked spots, A1–C3 (`SPOTS`), plus the
   cup (`CUP`). Positions are in mm, matching the printed plate.
2. **Layout** (`layout()`, `pairs()`). For a recording, it randomly picks which object
   to pick (about half and half) and two spots ≥ 60 mm apart, so the gripper can't hit the
   other object.
3. **Plan** (`plan()`). A list of 8 *waypoints*: home → above the object → down to grip
   height → close → lift → over the cup → open → home. Each one has a time.
   `best_pitch()` picks the most downward gripper tilt that IK can reach at each point.
4. **Trajectory** (`trajectory()`). Between waypoints it makes 30 in-between poses per
   second along a straight line, with ease-in/ease-out so it starts and stops gently. Each
   pose goes through IK. If one can't be reached at that exact tilt, `solve_near()` tilts
   the gripper a degree or two until it can, so the arm never jumps.
5. **Run** (`run()`). The whole trajectory is worked out *before* the arm moves, so a bad
   plan fails safely. Then each command is sent to the Nano, and if recording, the camera
   frame + angles are saved at the same moment.
6. **Save.** After you answer `keep it?`, `recorder.py` writes the frames, a `steps.csv` of
   angles and a `meta.json` with the sentence ("pick up the block"). That's one training
   example for GR00T or ACT. (`n` deletes the folder instead.)

The key idea: **the script is the teacher**. It knows where things are because you told
it; it does the pick perfectly 150 times while the camera watches; then GR00T learns to do
the same from the *image and the sentence alone*.

### Try it / questions
- Run the sim command above and read the printed output next to `plan()`.
- What happens if you set `SAFE_Z` too low? (Run the tests.)
- Why does the recorder save both "state" (last command) and "action" (this command)?

---

## 4. Simulation (`sim.py`)

A virtual copy of the arm, the plate, the cup, the ball, the block and the webcam, in the
free physics engine **MuJoCo**. Every script that takes `--mujoco` runs in it instead of on
the real arm. It's for after the real demos exist: it makes extra demos with the objects
anywhere on the plate, and it tests trained models automatically.

### What "training in simulation" means
Training doesn't change: ACT and GR00T still learn by **copying recorded demos** (imitation
learning), on the laptop or the cloud GPU. The simulator *produces* recordings (the same
format as the real arm's) and *checks* results (it knows where the ball ended up). GR00T's
own pretraining used simulated robots the same way, mixed with real robot data. Your real
demos stay in the mix: they're what teaches the model what the real camera and servos are
like.

### How it works
1. **The world.** `model_xml()` writes a description MuJoCo reads: bodies (the arm's parts,
   the ball, the block), joints (base, shoulder, elbow, wrist, two fingers), shapes for
   collisions, a light and a camera. MuJoCo then steps the physics 500 times per simulated
   second: gravity, joint motors, contacts and friction.
2. **The arm's shape** is the author's 3D model. `cad/sim_meshes.py` cut it into the moving
   pieces once and saved them in `software/sim_assets/`. The joints sit where `config.py`
   says (54 mm shoulder height, two 60 mm links, tool point 86.5 mm along the gripper and
   24.75 mm below the wrist), so the simulated gripper ends up where `kinematics.fk` says.
   `test_sim.py` checks this to within a few mm.
3. **Same commands as the real arm.** Scripts call `arm.send([5 servo angles])` exactly as
   for the Nano. `controls()` turns them into joint angles with `kinematics.from_servo`, which
   uses *your* `calibration.json`. That's why a simulated recording has the same numbers as a
   real one, and why both can be trained on together.
4. **The servos.** `_servo_tick()` does what the firmware does every 10 ms: moves each
   servo's output toward its target by at most 180°/s × 10 ms. Then a MuJoCo *position
   actuator* (a motor that pushes toward a target angle) follows it, with at most 0.18 N·m,
   an SG90's stall torque. Limited strength means the arm sags a few mm under gravity, as the
   real one does.
5. **The gripper.** Two hinged fingers on their pins; an *equality constraint* keeps them
   mirrored, like the meshed gears. `finger_angle()` maps your calibrated open/closed servo
   angles to 30°/6° per finger; 6° is narrower than the block, so the fingers stop on the
   object and squeeze with limited force.
6. **The camera.** `MujocoCamera.read()` renders 640×480 RGB from where the webcam mount
   holds the webcam, the same kind of picture `arm_io.Camera.read()` returns.
7. **Variety.** With `randomize`, every new layout changes the light, the colours and the
   camera pose a little (`_randomize()`), so a model can't depend on one exact look.
8. **Layouts.** `random_layout()` places the objects round the spots (`--offset`) or
   anywhere (`--anywhere`), then keeps a layout only if both objects are clear of the drum,
   cup and mast (`clear_of_fixed_parts()`), inside the camera's picture (`in_view()`), and
   autodemo's scripted pick works there, tried first in a hidden second world
   (`_pick_works()`). So every layout is one the real arm could do.
9. **Results.** `result()` looks at where the objects ended up: `ok`, `knocked`,
   `wrong object` or `missed`. That replaces the `keep it? [Y/n]` question you answer on the
   real arm.
10. **Two clocks.** In *real time* a background thread steps the physics to keep up with the
   wall clock (teleop, run_policy: a slow model call delays the arm, as on the real arm).
   In *lockstep* every `send()` moves the world 1/30 s (autodemo: as fast as the computer
   can draw).

### How to run it
From `software/`:

| What | Command |
|---|---|
| See what the simulated webcam sees | `uv run sim.py --view` (writes `data/sim_view.png`) |
| Compare with a real webcam frame | `uv run sim.py --compare <photo.jpg>` |
| Look around in 3D | `uv run sim.py --viewer` |
| Drive it with the keyboard | `uv run teleop.py --mujoco` |
| One scripted pick | `uv run autodemo.py --mujoco --target block --spot C1 --other-spot A3` |
| Record 300 simulated demos, objects anywhere | `uv run autodemo.py --mujoco --record 300 --anywhere` |
| Score a trained ACT (from `software/act`) | `uv run ../run_policy.py act --mujoco --trials 20 --anywhere --checkpoint ...` |
| Train on real + simulated demos (from `software/act`) | `uv run export_act.py --with pick_ball_mujoco --out pick_ball_mix` |

### What it gets wrong
Drawn pictures aren't photos (no shadows, no blue servo bodies, no cables, simpler light); SG90 dead band
and gear slop aren't modelled; the foam ball is a firm sphere. So a simulator score compares
models, and the real arm has the final say.

### Try it / questions
- Run `sim.py --viewer`, then read `model_xml()`: find the shoulder joint and its axis.
- Why does the simulator use your `calibration.json` instead of the default angles?
- What would happen to a model trained only on simulated demos when it sees the real camera?
- Why are only `ok` runs kept, never `knocked` ones?

---

## Bonus: inverse kinematics (IK)

*Optional. Nice to know for robotics in general, but nothing else in this guide depends on it.*

### The problem
- **Forward kinematics (FK):** *given the joint angles, where is the gripper?* Easy:
  add up the arm segments with sines and cosines.
- **Inverse kinematics (IK):** *given where I want the gripper, what joint angles
  get it there?* Harder, and it's what you actually need: you know where the ball
  is, not what angles reach it.

Our arm has 4 joints (base, shoulder, elbow, wrist) and we ask for 4 things: the
gripper's x, y, z, and its **pitch** (tilt; −90° = pointing straight down). Four
unknowns and four targets, so there's a clean answer.

### How `kinematics.ik()` solves it
Read `software/kinematics.py` alongside this. Measurements are in `config.py`:
shoulder 54 mm above the plate, upper arm 60 mm, forearm 60 mm, tool point 86.5 mm past
the wrist and 24.75 mm below it.

1. **Base angle (looking from above).** `atan2(y, x)` points the arm at the target. There's a
   small correction because the fingers sit 1.8 mm off to the side (`LATERAL`).
2. **Work in the arm's own flat plane.** After the base turns, the rest is a 2D problem:
   `r` = distance out from the base, `z` = height.
3. **Find the wrist.** You know where the tool point must be and how it's tilted (pitch), so
   step *backwards* along the gripper by its length to find where the wrist axis must be.
4. **Two-link triangle.** Shoulder → elbow → wrist is a triangle with two known sides (60, 60)
   and a known third side (shoulder to wrist). The **law of cosines** gives the elbow angle:
   `d = (rw² + zw² − L1² − L2²) / (2·L1·L2)`, `elbow = −acos(d)`. If `|d| > 1` the
   point is too far away (or too close): unreachable, return `None`. The minus sign picks
   "elbow up" out of the two possible solutions.
5. **Shoulder angle** = angle to the wrist, minus the angle the forearm adds:
   `atan2(zw, rw) − atan2(L2·sin(elbow), L1 + L2·cos(elbow))`.
6. **Wrist angle** is whatever is left to make the total tilt equal the pitch:
   `wrist = pitch − shoulder − elbow`.

This "split off the base, find the wrist, solve a triangle" recipe is the standard way to
do IK for most hobby and industrial arms, so it carries over to other robots.

### From joint angles to servo angles
Maths angles aren't what the servos take. `to_servo()` converts with
`servo = OFFSET + SIGN × joint_angle`, where OFFSET and SIGN come from how each horn was
mounted (that's what `calibrate.py` measures). It also refuses anything outside the
**soft limits** (where the arm would hit itself, found by the collision sweep in
`cad/assembly.py`). `solve()` = `ik()` + "doesn't hit the plate" + `to_servo()`, which is
the one function the rest of the code calls.

### Try it
```
cd software
uv run python -c "import math, kinematics as K; print(K.solve(110, 40, 20, math.radians(-45)))"
```
That prints the 4 servo angles (about `111, 104, 89, 122`) to put the gripper at spot B2,
20 mm up, tilted 45° down. Now try `-90` (straight down) at the same spot: you get
`None`. The arm is too short to reach that far out while pointing straight down. That's
why `autodemo.best_pitch()` searches for the most downward tilt that works (−65° here).
A far point like `(300, 0, 20, …)` is `None` at any tilt.
- `test_kinematics.py` checks IK with FK: solve for angles, then run FK on them, and you
  should land back on the target (`test_ik_then_fk_round_trips`). That's a useful habit for
  any IK you write.
- Questions to test yourself: what's the other solution that "elbow down" would give? Why
  does `best_pitch()` in autodemo try angles from −90° upward, and what does it give up?

---

## Where to go next
- `docs/guide.pdf`: the full build + GR00T guide (Part I is the GR00T explainer).
- `README.md`: milestones, what's done and what's left.
- Ask Claude to walk through any file line by line, or to quiz you on a section.
