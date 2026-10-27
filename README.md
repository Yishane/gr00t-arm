# gr00t-arm

A small 3D-printed desk arm (SG90 servos, Arduino Nano) used to show how
NVIDIA GR00T learns robot skills. A ball and a block sit on the plate. You
type **"pick up the ball"** or **"pick up the block"**, and the arm picks that
object and drops it in the cup. You record demos of both (scripted or by
keyboard), export them as a GR00T/LeRobot dataset with the instruction saved
per episode, train ACT on the laptop or fine-tune GR00T N1.7 on a cloud GPU,
and run the learned policy back on the arm.

> **Status (2026-10-22):** the arm is now an **open-source arm** (mobiobi's
> *Easy Robot Arm*). The software, firmware and GR00T pipeline are done and
> verified, and they're matched to the new arm's measurements and checked against
> its 3D model. **New demo (2026-10-23):** ball + block on the plate, and the
> typed instruction picks which one goes in the cup (milestone 2b). What's left
> is building, recording and training. See [Milestones](#milestones).

---

## The arm

> **"Easy Robot Arm (SG90 servos & now steppers too)"** by
> [**mobiobi**](https://www.printables.com/@mobiobi), from
> [Printables model 97370](https://www.printables.com/model/97370-easy-robot-arm-sg90-servos-now-steppers-too-update),
> licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/).
> The files are **unmodified**. The arm design is mobiobi's work, not ours.

The files are in [`cad/stl_opensource/easy_robot_arm/`](cad/stl_opensource/easy_robot_arm/),
with full credit and license notes in
[`ATTRIBUTION.md`](cad/stl_opensource/easy_robot_arm/ATTRIBUTION.md).
The license means: always credit mobiobi, no commercial use, and if the files
are ever modified and shared, the changed versions must keep the same license.

Why this one: it is all SG90, it needs only the screws that come with the
SG90s, and it has the same joints as our design (base, shoulder, elbow, wrist,
gripper), so the code stays the same.

### Decisions
- **5 servos on the arm**: base, shoulder, elbow, wrist, gripper. The arm's
  design also has a wrist-**roll** servo inside the screw-cap disc. We leave that
  one **out** (the 6th servo just sits on the desk, unplugged), so the firmware,
  dataset (5 numbers per step) and GR00T config don't change.
- **Both arm links are the 60 mm version** (`Arm60mm.stl` ×2). This is the
  author's advice, and it keeps the load on the shoulder servo low.
- **No stepper motor** for now. The author swapped in a 28BYJ-48 at the shoulder,
  but it has less torque than an SG90, needs a driver board and homing, and
  would need firmware changes. If the shoulder sags or jitters, first swap that
  one servo for an **MG90S** (same size, metal gears, no code changes).
- Wiring stays the same: Nano pins **D3, D5, D6, D9, D10** = base, shoulder,
  elbow, wrist, gripper. 5.5 V / 3 A bench supply for the servos.

### What to print (from `cad/stl_opensource/easy_robot_arm/`)

| File | Qty | Role |
|---|---|---|
| `Base.stl` | 1 | threaded drum; the base (yaw) servo sits inside |
| `BaseCap.stl` | 1 | screw-on ring that holds the rotator on the base |
| `BaseRotator.stl` | 1 | turntable on the base servo; holds the shoulder joint |
| `Arm60mm.stl` | 2 | upper arm and forearm (each carries the next servo) |
| `offAxis_gTrunkUndr.stl` | 1 | wrist piece; hangs the gripper below the wrist axis (the version in the author's assembly) |
| `gripperMount.stl` | 1 | holds the gripper servo; clamps onto the trunk disc |
| `gripperCap.stl` | 1 | screw-on ring that clamps gripperMount to the trunk |
| `gripperOne.stl`, `gripper_Two.stl` | 1 each | geared fingers (they open sideways) |
| `gripperPinOne.stl`, `gripperPinTwo.stl` | 1 each | finger pivot pins |

Not needed: `BaseLeg` (the drum stands in a ring on our baseplate instead, and
the legs would hang off its edge), `Arm75mm`, `Arm90mm` (longer links), `gripperTrunk`,
`offAxis_gTrunk` (alternative wrist pieces), `twistMount`, and the two
`28BYJ-48_*` stepper parts. `E-Z_RoboArm_v23.step` is the author's assembled
model; we use it for measurements.

Author's print settings: PLA, 0.2 mm layers, supports on, low infill (10 %
gyroid). Keep the gripper parts light so the servos don't struggle.
Without the roll servo, `gripperCap` just clamps `gripperMount` to the trunk.
If the gripper still turns, put a drop of superglue between the two discs.

### Measured from the author's model (mm)
These are in `software/config.py`. `cad/assembly.py` builds the arm from the
author's STEP file to check them.
- baseplate top to shoulder axis: 54.05 (drum standing on the plate, no legs)
- shoulder axis to elbow axis: 60 with Arm60 (the author's model uses Arm90 = 90)
- elbow axis to wrist axis: 60
- wrist axis to the point between the finger pads: 86.5 forward, 24.75 below
  (the "off-axis" trunk hangs the gripper down); 1.8 mm to the side
- the fingers turn on vertical pins 15.6 apart, so the jaws open **left/right**
  when the gripper is level (our old gripper opened up/down)
- joint limits from the collision sweep: elbow −125…0°, wrist −85…90° (the sweep
  allows 125°, but the wrist servo runs out at 90°), and the
  gripper can't fold back into the upper arm (elbow + wrist ≥ −175°)

---

## Repo layout

- `cad/stl_opensource/easy_robot_arm/` — **the arm we're building** (open source, see above)
- `cad/stl/` — our original self-designed parts. Arm parts `00`–`06` are now
  replaced by the open-source arm. The work-cell parts are still used: `07` baseplate,
  `08` camera mast, `09` phone cradle, `10` pin, `11` cup, `12` ball half,
  `13` block (the second object).
- `cad/arm.py` — CadQuery source for the work-cell parts (and the old arm parts)
- `cad/assembly.py` — builds the Easy Robot Arm from the author's STEP file;
  collision sweep (`--check`), reach check of every ball/block spot and the cup,
  plus every two-object layout (`--cell`), renders and guide figures (`--docs`)
- `firmware/arm/arm.ino` — serial servo firmware; `firmware/upload.ps1 COMx` uploads it
- `software/` — Python: `config.py` (geometry + calibration), `kinematics.py`
  (+ tests), `calibrate.py`, `teleop.py`, `autodemo.py`, `recorder.py`,
  `export_dataset.py` (GR00T format), `run_policy.py`, `policy_client.py`
- `software/act/` — ACT export/train (own Python env, runs on the RTX 5050)
- `software/gr00t/` — GR00T modality config + cloud scripts (`setup_cloud.sh`,
  `finetune.sh`, `serve.sh`)
- `docs/guide.tex` → `docs/guide.pdf` — the build + GR00T guide
- `docs/learning_guide.md` — what to learn from this project (GR00T, IK, serial, autodemo)
- `docs/gr00t_dry_run.md` — run the GR00T cloud pipeline once on a fake dataset (`software/gr00t/dryrun/`)
- `docs/computer_to_nano.md` — how the laptop (and GR00T) talks to the Nano over USB
- `context.md` — running notes for Claude, where work left off

---

## Milestones

Legend: ✅ done · 🔄 in progress · ⬜ to do. "Claude" = done by Claude Code,
"You" = needs your hands, accounts or hardware.

### ✅ 0. Already done (before the switch)
- ✅ Firmware for the Nano (5 servos, serial protocol), compiles; upload script.
- ✅ Kinematics + 12 passing tests, including a floor-clearance check.
- ✅ Teleop (keyboard; gamepad optional), scripted auto-demos that also record,
  recorder, calibration tool.
- ✅ Dataset export in GR00T's LeRobot v2.1 format. **Verified** by loading it
  with GR00T's own loader.
- ✅ ACT pipeline (export → train → run) **verified** on the laptop GPU
  (~2.3 steps/s, ~23 ms per action).
- ✅ GR00T client **verified** against NVIDIA's real PolicyServer; cloud
  fine-tune/serve scripts written (not yet run, since that needs a cloud GPU).
- ✅ Work cell: baseplate, camera mast (phone cradle; webcam mount to come), cup, ball.
- ✅ LaTeX guide (13 pages): GR00T explainer, talk script, demo plan, build steps.
- ✅ Our own arm CAD (now replaced by the open-source arm, files kept).

### ✅ 1a. Pick and download an open-source arm
- ✅ Compared three Printables SG90 arms and picked mobiobi's Easy Robot Arm.
- ✅ Downloaded all files into `cad/stl_opensource/easy_robot_arm/`.
- ✅ Read the joint geometry from the author's STEP model (numbers above).

### ✅ 1b. Match the software to the new arm (Claude)
- ✅ Measured from the STEP + STLs (numbers above); shoulder axis confirmed from
  the SG90 shaft hole in `BaseRotator`, so it holds without the author's stepper.
- ✅ `software/config.py` updated (geometry, gripper outline, joint limits);
  `kinematics.py` also blocks the gripper folding into the upper arm.
- ✅ `software/test_kinematics.py` updated: 13 tests pass.
- ✅ The 40 mm ball should fit: each finger opens about 30°, and the ball stays
  clear of the gripper mount (checked on the model's dimensions; confirm on the real arm).
- ✅ Reach: all 9 ball spots, the lift above each, and the cup drop pass with no
  collisions. The cup moved to x = 20 (at x = 0 the base would need just over 90°).
  `autodemo.py` plans every spot cleanly in simulation.
- ✅ Baseplate: the old base pegs are replaced by a ring the 75 mm drum stands in
  (tape or hot-glue it so it can't turn). `BaseLeg` isn't printed.

### ✅ 2. Update the guide and checks (Claude)
- ✅ `docs/guide.tex` + PDF: credit on the title page and in the references, new
  parts and print list, assembly steps for this arm, "no roll servo" note, new
  figures, tool-point check after calibration.
- ✅ `cad/assembly.py` builds the real arm from the author's STEP file (voxel
  collision checks, so a sweep takes seconds).
- ✅ `context.md` up to date.

### ✅ 2b. New demo: ball + block, pick the one you name (Claude)
- ✅ The task: a ball and a block are both on the plate. You type
  **"pick up the ball"** or **"pick up the block"**, and the arm picks that
  object and drops it in the cup. The words are the only way to know which
  object to pick, so this shows GR00T using the language input.
- ✅ Block: `cad/stl/13_block.stl`, a 25 mm cube with chamfered edges. Print it
  in a colour that stands out from the ball and the plate (our scheme: black
  arm, light grey plate, orange ball, blue block). Set it with a face pointing at the arm's base, since the
  fingers close sideways.
- ✅ `autodemo.py --record N`: each episode puts both objects on random spots,
  picks the ball or the block (about 50/50), and saves the instruction in
  that episode's `meta.json`. It tells you where to put each object and which
  one it will pick. Single run: `autodemo.py --target block --spot C1 --other-spot A3`.
- ✅ Grip heights: the ball at its centre (z 20), the block at z 18 (lower, and
  the fingertips hit the plate or the gripper hits the base drum).
- ✅ Layouts: the two spots are ≥ 60 mm apart and C3 + A2 is left out, so the
  gripper never touches the other object (`cad/assembly.py --cell` checks
  all 24 spot pairs × 2 targets: PASS). The middle spot B2 isn't used for
  recording (it's too close to every other spot).
- ✅ `run_policy.py` asks `instruction>` in a loop: type the sentence (or just
  `ball` / `block`). It runs for 20 s (or until Ctrl+C), goes home, asks
  again; `q` quits. `--task "pick up the block"` runs one without asking.
- ✅ `teleop.py`: **T** switches the instruction saved with the next recording.
- ✅ GR00T gets the instruction as words (export already stores one task per
  episode; `tasks.jsonl` now has both). ACT has no language input, so
  `act/export_act.py` adds the target as a 6th state number (0 ball, 1
  block) and `run_policy.py act` adds the same number from what you type.
- ✅ Checked in simulation: recorded 4 two-object episodes, exported for GR00T
  (2 tasks) and ACT, trained ACT 20 steps, and ran the `run_policy.py act`
  loop. 40 tests pass. Guide updated (print list, calibration, recording, demo
  plan, talk script).

### ✅ 2c. Code-review fixes (Claude, 2026-10-23)
- ✅ When GR00T or ACT drives, every command is checked first (joint limits,
  gripper folding into the arm, gripper into the plate). Unsafe ones aren't
  sent; the arm holds still and `run_policy.py` prints `N blocked`.
- ✅ A GR00T timeout, server error or empty answer is retried. After 3 in a
  row the run stops and asks for the next instruction, instead of quitting.
- ✅ The policy gets a fresh camera frame each time (old buffered frames are dropped),
  and the camera opens on Linux/macOS too.
- ✅ `autodemo.py` no longer jumps at spot A1. It tilts the gripper a little
  instead (biggest move per frame went from 51.6° to under 4°). It plans the
  whole pick before moving, and Ctrl+C while recording throws away that half episode.
- ✅ `calibrate.py`: `flip` keeps an earlier `trim`. The tests ignore your
  `calibration.json`, so they still pass after you calibrate.
- ✅ `finetune.sh` checks the newest checkpoint; `upload.ps1` says when the
  *compile* failed (not run here: Windows-only).
- ✅ 61 tests pass. Plan: `docs/superpowers/plans/2026-10-23-code-review-fixes.md`.

### ⬜ 3. Print (You)
- Print the parts in the table above on the Bambu (PLA, supports on). No `BaseLeg`.
- Print the work cell from `cad/stl/`: `07_baseplate` in light grey (reprint it if you printed
  the old one: it now has the ring for the drum), `08_camera_mast`,
  `11_cup` (black), **`13_block`** (blue)
  (`12_ball_half` ×2 only if you have no ball). The camera is a **USB webcam** (monitor
  clip, 90° view); its mount for the mast gets designed once the webcam arrives and is
  measured. `09_phone_cradle` + `10_pin` are only for using a phone as the camera instead.
- Buy a **40 mm orange soft foam ball** (stress-ball type). It grips better and rolls
  less than a ping-pong ball. Keep it 40 mm: the code grips it at its centre (20 mm up).

### ⬜ 4. Wire and flash (You, guide "Wiring")
- **Check every servo is a 180° one** before it goes in: send it 0 then 180. It must
  turn and stop. One that keeps spinning is a 360° "continuous" servo and won't work.
- 5 servo signal wires to D3/D5/D6/D9/D10. Every servo's red to the bench supply +
  (5.5 V, 3 A limit), brown to supply −, and **one wire from Nano GND to supply −**
  (common ground). Never power the servos from the Nano.
- Flash: `firmware\upload.ps1 COM3` (use your port), or open `firmware/arm/arm.ino` in the
  Arduino IDE (Board: Arduino Nano, Processor: ATmega328P (Old Bootloader)) and Upload.

### ⬜ 5. Assemble (You)
- Before fitting each horn, run `calibrate.py` → `center` so every servo sits at 90°.
  Then put each part on with the arm in the home pose: upper arm straight up,
  forearm pointing forward, gripper level and in line with the forearm.
- Leave the roll servo out. Clamp `gripperMount` to the trunk with
  `gripperCap` (glue if it slips).
- Stand the drum in the baseplate ring and tape or hot-glue it.
- The guide's "Assembly" section has the step-by-step.

### ⬜ 6. Calibrate (You + `calibrate.py`)
- `cd software; uv run calibrate.py`. Check each joint's direction, then set
  the gripper open/closed angles. Set "closed" so it squeezes the **block**
  (the smaller object), then check it still holds the ball without the servo
  buzzing hard. This saves `calibration.json`.
- Check the real tool point: command a spot and see if the ball lands under
  the fingers. Tell Claude if it's off, and the numbers in `config.py` get tuned.

### ⬜ 7. Record demos (You)
- Test: `uv run autodemo.py --spot B2 --camera 1` (webcam), then the
  block: `uv run autodemo.py --target block --spot C1 --other-spot A3 --camera 1`.
- **Record 100–200** (about 150 is a good target):
  `uv run autodemo.py --record 150 --camera 1`. Each time it prints e.g.
  `block on C1, ball on A3 -> "pick up the block"`; place both, press Enter.
  About 20 s each, so 150 take under an hour. You can split it into sessions:
  it keeps numbering where it left off.
- Optional keyboard demos: `uv run teleop.py --camera 1`, press **T** to pick
  the instruction, then pick that object.

### ⬜ 8. Train ACT locally (You run it; optional but it checks the data)
- `cd software/act; uv sync; uv run export_act.py; uv run train_act.py --steps 20000`
  (about 2.5 hours on the RTX 5050).
- Run it on the arm:
  `uv run ..\run_policy.py act --camera 1 --checkpoint ..\..\outputs\act\pick_ball\checkpoints\020000\pretrained_model`,
  then type `pick up the ball` or `pick up the block` at the `instruction>` prompt.

### ⬜ 9. GR00T fine-tune in the cloud (You: accounts; Claude: helps)
> **This is the only milestone that can cost money.** Fine-tuning GR00T needs a
> GPU with ≥ 40 GB of memory (the laptop has 8 GB), so it has to be rented.
> Paid, that's about $10–20 for the first run. Milestones 3–8 are all free, so
> do those first and decide on this one once ACT works.
>
> **Free options to check first** (terms change, so check their pricing pages):
> - **Lightning AI** free plan: monthly credits and a Studio you can SSH into,
>   so the scripts below work as they are. The credits buy fewer hours on big GPUs.
> - **Modal** Starter plan: about $30/month of free compute. It isn't an SSH box,
>   so Claude would need to write a small Modal wrapper for the scripts first.
> - Free 16 GB GPUs (Colab, Kaggle) are too small for this fine-tune as set up.
>
> **If you skip it:** ACT (milestone 8) runs the same pipeline for free on the
> laptop. In the talk, explain GR00T with the guide's diagrams and do the live
> demo with ACT, saying it's the small model trained on the same data.

- Make a Hugging Face account and request access to `nvidia/Cosmos-Reason2-2B`
  (free). Get GPU time: free credits from above, or a paid ≥ 40 GB GPU.
- Export the dataset (`cd software; uv run export_dataset.py`), rent a Linux GPU
  box, copy `software/gr00t` + `data/lerobot/pick_ball` up with `scp`.
- On the cloud box: `bash ~/gr00t/setup_cloud.sh` (needs `HF_TOKEN`), then
  `bash ~/gr00t/finetune.sh ~/pick_ball` (1–2 h, in `tmux`), then `bash ~/gr00t/serve.sh`.
- On the laptop: SSH tunnel `ssh -p <port> -L 5555:localhost:5555 root@<ip>`, then
  `uv run run_policy.py gr00t --host localhost --camera 1` and type the instruction.
- Stop the cloud machine afterwards (it bills by the hour; budget about $10–20).
  The guide's "Fine-tuning and running GR00T in the cloud" section has every step.
- This has never been run for real, so expect the first run to need small fixes.

### ⬜ 10. Present
- Guide Part I has the 3-minute GR00T talk script, the live demo plan and
  likely questions. The arm is the live demo: put both objects down, ask the
  audience "ball or block?", type it, then swap them and do the other one.
- Credit the arm on a slide or out loud: "Arm: Easy Robot Arm by mobiobi,
  Printables, CC BY-NC-SA". The license requires credit wherever it's shown.

---

## Credits and licenses
- **Arm hardware design:** "Easy Robot Arm (SG90 servos & now steppers too)"
  by [mobiobi](https://www.printables.com/@mobiobi),
  [Printables 97370](https://www.printables.com/model/97370-easy-robot-arm-sg90-servos-now-steppers-too-update),
  [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/). Used unmodified.
  See [`ATTRIBUTION.md`](cad/stl_opensource/easy_robot_arm/ATTRIBUTION.md).
- **Robot model:** [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) (N1.7).
- **ACT policy and dataset format:** [Hugging Face LeRobot](https://github.com/huggingface/lerobot).
- **Everything else** (firmware, Python software, work-cell parts in `cad/stl/`,
  the guide) was written for this project by Yishane Kao with Claude Code.
