# gr00t-arm

A small 3D-printed desk arm (SG90 servos, Arduino Nano) used to show how
NVIDIA GR00T learns robot skills. You record keyboard-teleoperated demos of
"pick up the ball, drop it in the cup", export them as a GR00T/LeRobot
dataset, train ACT on the laptop or fine-tune GR00T N1.7 on a cloud GPU, and
run the learned policy back on the arm.

> **Status (2026-10-22):** we are switching from our own self-designed arm to an
> **open-source arm** (mobiobi's *Easy Robot Arm*). The software, firmware and
> GR00T pipeline are done and verified. What's left is matching them to the new
> arm's measurements, then building, recording and training. See
> [Milestones](#milestones).

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
| `BaseLeg.stl` | 3 | feet (the author's photo shows a tripod) |
| `Arm60mm.stl` | 2 | upper arm and forearm (each carries the next servo) |
| `offAxis_gTrunkUndr.stl` | 1 | wrist piece; hangs the gripper below the wrist axis (the version in the author's assembly) |
| `gripperMount.stl` | 1 | holds the gripper servo; clamps onto the trunk disc |
| `gripperCap.stl` | 1 | screw-on ring that clamps gripperMount to the trunk |
| `gripperOne.stl`, `gripper_Two.stl` | 1 each | geared fingers (they open sideways) |
| `gripperPinOne.stl`, `gripperPinTwo.stl` | 1 each | finger pivot pins |

Not needed: `Arm75mm`, `Arm90mm` (longer links), `gripperTrunk`,
`offAxis_gTrunk` (alternative wrist pieces), `twistMount`, and the two
`28BYJ-48_*` stepper parts. `E-Z_RoboArm_v23.step` is the author's assembled
model; we use it for measurements.

Author's print settings: PLA, 0.2 mm layers, supports on, low infill (10 %
gyroid). Keep the gripper parts light so the servos don't struggle.
Without the roll servo, `gripperCap` just clamps `gripperMount` to the trunk.
If the gripper still turns, put a drop of superglue between the two discs.

### Measured from the author's model (mm)
- shoulder axis to elbow axis: 60 with Arm60 (the author's model uses Arm90 = 90)
- elbow axis to wrist axis: 60
- wrist axis to fingertips: about 87 forward; the gripper sits about 20–25 below
  the wrist axis (the "off-axis" trunk)
- the fingers turn on vertical pins 15.6 apart, so the jaws open **left/right**
  when the gripper is level (our old gripper opened up/down)

---

## Repo layout

- `cad/stl_opensource/easy_robot_arm/` — **the arm we're building** (open source, see above)
- `cad/stl/` — our original self-designed parts. Arm parts `00`–`06` are now
  replaced by the open-source arm. The work-cell parts are still used: `07` baseplate,
  `08` camera mast, `09` phone cradle, `10` pin, `11` cup, `12` ball half.
- `cad/arm.py`, `cad/assembly.py`, `cad/render.py` — CadQuery source, collision
  and reach checks, and renders, still for the **old** arm (see milestone 1)
- `firmware/arm/arm.ino` — serial servo firmware; `firmware/upload.ps1 COMx` uploads it
- `software/` — Python: `config.py` (geometry + calibration), `kinematics.py`
  (+ tests), `calibrate.py`, `teleop.py`, `autodemo.py`, `recorder.py`,
  `export_dataset.py` (GR00T format), `run_policy.py`, `policy_client.py`
- `software/act/` — ACT export/train (own Python env, runs on the RTX 5050)
- `software/gr00t/` — GR00T modality config + cloud scripts (`setup_cloud.sh`,
  `finetune.sh`, `serve.sh`)
- `docs/guide.tex` → `docs/guide.pdf` — the build + GR00T guide (still describes the old arm)
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
- ✅ Work cell: baseplate, phone-camera mast and cradle, cup, ball.
- ✅ LaTeX guide (13 pages): GR00T explainer, talk script, demo plan, build steps.
- ✅ Our own arm CAD (now replaced by the open-source arm, files kept).

### ✅ 1a. Pick and download an open-source arm
- ✅ Compared three Printables SG90 arms and picked mobiobi's Easy Robot Arm.
- ✅ Downloaded all files into `cad/stl_opensource/easy_robot_arm/`.
- ✅ Read the joint geometry from the author's STEP model (numbers above).

### 🔄 1b. Match the software to the new arm (Claude)
The code stays the same. Only the numbers in `software/config.py` change.
1. Finish measuring from the STEP + STLs: shoulder height above the desk,
   sideways offset of the gripper, the exact gripper drop below the wrist,
   where a 40 mm ball sits between the fingers, and the gripper's outline
   (for the floor check).
2. Put them into `software/config.py`: `SHOULDER_HEIGHT`, `L_UPPER=60`,
   `L_FORE=60`, `L_TOOL`, `TOOL_UP` (negative, because the gripper hangs below),
   `LATERAL`, `GRIPPER_POINTS`, and the joint limits.
3. Update `software/test_kinematics.py` for the new numbers; all tests must pass.
4. Check that the gripper can hold the 40 mm ball. If not, use a smaller ball.
5. Re-check reach: every ball spot and the cup must be reachable. If not,
   move the spots and cup (the `SPOT_X`/`SPOT_Y`/`CUP_XY` values in
   `cad/arm.py`) and rebuild the baseplate STL.
6. Check how the new base sits on our baseplate: the new base is a 75 mm drum
   with feet, and the baseplate was made for our old base.

### ⬜ 2. Update the guide and checks (Claude)
- `docs/guide.tex`: new parts list, print table, assembly steps for this arm,
  "roll servo stays off" note, new figures, and a credit line for mobiobi's
  arm (title, author, link, CC BY-NC-SA) on the title page and parts section;
  rebuild the PDF.
- `cad/assembly.py`: load the open-source STLs so the collision and reach
  checks and the renders use the real arm.
- `context.md` kept up to date.

### ⬜ 3. Print (You)
- Print the parts in the table above on the Bambu (PLA, supports on).
- Print the work cell from `cad/stl/`: `07_baseplate`, `08_camera_mast`,
  `09_phone_cradle`, `10_pin`, `11_cup` (`12_ball_half` ×2 only if you have no
  ping-pong ball).

### ⬜ 4. Wire and flash (You, guide "Wiring")
- 5 servos to D3/D5/D6/D9/D10, servo power from the bench supply (5.5 V, 3 A
  limit), and **common ground** between the supply and the Nano.
- Flash: `firmware\upload.ps1 COM5` (use your port).

### ⬜ 5. Assemble (You)
- Before fitting each horn, run the firmware so every servo sits at 90°.
  Then put each part on with the arm in the home pose: upper arm straight up,
  forearm pointing forward, gripper level and in line with the forearm.
- Leave the roll servo out. Clamp `gripperMount` to the trunk with
  `gripperCap` (glue if it slips).

### ⬜ 6. Calibrate (You + `calibrate.py`)
- `cd software; uv run calibrate.py`. Check each joint's direction, then set
  the gripper open/closed angles on the real ball. This saves `calibration.json`.
- Check the real tool point: command a spot and see if the ball lands under
  the fingers. Tell Claude if it's off, and the numbers in `config.py` get tuned.

### ⬜ 7. Record demos (You)
- Test: `uv run autodemo.py --spot B2 --camera 1` (phone as webcam).
- Record about 60: `uv run autodemo.py --record 60 --camera 1`. Add some
  keyboard ones with `uv run teleop.py --camera 1`.

### ⬜ 8. Train ACT locally (You run it; optional but it checks the data)
- `cd software/act; uv sync; uv run export_act.py; uv run train_act.py --steps 20000`
  (about 2.5 hours on the RTX 5050).
- Run it on the arm:
  `uv run ..\run_policy.py act --camera 1 --checkpoint ..\..\outputs\act\pick_ball\checkpoints\020000\pretrained_model`.

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
  `uv run run_policy.py gr00t --host localhost --camera 1`.
- Stop the cloud machine afterwards (it bills by the hour; budget about $10–20).
  The guide's "Fine-tuning and running GR00T in the cloud" section has every step.
- This has never been run for real, so expect the first run to need small fixes.

### ⬜ 10. Present
- Guide Part I has the 3-minute GR00T talk script, the live demo plan and
  likely questions. The arm is the live demo.
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
