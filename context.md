# gr00t-arm — context / where we left off

Read this first every session. Update the "Status" and "Next" sections before stopping.

## SWITCH TO OPEN-SOURCE ARM (2026-10-22, Yishane's decision) — READ FIRST
- Arm = "Easy Robot Arm" by mobiobi, Printables 97370, CC BY-NC-SA, files unmodified in
  cad/stl_opensource/easy_robot_arm/ (19 STLs + author's assembled STEP). He does NOT want
  Claude to design arm STLs. Keep cad/stl/ (old arm parts kept; work-cell parts 07-12 still used).
- Decisions: 5 servos on the arm (base, shoulder, elbow, wrist, gripper). The design's wrist-ROLL
  servo is left OFF (sits on the desk) -> no firmware/dataset/GR00T changes, pins unchanged.
  Arm60mm x2 for upper arm + forearm. No stepper (he could buy one; advised MG90S at shoulder first).
- Do NOT recode for a 6th servo. Only numbers in software/config.py change.
- README.md has the full milestone list (done / to do / how). Keep it updated with context.md.
- STEP measurements (STEP frame: pitch axes along x, arm plane y-z): shoulder axis (y22.7,z8.2),
  Arm90 elbow (22.7,98.2), Arm60 wrist at 60 from elbow; off-axis trunk roll axis z=78.3 (~20 below
  wrist axis); finger gear pins vertical at x -26.9/-42.5 (15.6 apart), pads ~y -123.8 => ~87 mm
  from wrist axis. Fingers open sideways. Base yaw axis x=-32.9,y=22.7.
- DONE 2026-10-22: milestones 1b + 2. config.py from the STEP: SHOULDER_HEIGHT 54.05 (drum on
  plate, no legs; shoulder axis = SG90 shaft hole in BaseRotator z=8.3), L 60/60, L_TOOL 86.5,
  TOOL_UP -24.75, LATERAL -1.8, new GRIPPER_POINTS, limits elbow -125..0, wrist -85..125,
  FOLD_LIMIT elbow+wrist >= -175 (kinematics.within_limits). 13 tests pass.
- cad/assembly.py now builds the arm from the STEP (upper arm = Arm60 turned upright; forearm +
  gripper dropped 29.9 for Arm90->Arm60), voxel collision checks (fast). --check / --cell / --docs.
  Cell check PASS for 9 spots (z 20 and 85) + cup drop (z 80).
- Baseplate: pegs/outline replaced by a ring for the 74.6 mm drum (tape/hot glue). CUP_XY moved
  to (20,110) in arm.py AND autodemo.py (x=0 needed >90 deg yaw). autodemo holds last pose if an
  interpolated point is unsolvable.
- Guide rewritten for the new arm (credit on title + bib, print list, assembly, tool-point check).
- Author's page (read via Chrome; WebFetch gets 403) has no step-by-step: tips = use 60 mm arms,
  elastic bands at joints, shoulder servo works hardest, servos hot-swappable.
- He HAS a breadboard + jumper wires (confirmed 2026-10-22).

## NEW DEMO: BALL + BLOCK (2026-10-23, Yishane's request) — DONE (milestone 2b)
- Task: ball + block both on the plate; he types "pick up the ball" / "pick up the block";
  arm picks that one and drops it in the cup. Record target 100-200 episodes (~150).
- Block = cad/stl/13_block.stl (arm.py block(), BLOCK=25 mm cube, 1 mm chamfer), print in a
  contrasting colour; face toward the base (fingers close sideways).
- recorder.TASKS = {"ball": "pick up the ball", "block": "pick up the block"}; TASK = ball's;
  target_of(text) -> "ball"/"block"/None. Recorder.start(task=...) saves per-episode meta.json task.
- autodemo: OBJECTS grip z {ball 20, block 18}; pairs() = spots >= MIN_GAP 60 apart minus
  TOO_CLOSE {C3,A2} -> 24 pairs, B2 never used when recording; layout() picks target 50/50.
  Flags --target --spot --other-spot; --task flag removed. Block z 16 failed (A1 gripper hits
  base drum; tilted gripper hits the other object), z 18 OK. Pair check uses fingers as in the
  STEP model (not fully open) -> watch the first real runs for the open fingers brushing the other object.
- cad/assembly.py --cell: block/ball grip z from autodemo.OBJECTS + pair check (voxel overlap of
  arm vs the other object, all pairs x 2 targets): PASS. Needs --with trimesh --with scipy
  --with opencv-python-headless --with pyserial (it imports autodemo). Work-cell figure now
  ball on C1 + block on A3.
- run_policy.py: instruction> loop (q/empty quits, "ball"/"block" expand to the full sentence,
  unclear text re-asks), --seconds 20 per instruction, Ctrl+C stops one run -> home -> ask again;
  --task runs one non-interactively. Backends take task per call.
- ACT has NO language input -> act/export_act.py adds a 6th state dim "target" (0 ball, 1 block,
  index in sorted(TASKS)); run_policy ActBackend appends it if the checkpoint's state is 6-D.
  GR00T state stays 5-D (gr00t_arm_config unchanged); language goes in via task_index.
- teleop.py: T toggles rec.task (shown top-left).
- One GRIPPER_CLOSED: calibrate it on the block (smaller); if the servo buzzes hard on the ball,
  add a per-object closed angle (not built).
- Verified in sim: 4 episodes -> export_dataset (2 tasks) + export_act + train_act 20 steps +
  run_policy act loop. 40 pytest pass (new software/test_autodemo.py). Guide updated + rebuilt (14 pp).
- Dataset name kept as "pick_ball" (paths everywhere) even though it now has both objects.

## CODE-REVIEW FIXES (2026-10-23) — DONE (milestone 2c)
- 15 findings from two /code-review runs, plan in docs/superpowers/plans/2026-10-23-code-review-fixes.md.
- kinematics.servo_ok(servo) = finite + servo range + within_limits + clear_of_floor. run_policy
  skips (holds) any policy command that fails it, prints "N blocked"; MAX_FAILS=3 policy errors
  (TimeoutError/RuntimeError/empty chunk) in a row -> back to the prompt. --horizon >= 1.
- LIMITS wrist now (-85, 90): +90 is where the wrist servo hits 0 (was 125, unreachable).
- arm_io: camera_backend() = DSHOW only on win32; Camera.read(fresh=True) grabs FLUSH=2 frames first
  (run_policy uses it); CAP_PROP_BUFFERSIZE 1.
- autodemo.solve_near: nearest pitch within PITCH_SEARCH=30 deg; trajectory raises ValueError instead
  of hold/jump; run() plans fully before moving and rec.stop(keep=False) on any exception.
- calibrate.flip(name) keeps trims (HOME_DEG). config skips calibration.json if env
  GR00T_ARM_DEFAULT_CALIBRATION is set; software/conftest.py sets it (+ RecordingSimArm helper).
- finetune.sh evaluates newest checkpoint; upload.ps1 stops on compile error (not run: no pwsh).
- 61 pytest pass. Policy commands with NaN or not 5 values are blocked too (final review).

## Goal
3D-printed 4-DOF + gripper desk arm driven by 5× SG90 servos, teleoperated with a
gamepad, recording demos in LeRobot format, training ACT locally and fine-tuning
NVIDIA GR00T (N1.7) on a rented cloud GPU. No leader arm.

## Decisions (made with Yishane)
- SG90 only (no STS3215 purchase for now). MG90S is a drop-in upgrade if jitter is bad.
  He has as many SG90s as needed (design uses 5).
- PRIORITY: keep the arm as simple as possible — just functional. The point of the
  project is the GR00T connection. Don't add features/parts that don't serve that.
- No leader arm: gamepad + inverse kinematics teleop. State = commanded servo angles.
- Microcontroller: Arduino NANO (likely CH340 clone; may need 'old bootloader'). Firmware
  pins D3,D5,D6,D9,D10 = base,shoulder,elbow,wrist,gripper.
- Printer: Bambu (A1/P1/X1, 256 mm bed).
- Power: adjustable bench supply → set 5.5 V, 3 A current limit.
- Camera: NO USB webcam (don't make him buy one). Use his phone as webcam (Phone Link /
  Iriun / DroidCam) on a printed phone stand; laptop webcam = fallback. Code takes a camera index.
- No gamepad → keyboard teleop is primary (gamepad optional if pygame finds one).
- Presentation: class talk about humanoids; he HAS slides already — no slides from us. Guide
  needs a short "how to explain the arm / GR00T" section + live demo plan. Arm will be present.
- GPU: laptop RTX 5050 8 GB → ACT trains locally; GR00T fine-tune + inference on cloud GPU (≥40 GB).
- Claude does all coding; Yishane doesn't need to understand the code. Claude may download
  software / use Chrome, but must never purchase anything.

## Deliverables (requested 2026-10-22)
1. STL files (find or design)  2. all code  3. LaTeX doc: how parts connect to GR00T +
   setup, assembly, wiring, all instructions  4. this file.
5. He wants to PRESENT "how GR00T works" using this arm → the guide gets a presentable
   GR00T explainer mapped onto the arm (talking points + demo script). NO slide deck.

## Repo
- GitHub: https://github.com/Yishane/gr00t-arm (PRIVATE). Commit + push after each milestone.

## Layout
- `cad/arm.py` — parametric CadQuery source for all parts → `cad/stl/`, `cad/render.py` → `cad/renders/`
- `firmware/arm/arm.ino` — serial servo firmware (Uno/Nano/ESP32)
- `software/` — Python: config, kinematics (+ tests), calibrate, teleop/record, export, train/run
- `docs/` — LaTeX guide

## CAD facts: OLD self-designed arm (superseded; numbers below are NOT the current arm)
- 13 STLs in cad/stl: 00 fit test, 01–06 arm, 07 baseplate (250x200), 08 camera mast,
  09 phone cradle, 10 pin, 11 cup, 12 ball half (print 2 only if no ping-pong ball).
- assembly.py: places parts + dummy SG90s; `--check` self-collision sweep, `--cell`
  reach check of all ball spots + cup against the camera rig. All pass (2026-10-22).
- Fixes made: fixed finger height, full-length horn slots in gripper, shoulder axis +2 mm
  (TT_SHAFT_Z), fingers shortened (FINGER_END=45), spot grid x=85/110/135, y=0/40/80.
- Measured kinematics: shoulder height 55.25, tool 75 along + 10.5 "up", lateral 52.8.
  Software angle -> CAD horn angle is a sign flip (horn = -joint) for shoulder/elbow/wrist.
- Joint soft limits from sweep: elbow -150..0, wrist ±90; elbow fold >150 collides.

## Status
- [x] Firmware written (not yet compiled)
- [x] kinematics.py + tests (12 passing), incl. floor-clearance check
- [x] CAD fixes + work cell (baseplate/mast/cradle/cup) + collision + reach checks + renders
- [x] Python: arm_io (real+sim), recorder, teleop (keyboard/gamepad), autodemo (scripted
      pick->cup, also auto-collects demos), calibrate, export_dataset (GR00T LeRobot v2.1)
- [x] software/gr00t/gr00t_arm_config.py (NEW_EMBODIMENT modality config)
- [x] VERIFIED: GR00T's own LeRobotEpisodeLoader + stats.py load our exported sim dataset
      (test script pattern: sparse clone of Isaac-GR00T in %TEMP%, OpenCV decode shim because
      torchcodec lacks FFmpeg DLLs on Windows)
- [x] ACT path VERIFIED on RTX 5050: software/act (own uv env, lerobot 0.6.1[dataset,training],
      torch 2.11 cu128). export_act.py (raw -> LeRobot v3), train_act.py (patches Windows
      'last' symlink crash; ~2.3 step/s @ batch 8, 2.1 GB), run_policy.py act (~23 ms/call).
- [x] GR00T client VERIFIED: policy_client.py talks to NVIDIA's real PolicyServer (ReplayPolicy
      test harness in %TEMP%); server replies msgpack_numpy {nd,type,shape,data} arrays.
      run_policy.py gr00t --host --server-port. Cloud scripts: software/gr00t/{setup_cloud,
      finetune,serve}.sh (NOT run � needs cloud GPU + HF access).
- [x] Firmware compiles for Nano (arduino-cli via winget); firmware/upload.ps1 <COMx>
- [x] LaTeX guide: docs/guide.tex -> docs/guide.pdf (13 pp, compiles clean with MiKTeX latexmk).
      Part I = GR00T explainer + 3-min talk script + demo plan + Q&A; Part II = BOM, printing,
      setup, wiring (TikZ), assembly, calibration, demos, ACT, GR00T cloud, troubleshooting.
      Figures from `cad/assembly.py --docs`.

## Next (all build deliverables done 2026-10-22; waiting on Yishane)
1. He prints (Easy Robot Arm parts + work cell incl. 13_block, no BaseLeg), wires, assembles, calibrates � guide sections 7�11.
2. Record 100-200 demos, two objects (autodemo --record 150), export, train ACT locally to
   validate data (check it picks the RIGHT object for each instruction).
3. He makes HF account + requests nvidia/Cosmos-Reason2-2B access; gets cloud GPU credit.
   COST (told him 2026-10-22): milestone 9 is the ONLY step that can cost money (~$10-20 paid).
   Free options documented in README m9 + guide sec:cloud: Lightning AI free credits (SSH Studio,
   scripts work as-is), Modal ~$30/mo free (needs a Modal wrapper Claude would write - not built).
   Colab/Kaggle 16 GB too small. Fallback: stop at ACT, demo with ACT. Recommended: do 3-8 first.
   He has a breadboard + jumpers.
4. Then: setup_cloud.sh, finetune.sh, serve.sh, run_policy.py gr00t (never run for real yet �
   first real run may need fixes; check open_loop_eval plot).
Untested on real hardware: firmware on a physical Nano, servo SIGN defaults, gripper angles,
phone-camera view framing (CAM_DOWN=65 in cad/arm.py if the view needs changing).
- Colours (Yishane, 2026-10-23, wants it to look good): arm already printed BLACK (keep it).
  Scheme: plate LIGHT GREY (not pure white: phone exposure/glare), ORANGE 40 mm ping-pong ball
  (to buy), BLUE block (not red: too close to orange). Code never uses colour (policy learns
  from images + joint angles). Keep ball/block/plate/lighting the same for recording and the demo.

## WORKING MODE until 2026-11-04 (Yishane, 2026-10-23)
- Do all work in Claude Code cloud sessions (claude.ai/code) on the GitHub repo, not on the laptop.
- Claude may commit (and push, so the next cloud session sees it) without asking. After 2026-11-04
  go back to asking before committing.
