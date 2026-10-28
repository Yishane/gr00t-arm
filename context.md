# gr00t-arm — context / where we left off

Read this first every session. Update the "Status" and "Next" sections before stopping.

## >>> LATEST STATE (2026-10-28) — READ THIS FIRST <<<
- Milestones 3-6 (print, wire + flash, assemble, calibrate) are DONE (Yishane, 2026-10-28).
  Next: 7 record demos, 8 ACT, 9 GR00T, 10 simulation (new), 11 present.
- Branch claude/autodemo-keep-prompt (cloud session, NOT yet merged into main):
  - autodemo --record N asks `keep it? [Y/n]` after each pick (recorder.ask_keep); n deletes the
    episode; N counts KEPT episodes (Recorder.last_kept). --yes keeps everything (tests).
  - run_policy --record [DATASET] (default pick_ball_policy; `--record pick_ball` = next to the demos):
    every played step saved (frame read each step, state = last command, action = command; blocked
    step -> held command), source "policy"; Ctrl+C when the object is in the cup, then keep? prompt
    after go_home. run() wraps the old loop (now _drive) and drops a half episode on any error.
  - GOTCHA: train_act.py fails if outputs/act/<name> exists (LeRobot 0.6.1 TrainPipelineConfig raises
    FileExistsError unless resume). Docs say to rename the old folder (pick_ball_round1) first.
    Offered to automate (rename / warm-start from the previous model); not built.
  - Explained: filtered self-imitation only helps on kept SUCCESSES; corrections via keyboard
    takeover (DAgger) would help more on failures (not built). autodemo only shows the 9 spots
    (jitter = heights/timing, NOT position) -> unseen positions likely fail; sim fixes that.
- context.md had 4 stray Windows-1252 dashes; converted to UTF-8.

## >>> LATEST STATE (2026-10-27 night) — previous <<<
Newest notes are here; older dated sections below give the history.

### Where everything is
- Branch claude/relaxed-dirac-y3xj6m. Yishane merges it into main via PRs himself (PR #1-#8 merged).
  main has everything up to 324cb88 (first no-reset connect). NOT yet in main: the quiet-boot
  connect fix (arm_io._wait_ready/_after_boot) -> needs one more PR merge.
- His laptop clone: C:\Users\yskao\Projects\gr00t-arm (Windows, PowerShell). Old clone
  C:\Users\yskao\gr00t-arm holds his calibration.json (not in git) -> copy into the new clone.
- Nano on COM3 (CH340 clone). Arduino IDE 2.3.10 works for uploads; upload.ps1 needs
  `powershell -ExecutionPolicy Bypass -File .\upload.ps1 COM3`. Upload with servos unplugged.

### Hardware (all built except the webcam mount)
- Arm assembled + wired; firmware on the Nano (READY, "?" works, commands need commas).
- He once wired the bench supply reversed -> some servos may be damaged; told him to test each.
- Motion is "jumpy" (stop-start, not wrong directions): explained as SG90 deadband/gear slop,
  harmless for data; suggested 3 A limit, capacitor, MG90S shoulder, light rubber bands. No PID
  possible (SG90s have no position feedback; feedback servo = 4-wire/STS3215/encoders).
- Calibration saved; gripper reversed vs defaults (open ~10, closed ~90). Unclear if `test`/flip
  was fully checked in the new clone (calibration.json copy!).
- Webcam mount 14_webcam_mount.stl designed (see section below); he still has to print it and buy a
  1/4"-20 x ~3/8" screw. Scripted picks can be tested now with `--camera -1` (fake camera).
- Block printed GREEN. Ball: orange 40 mm foam or ping-pong (not white).

### Connection change (arm_io.Arm) — status: second version UNTESTED on his Nano
- v1 (DTR low + "?" every 0.5 s) FAILED on his Nano ("doesn't answer"): likely the CH340 driver still
  resets it and the old bootloader hangs if it keeps receiving bytes.
- v2 (pushed, not yet in main): ask once; stay silent until READY or BOOT_S=3 s; ask again;
  last resort pulse DTR and repeat. Sets speed 180 after connecting. 65 tests pass.
- If v2 still fails on his hardware: revert to plain open (reset) + 2 s wait; optionally make the
  firmware ease to 90 at boot (needs a re-upload).

### GR00T cloud (MAIN PLAN; ACT = backup) — DRY RUN DONE (see result below)
- Provider: Lambda Cloud (signed up), 1x A100 SXM 40 GB, $1.99/h, billed per minute, NO stop
  (terminate = everything deleted; optional persistent filesystem ~$0.20/GB/mo, region-locked).
  Base image Lambda Stack 22.04, region Virginia (us-east-1) this time (Arizona us-west-2 preferred
  later for lower latency). SSH user `ubuntu`, key ~/.ssh/id_ed25519 (added to Lambda).
- Instance IP this session: 132.226.96.174 (will change on every new launch).
- Done on it: scp software/gr00t up, sed CRLF fix, setup_cloud.sh (first try failed: invalid
  HF_TOKEN; fixed with a new Read token) -> "setup done".
- NOW: in tmux, `MAX_STEPS=200 SAVE_STEPS=100 BATCH=16 bash ~/gr00t/finetune.sh
  ~/gr00t/dryrun/pick_ball_sim` (BATCH=16 for the 40 GB card; BATCH=8 if OOM).
  Expect "done: newest checkpoint .../checkpoint-200".
- NEXT in the dry run: `bash ~/gr00t/serve.sh`; laptop window 1:
  `ssh -N -L 5555:localhost:5555 ubuntu@<ip>`; window 2 (software/):
  `uv run run_policy.py gr00t --host localhost --sim --task "pick up the ball" --seconds 5`
  -> expect "connected to GR00T" + "policy call N ms". Then TERMINATE the instance.
- Real run later: download the checkpoint (scp -r ubuntu@<ip>:~/gr00t_arm_ckpt/checkpoint-N .)
  before terminating. Budget ~ $11-13 total (dry run ~$3, real ~$6-8, demo ~$2).
- HF: account + Cosmos-Reason2-2B access granted; GR00T-N1.7-3B open. Token never in chat/repo.

### DRY RUN RESULT (2026-10-27) — PIPELINE WORKS END TO END
- Fine-tune 200 steps (BATCH=16) on the fake dataset OK; open-loop eval ran: MAE 14.5 deg (expected
  for fake data). serve.sh + SSH tunnel + `run_policy.py gr00t --sim` OK: first call 1019 ms (warm-up),
  then ~280 ms per call from Vancouver to Virginia; "2 blocked" = safety check working.
- 280 ms call + 8 steps (~270 ms) -> arm would pause ~0.28 s between chunks. Options (not done yet):
  Arizona region, --horizon 16, or pipelining (request the next chunk while playing the current one).
- Told him to terminate the instance after the test.

### NEXT CODE TASK — HANDED TO HIS LOCAL AGENT (2026-10-27)
- Cloud session ended (out of cloud credits). Implement docs/superpowers/plans/2026-10-27-pipelined-policy.md
  with superpowers:executing-plans (he chose that skill): run_policy.run(..., pipeline=True) requests the
  next chunk in a 1-thread ThreadPoolExecutor while the current chunk plays (submit when remaining play
  time <= last call latency; observation = fresh frame + last sent angles), skips the steps that
  elapsed during the call (keep >= 1), shutdown(wait=False, cancel_futures=True) so Ctrl+C never waits;
  tests in a NEW software/test_pipeline.py with real sleeps (test_run_policy.py patches time.sleep
  globally). Then Task 2: --no-pipeline flag + docs + context. DONE 2026-10-27 by the local agent (see PIPELINING DONE below).
- Also not yet merged into main: the quiet-boot Nano connect fix (arm_io) and later context/plan commits
  on claude/relaxed-dirac-y3xj6m -> merge that branch (PR) before working on his laptop.
- Lambda: make sure the dry-run instance is TERMINATED.

### PIPELINING DONE (2026-10-27, local agent, branch claude/relaxed-dirac-y3xj6m)
- run_policy.run(..., pipeline=True) is the default; `--no-pipeline` turns it off. Tests in
  software/test_pipeline.py (real sleeps). Extra: _CALL_LOCK serializes policy calls (one ZMQ socket;
  a call left running by Ctrl+C must not overlap the next instruction). Docs: computer_to_nano.md sec 5,
  guide 'The loop when GR00T drives'. Untested against the real cloud server / real arm.
- LIMIT: GR00T's chunk is 16 steps (0.53 s). Pipelining hides the call only if call + webcam read
  <= ~0.27 s. Dry run was 280 ms + ~50 ms read -> expect a short (~60-80 ms) hitch per chunk instead of
  ~330 ms. A closer region (Arizona) would remove it. Deferred review minors in the 2026-10-27 final message.

### Schedule (presentation Nov 4)
- Tue Oct 27: finish HW5, arm (calibrate test, scripted picks --camera -1, send offsets), print +
  mount webcam, framing check; start MATH 255 HW6 (due Thu 5:59 pm); sleep by 23:00.
- Wed Oct 28: record ~150 demos (autodemo.py --record 150 --camera 1), ACT overnight.
- Oct 30-Nov 1 GR00T real fine-tune + test; Nov 2 backup videos; nothing new after Nov 1.
- Notion To-Do List page updated by Claude (Tue/Wed blocks). He finished APSC 278 Assignment 3
  (he declined Claude ticking it in Notion — don't edit Notion without asking).

### Talk / video notes (from chat)
- Explained: VLA (ACT is NOT a VLA: no language; ACT = Action Chunking with Transformers, Stanford
  2023, ALOHA), System 2 = VLM -> embeddings, System 1 = DiT action head generating 16-step chunks by
  denoising (flow matching, 4 steps), imitation learning not RL, closed loop via camera.
  Biggest GR00T vs ACT difference = pretraining (not speed). He prefers GR00T; fine.
- Video: starts as a fake boring "what is a humanoid/VLA" lecture -> "nobody cares" -> sunglasses
  throw -> Spectrum "F-L-Y" + "building a robotic arm" title; Michael Reeves-style humour; a
  "here's the plan" script was drafted in chat.

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

## CAMERA FRAMING (2026-10-24)
- Yishane uses the phone's 0.5x ULTRA-WIDE lens. Estimate (CAD camera at mast (178,55,185), 65 deg down,
  assumed ~108x85 deg landscape FOV): tool point in frame 100% of autodemo frames; spots/cup/base in frame;
  elbow/wrist/back of gripper leave the TOP of the frame in ~60-80% of frames (fine: state has joint angles).
  With 1x (~64x50 deg) the tool point would be out ~23% of frames. Autodemo max arm height 174 mm, gripper
  tool z <= ~93; fully straight up would be 277. teleop R and a policy can go higher: offered a z cap, not built.
- He may use 1x or 0.5x. Guide sec:camsetup has the setup + teleop framing check (fingers, 9 spots, cup in view; z up to ~95).
- Tell him: check the webcam app actually uses the chosen lens; same lens for recording and running.

## 2026-10-28 NO RESET ON CONNECT
- He saw a twitch when autodemo starts (Nano reset by DTR when the port opens). arm_io.Arm now opens the
  port with dtr=rts=False, then _wait_ready() sends "?" until "P ..." (4 s timeout), then speed(180)
  (calibrate.py's speed(60) would otherwise persist). No firmware change -> no re-upload. If his CH340
  driver still pulses DTR it just takes ~2 s.
- FIRST VERSION FAILED on his Nano ("doesn't answer"): likely his CH340 driver still resets it and the
  "?" sent every 0.5 s during boot kept the OLD bootloader from starting the firmware. Fixed: ask once,
  then stay silent BOOT_S=3 s (or until READY), ask again; last resort pulse DTR to reset on purpose
  (un-sticks a hung bootloader) and repeat. 65 tests. If this still fails: revert to the plain
  reset + 2 s wait.
- He moved to a new clone at C:\Users\yskao\Projects\gr00t-arm: calibration.json (not in git) must be
  copied from C:\Users\yskao\gr00t-arm\software\.

## 2026-10-28 HARDWARE BUILT (all but the webcam mount)
- Arm assembled and wired; firmware on the Nano (READY, "?" works; commands need commas).
  Earlier he wired the bench supply reversed: told him to damage-test each servo one by one.
  Uploads fail with 2+ servos attached -> upload with servos unplugged (only needed once).
- Base: drum notch (cable slot, ~4 mm) faces backwards. Gripper servo: body on top of gripperMount,
  shaft DOWN through the top plate into the finger gear (from slicing the STEP; tab holes 28 mm apart).
- Calibration saved (calibration.json on his laptop, not in git). His gripper is REVERSED vs defaults:
  open ~10 (advised: smallest angle clearing the 40 mm ball), closed ~90 (squeeze the block). Unknown if
  `test`/flip/trim were all done.
- NEXT: confirm calibrate.py test passed, then autodemo with --camera -1 (fake camera, real
  arm) to check pick accuracy; he reports offsets -> tune config.py. Print 14_webcam_mount + 1/4"-20 screw.

## 2026-10-27 STATUS
- Block printed GREEN. He suggested a white ping-pong ball: advised against (white on light grey plate =
  low contrast); colour it with marker or get orange. Ball still undecided.
- upload.ps1: execution policy fixed with -ExecutionPolicy Bypass; then both bootloaders "not in sync"
  on COM3 although the IDE uploaded with the same cable earlier -> told him: unplug servos/power and
  D0/D1, close IDE, upload arm.ino from the IDE with the settings that worked. Waiting on result.

## WEBCAM MOUNT DONE 2026-10-27: cad/stl/14_webcam_mount.stl (arm.webcam_mount / webcam_dummy)
- Webcam measured by Yishane: head 91x28x25, lens centred; clip bar 61 long x 42 wide (tapers); head tilt
  hinge only ~20 deg; clip leg hinge stays where left; 1/4"-20 hole (~6 mm) on the folding back leg,
  centred, ~17 mm from hinge end. Estimates: folded clip 14 thick, lens 34 above plate, 13 ahead of hole.
- Design: clip folded shut, leg screwed to a plate standing square to the mast's 25-deg top face (on the
  pin) -> camera looks 65 deg down, lens ~(154, 55, 242). Slot +-10 mm for the hole, front lip (gravity
  pushes the clip onto it), zip-tie slots. Prints face down, no supports. Needs a 1/4"-20 x ~3/8" screw.
- Framing (FOV assumed 64x48 at 640x480): needs 16 up / 22 down / 22 side -> margins small; head hinge
  +-20 deg fine-tunes. assembly.py scene now uses mount + webcam (no phone); --cell PASS; figures redone.

## GR00T IS THE MAIN PLAN (Yishane, 2026-10-27); ACT = cheap safety net
- Dry-run kit: docs/gr00t_dry_run.md + software/gr00t/dryrun/pick_ball_sim (20 sim episodes, 1.4 MB,
  2 tasks, exported with export_dataset.py). finetune.sh now takes SAVE_STEPS (default 1000) and exits
  clearly if no checkpoint. Dry run: MAX_STEPS=200 SAVE_STEPS=100; laptop test with run_policy --sim.
  He does it while assembling the arm; waiting on his results (policy ms, errors, provider/GPU).

## GUIDE/README UPDATED 2026-10-26 to current plan
- Camera = USB webcam everywhere (phone + cradle kept as fallback); webcam mount listed as "not in this
  version yet". Servo 180 vs 360 check (guide sec:servotest + troubleshooting row); he hit a 360 servo
  and is buying Miuzei SG90 180 (or MG90S). Arduino IDE upload option (he uses IDE 2.3.10, COM3).
  Part I: System 2 -> embeddings (not words/positions), System 1 generates commands; imitation learning.
  Cloud: on only when used; serving ~16 GB (laptop 8 GB probably too small, worth one try).

## SCHEDULE — PRESENTATION 2026-11-04 (set 2026-10-24)
- Oct 24-26 build (print, wire, flash, assemble) | Oct 27 calibrate + camera framing check |
  Oct 28 scripted pick tests + config tuning | Oct 29 record ~150 demos, ACT overnight |
  Oct 30 test ACT, more demos if needed; EVENING = GR00T go/no-go (only if ACT works + cloud access) |
  Oct 31-Nov 1 GR00T cloud | Nov 2 backup videos + rehearse | Nov 3 buffer, nothing new | Nov 4 present.
- Update: webcam arrives Oct 26; all physical hardware built by Oct 26. On the 26th he sends webcam
  side photo (head tilted fully down) + clip grip range + size -> design mount same day; he calibrates
  while it prints (calibration moved up to Oct 26).
- Live demo = whichever model works best on Nov 2 (ACT fallback is fine). No changes after Nov 1.
- DONE 2026-10-24: HF account made, access to nvidia/Cosmos-Reason2-2B GRANTED. Token: make a Read token
  only when setting up the cloud box (never paste it in chat/repo). Still to do: cloud GPU (Lightning AI credits?).
- Confidence given: serial/firmware high; arm motion med-high; scripted picks medium (first real test);
  ACT medium; GR00T lowest (never run end to end).

## WEBCAM (2026-10-24) — Yishane bought a USB webcam (supersedes "no webcam" decision)
- KitGody 1080P, 90 deg diagonal FOV, "rotatable", plug and play (amazon.ca B0FHKFZBTR). Mount type unknown.
- Model check (FOV assumed ~66x50 at 640x480): from the current mast spot (178,55) the best pose is
  height ~230 mm, tilt 60 deg down: needs 18 up/19 down/22 side vs ~25/25/33 available -> gripper,
  9 spots and cup in frame in every autodemo frame. Current phone pose (185 mm, 65 deg) would lose the
  gripper ~23% of frames. Phone at 0.5x would also work on the current mast.
- NEXT: when it arrives, he sends webcam size + mount (clip? 1/4-20 tripod hole?) -> design a webcam
  mount for the mast (taller mast ~230, tilt 60), rerun cell checks, update guide sec:camsetup.
- His photos (handheld, from beside the base) show the printed light grey plate + drum in its ring.

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
  09 phone cradle, 10 pin, 11 cup, 12 ball half (print 2 only if no 40 mm ball).
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
      finetune,serve}.sh (NOT run — needs cloud GPU + HF access).
- [x] Firmware compiles for Nano (arduino-cli via winget); firmware/upload.ps1 <COMx>
- [x] LaTeX guide: docs/guide.tex -> docs/guide.pdf (13 pp, compiles clean with MiKTeX latexmk).
      Part I = GR00T explainer + 3-min talk script + demo plan + Q&A; Part II = BOM, printing,
      setup, wiring (TikZ), assembly, calibration, demos, ACT, GR00T cloud, troubleshooting.
      Figures from `cad/assembly.py --docs`.

## Next (all build deliverables done 2026-10-22; waiting on Yishane)
1. He prints (Easy Robot Arm parts + work cell incl. 13_block, no BaseLeg), wires, assembles, calibrates — guide sections 7–11.
2. Record 100-200 demos, two objects (autodemo --record 150), export, train ACT locally to
   validate data (check it picks the RIGHT object for each instruction).
3. He makes HF account + requests nvidia/Cosmos-Reason2-2B access; gets cloud GPU credit.
   COST (told him 2026-10-22): milestone 9 is the ONLY step that can cost money (~$10-20 paid).
   Free options documented in README m9 + guide sec:cloud: Lightning AI free credits (SSH Studio,
   scripts work as-is), Modal ~$30/mo free (needs a Modal wrapper Claude would write - not built).
   Colab/Kaggle 16 GB too small. Fallback: stop at ACT, demo with ACT. Recommended: do 3-8 first.
   He has a breadboard + jumpers.
4. Then: setup_cloud.sh, finetune.sh, serve.sh, run_policy.py gr00t (never run for real yet —
   first real run may need fixes; check open_loop_eval plot).
Untested on real hardware: firmware on a physical Nano, servo SIGN defaults, gripper angles,
phone-camera view framing (CAM_DOWN=65 in cad/arm.py if the view needs changing).
- Colours (Yishane, 2026-10-23, wants it to look good): arm already printed BLACK (keep it).
  Scheme: plate LIGHT GREY (not pure white: phone exposure/glare), ORANGE 40 mm SOFT FOAM ball
  (to buy; Yishane 2026-10-23: foam instead of ping-pong, grips better/rolls less; must stay 40 mm
  or update autodemo.OBJECTS grip z + cad checks)
  PENDING: he'll measure the foam ball he buys (may not be 40 mm). When he sends the diameter D:
  autodemo.OBJECTS['ball'] = D/2 (grip at centre), cad/arm.py BALL = D, rerun cad/assembly.py --cell
  (all spot pairs + gripper opening), tests, and check the fingers open wide enough., GREEN block (2026-10-27: out of blue filament; was blue), BLACK cup (fixed in place, Yishane 2026-10-24). Code never uses colour (policy learns
  from images + joint angles). Keep ball/block/plate/lighting the same for recording and the demo.

## WORKING MODE until 2026-11-04 (Yishane, 2026-10-23)
- Do all work in Claude Code cloud sessions (claude.ai/code) on the GitHub repo, not on the laptop.
- Claude may commit (and push, so the next cloud session sees it) without asking. After 2026-11-04
  go back to asking before committing.
