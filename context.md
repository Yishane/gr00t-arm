# gr00t-arm — context / where we left off

Read this first every session. Update the "Status" and "Next" sections before stopping.

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

## Open-source gripper (2026-10-22, Yishane's request)
- He wants an open-source gripper, NOT one we design. Never delete cad/stl/ files.
- `cad/stl_opensource/simple_sg90_claw/`: "Simple SG90 servo/micro servo Claw" by
  CtrlPlusAlt, Printables 1778845, CC BY-NC, unmodified. README there has attribution + fit notes.
- Mount: glue the wrist horn to the back of Claw_connector_V4 (replaces 05+06). Fingers are
  ~58 mm vs our FINGER_END=45, so tool length/gripper angles need recalibration. Guide not yet updated.
- Alternative if needed: Printables 1871958 (SG90 rack-and-pinion, 60 mm opening, CC-BY,
  STEP only, needs 4x M3x35 + heat-set inserts).

## CAD facts (cad/arm.py, cad/assembly.py)
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
1. He prints (fit test first), wires, assembles, calibrates � guide sections 7�11.
2. Record ~60 demos (autodemo --record 60), export, train ACT locally to validate data.
3. He makes HF account + requests nvidia/Cosmos-Reason2-2B access; gets cloud GPU credit.
4. Then: setup_cloud.sh, finetune.sh, serve.sh, run_policy.py gr00t (never run for real yet �
   first real run may need fixes; check open_loop_eval plot).
Untested on real hardware: firmware on a physical Nano, servo SIGN defaults, gripper angles,
phone-camera view framing (CAM_DOWN=65 in cad/arm.py if the view needs changing).
