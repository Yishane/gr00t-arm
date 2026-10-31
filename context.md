# gr00t-arm — context / where we left off

Read this first every session. Update the "Status" and "Next" sections before stopping.
RULE (Yishane, 2026-10-30): EVERY commit updates context.md (what changed, why, what's next) -
including small fixes. Put the note in the same commit, not a later one.

## 2026-10-31: HOME TURNED 25 deg (branch claude/home-turned)
- His real webcam frame at home: the gripper + its servo hide A1 completely (worse than the sim's
  30%: real servo bodies/cables). config.HOME_BASE = -25 (deg, toward the plate's front edge y<0);
  arm_io.HOME = to_servo([HOME_BASE, 90, HOME_ELBOW, 0]) -> base servo 70 with his offset 95.
  Sim: row 1 fully visible from -15 (-5: 64%, -10: 91%); -25 = margin for the real arm's bulk.
  CAD: no collisions at the new home (assembly.collisions with the cell). Everything that uses
  HOME follows (autodemo, run_policy, teleop H, sim, calibrate.py's start pose; calibrate's own
  `home` command is still the level calibration pose, base 0).
- His push c3b9b92 "hi": SPOT_FIX = {}, all old numbers moved to OBJECT_SPOT_FIX["ball"], block
  only A1 (4, 0, 0). That broke 2 tests that clear SPOT_FIX to test one fix (ball's B2 +4 leaked
  in): they now also clear OBJECT_SPOT_FIX. Note: the block needs ~no corrections where the ball
  needs up to 16 mm up -> the big fixes are about the BALL (size/softness/rolling), not arm slop
  as I had said; the sim's ball demos don't have them.
- NEXT for him: merge, test single picks from the new home (ball + block, a few spots), re-record
  the sim set (old one used the old home), then the 105 real demos. 144 tests.

## 2026-10-31: per-object spot fixes; home hides row 1 (branch claude/object-spot-fix)
- autodemo.OBJECT_SPOT_FIX = {"block": {spot: (dx, dy, dz)}, "ball": {...}}: extra real-arm
  correction for ONE object at a named spot, added on top of SPOT_FIX (+ GRIP_FIX height).
  Starts empty. Like SPOT_FIX: named spots only (real arm), never sim (x, y) layouts. 144 tests.
- Occlusion at home (sim camera = fitted to his webcam, segmentation-rendered): the forearm +
  gripper hang over ROW 1, so at home the camera sees only ~30% of an object on A1, ~50% on B1,
  ~75% on C1; rows 2-3 fully visible. He asked if the model learns "block not visible -> A1":
  partly yes (a shortcut by elimination; brittle), softened by partial visibility + re-planning.
  Turning the home base -20 deg (toward the empty front edge, y<0; tool point ~(139, -53, 103),
  base servo ~75 with his offset 95) makes every spot 100% visible. OFFERED, NOT DONE: it changes
  home again -> do it before the 105 real demos, stop/re-record the sim set (pick_ball_sim, then
  in progress with the current home). Waiting for his yes.

## 2026-10-31: single named-spot picks missed in the sim (branch claude/sim-single-run-no-spot-fix)
- Yishane ran `autodemo.py --mujoco --watch --spot X --target ball` per spot: only A1 + B1 ok.
  Cause: the single-run path passed the spot NAME to plan(), which adds the REAL-arm SPOT_FIX
  (up to 12 mm sideways, 16 mm up) + GRIP_FIX ball +4 -> the sim arm gripped beside/above the
  ball (A1/B1 have no z fix). By coordinates all 9 spots pick ok (reproduced with his
  calibration). Fix: in the sim the single run goes by SPOTS[spot] coordinates. Regression test
  (fails without the fix). 143 tests.
- NOT the cause of model B's low scores: --record and run_policy trials place objects by
  coordinates (no SPOT_FIX) and every saved sim demo was checked "ok". His recording confirms it
  (`--record 300 --ball-colour red --dataset pick_ball_sim`: all ok).
- His pick_ball_sim dataset already had 428 episodes when that run started (numbering began at
  episode_000428): check they're the same settings (ball colour, shadows) before training on it.
  That run has no --offset -> objects exactly on the 9 spots.

## 2026-10-30 evening: model B results, one ball colour (branch claude/sim-ball-colour)
- Model B (pick_ball_mixed, 2153 eps, 98% sim) after the merged fixes, --mujoco --trials 20
  --anywhere: 60k 3/20, 80k 0/20, 100k 7/20 (ball 4/10, block 3/10). Earlier 4 spot trials at
  the newest: block 2/2 (both survived "8 limited" at the grip = cut-short fix works), ball 0/2.
  100k training log: epch 1.21, loss 0.070 (l1 0.069, kld 0.000), 2.1 GB, ~3 step/s.
  Read: undertrained for its data (1.2 passes) + hardest task variant (anywhere, random ball
  colour, randomized look); checkpoints swing -> not converged. Advised: stop spending nights on
  B; record 105 real on the spots -> real-only A-150 (~60k steps); GR00T main attempt on 150 real
  + ~300 sim (--with-max 300); scripted demo fallback. Asked him to score B@100k on spots
  (no --anywhere) to see if a spot-only sim model is worth one night.
- Shadows: ~1800 of his 2108 sim demos have shadows, ~300 don't -> trials (shadows on) match;
  shadows ruled out as the cause.
- --ball-colour NAME|#rrggbb (add_hw_args, --mujoco): sim.World(ball_colour=...) keeps the ball
  that colour (still shaded +-0.04 like other materials); default = random hue each layout.
  He may use ONE real ball for the 105 demos + the live demo; then record new sim demos with the
  same colour into a new dataset (e.g. --dataset pick_ball_mujoco_orange). 142 tests.

## 2026-10-30: --step to pick/compare checkpoints (branch claude/compare-checkpoints)
- run_policy/run_act `--step 60k` (one), `40k,60k,80k` or `all` (needs --mujoco --trials): each
  checkpoint scored on the SAME layouts (trials seed random + world.rng with the trial index,
  always, so separate runs compare too), table at the end; GPU cache freed between models.
  checkpoints(name, which) replaces the glob; latest_checkpoint wraps it. 140 tests.
  e.g. `uv run run_act.py --model pick_ball_mixed --step all --mujoco --trials 20 --anywhere`.
- Commit 9a5f980 went out without this note (the note's anchor only existed on the resume
  branch); fixed in the next commit, no history rewrite. This branch also merges
  claude/train-resume-total-steps: merging this one PR brings both.
- Steps advice given: ~10-20 passes for small sets (150 real ~46k frames -> 60-115k steps);
  big sim sets limited by time; watch loss; pick the checkpoint where the score levels off.
  Real-only models: compare checkpoints on the real arm (the sim is the wrong judge for them).

## 2026-10-30 13:18: model B is already at 80,000 steps
- His `train_act.py --resume --name pick_ball_mixed --steps 36000` ended at once: LeRobot printed
  "Resuming data order at epoch 0, sample 640000" = step 80,000 (x batch 8) > 36,000 total.
  --steps is the TOTAL. train_act --resume now refuses a total <= the saved step with a clear
  message (branch claude/train-resume-total-steps). Next: score B@80k in the sim (--trials 20
  --anywhere); continue only if it improved (e.g. --steps 116000 --save_freq=5000).

## 2026-10-30 (cloud session, later): fewer sim demos per real
- export_act.py / export_dataset.py --with-max N: at most N episodes from each --with set, spread
  evenly (recorder.episodes_in(raws, max_extra)). Advice: keep all 2108 sim demos on disk (~13 GB,
  laptop has room), export a new mix, e.g. `--with pick_ball_mujoco --with-max 300 --out
  pick_ball_mix300` (~1-2 sim per real), train under that name (train_act refuses an existing
  outputs/act/<name>). Model B (pick_ball_mixed, 47 sim per real) stays for comparison.
- Merged by Yishane: PR #17 (shadows back on) and #18 (cut unsafe moves short + gripper snap).
- He hoped GR00T would work zero-shot; explained: new embodiment = arm-specific layers start
  empty, so fine-tuning is required; pitch = few-shot (~150 demos), generalization from pretraining.

## 2026-10-30 (cloud session): sim trials "not working at all"
- Model B (pick_ball_mixed, 98% sim demos) now grips in the sim but twitches after the grip.
  Trials code itself verified on main with an oracle policy + his calibration (real-time world):
  2/2 ok; layouts vary. MISTAKE: context said all 2108 sim demos were recorded without shadows
  (before f96a42e turned them on at 10-29 17:00), so I defaulted shadows OFF; Yishane says the
  1000+ sim demos were recorded WITH shadows -> reverted: shadows ON by default again,
  SIM_SHADOWS=0 turns them off. Nothing recorded/trained was affected (only new renders).
  Which episodes have shadows: compare each episode's meta.json time with 10-29 17:00, or look.
  Cause of "trials not working at all" still unknown: asked for his terminal output.
- 10-30 LOGS from Yishane: SIM trials (model B, pick_ball_mixed): 0/4; gripper never left 49-50
  (open); trials 1+4 froze with "8 blocked" every call for ~15 s (deadlock: blocked -> hold ->
  same state+image -> same unsafe move). REAL arm (`instruction>` loop): no blocks; grip rises
  49 -> 64 -> ~80-87 (closed = 90) and then sits ~15 s without lifting.
  Measured: scripted demos grip with the lowest gripper point only ~1.6 mm (median) above the
  2 mm floor limit (22/60 within 1 mm) -> ~26% of near-grip commands with ~2 deg error fail
  servo_ok. Changes (branch claude/safe-limit-not-freeze): run_policy.make_safe cuts an unsafe
  move short (bisection toward it from the last safe command), gripper always applied and
  snap_gripper()'d to open/closed (nearest); log says "N limited". NaN/short -> hold.
  test_pipeline keeps raw gripper tags (monkeypatch snap_gripper=float). 137 tests.
  HONEST: an open-loop noisy-oracle sim test (2 deg bias) gave 11/20 old vs 10/20 new - it
  can't reproduce the closed-loop deadlock, so no measured gain yet; his model in the sim is the test.
  Suggested now: `--horizon 30` (fewer re-plans) as a no-code test; temporal ensembling not
  built (his ACT calls take ~57 ms > 33 ms per step, so per-step queries would halve the rate;
  chunk-level ensembling would work). Real arm stuck after grip = B is 98% sim: more REAL demos.
- Twitching after the grip, explained (not built yet, offered): ACT chunk 100, run_policy plays
  8 then re-plans -> flips between plans when unsure; fix = temporal ensembling (query every step,
  average overlapping chunks) + snap gripper to open/closed; data fix = demos with recoveries.
- test_sim trials test broke when run_policy started calling latest_checkpoint() before
  ActBackend (no models in the cloud): test now passes --checkpoint. 135 tests pass.

## >>> LATEST STATE (2026-10-29 22:20, local agent on his laptop) — READ THIS FIRST <<<
- DATA: 45 real demos (data/raw/pick_ball) + 2108 SIM demos (data/raw/pick_ball_mujoco, all
  recorded 10-29 12:37-16:00, after every sim change; random ball colours, --anywhere, no shadows -
  shadows were turned on later). Real episode_000045 lacks frames/000000.jpg (only broken episode).
- EXPORTS (data/act): pick_ball = model A, 45 eps / 14,355 frames / 117 MB (re-exported with the
  fixed export_act). pick_ball_mixed (NOTE the -ed name he chose) = model B, 2153 eps (45 real +
  2108 sim) / 663,770 frames / 4.1 GB -> ~47 sim per real, B is essentially a sim model.
  export_act now uses streaming_encoding (~3 s/demo, was ~17 s) and skips steps with a missing
  frame. The long torchcodec/libtorchcodec error block at startup is harmless (falls back to pyav).
- TRAINING (train_act.py, batch 8, 20k steps, save every 4000, ~2.65 step/s):
  model A (outputs/act/pick_ball) started 20:18; checkpoints 004000..016000 exist (016000 at 21:22),
  NO 020000. Then a run died with `torch.AcceleratorError: CUDA error: out of memory` at step
  479/20000 after 3 min (his screenshot). Unclear which run: most likely model B started right
  after A (`a; b` in one line) - outputs/act/pick_ball_mixed was never created - and A itself may
  have stopped between 16k and 20k. GPU was empty (0 MiB) afterwards; sim recording had ended at
  16:00, so nothing else held the GPU. Earlier ACT runs used ~2.1 GB of the 8 GB RTX 5050.
  NEXT: (1) ask him which run crashed / what A's terminal printed last; (2) A's 016000 checkpoint
  is usable: `uv run ../run_policy.py act --camera 1 --checkpoint
  ../../outputs/act/pick_ball/checkpoints/016000/pretrained_model`; (3) retry B; if OOM again,
  `uv run train_act.py --name pick_ball_mixed --batch_size=4` (extra args pass through).
  A re-run of A needs outputs/act/pick_ball renamed first (lerobot FileExistsError).
- Untracked `cad/stl_opensource/easy_robot_arm/* - Copy.stl` (his copies) left out of git.
- 10-29 late: `software/act/run_act.py` launcher (`uv run run_act.py [--model NAME]`, newest
  checkpoint; newer uv picks the env from the SCRIPT's folder, so `uv run ../run_policy.py` from act/
  ran without torch). `train_act.py --resume` continues from the newest checkpoint (tested 30->40).
  Model A (45 demos, ckpt 16k) on the real arm: tweaks at start (start pose: demos began at
  ~[90,90,94,90], home is now [95,91,83,92]), goes to the object, hovers without grasping, only
  manages the ball. Checked: ball vs block label changes its commands ~10 deg (label is used).
  Verdict: too few demos -> record ~105 more real, retrain A (rename old outputs/act/pick_ball).
- Live view for class: `run_act.py --view` / `--fullscreen` (also run_policy gr00t): pygame window,
  camera + instruction banner, F11 toggles; frames read every step (like --record). pygame because
  the ACT env's OpenCV is headless (lerobot) - pygame-ce added to software/act. teleop --fullscreen
  + F11, text scales with window. OOM cause likely his video editing on the GPU at the same time.
  His new SPOT_FIX (C3 8,5,15; B3 9,9,12; A3 2,-5,12; C2 7,7,16; A1/B1 3,0,0; C1 12,-2,9) - sim
  tests that use named spots now ignore SPOT_FIX/GRIP_FIX.

## (previous) 2026-10-30 afternoon
- NOW: away from the arm. He is recording ~1000 SIM demos (`autodemo.py --mujoco --record 770
  --anywhere`; 305 in data/raw/pick_ball_mujoco so far, all recorded after every sim change below),
  then trains overnight, in software/act:
    model A = 45 real only:  `uv run export_act.py` + `uv run train_act.py`
    model B = 45 real + all sim: `uv run export_act.py --with pick_ball_mujoco --out pick_ball_mix`
      + `uv run train_act.py --name pick_ball_mix` (~22 sim per real; ~2.5 h each, 20k steps)
  Later at the arm: `uv run ../run_policy.py act --camera 1 --checkpoint
  ../../outputs/act/<name>/checkpoints/020000/pretrained_model`, same ~10 layouts for A and B.
  Told him: 45 real is thin (A will be unreliable); B may suffer sim-to-real; keep recording real
  demos (target ~150) and retrain A for the class. If B wins, add export_act option to cap #sim
  (not built; asked for it only if needed). train_act fails if outputs/act/<name> exists -> rename.
- CALIBRATION NOW (true): base 95, shoulder 181, elbow 0, wrist 92; open 50 / closed 90. HOME is
  NOT the calibration pose: config.HOME_ELBOW = -83 (elbow servo 83, forearm rests on the horn so
  the elbow idles - his idea; done in the home pose, not the calibration, so IK stays accurate).
  arm_io.HOME = [95, 91, 83, 92]; calibrate.py `home` still = level pose [95, 91, 90, 92].
  Backups (git-ignored): software/calibration_backup_2026-10-29.json (95/178/-1/92),
  calibration_defaults_2026-10-30.json (90/180/0/90), calibration_elbow-7_2026-10-30.json.
  Rule: SPOT_FIX/GRIP_FIX changes never affect recordings; only re-mounting/trim needs
  convert_calibration.py (changing a number WITHOUT a physical change: don't convert).
- SIM (10-30): ball gets a random strong colour every layout (no green 80-170 deg hue, no grey);
  no object nearer the base than A1 (sim.NEAREST = 85 mm), --anywhere still ~60% off-spot over
  x 63-143, y -20-130; `--watch [SPEED]` shows the sim camera while recording (his default 15x).
  Sim demo ~2 s each, ~6 MB; real demo ~14 MB. Laptop has ~273 GB free.
- DATA STORAGE: don't put demos in git (size, ~300k files). Suggested a private Hugging Face
  dataset repo for the EXPORTED datasets (cloud GR00T box can download it); raw zip on Drive.
  Not set up. He makes the HF write token himself (never in chat).
- GR00T explained to him (training = flow-matching action head on (image, state, instruction) ->
  next 16 actions; VLM frozen). He pushed back hard on "only does what demos show": phrase it as
  pretrained perception/language generalizes, positions interpolate between spots, but THIS
  arm's motor mapping and recovery behaviour come from the demos. Offered (not built):
  deliberate off-course-then-back moves in autodemo so demos contain corrections.

## (previous) 2026-10-29 night
- REAL DEMOS: data/raw/pick_ball has 45 kept real demos (episodes 1-45, recorded
  10-28 22:54 to 10-29 00:23). Target ~150 -> `uv run autodemo.py --camera 1 --record 105`.
  episode_000000 (10-28 01:44, OLD camera pose, washed out, no objects) moved to
  data/raw/_set_aside/ - never train on it. No ACT/GR00T model on real data yet (outputs/act has
  only sim_test). Next: finish recording -> export_act + train_act (model A, real only) -> test on arm.
- 10-30 ~02:30: elbow servo RE-MOUNTED. calibration.json now base 95, shoulder 181, elbow 0, wrist 92
  (open 50, closed 90). All 45 demos were recorded with 95/181/-4/92 (inferred exactly from the
  fixed cup-drop pose) -> converted with `convert_calibration.py pick_ball --old-offsets 95,181,-4,92`
  (elbow +4); originals in data/raw/_set_aside/pick_ball_before_conversion_*. Verified 45/45 now
  imply the current calibration. He had reset all offsets to 90/180/0/90 without re-mounting the
  others -> restored. SPOT_FIX values were tuned on the old setup: retest spots. New demos store
  their calibration in meta.json. Lighting at 02:30 was a warm lamp, darker than demos 1-45.
- He swaps ball COLOUR every demo (several 40 mm balls; no green, no white). Ball, block green, cup black.
  Remind: other object goes on its named spot (ep 45 started with the block already in the cup).
- HARDWARE TROUBLE (10-29): arm drooped/folded at far reach and later even with an empty plate;
  current spikes (he read "50 A"/"70 A" = surely 0.50/0.70 A) = servo stall -> supply limit (3 A) ->
  voltage sag. Holding home = 0.18 A. `s 2 110` (elbow, forearm down) -> ~0.7 A stall. He added a
  shoulder rubber band (DON'T tell him to again). Suspects: band crossing the elbow, damaged elbow
  SG90, slipped horns. Advised: 5 A limit, band upper arm->base only, swap elbow if still ~0.7 A.
  Status of these checks unknown. He gets frustrated by repeats: ask what he already did.
- His tuning (autodemo.py, all real-arm only): GRIP_FIX {"ball": 4} (OBJECTS stays geometry 20/18
  for the sim), SPOT_FIX C3 (-3,18,18), A3 (-5,3,12), A2 (-3,-3,0), C2 (0,0,9), A1 (-3,0,0),
  B2 (3,12,4). B2/C3 sideways fixes ~5-6 deg of base = likely base horn/drum slipped after the
  folds (heights creeping up = shoulder sag/slip). He prefers SPOT_FIX over trims ("only changing
  the markers") - fine: demos stay consistent. Calibration unchanged: offsets base 95, shoulder 178,
  elbow -4, wrist 92; signs 1/-1/-1/-1; gripper open 50 (small = open), closed 84-90.
  Backup: software/calibration_backup_2026-10-29.json (elbow -1 then).
- Code today (branch claude/autodemo-preview, pushed to main):
  - run_policy pipelining (plan docs/superpowers/plans/2026-10-27-pipelined-policy.md DONE) +
    --no-pipeline; _CALL_LOCK; frame read after a step's send. Deferred minors: calls slower than a
    16-step chunk collapse to 1 step; exit can wait 15 s after Ctrl+C; lock-leak race; camera errors
    in prefetch escape run(); ACT (~20 ms) never prefetches.
  - arm_io.HOME = K.to_servo(home joints) from the calibration (was fixed 90s -> arm jumped ~5 deg
    when autodemo started after calibrate.py). Kept demos 1-45 start ~5 deg off the new home: fine.
  - autodemo: live OpenCV preview (--no-preview; off with --yes/--mujoco); SPOT_FIX/GRIP_FIX apply
    only to NAMED spots (real arm), not sim (x, y) layouts; MAX_SPEED 60 deg/s (trajectory stretches
    fast legs; was ~100 deg/s home<->object, arm wobbled; stops stretching when a peak is a
    solution jump - a first version looped forever).
  - recorder meta.json now stores the calibration (offset/sign/gripper) per episode (eps 1-45 don't).
  - teleop window resizable (picture scales; frames stay 640x480).
  - sim camera FITTED to his real webcam from the 9 spot crosses (0.6 px rms), lens x pinned 154:
    CAMERA_POS (154, 57.9, 307.9), DOWN 70.7, new CAMERA_YAW -2.3, FOVY 39.6 (was 52: 640x480 is the
    sensor centre). Crosses alone can't separate height from FOV; x pin decides. Don't move the camera.
- Video script (his): don't claim sim demos unless model B (real+sim) beats A on the real arm;
  ACT has no language input (VLM joke only fits GR00T); film demos with run_policy, not autodemo.
  Arm credit already in the video intro.
- Plan for sim: model A = real only, model B = real + pick_ball_mujoco (export_act --with
  pick_ball_mujoco --out pick_ball_mix; train_act --name pick_ball_mix), compare 10 picks each.

## (previous) LATEST STATE (2026-10-28)
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
- MUJOCO SIMULATION BUILT (milestone 10; Yishane asked to train GR00T + ACT "in simulation" after
  the real recordings): software/sim.py (World / MujocoArm / MujocoCamera), --mujoco on every script
  via arm_io.open_hardware(args, realtime). Facts:
  - Meshes: cad/sim_meshes.py (needs cadquery env like assembly.py) -> software/sim_assets/*.stl
    (arm parts from the STEP via assembly.home_pieces(), in each joint's frame at joint zero, metres;
    baseplate + cup from cad/arm.py). CC BY-NC-SA derived -> sim_assets/ATTRIBUTION.md.
  - Joints: base z; shoulder/elbow/wrist axis 0 -1 0, qpos = software joint angle (config
    conventions). Finger pivots (home pose) (103.5, 6.04) / (103.5, -9.64), z 89.25; fingers mirrored
    by an equality; finger angle 30 deg at GRIPPER_OPEN, 6 at GRIPPER_CLOSED (linear, uses calibration).
    Holds block at ~12.5 deg, ball 13-16.
  - Servos: firmware slew 180 deg/s in 10 ms ticks, position actuators kp 3, 0.18 N m, armature 5e-4
    -> ~4 mm tool sag at home. Commands go through kinematics.from_servo = calibration.json.
  - Collisions: arm contype 1/conaff 0 (no self-collision); plate/cup/desk contype 2/conaff 1; drum
    contype 2/conaff 2 (objects only: the upper-arm hull rubbed it); objects 3/3; maxhullvert 32.
    Ball condim 6 (rolling friction), block condim 4, elliptic cone, impratio 10, dt 2 ms.
  - Camera "front" at (154, 55, 242) mm, 65 deg down, fovy 52 (estimate; sim.py --compare to tune).
    Randomize per layout: cam +-4 mm, tilt +-1.5, fovy +-1.5, light, colours.
  - result(): ok / knocked (other moved > 15 mm) / wrong object / missed. random_layout(offset):
    autodemo.layout() spots + offset, >= MIN_GAP, autodemo.plan must succeed.
  - realtime=True: thread keeps sim time = wall time (teleop, run_policy). realtime=False
    (autodemo): MujocoArm.lockstep, send() advances 1/30 s; arm.wait() used by go_home/autodemo.
  - autodemo --mujoco --record N --offset MM -> data/raw/pick_ball_mujoco, meta "sim": true, auto-keep,
    stops after 20 fails in a row. Scripted picks: 30/30 ok at offset 0 and 15, 27/30 at 25.
  - run_policy --mujoco [--trials N] [--offset] [--record]: random layout per instruction, done =
    any object in cup, prints "ball: 9/10 ok (1 missed)"; --record keeps ok runs (pick_ball_mujoco).
  - export_act.py / export_dataset.py --with NAME (merge raw sets; recorder.episodes_in), export_act --out.
  - Rendering: ~55 ms/frame on this CPU-only cloud box (EGL, set automatically when no DISPLAY);
    ~14 s per recorded demo here. Laptop GPU should be much faster (not measured).
  - mujoco added to software/ and software/act/ pyproject. software/act/uv.lock NOT regenerated
    (download.pytorch.org blocked from the cloud container) -> uv updates it on the first sync on the laptop.
  - 103 tests pass (test_sim.py 23). Guide section "Simulation (MuJoCo)" + figure sim_views.jpg;
    guide also got \DisableLigatures for tt (-- showed as one dash) and the troubleshooting table [H].
  - MAST +80 mm (Yishane asked "make the mast 80mm taller", mast already printed): new part
    cad/stl/15_mast_extension.stl (arm.mast_extension(), MAST_EXT = 80, MAST_TOP = 265): 30x30x94,
    slanted faces matching the mast top (25 deg), pin socket in each; sits ON TOP (mast pin ->
    extension -> 2nd 10_pin -> webcam mount). Print lying on a slanted face. A first version was a
    riser UNDER the mast (98 mm, 28 mm socket at z 70-96): REPLACED because that socket sits at the
    arm's travel height and the open right finger hit it just past C3 in the sim. One 274 mm mast
    would not fit the Bambu standing. assembly --cell PASS with the extension; figures redone.
    sim CAMERA_POS z 242 -> 322. Projection over all scripted picks (fovy 52, 65 deg): spots/cup/tool
    stay in frame (rows 199-450 vs 112-441), elbow/wrist in view 100% (was 77%/64%); ball ~64 px
    (was ~86). Suggested tilting the webcam head 5-10 deg more down.
  - --anywhere (autodemo + run_policy, sim only, not with --offset): sim.random_layout samples
    ANYWHERE_X (50-180) x ANYWHERE_Y (-30-130); EVERY layout (offset mode too) must pass
    clear_of_fixed_parts (plate edge, drum r39, cup, mast socket 28 mm square, CLEAR 22 mm),
    World.in_view (nominal camera, 20 px margin, object centre z 15) and _pick_works (scripted pick
    in a hidden lockstep World, jitter 0, ~0.1 s). Accepted area ~x 51-149, y -23-130 (front edge
    = camera view, right edge near the mast = fingers hit the mast socket). Figure sim_layouts.png.
  - Sim collisions now: gripper+fingers contype 5 also hit the drum (conaffinity 6); mast bar,
    mast head+extension box (z 145-272) and the baseplate's mast socket (28 mm, z 0-16) solid.
  - SHADOWS = False in sim.py: MuJoCo shadow maps speckled the desk/plate with EGL software
    rendering (spot and directional, shadowscale/clip/4096 tried). Try True on the laptop GPU.
  - FLAKY (pre-existing timing test): test_pipeline.py::test_a_slow_frame_read_does_not_stall_the_chunk
    failed 2x in ~15 full-suite runs on this branch under heavy CPU load (gap 0.116 s > 0.07); main
    passed 5/5 under load; branch passes it alone. Not changed (run loop only gained per-step checks).
  - NOT done / ideas: servo bodies (blue SG90s) not in the model; camera never compared with a real
    frame yet; ACT/GR00T never trained on sim data yet; DAgger-style keyboard takeover not built.

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
