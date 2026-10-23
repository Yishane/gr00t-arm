# Code-Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every finding from the two `/code-review` runs of 2026-10-23, so the arm software is safe when a policy drives it, robust to network and camera hiccups, and correct on every platform.

**Architecture:** One shared safety check in `kinematics.py` (`servo_ok`) that `run_policy.py` uses on every command. The camera picks its backend per OS and can drop stale frames. `autodemo.trajectory` never jumps: an unsolvable in-between pose uses the nearest pitch that solves. The other fixes are small and local: calibrate, tests, the GR00T client, two scripts.

**Tech Stack:** Python 3.12 (uv), pytest, OpenCV, pyzmq/msgpack, bash (cloud scripts), PowerShell (firmware upload).

**Spec:** the findings table below, merged from the two `/code-review` outputs in this session. Each task lists the finding IDs it closes.

| ID | Where | Problem |
|---|---|---|
| F1 | `run_policy.py` `run()` | Policy commands are only clipped to 0–180°; LIMITS, FOLD_LIMIT and floor clearance are never checked |
| F2 | `autodemo.py` `trajectory()` | An unsolvable in-between pose holds the last command, then jumps (up to 51.6° in one step at A1). The jump path is never checked |
| F3 | `autodemo.py` `trajectory()` | `prev or K.solve(...)` can be `None`, which crashes with `None + [grip]` |
| F4 | `run_policy.py` `run()` | One camera read per chunk returns a stale buffered frame |
| F5 | `arm_io.py` `Camera` | `cv2.CAP_DSHOW` is used always (Windows-only) |
| F6 | `run_policy.py` docstring | The ACT checkpoint path doesn't exist |
| F7 | `config.py` `LIMITS` | wrist +125 is unreachable (OFFSET 90, SIGN −1 → servo < 0); the real limit is +90 |
| F8 | `calibrate.py` `flip` | Throws away an earlier `trim` on that joint |
| F9 | `run_policy.py` `run()` | TimeoutError/RuntimeError from the GR00T client ends the whole session |
| F10 | `run_policy.py` `run()` | `steps[0][4]` raises IndexError on an empty chunk / `--horizon 0` |
| F11 | `autodemo.py` `run()` | An interrupted recording leaves a half-written episode folder |
| F12 | `policy_client.py` `ping()` | A RuntimeError from the server escapes `ping()` |
| F13 | `test_*.py` | Tests assert default calibration but load the user's `calibration.json` |
| F14 | `gr00t/finetune.sh` | Assumes `checkpoint-$MAX_STEPS` exists |
| F15 | `firmware/upload.ps1` | Ignores the compile exit code; a compile error gets reported as a port/cable problem |

## Global Constraints

- All commands run from `software/` with `uv run pytest -q` (dev group has pytest). The existing 40 tests must keep passing.
- Match the surrounding code: short docstrings, plain comments, no new dependencies.
- Wire protocol and firmware (`arm.ino`) don't change. The pins and the 5-number line format stay the same.
- `HOME` must still map to servo 90 on all four joints with default calibration.
- Working mode until 2026-11-04: commit and push to `claude/relaxed-dirac-y3xj6m` after each task without asking.

## Review Focus

1. **Policy outputs NaN** (a diverged model): it must be blocked, not sent as `nan` to the Nano. Test in Task 1 (`servo_ok` with NaN is False).
2. **autodemo plan unreachable after a calibration change**: it must fail *before* the arm moves, not halfway through a pick. Test in Task 4 (`run` raises before any `arm.send`).
3. **Every command in a chunk blocked**: the arm must hold still at the last safe command, not snap anywhere. Test in Task 3.
4. **The user calibrates and then runs the tests**: the tests must still pass. Test in Task 5 (conftest isolation).
5. **GR00T server answers with an error dict on every call**: after 3 failures in a row, `run` returns to the prompt instead of looping forever. Test in Task 3.

---

### Task 1: Kinematics safety check + wrist limit (F1 core, F7)

**Files:**
- Modify: `software/kinematics.py` (add `servo_ok` after `from_servo`)
- Modify: `software/config.py` (`LIMITS["wrist"]`)
- Test: `software/test_kinematics.py`

**Interfaces:**
- Produces: `kinematics.servo_ok(servo_deg) -> bool`, where `servo_deg` is the 4 arm servo angles (a 5th gripper value, if present, is ignored). True only if every value is finite and inside SERVO_MIN..SERVO_MAX, `within_limits(from_servo(...))` holds, and `clear_of_floor(...)` holds.

- [ ] **Step 1: Write the failing tests** in `test_kinematics.py`:

```python
from kinematics import servo_ok

def test_servo_ok_accepts_home_and_rejects_unsafe():
    assert servo_ok([90, 90, 90, 90])
    assert servo_ok([90, 90, 90, 90, 125])          # gripper value ignored
    assert not servo_ok([90, 90, 120, 150])         # elbow -120 + wrist -60 < FOLD_LIMIT
    assert not servo_ok([90, 90, float("nan"), 90])
    assert not servo_ok([90, 90, 90, 190])          # outside servo range

def test_servo_ok_rejects_gripper_in_the_plate():
    j = ik(120, 0, 5, -math.pi / 2)    # pointing down, fingertips ~16 mm past the tool point
    assert j is not None and not servo_ok(to_servo_unchecked(j))

def test_wrist_limit_matches_servo_range():
    lo, hi = C.LIMITS["wrist"]
    assert hi == 90
    assert to_servo([0, math.pi / 2, -math.pi / 2, math.radians(hi)]) is not None
```

`to_servo_unchecked(j)` is a 3-line helper defined in the test file: `[C.OFFSET[n] + C.SIGN[n] * math.degrees(a) for n, a in zip(C.JOINTS, j)]`.

- [ ] **Step 2: Run** `uv run pytest -q test_kinematics.py`. Expected: FAIL (ImportError `servo_ok`, `hi == 125`).
- [ ] **Step 3: Implement** `servo_ok` in `kinematics.py`. Use `math.isfinite`. In `config.py`, set `"wrist": (-85, 90)` and change its comment to say +90 is where the servo (OFFSET 90, SIGN −1) hits 0.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -am "kinematics: servo_ok safety check; wrist limit +90 to match the servo range"`

### Task 2: Camera backend per OS + fresh frames (F4 camera side, F5)

**Files:**
- Modify: `software/arm_io.py` (`Camera`, `FakeCamera`, new module-level `camera_backend`)
- Test: create `software/test_arm_io.py`

**Interfaces:**
- Produces: `arm_io.camera_backend(cv2, platform=sys.platform) -> int`, which returns `cv2.CAP_DSHOW` when `platform == "win32"` and `cv2.CAP_ANY` otherwise.
- Produces: `Camera.read(fresh=False)` and `FakeCamera.read(fresh=False)`. With `fresh=True`, `Camera` calls `self.cap.grab()` `FLUSH` times (module constant `FLUSH = 2`) before reading. `Camera.__init__` also sets `cv2.CAP_PROP_BUFFERSIZE` to 1 (backends that ignore it are covered by the flush).

- [ ] **Step 1: Write the failing tests**:

```python
import cv2
import arm_io

def test_backend_is_dshow_only_on_windows():
    assert arm_io.camera_backend(cv2, "win32") == cv2.CAP_DSHOW
    assert arm_io.camera_backend(cv2, "linux") == cv2.CAP_ANY
    assert arm_io.camera_backend(cv2, "darwin") == cv2.CAP_ANY

def test_fresh_read_flushes_buffered_frames():
    cam = arm_io.Camera.__new__(arm_io.Camera)        # skip opening a device
    cam.cv2, cam.size = cv2, (8, 6)
    cam.cap = StubCap()            # records grab()/read() calls, read() returns a 6x8 BGR frame
    cam.read(fresh=True)
    assert cam.cap.calls == ["grab"] * arm_io.FLUSH + ["read"]
    cam.cap.calls.clear(); cam.read()
    assert cam.cap.calls == ["read"]

def test_fake_camera_accepts_fresh():
    assert arm_io.FakeCamera().read(fresh=True).shape == (480, 640, 3)
```

`StubCap` is defined in the test file: `calls` list, `grab()` appends "grab", `read()` appends "read" and returns `(True, np.zeros((6, 8, 3), np.uint8))`.

- [ ] **Step 2: Run** `uv run pytest -q test_arm_io.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the changes in `arm_io.py` (`import sys`).
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `"arm_io: pick the camera backend per OS; read(fresh=True) drops stale frames"`

### Task 3: run_policy and the GR00T client hold up (F1, F4, F6, F9, F10, F12)

**Files:**
- Modify: `software/run_policy.py` (docstring, `run()`, `main()` argparse)
- Modify: `software/policy_client.py` (`ping`)
- Test: create `software/test_run_policy.py`

**Interfaces:**
- Consumes: `kinematics.servo_ok` (Task 1), `cam.read(fresh=True)` (Task 2).
- Produces: `run(policy, arm, cam, task, horizon, seconds)` keeps its signature. Behaviour:
  - Every chunk starts with `cam.read(fresh=True)`.
  - `policy.chunk` raising `TimeoutError` or `RuntimeError`, or returning an empty list, counts as one failure: print it and try again. After `MAX_FAILS = 3` failures in a row (module constant), print `"policy failed 3 times, stopping this run"`, go home and return. A successful chunk resets the count.
  - Each command: clip to SERVO_MIN..SERVO_MAX as now. If `not K.servo_ok(cmd[:4])`, don't send it. Hold the last sent command and count it as blocked. The per-chunk print adds `", N blocked"` when N > 0. The status line reads the gripper from the first *sent* step, so an empty or all-blocked chunk can't raise IndexError.
- `--horizon` uses an argparse type that rejects values < 1.
- Docstring ACT example becomes: `cd act && uv run ../run_policy.py act --checkpoint ../../outputs/act/pick_ball/checkpoints/020000/pretrained_model`.
- `PolicyClient.ping` also catches `RuntimeError` and returns False.

- [ ] **Step 1: Write the failing tests**. Patch `run_policy.time.sleep` to a no-op and use `SimArm` + `FakeCamera`. `ScriptedPolicy(results)` is a test-file class whose `chunk()` pops the next item from the list and raises it if it's an exception. Each item can be an exception, a list of steps, or `[]`.

```python
HOME5 = arm_io.HOME + [C.GRIPPER_OPEN]

def test_timeouts_and_empty_chunks_stop_the_run_not_the_program():
    pol = ScriptedPolicy([TimeoutError(), [], RuntimeError("server error")])
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the ball", 8, 60)
    assert pol.calls == 3                             # MAX_FAILS, then returned

def test_a_success_resets_the_failure_count():
    pol = ScriptedPolicy([TimeoutError(), TimeoutError(), [HOME5], TimeoutError(),
                          TimeoutError(), TimeoutError()])
    run_policy.run(pol, SimArm(), FakeCamera(), "pick up the ball", 8, 60)
    assert pol.calls == 6

def test_unsafe_commands_are_never_sent():
    arm = RecordingSimArm()                           # SimArm that appends every send() to .sent
    fold = [90, 90, 120, 150, 125]
    pol = ScriptedPolicy([[fold, fold]] + [TimeoutError()] * 3)
    run_policy.run(pol, arm, FakeCamera(), "pick up the ball", 8, 60)
    assert fold not in arm.sent
    assert all(K.servo_ok(c[:4]) for c in arm.sent)

def test_ping_is_false_on_a_server_error(monkeypatch):
    c = PolicyClient("localhost", 5999, timeout_ms=10)
    monkeypatch.setattr(c, "call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert c.ping() is False

def test_docstring_act_path_matches_train_act():
    assert "outputs/act/pick_ball/checkpoints/" in run_policy.__doc__
```

- [ ] **Step 2: Run** `uv run pytest -q test_run_policy.py`. Expected: FAIL.
- [ ] **Step 3: Implement** as specified above.
- [ ] **Step 4: Run** `uv run pytest -q` and the sim smoke run `uv run run_policy.py act --help`. Expected: all pass; help shows the new path.
- [ ] **Step 5: Commit** `"run_policy: block unsafe commands, survive timeouts/empty chunks, fresh camera frames"`

### Task 4: autodemo never jumps, fails before moving, cleans up recordings (F2, F3, F11)

**Files:**
- Modify: `software/autodemo.py` (`trajectory`, `run`, new `solve_near`)
- Test: `software/test_autodemo.py`

**Interfaces:**
- Produces: `autodemo.solve_near(x, y, z, pitch) -> list | None`. It tries `K.solve` at `pitch`, then at pitch ±1°, ±2°, … up to ±`PITCH_SEARCH` (module constant 30°), trying + before −, and returns the first solution. (A prototype of this brought the worst step at A1 down from 51.6° to 3.9°.)
- `trajectory(waypoints, fps)` uses `solve_near` for every in-between pose and raises `ValueError(f"can't reach {pose}")` if it returns None. There is no more hold/fallback branch.
- `run(...)`: build `cmds = list(trajectory(...))` **before** `rec.start` or any `arm.send`. Wrap the send loop in `try/except BaseException: rec.stop(keep=False); raise` when recording.

- [ ] **Step 1: Write the failing tests**:

```python
MAX_STEP = 6.0   # deg per 1/30 s step on any arm servo (the firmware's slew is 180 deg/s)

@pytest.mark.parametrize("seed", range(5))
def test_no_step_jumps(seed):
    random.seed(seed)
    for target in A.OBJECTS:
        for spot in A.SPOTS:
            c = list(A.trajectory(A.plan(spot, target, jitter=8.0), 30))
            assert max(max(abs(a - b) for a, b in zip(p[:4], q[:4]))
                       for p, q in zip(c, c[1:])) <= MAX_STEP, (spot, target)

def test_unreachable_plan_fails_before_the_arm_moves(monkeypatch):
    arm = RecordingSimArm()
    monkeypatch.setattr(A, "solve_near", lambda *a: None)
    with pytest.raises(ValueError):
        A.run(arm, FakeCamera(), "B2", "ball")
    assert arm.sent == []

def test_interrupted_recording_leaves_no_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "DATA", tmp_path)
    rec = Recorder("t", fps=30, source="autodemo")
    with pytest.raises(KeyboardInterrupt):
        A.run(SimArm(), CrashingCamera(after=20), "B2", "ball", rec)   # read() raises KeyboardInterrupt on call 21
    assert list((tmp_path / "t").glob("episode_*")) == []
```

Patch `autodemo.time.sleep` to a no-op in these tests. Put `RecordingSimArm` in `software/conftest.py` (Task 5 creates that file; if Task 4 runs first, create it here and Task 5 adds to it).

- [ ] **Step 2: Run** `uv run pytest -q test_autodemo.py`. Expected: `test_no_step_jumps` fails at A1, and the other two fail.
- [ ] **Step 3: Implement** as specified above.
- [ ] **Step 4: Run** `uv run pytest -q` and `uv run autodemo.py --sim --record 2 --yes --dataset tmp_check`, then delete `../data/raw/tmp_check`. Expected: all pass; 2 episodes saved.
- [ ] **Step 5: Commit** `"autodemo: nearest-pitch fallback instead of jumps; plan before moving; discard interrupted episodes"`

### Task 5: calibrate keeps trims; tests ignore calibration.json (F8, F13)

**Files:**
- Modify: `software/calibrate.py` (module-level `HOME_DEG` dict + `flip(name)` function; the `flip` command calls it)
- Modify: `software/config.py` (load only when the env var is unset)
- Create: `software/conftest.py`
- Test: create `software/test_calibrate.py`

**Interfaces:**
- Produces: `calibrate.HOME_DEG = {"base": 0, "shoulder": 90, "elbow": -90, "wrist": 0}` and `calibrate.flip(name) -> None`. `flip` reverses `C.SIGN[name]` and sets `C.OFFSET[name]` so the servo angle at `HOME_DEG[name]` is unchanged: `at_home = OFFSET + SIGN*home` before the flip, `OFFSET = at_home - SIGN*home` after.
- `config.py` ends with `if not os.environ.get("GR00T_ARM_DEFAULT_CALIBRATION"): load_calibration()`.
- `conftest.py` sets `os.environ["GR00T_ARM_DEFAULT_CALIBRATION"] = "1"` at import time, before any test imports `config`. It also holds `RecordingSimArm` (from Task 4).

- [ ] **Step 1: Write the failing tests**:

```python
def test_flip_keeps_an_earlier_trim(monkeypatch):
    monkeypatch.setitem(C.OFFSET, "shoulder", C.OFFSET["shoulder"] + 4)   # trim 1 4
    monkeypatch.setitem(C.SIGN, "shoulder", C.SIGN["shoulder"])
    before = C.OFFSET["shoulder"] + C.SIGN["shoulder"] * 90
    calibrate.flip("shoulder")
    assert C.OFFSET["shoulder"] + C.SIGN["shoulder"] * 90 == pytest.approx(before)
    assert C.SIGN["shoulder"] == 1

def test_tests_use_default_calibration(tmp_path):
    # a calibration.json from the user's arm must not leak into the tests
    assert os.environ["GR00T_ARM_DEFAULT_CALIBRATION"] == "1"
    assert C.OFFSET == {"base": 90.0, "shoulder": 180.0, "elbow": 0.0, "wrist": 90.0}
```

- [ ] **Step 2: Run** `uv run pytest -q test_calibrate.py`. Expected: FAIL.
- [ ] **Step 3: Implement** as specified above.
- [ ] **Step 4: Verify** isolation for real: write a throwaway `software/calibration.json` with `{"offset": {"shoulder": 170}}`, run `uv run pytest -q` (expected: all pass), then delete the file.
- [ ] **Step 5: Commit** `"calibrate: flip keeps trims; tests ignore the user's calibration.json"`

### Task 6: Cloud and upload scripts (F14, F15)

**Files:**
- Modify: `software/gr00t/finetune.sh` (lines after training)
- Modify: `firmware/upload.ps1`

- [ ] **Step 1: finetune.sh.** After training, set `CKPT="$(ls -d "$OUT"/checkpoint-* | sort -V | tail -1)"` (same as `serve.sh`). Use `$CKPT` in `open_loop_eval` and in the final line `echo "done: newest checkpoint $CKPT"`.
- [ ] **Step 2: Verify** with `bash -n software/gr00t/finetune.sh`, then run the `CKPT=` line by hand against a scratch dir containing `checkpoint-1000 checkpoint-2000 checkpoint-10000`. Expected: no syntax error; it picks `checkpoint-10000`.
- [ ] **Step 3: upload.ps1.** Inside the loop, after `arduino-cli compile`: `if ($LASTEXITCODE -ne 0) { Write-Host "compile failed: the sketch or the Servo library is the problem, not the port"; exit 1 }`. The compile doesn't depend on the port, so one failure is enough to stop.
- [ ] **Step 4: Verify** with `pwsh -NoProfile -Command "[ScriptBlock]::Create((Get-Content firmware/upload.ps1 -Raw)) | Out-Null"` if `pwsh` exists. Otherwise re-read the diff and note in the commit that it isn't run here (Windows-only).
- [ ] **Step 5: Commit** `"finetune.sh evaluates the newest checkpoint; upload.ps1 reports compile errors as compile errors"`

### Task 7: Docs and wrap-up

**Files:**
- Modify: `context.md` (Status: a line on these fixes and the new test count; note that `servo_ok` now guards policy commands)
- Modify: `README.md` milestone 0 test count if it's stated
- Modify: `docs/guide.tex` + rebuild `docs/guide.pdf` only if it states the wrist limit as 125 or the old ACT path (grep for `125` and `checkpoints/last`)

- [ ] **Step 1:** `uv run pytest -q`. Expected: all pass (40 old + new).
- [ ] **Step 2:** Update the docs as listed and rebuild the guide if it was touched (`latexmk -pdf` in `docs/`, 0 undefined refs).
- [ ] **Step 3: Commit and push** `git push -u origin claude/relaxed-dirac-y3xj6m`.
