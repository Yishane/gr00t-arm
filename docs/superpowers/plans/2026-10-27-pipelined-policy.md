# Pipelined Policy Calls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the arm pausing ~280 ms between action chunks when GR00T (or ACT) drives it, by requesting the next chunk while the current one is still playing.

**Architecture:** `run()` in `software/run_policy.py` gets a one-thread `ThreadPoolExecutor`. While a chunk plays, once the remaining play time is no longer than the last measured call time, it reads a fresh frame, sends it with the current commanded angles to the policy in the background, and keeps playing. When the chunk ends, it takes the new chunk and drops the steps that "happened" while the request was in flight (latency compensation). Failures, the safety check and the printout keep working as now.

**Tech Stack:** Python 3.12 stdlib (`concurrent.futures`, `time`), pytest. No new dependencies.

**Spec (from the chat, 2026-10-27):** GR00T dry run measured ~280 ms per call from Vancouver; `run()` plays 8 steps (~270 ms) then blocks on the next call, so the arm freezes ~0.28 s per chunk. Requirements:
1. No gap between chunks beyond normal step timing when the call is shorter than a chunk's play time.
2. Observation sent for the next chunk = a fresh camera frame + the angles last sent at that moment (consistent pair).
3. Steps the model planned for the time the request was in flight are skipped.
4. Timeouts / errors / empty chunks still count toward `MAX_FAILS = 3`; unsafe commands are still blocked; Ctrl+C still sends the arm home.
5. Works for both backends (GR00T over the network, ACT on the laptop). Option to turn it off.

## Global Constraints
- No new dependencies. Windows laptop, Python 3.12.
- All 65 existing tests keep passing (`cd software; uv run pytest -q`).
- `run(policy, arm, cam, task, horizon, seconds)` keeps working with those 6 arguments.
- The camera is only read on the main thread (the worker gets the frame as an argument).
- Commit + push to `claude/relaxed-dirac-y3xj6m` after each task.

## Review Focus
1. **Ctrl+C while a slow request is pending:** the arm must go home right away, not hang for the 15 s client timeout. Test in Task 1 (`test_ctrl_c_with_a_slow_call_pending_returns_quickly`).
2. **Call slower than a whole chunk** (e.g. 0.6 s with horizon 8 = 0.27 s): the request starts at the chunk's first step; skipping must leave at least one step; no IndexError. Test in Task 1 (`test_call_slower_than_a_chunk_still_plays`).
3. **A prefetched call fails:** it counts as one failure and the next try is a normal (blocking) call. Existing `test_a_success_resets_the_failure_count` covers it with pipelining on by default.
4. **Unsafe steps inside a prefetched chunk:** still blocked. Existing `test_unsafe_commands_are_never_sent` covers it.
5. **`time.sleep` patched to a no-op** (existing tests): the loop must not spin forever or fail. The existing tests run with pipelining on and must pass unchanged.

---

### Task 1: Pipelined `run()` with latency compensation

**Files:**
- Modify: `software/run_policy.py` (`run()`, new helper `_timed_chunk`)
- Test: create `software/test_pipeline.py` (real sleeps; no autouse sleep patch, unlike `test_run_policy.py`)

**Interfaces:**
- Produces: `run(policy, arm, cam, task, horizon, seconds, pipeline=True)`. With `pipeline=False` it behaves exactly as today.
- Produces: `_timed_chunk(policy, frame, servo, task) -> (list_of_steps, ms)`, which calls `policy.chunk` and measures it. It runs on the worker thread.

- [ ] **Step 1: Write the failing tests** in `software/test_pipeline.py`:

Test helpers (in the test file):
- `SlowPolicy(delay)`: its `chunk()` sleeps `delay`, increments `self.calls`, and returns 16 steps. Step k = `[80 + k, 90, 90, 90, 100 + self.calls]` (base encodes k, gripper encodes the call number; all safe).
- `TimedArm(SimArm)`: `send()` appends `(time.perf_counter(), list(angles))` to `self.log`.
- `run_policy.go_home` is monkeypatched to send HOME5 without the 1 s sleep.
- `call_of(a)`: `int(a[4]) - 100` if `101 <= a[4] < 120` else `None` (HOME5's gripper is 125, so home commands are excluded).
- `CrashArm(TimedArm)`: its 5th `send()` raises `KeyboardInterrupt` once; later sends work.

```python
def test_no_pause_between_chunks():
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.28), arm, FakeCamera(), "pick up the ball", 8, 2.0)
    times = [t for t, a in arm.log if (call_of(a) or 0) >= 2]   # chunks 2+ only
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert max(gaps) < 0.1                                   # was ~0.3 s

def test_sync_mode_still_pauses():                           # proves the test above measures something
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.28), arm, FakeCamera(), "pick up the ball", 8, 2.0, pipeline=False)
    times = [t for t, a in arm.log if call_of(a)]
    assert max(b - a for a, b in zip(times, times[1:])) > 0.25

def test_steps_planned_during_the_call_are_skipped():
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.28), arm, FakeCamera(), "pick up the ball", 8, 2.0)
    first_k = {}                                             # call number -> first k played
    for t, a in arm.log:
        if call_of(a):
            first_k.setdefault(call_of(a), a[0] - 80)
    assert all(k >= 5 for call, k in first_k.items() if call >= 2)

def test_call_slower_than_a_chunk_still_plays():
    arm = TimedArm()
    run_policy.run(SlowPolicy(0.6), arm, FakeCamera(), "pick up the ball", 8, 2.5)
    assert any((call_of(a) or 0) >= 2 for t, a in arm.log)   # later chunks were played

def test_ctrl_c_with_a_slow_call_pending_returns_quickly():
    arm = CrashArm()                                         # Ctrl+C mid-chunk, prefetch already running
    t0 = time.perf_counter()
    run_policy.run(SlowPolicy(2.0), arm, FakeCamera(), "pick up the ball", 8, 30)
    assert time.perf_counter() - t0 < 3.0                   # 2 s first call + a few steps; waiting would be ~4 s
    assert arm.log[-1][1] == HOME5                           # went home
```
- [ ] **Step 2: Run** `uv run pytest -q test_pipeline.py`. Expected: FAIL (`run()` has no `pipeline` argument; there are gaps of ~0.3 s).

- [ ] **Step 3: Implement** in `run_policy.py`. Rules the tests don't fully pin down:
  - Executor: `concurrent.futures.ThreadPoolExecutor(max_workers=1)`, created per `run()` call. In a `finally`: `shutdown(wait=False, cancel_futures=True)`. Never wait on a pending future when leaving.
  - `lat` = ms of the last successful call / 1000 (it starts from the first synchronous call).
  - While playing step `i` of `n`, if `pipeline` and nothing is pending and `(n - i) * dt <= lat`: read `frame = cam.read(fresh=True)`, submit `_timed_chunk(policy, frame, servo, task)` with the current `servo` (last sent command), and record `t_req = time.perf_counter()`.
  - After the chunk ends: if a request is pending, `steps, ms = pending.result()`. A `TimeoutError`/`RuntimeError`/empty result goes through the existing failure path. Then skip `min(round((time.perf_counter() - t_req) / dt), len(steps) - 1)` steps from the front, then `[:horizon]`. If nothing is pending (first chunk, or after a failure), make a normal blocking call with no skip.
  - Keep the per-chunk print. Add `", prefetched"` when the chunk came from the background.
  - The safety check, `blocked` counting, `MAX_FAILS` and `go_home` behave as today.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass (65 old + 5 new).

- [ ] **Step 5: Commit + push** `"run_policy: request the next chunk while the current one plays (no pause between chunks)"`

### Task 2: CLI switch + docs

**Files:**
- Modify: `software/run_policy.py` (`main()` argparse, module docstring)
- Modify: `docs/computer_to_nano.md` (section 5), `docs/guide.tex` (subsection "The loop when GR00T drives") + rebuild `docs/guide.pdf`, `context.md`

**Interfaces:**
- Consumes: `run(..., pipeline=...)` from Task 1.
- Produces: CLI flag `--no-pipeline` (default: pipelining on).

- [ ] **Step 1: Write the failing test** in `software/test_pipeline.py`:
```python
def test_no_pipeline_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_policy, "run", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(run_policy, "Gr00tBackend", lambda *a: object())
    monkeypatch.setattr(sys, "argv", ["run_policy.py", "gr00t", "--sim", "--task", "pick up the ball", "--no-pipeline"])
    run_policy.main()
    assert seen["pipeline"] is False
```
- [ ] **Step 2: Run** it. Expected: FAIL (unrecognized argument).
- [ ] **Step 3: Implement** the flag and pass `pipeline=not args.no_pipeline` to both `run(...)` calls. Docs: describe the overlapped loop in one or two sentences each (request the next chunk while playing; drop the steps that played during the call; `--no-pipeline` turns it off). Context: one bullet.
- [ ] **Step 4: Run** `uv run pytest -q` and rebuild the guide (`latexmk -pdf`, 0 undefined refs). Expected: all pass, guide builds.
- [ ] **Step 5: Commit + push** `"run_policy --no-pipeline; docs for the overlapped policy loop"`
