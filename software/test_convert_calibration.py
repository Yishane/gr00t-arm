import csv
import json

import config as C
import convert_calibration as CC


def make_episode(root, name, rows, meta):
    ep = root / name
    ep.mkdir(parents=True)
    with open(ep / "steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"state_{i}" for i in range(5)] + [f"action_{i}" for i in range(5)])
        w.writerows(rows)
    (ep / "meta.json").write_text(json.dumps(meta))
    return ep


def test_converts_servo_numbers_to_the_current_calibration(tmp_path, monkeypatch):
    monkeypatch.setitem(C.OFFSET, "elbow", 0.0)
    old = {"offset": {"base": 95.0, "shoulder": 181.0, "elbow": -4.0, "wrist": 92.0},
           "sign": dict(C.SIGN), "gripper_open": C.GRIPPER_OPEN,
           "gripper_closed": C.GRIPPER_CLOSED}
    monkeypatch.setitem(C.OFFSET, "base", 95.0)
    monkeypatch.setitem(C.OFFSET, "shoulder", 181.0)
    monkeypatch.setitem(C.OFFSET, "wrist", 92.0)
    ep = make_episode(tmp_path, "episode_000001",
                      [[0.0, 95, 91, 86, 92, 50, 95, 91, 86, 92, 50]],
                      {"task": "pick up the ball"})
    CC.convert(ep, old)
    row = next(csv.DictReader(open(ep / "steps.csv")))
    # same physical pose: elbow servo number moves by new - old offset (+4)
    assert float(row["state_2"]) == 90.0 and float(row["action_2"]) == 90.0
    assert float(row["state_0"]) == 95.0 and float(row["action_4"]) == 50.0
    meta = json.loads((ep / "meta.json").read_text())
    assert meta["calibration"]["offset"] == C.OFFSET
    assert meta["converted_from"]["offset"]["elbow"] == -4.0
    assert meta["task"] == "pick up the ball"


def test_an_episode_already_in_the_current_calibration_is_left_alone(tmp_path):
    cur = {"offset": dict(C.OFFSET), "sign": dict(C.SIGN),
           "gripper_open": C.GRIPPER_OPEN, "gripper_closed": C.GRIPPER_CLOSED}
    ep = make_episode(tmp_path, "episode_000002",
                      [[0.0, 90, 90, 90, 90, 50, 90, 90, 90, 90, 50]],
                      {"task": "pick up the ball", "calibration": cur})
    before = (ep / "steps.csv").read_text()
    assert CC.convert(ep, cur) is False
    assert (ep / "steps.csv").read_text() == before
