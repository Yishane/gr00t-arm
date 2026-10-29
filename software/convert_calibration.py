"""Rewrite recorded episodes so their servo numbers match the CURRENT
calibration (calibration.json), after a joint was re-mounted or re-trimmed.

A recorded servo angle only means a pose together with the calibration it was
recorded with: servo = offset + sign * joint. Converting keeps every pose
(and the gripper's open/closed) and changes only the numbers.

    uv run convert_calibration.py pick_ball --old-offsets 95,181,-4,92

--old-offsets is the calibration of episodes whose meta.json doesn't say
(recorded before meta.json kept it); signs and gripper angles are taken as
the current ones unless meta.json has them. The originals are copied to
data/raw/_set_aside/<dataset>_before_conversion_<time>/ first.
"""
import argparse
import csv
import json
import shutil
import time

import config as C
from recorder import DATA


def current():
    return {"offset": dict(C.OFFSET), "sign": dict(C.SIGN),
            "gripper_open": C.GRIPPER_OPEN, "gripper_closed": C.GRIPPER_CLOSED}


def servo_map(old, new):
    """Functions old servo deg -> new servo deg, one per servo (4 joints + gripper)."""
    fns = []
    for name in C.JOINTS:
        o, s = old["offset"][name], old["sign"][name]
        n, t = new["offset"][name], new["sign"][name]
        fns.append(lambda v, o=o, s=s, n=n, t=t: n + t * (v - o) * s)
    go, gc = old["gripper_open"], old["gripper_closed"]
    no, nc = new["gripper_open"], new["gripper_closed"]
    fns.append(lambda v: no + (v - go) * (nc - no) / (gc - go) if gc != go else v)
    return fns


def convert(ep, old):
    """Convert one episode folder from `old` to the current calibration.
    Returns False if it is already in the current calibration."""
    meta = json.loads((ep / "meta.json").read_text())
    old = meta.get("calibration", old)
    new = current()
    if old == new:
        return False
    fns = servo_map(old, new)
    with open(ep / "steps.csv", newline="") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        for k in range(5):
            for col in (1 + k, 6 + k):                  # state_k, action_k
                row[col] = f"{fns[k](float(row[col])):.4f}"
    with open(ep / "steps.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    meta["calibration"] = new
    meta["converted_from"] = old
    (ep / "meta.json").write_text(json.dumps(meta, indent=2))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--old-offsets", required=True, metavar="B,S,E,W",
                    help="offsets of episodes whose meta.json has no calibration")
    args = ap.parse_args()
    offsets = [float(v) for v in args.old_offsets.split(",")]
    old = current()
    old["offset"] = dict(zip(C.JOINTS, offsets))
    src = DATA / args.dataset
    backup = DATA / "_set_aside" / f"{args.dataset}_before_conversion_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(src, backup)
    print(f"originals copied to {backup}")
    done = [ep.name for ep in sorted(src.glob("episode_*")) if convert(ep, old)]
    print(f"converted {len(done)} episodes to offsets {C.OFFSET}")


if __name__ == "__main__":
    main()
