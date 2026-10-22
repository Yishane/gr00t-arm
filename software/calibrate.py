"""Calibration helper. Type commands, watch the arm, save when it's right.

    uv run calibrate.py

Commands
    center              all servos to 90 (do this BEFORE gluing any horn)
    s <joint> <deg>     move one servo, joint = 0 base .. 3 wrist, 4 gripper
    home                arm to the home pose using the current calibration
    test                small moves to check each joint turns the right way
    flip <joint>        reverse a joint's direction (if `test` moved it backwards)
    trim <joint> <deg>  shift a joint's zero by deg (if home is a bit off)
    grip open <deg>     set the gripper's open angle
    grip closed <deg>   set the gripper's closed angle (squeezes a ball)
    save                write calibration.json
    quit
"""
import argparse
import math
import time

import config as C
import kinematics as K
from arm_io import HOME, Arm, SimArm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()
    arm = SimArm() if args.sim else Arm(args.port)
    arm.speed(60)
    cur = HOME + [C.GRIPPER_OPEN]
    arm.send(cur)
    print(__doc__)
    while True:
        try:
            parts = input("> ").split()
        except EOFError:
            break
        if not parts:
            continue
        cmd = parts[0]
        try:
            if cmd == "quit":
                break
            elif cmd == "center":
                cur = [90.0] * 5
            elif cmd == "s":
                cur[int(parts[1])] = float(parts[2])
            elif cmd == "home":
                cur = K.to_servo([0, math.pi / 2, -math.pi / 2, 0]) + [C.GRIPPER_OPEN]
            elif cmd == "test":
                base = K.to_servo([0, math.pi / 2, -math.pi / 2, 0]) + [C.GRIPPER_OPEN]
                checks = [
                    (0, [0.35, math.pi / 2, -math.pi / 2, 0], "base: arm turns LEFT (counter-clockwise from above)"),
                    (1, [0, math.pi / 2 - 0.35, -math.pi / 2, 0], "shoulder: upper arm tips FORWARD"),
                    (2, [0, math.pi / 2, -math.pi / 2 + 0.35, 0], "elbow: forearm tips UP"),
                    (3, [0, math.pi / 2, -math.pi / 2, 0.35], "wrist: gripper tips UP"),
                ]
                for j, pose, what in checks:
                    arm.send(base)
                    time.sleep(1.0)
                    arm.send(K.to_servo(pose) + [C.GRIPPER_OPEN])
                    input(f"  joint {j} {what}? (if not: flip {j})  Enter = next ")
                cur = base
            elif cmd == "flip":
                name = C.JOINTS[int(parts[1])]
                C.SIGN[name] *= -1
                # keep the home pose where it is: servo 90 at the home angle
                home_deg = {"base": 0, "shoulder": 90, "elbow": -90, "wrist": 0}[name]
                C.OFFSET[name] = 90 - C.SIGN[name] * home_deg
                print(f"  {name}: sign {C.SIGN[name]}, offset {C.OFFSET[name]}")
            elif cmd == "trim":
                name = C.JOINTS[int(parts[1])]
                C.OFFSET[name] += float(parts[2])
                print(f"  {name}: offset {C.OFFSET[name]}")
                cur = K.to_servo([0, math.pi / 2, -math.pi / 2, 0]) + [cur[4]]
            elif cmd == "grip":
                deg = float(parts[2])
                if parts[1] == "open":
                    C.GRIPPER_OPEN = deg
                else:
                    C.GRIPPER_CLOSED = deg
                cur[4] = deg
            elif cmd == "save":
                C.save_calibration()
                print(f"  saved {C.CALIBRATION_FILE}")
                continue
            else:
                print("  unknown command")
                continue
        except (IndexError, ValueError, TypeError) as e:
            print("  ?", e)
            continue
        arm.send(cur)
        print("  servos:", " ".join(f"{a:.0f}" for a in cur))
    arm.close()


if __name__ == "__main__":
    main()
