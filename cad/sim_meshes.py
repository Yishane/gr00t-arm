"""
Meshes for the MuJoCo simulation (software/sim.py): the Easy Robot Arm's
parts from the author's STEP model (via assembly.py), plus our baseplate and
cup, written as STL files in software/sim_assets/.

Each arm piece is moved into the frame of the joint that turns it, with the
joint at zero (software/config.py conventions: shoulder 0 = upper arm level,
elbow 0 = straight, wrist 0 = in line), in metres. The two fingers get their
own frames on their pivot pins (sim.FINGER_PIVOTS), so they can open.

Run (once, or after the arm's model changes):
    uv run --python 3.12 --with cadquery --with trimesh --with scipy
           --with opencv-python-headless --with pyserial python sim_meshes.py
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

import arm
import assembly as A

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "software"))
import sim  # noqa: E402

OUT = HERE.parent / "software" / "sim_assets"
TOL, ANG = 0.25, 0.4      # tessellation: mm deviation, radians

# (group, STEP name, solid index) -> sim mesh name
NAMES = {
    ("base", "Base", 1): "drum",
    ("base", "Base", 2): "base_cap",
    ("turntable", "Base", 0): "turntable",
    ("upper_arm", "Arm60mm", 0): "upper_arm",
    ("forearm", "Arm60mm", 0): "forearm",
    ("gripper", "ArmWrist", 5): "gripper_trunk",
    ("gripper", "ArmWrist", 7): "gripper_mount",
    ("gripper", "ArmWrist", 2): "gripper_cap",
    ("gripper", "ArmWrist", 3): "finger_pin_l",
    ("gripper", "ArmWrist", 4): "finger_pin_r",
    ("gripper", "ArmWrist", 8): "finger_l",
    ("gripper", "ArmWrist", 9): "finger_r",
}


def to_local(name, p):
    """Home-pose points (mm, our frame) -> that piece's joint frame at zero (m)."""
    sh = np.array([0.0, 0.0, A.SHOULDER[1]])
    el = np.array([0.0, 0.0, A.ELBOW[1]])
    wr = np.array([A.WRIST[0], 0.0, A.WRIST[1]])
    if name == "upper_arm":
        # home shoulder = 90 (straight up); at 0 the link points along +x
        q = p - sh
        p = np.stack([q[:, 2], q[:, 1], -q[:, 0]], axis=1)
    elif name == "forearm":
        p = p - el
    elif name in ("finger_l", "finger_r"):
        x, y = sim.FINGER_PIVOTS[name]
        p = p - np.array([x, y, sim.FINGER_Z])
    elif name.startswith(("gripper", "finger_pin")):
        p = p - wr
    return p / 1000.0


def mesh(shape):
    v, t = shape.tessellate(TOL, ANG)
    return np.array([(q.x, q.y, q.z) for q in v]), np.array(t)


def main():
    OUT.mkdir(exist_ok=True)
    for g, name, k, shape in A.home_pieces():
        out = NAMES[(g, name, k)]
        v, t = mesh(shape)
        trimesh.Trimesh(to_local(out, v), t).export(OUT / f"{out}.stl")
        print(f"{out:14s} {len(t):6d} triangles")
    for out, shape in (("baseplate", arm.baseplate().translate((0, 0, -arm.BP_T))),
                       ("cup", arm.cup())):
        v, t = mesh(shape.val())
        trimesh.Trimesh(v / 1000.0, t).export(OUT / f"{out}.stl")
        print(f"{out:14s} {len(t):6d} triangles")


if __name__ == "__main__":
    main()
