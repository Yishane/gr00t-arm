"""
Assembled Easy Robot Arm (mobiobi, Printables 97370, CC BY-NC-SA) in the
work cell: places the author's parts at a set of joint angles, checks for
interference and renders the assembly.

The parts come from the author's assembled model E-Z_RoboArm_v23.step,
unmodified. That model uses a 90 mm upper arm and a stepper at the shoulder;
we build it with two 60 mm links and an SG90 at the shoulder, so the upper
arm here is the author's Arm60mm turned upright, and everything above the
elbow is moved down by the 30 mm difference. The 3 BaseLeg feet are left
out: the base drum stands in a ring on our baseplate instead.

Run:  uv run --python 3.12 --with cadquery --with matplotlib --with trimesh --with scipy
             --with opencv-python-headless --with pyserial python assembly.py
      --check   sweep elbow/wrist/shoulder for self-collisions
      --cell    check every ball/block spot and the cup against the camera rig,
                and every target/other-object pair autodemo.py can lay out
      --docs    also write the figures used by docs/guide.tex

Joint angles are the software ones (software/config.py): base yaw, shoulder
elevation (90 = straight up), elbow relative to the upper arm (-90 = forearm
level when the upper arm is up), wrist relative to the forearm (0 = in line).
Frame: origin on the baseplate top under the base axis, +x forward, z up.
"""
import itertools
import sys
from pathlib import Path

import cadquery as cq

import arm

HERE = Path(__file__).parent
STEP = HERE / "stl_opensource" / "easy_robot_arm" / "E-Z_RoboArm_v23.step"

# measured from the STEP (its frame: pitch axes along X, arm forward = -Y)
STEP_YAW_XY = (-32.9, 22.7)    # base yaw axis
STEP_DESK_Z = -45.75           # underside of the base drum (no legs)
STEP_SHOULDER_Z = 8.3          # shoulder axis (SG90 shaft hole in BaseRotator)
STEP_ELBOW_Z = 98.2            # elbow axis with the author's 90 mm upper arm
L_UPPER = 60.0                 # we use Arm60mm for the upper arm too
L_FORE = 60.0

SHOULDER = (0.0, STEP_SHOULDER_Z - STEP_DESK_Z)          # (x, z) in our frame
ELBOW = (0.0, SHOULDER[1] + L_UPPER)
WRIST = (L_FORE, ELBOW[1])

# STEP solids we use: (name in the STEP, index of the solid, our group).
# The STEP also holds spare parts lying beside the arm; those are skipped.
PIECES = [
    ("Base", 1, "base"),          # drum
    ("Base", 2, "base"),          # BaseCap, screwed onto the drum
    ("Base", 0, "turntable"),     # BaseRotator, holds the shoulder servo
    ("Arm60mm", 0, "forearm"),
    ("ArmWrist", 5, "gripper"),   # offAxis_gTrunkUndr
    ("ArmWrist", 7, "gripper"),   # gripperMount
    ("ArmWrist", 2, "gripper"),   # gripperCap
    ("ArmWrist", 3, "gripper"),   # finger pins
    ("ArmWrist", 4, "gripper"),
    ("ArmWrist", 8, "gripper"),   # fingers
    ("ArmWrist", 9, "gripper"),
]
GROUPS = ("base", "turntable", "upper_arm", "forearm", "gripper")


def _read_step():
    """{name: [solids]} from the STEP, in the STEP's own frame."""
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_Label, TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.ReadFile(str(STEP))
    reader.Transfer(doc)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    top = TDF_LabelSequence()
    tool.GetFreeShapes(top)
    comps = TDF_LabelSequence()
    tool.GetComponents_s(top.Value(1), comps)
    out = {}
    for i in range(1, comps.Length() + 1):
        c = comps.Value(i)
        ref = TDF_Label()
        tool.GetReferredShape_s(c, ref)
        attr = TDataStd_Name()
        ref.FindAttribute(TDataStd_Name.GetID_s(), attr)
        shape = cq.Shape.cast(tool.GetShape_s(ref).Moved(tool.GetLocation_s(c)))
        out[attr.Get().ToExtString()] = shape.Solids()
    return out


def _to_cell(shape):
    """STEP frame -> our frame: forward (-Y) becomes +x, desk at z = 0."""
    return (shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 90)
            .translate(cq.Vector(STEP_YAW_XY[1], -STEP_YAW_XY[0], -STEP_DESK_Z)))


def home_pieces(cache=[]) -> list:
    """[(group, STEP name, solid index, Shape)] in the home pose: upper arm
    up, forearm forward, gripper level."""
    if cache:
        return cache
    solids = _read_step()
    drop = STEP_ELBOW_Z - (STEP_SHOULDER_Z + L_UPPER)    # Arm90 -> Arm60
    for name, k, g in PIECES:
        s = _to_cell(solids[name][k])
        if g in ("forearm", "gripper"):
            s = s.translate(cq.Vector(0, 0, -drop))
        cache.append((g, name, k, s))
    # upper arm = the forearm link turned upright about the elbow, then put
    # on the shoulder
    fore = next(s for g, _, _, s in cache if g == "forearm")
    up = fore.rotate(cq.Vector(0, 0, ELBOW[1]), cq.Vector(0, 0, ELBOW[1]) + cq.Vector(0, 1, 0), -90)
    cache.append(("upper_arm", "Arm60mm", 0, up.translate(cq.Vector(0, 0, -L_UPPER))))
    return cache


def home_parts(cache={}) -> dict:
    """{group: Compound} in the home pose (see home_pieces)."""
    if not cache:
        cache.update({g: cq.Compound.makeCompound([s for gg, _, _, s in home_pieces() if gg == g])
                      for g in GROUPS})
    return cache


def _ry(shape, pivot, deg):
    """Rotate about the y axis through (pivot x, pivot z). Positive deg
    turns +x toward -z (tips the link down)."""
    p = cq.Vector(pivot[0], 0, pivot[1])
    return shape.rotate(p, p + cq.Vector(0, 1, 0), deg)


def build(base=0.0, shoulder=90.0, elbow=-90.0, wrist=0.0) -> dict:
    """{group: Shape} with the arm at these joint angles (degrees)."""
    h = home_parts()

    def chain(shape, upto):
        if upto >= 3:
            shape = _ry(shape, WRIST, -wrist)
        if upto >= 2:
            shape = _ry(shape, ELBOW, -(elbow + 90))
        if upto >= 1:
            shape = _ry(shape, SHOULDER, 90 - shoulder)
        return shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), base)

    return {"base": h["base"], "turntable": chain(h["turntable"], 0),
            "upper_arm": chain(h["upper_arm"], 1), "forearm": chain(h["forearm"], 2),
            "gripper": chain(h["gripper"], 3)}


def scene(parts: dict, ball_at=None, block_at=None, cache={}) -> dict:
    """Add the work cell (baseplate, camera mast + extension, webcam mount with a dummy
    webcam, cup, ball, block) to an assembled arm. ball_at is the ball centre,
    block_at the centre of the block's bottom face."""
    if not cache:
        cache.update({"plate": arm.baseplate(), "mast": arm.camera_mast(),
                      "ext": arm.mast_extension(),
                      "mount": arm.webcam_mount(), "webcam": arm.webcam_dummy(),
                      "cup": arm.cup()})
    c = cache
    out = dict(parts)

    def place(name, shape, at):
        out[name] = shape.val().moved(at)

    place("baseplate", c["plate"], cq.Location(cq.Vector(0, 0, -arm.BP_T)))
    mx, my = arm.MAST_XY
    place("camera_mast", c["mast"], cq.Location(cq.Vector(mx, my, 0)))
    place("mast_extension", c["ext"], cq.Location(cq.Vector(mx, my, 0)))
    face = (cq.Location(cq.Vector(mx, my, arm.MAST_TOP))
            * cq.Location(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), 90.0 - arm.CAM_DOWN))
    place("webcam_mount", c["mount"], face)
    place("webcam", c["webcam"], face)
    place("cup", c["cup"], cq.Location(cq.Vector(*arm.CUP_XY, 0)))
    if ball_at is not None:
        place("ball", cq.Workplane("XY").sphere(arm.BALL / 2), cq.Location(cq.Vector(*ball_at)))
    if block_at is not None:
        place("block", arm.block(), cq.Location(cq.Vector(*block_at)))
    return out


# ------------------------------------------------------------ collisions
# Exact solid booleans on the author's NURBS model take minutes per pose, so
# the checks fill every part with 1 mm voxels once and move those points.
VOXEL = 1.0


def _voxels(shape):
    import numpy as np
    import trimesh
    verts, tris = shape.tessellate(0.2, 0.3)
    m = trimesh.Trimesh(np.array([(v.x, v.y, v.z) for v in verts]), np.array(tris))
    return m.voxelized(VOXEL).fill().points


def _matrices(base=0.0, shoulder=90.0, elbow=-90.0, wrist=0.0) -> dict:
    """Same motion as build(), as 4x4 matrices per group."""
    import numpy as np

    def ry(pivot, deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        r = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        t = np.eye(4); t[[0, 2], 3] = pivot
        ti = np.eye(4); ti[[0, 2], 3] = [-pivot[0], -pivot[1]]
        return t @ r @ ti

    a = np.radians(base)
    rz = np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    sh = rz @ ry(SHOULDER, 90 - shoulder)
    el = sh @ ry(ELBOW, -(elbow + 90))
    return {"base": np.eye(4), "turntable": rz, "upper_arm": sh, "forearm": el,
            "gripper": el @ ry(WRIST, -wrist)}


_VOX = {}       # name -> points in the home pose / their fixed place
_BASELINE = {}  # overlap between parts in the home pose (where they mate)


def _points(name, shape, m):
    if name not in _VOX:
        _VOX[name] = _voxels(shape)
    p = _VOX[name]
    return p if m is None else p @ m[:3, :3].T + m[:3, 3]


def _overlap(pa, pb):
    """Approximate overlap volume (mm^3) of two voxel clouds."""
    import numpy as np
    from scipy.spatial import cKDTree
    lo, hi = np.maximum(pa.min(0), pb.min(0)), np.minimum(pa.max(0), pb.max(0))
    if (lo > hi + VOXEL).any():
        return 0.0
    d, _ = cKDTree(pb).query(pa, distance_upper_bound=0.5 * VOXEL)
    return float(np.isfinite(d).sum()) * VOXEL ** 3


def collisions(angles=(0.0, 90.0, -90.0, 0.0), extra=None, tol=30.0):
    """Pairs that overlap by more than tol mm^3 beyond their home-pose
    overlap. extra: {name: Shape} of fixed parts (the work cell)."""
    home = home_parts()
    if not _BASELINE:
        h = {g: _points(g, home[g], None) for g in GROUPS}
        for a, b in itertools.combinations(GROUPS, 2):
            _BASELINE[frozenset((a, b))] = _overlap(h[a], h[b])
    mats = _matrices(*angles)
    pts = {g: _points(g, home[g], mats[g]) for g in GROUPS}
    for name, shape in (extra or {}).items():
        pts[name] = _points(name, shape, None)
    out = []
    for a, b in itertools.combinations(list(pts), 2):
        if {a, b} <= set(extra or {}) | {"base"}:
            continue  # neither part moves
        v = _overlap(pts[a], pts[b]) - _BASELINE.get(frozenset((a, b)), 0.0)
        if v > tol:
            out.append((a, b, round(v)))
    return out


def cell_parts():
    """The fixed work-cell parts, without the arm."""
    return {k: v for k, v in scene({}).items()}


def render(parts: dict, path: Path, title="", views=((20, -60), (0, -90))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    arm_col = {"base": "#888888", "turntable": "#3b6fb6", "upper_arm": "#d9822b",
               "forearm": "#d9822b", "gripper": "#3b6fb6", "ball": "#f4f4f4", "block": "#d83a34",
               "cup": "#6aa36f", "baseplate": "#c8c8c8"}
    fig = plt.figure(figsize=(6 * len(views), 6))
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        # one collection for everything so faces are depth-sorted together
        allv, faces, cols = [], [], []
        for name, shape in parts.items():
            verts, tris = shape.tessellate(0.3, 0.4)
            v = np.array([(q.x, q.y, q.z) for q in verts])
            allv.append(v)
            faces.append(v[np.array(tris)])
            cols += [arm_col.get(name, "#bbbbbb")] * len(tris)
        ax.add_collection3d(Poly3DCollection(np.vstack(faces), facecolor=cols,
                                             edgecolor="none", alpha=1.0))
        v = np.vstack(allv)
        lo, hi = v.min(0), v.max(0)
        mid, half = (lo + hi) / 2, (hi - lo).max() / 2
        for setlim, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
            setlim(m - half, m + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev, azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _tool_pose(x, y, z):
    """First (pitch from -90 up) IK solution that passes the software checks."""
    import math
    import kinematics as K
    for p in range(-90, 1, 5):
        j = K.ik(x, y, z, math.radians(p))
        if j and K.clear_of_floor(j) and K.to_servo(j):
            return [math.degrees(a) for a in j]
    return None


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "software"))
    out = HERE / "renders"
    out.mkdir(exist_ok=True)
    home = build()
    print("home collisions:", collisions() or "none")
    render(home, out / "10_assembly_home.png", "Easy Robot Arm, home pose")
    reach = build(shoulder=40, elbow=-100, wrist=-30)
    render(reach, out / "11_assembly_reach.png", "reaching down")
    spot = (arm.SPOT_X[2], arm.SPOT_Y[0], 20)        # gripping the ball on C1, block on A3
    j = _tool_pose(*spot)
    cell = scene(build(*j) if j else home, ball_at=spot,
                 block_at=(arm.SPOT_X[0], arm.SPOT_Y[2], 0))
    render(cell, out / "12_work_cell.png", "work cell: arm, baseplate, webcam on its mast, cup")
    if "--docs" in sys.argv:
        docs = HERE.parent / "docs" / "figures"
        render(home, docs / "arm_home.png", views=((22, -55),))
        render(cell, docs / "work_cell.png", views=((25, -50),))
    if "--check" in sys.argv:
        print("elbow x wrist self-collisions (upper arm up):")
        for el in range(-170, 31, 20):
            row = []
            for wr in range(-150, 151, 30):
                c = collisions((0, 90, el, wr))
                c = [x for x in c if "base" not in x[:2] and "turntable" not in x[:2]]
                row.append("X" if c else ".")
            print(f"  elbow {el:5d}: " + "".join(row))
        print("  (columns: wrist -150..150 step 30)")
        print("shoulder sweep (forearm in line with the upper arm):")
        for sh in range(0, 181, 15):
            c = collisions((0, sh, 0, 0))
            print(f"  shoulder {sh:4d}:", c or "ok")
    if "--cell" in sys.argv:
        # ball and block grip on each spot, lifted above it, and the drop over
        # the cup (software/autodemo.py OBJECTS grip z, SAFE_Z + 10, DROP_Z)
        import autodemo as A
        zs = sorted(set(A.OBJECTS.values())) + [A.SAFE_Z + 10]
        targets = [(x, y, z) for z in zs for x in arm.SPOT_X for y in arm.SPOT_Y]
        targets += [(arm.CUP_XY[0], arm.CUP_XY[1], 80)]
        fixed = cell_parts()
        bad = 0
        for t in targets:
            j = _tool_pose(*t)
            if j is None:
                print(t, "UNREACHABLE")
                bad += 1
                continue
            hits = collisions(j, fixed)
            print(t, [round(a) for a in j], "ok" if not hits else hits)
            bad += bool(hits)
        # gripping one object with the other one on any spot autodemo.py
        # allows next to it: the arm must not touch the other object
        home = home_parts()
        objects = {"ball": _voxels(cq.Workplane("XY").sphere(arm.BALL / 2).val())
                   + (0, 0, arm.BALL / 2),
                   "block": _voxels(arm.block().val())}
        pairs = A.pairs()
        print(f"pair check: {len(pairs)} spot pairs x 2 targets")
        for target, other in (("ball", "block"), ("block", "ball")):
            for t_spot, o_spot in pairs:
                j = _tool_pose(*A.SPOTS[t_spot], A.OBJECTS[target])
                mats = _matrices(*j)
                pts = objects[other] + (*A.SPOTS[o_spot], 0)
                hits = [(g, round(v)) for g in GROUPS if g != "base"
                        for v in [_overlap(_points(g, home[g], mats[g]), pts)] if v > 0]
                if hits:
                    print(f"  {target} on {t_spot}, {other} on {o_spot}:", hits)
                    bad += 1
        print("cell check:", "PASS" if not bad else f"{bad} problem(s)")
