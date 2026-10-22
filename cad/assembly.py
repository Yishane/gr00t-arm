"""
Assembled arm: places every printed part plus dummy SG90 servos at a given
set of joint angles, checks for interference, and renders the assembly.

Run:  uv run --python 3.12 --with cadquery --with matplotlib python assembly.py
      (add --check to sweep poses for collisions)

Joint angles here are rotations of each servo horn about its own shaft, in
degrees, measured from the pose the parts are modelled in (link along +x of
the previous servo). They are NOT the servo command angles; see
software/config.py for that mapping.
"""
import itertools
import sys
from pathlib import Path

import cadquery as cq

import arm

HERE = Path(__file__).parent
SEAT = arm.STACK - arm.HORN_SLOT_D   # tab plane -> underside of the part on the horn


def sg90() -> cq.Workplane:
    """Rough SG90 envelope in its own frame: tab underside on z=0, shaft on
    the origin pointing +z, body extending toward -x."""
    cx = -arm.BODY_OFFSET
    body = (cq.Workplane("XY").workplane(offset=-arm.TAB_TO_BOTTOM).center(cx, 0)
            .rect(arm.BODY_L, arm.BODY_W).extrude(arm.TAB_TO_BOTTOM + arm.TAB_TO_TOP))
    tabs = cq.Workplane("XY").center(cx, 0).rect(arm.TAB_SPAN, arm.BODY_W).extrude(2.5)
    hump = (cq.Workplane("XY").workplane(offset=arm.TAB_TO_TOP)
            .circle(arm.BODY_W / 2 - 0.3).extrude(SEAT - arm.TAB_TO_TOP - 0.3))
    return body.union(tabs).union(hump)


def horn() -> cq.Workplane:
    """Double-arm horn in the frame of the part glued onto it (it turns with
    that part): outer face on z=HORN_SLOT_D, sitting in the slot."""
    return (cq.Workplane("XY").rect(32, 6).extrude(arm.HORN_SLOT_D - 0.2)
            .union(cq.Workplane("XY").circle(3.5).extrude(arm.HORN_SLOT_D - 0.2))
            .translate((0, 0, 0.2)))


def loc(x=0.0, y=0.0, z=0.0, axis=(0, 0, 1), deg=0.0) -> cq.Location:
    return cq.Location(cq.Vector(x, y, z)) * cq.Location(cq.Vector(0, 0, 0),
                                                         cq.Vector(*axis), deg)


def place(parts: dict, name: str, shape: cq.Workplane, at: cq.Location):
    parts[name] = shape.val().moved(at)


def driven(parts: dict, name: str, shape: cq.Workplane, at: cq.Location, horn):
    """A part glued onto a servo horn: place it and the horn together."""
    place(parts, name, shape, at)
    place(parts, "horn_" + name, horn, at)


def build(yaw=0.0, shoulder=-60.0, elbow=60.0, wrist=0.0, grip=0.0,
          cache={}) -> dict:
    """Return {name: Shape} for the assembled arm at the given horn angles."""
    if not cache:
        cache.update({
            "servo": sg90(), "horn": horn(), "base": arm.base(), "turntable": arm.turntable(),
            "upper": arm.link(arm.L_UPPER), "fore": arm.link(arm.L_FORE),
            "gbody": arm.gripper_body(), "jaw": arm.gripper_jaw(),
        })
    c = cache
    p = {}
    place(p, "base", c["base"], loc())
    s0 = loc(z=24.0)                                   # base servo, tabs on base top
    place(p, "servo_base", c["servo"], s0)
    t = s0 * loc(z=SEAT, deg=yaw)                       # turntable on base horn
    driven(p, "turntable", c["turntable"], t, c["horn"])
    z_shaft = arm.TT_SHAFT_Z
    s1 = t * loc(y=-4.0, z=z_shaft, axis=(1, 0, 0), deg=-90)   # shoulder servo
    place(p, "servo_shoulder", c["servo"], s1)
    u = s1 * loc(z=SEAT, deg=shoulder)
    driven(p, "upper_arm", c["upper"], u, c["horn"])
    s2 = u * loc(x=arm.L_UPPER, z=arm.LINK)
    place(p, "servo_elbow", c["servo"], s2)
    f = s2 * loc(z=SEAT, deg=elbow)
    driven(p, "forearm", c["fore"], f, c["horn"])
    s3 = f * loc(x=arm.L_FORE, z=arm.LINK)
    place(p, "servo_wrist", c["servo"], s3)
    g = s3 * loc(z=SEAT, deg=wrist)
    driven(p, "gripper_body", c["gbody"], g, c["horn"])
    s4 = g * loc(x=arm.GRIP_SERVO[0], y=arm.GRIP_SERVO[1], z=arm.LINK)
    place(p, "servo_gripper", c["servo"], s4)
    driven(p, "gripper_jaw", c["jaw"], s4 * loc(z=SEAT, deg=grip), c["horn"])
    return p


def scene(parts: dict, ball_at=None, cache={}) -> dict:
    """Add the work cell (baseplate, camera mast, phone cradle with a dummy
    phone, cup, ball) to an assembled arm."""
    if not cache:
        cache.update({"plate": arm.baseplate(), "mast": arm.camera_mast(),
                      "cradle": arm.phone_cradle(), "cup": arm.cup(), "pin": arm.pin()})
    c = cache
    out = dict(parts)
    place(out, "baseplate", c["plate"], loc(z=-arm.BP_T))
    mx, my = arm.MAST_XY
    place(out, "camera_mast", c["mast"], loc(mx, my))
    beta = 90.0 - arm.CAM_DOWN
    face = loc(mx, my, arm.MAST_H, axis=(0, 1, 0), deg=beta)
    cradle_at = face * loc(x=-19)
    place(out, "phone_cradle", c["cradle"], cradle_at)
    phone = cq.Workplane("XY").box(75, 160, 9, centered=False).translate((-75, -80, 3))
    place(out, "phone", phone, cradle_at)
    place(out, "cup", c["cup"], loc(*arm.CUP_XY))
    if ball_at is not None:
        place(out, "ball", cq.Workplane("XY").sphere(arm.BALL / 2), loc(*ball_at))
    return out


# parts that are meant to touch (servo in its pocket / horn in its slot)
MATED = {
    frozenset(x) for x in [
        ("base", "servo_base"), ("servo_base", "turntable"),
        ("turntable", "servo_shoulder"), ("servo_shoulder", "upper_arm"),
        ("upper_arm", "servo_elbow"), ("servo_elbow", "forearm"),
        ("forearm", "servo_wrist"), ("servo_wrist", "gripper_body"),
        ("gripper_body", "servo_gripper"), ("servo_gripper", "gripper_jaw"),
    ]
} | {
    frozenset(x) for d, s in [("turntable", "servo_base"), ("upper_arm", "servo_shoulder"),
                              ("forearm", "servo_elbow"), ("gripper_body", "servo_wrist"),
                              ("gripper_jaw", "servo_gripper")]
    for x in [(d, "horn_" + d), (s, "horn_" + d)]
}


def collisions(parts: dict, tol=1.0, include_mated=False):
    """Pairs whose overlap volume exceeds tol mm^3."""
    out = []
    names = list(parts)
    for a, b in itertools.combinations(names, 2):
        if not include_mated and frozenset((a, b)) in MATED:
            continue
        bba, bbb = parts[a].BoundingBox(), parts[b].BoundingBox()
        if (bba.xmax < bbb.xmin or bbb.xmax < bba.xmin or bba.ymax < bbb.ymin
                or bbb.ymax < bba.ymin or bba.zmax < bbb.zmin or bbb.zmax < bba.zmin):
            continue
        v = parts[a].intersect(parts[b]).Volume()
        if v > tol:
            out.append((a, b, round(v, 1)))
    return out


def render(parts: dict, path: Path, title="", views=((20, -60), (0, -90))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = {"servo": "#3b6fb6", "base": "#888888"}
    fig = plt.figure(figsize=(6 * len(views), 6))
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        allv = []
        for name, shape in parts.items():
            verts, tris = shape.tessellate(0.2, 0.3)
            v = np.array([(q.x, q.y, q.z) for q in verts])
            allv.append(v)
            col = colors["servo"] if name.startswith("servo") else (
                colors["base"] if name == "base" else "#d9822b")
            ax.add_collection3d(Poly3DCollection(v[np.array(tris)], facecolor=col,
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


if __name__ == "__main__":
    out = HERE / "renders"
    out.mkdir(exist_ok=True)
    home = build()
    print("home collisions:", collisions(home) or "none")
    print("mated overlaps (should be none):",
          [c for c in collisions(home, include_mated=True)
           if frozenset(c[:2]) in MATED] or "none")
    render(home, out / "10_assembly_home.png", "assembly, home pose")
    render(build(shoulder=-20, elbow=100, wrist=60, grip=30),
           out / "11_assembly_reach.png", "assembly, reaching down, gripper open")
    render(scene(build(shoulder=-60, elbow=60)), out / "12_work_cell.png",
           "work cell: arm, baseplate, phone camera mast, cup")
    if "--docs" in sys.argv:
        docs = HERE.parent / "docs" / "figures"
        render(build(), docs / "arm_home.png", views=((22, -55),))
        render(scene(build(shoulder=-60, elbow=60)), docs / "work_cell.png", views=((25, -50),))
    if "--check" in sys.argv:
        # elbow/wrist self-collision (shoulder angle doesn't change these)
        print("elbow x wrist self-collisions:")
        for el in range(-160, 161, 20):
            row = []
            for wr in range(-150, 151, 30):
                c = collisions(build(shoulder=-90, elbow=el, wrist=wr))
                c = [x for x in c if not {"base", "turntable", "servo_base"} & set(x[:2])]
                row.append("X" if c else ".")
            print(f"  elbow {el:5d}: " + "".join(row))
        print("  (columns: wrist -150..150 step 30)")
        # shoulder vs base/turntable with the rest of the arm straight
        print("shoulder sweep (arm straight):")
        for sh in range(-180, 1, 15):
            c = collisions(build(shoulder=sh, elbow=0, wrist=0))
            print(f"  shoulder {sh:5d}:", c or "ok")
    if "--cell" in sys.argv:
        # arm reaching every ball spot and the cup must not hit the camera rig
        sys.path.insert(0, str(HERE.parent / "software"))
        import math
        import kinematics as K
        targets = [(x, y, 20) for x in arm.SPOT_X for y in arm.SPOT_Y]
        targets += [(arm.CUP_XY[0], arm.CUP_XY[1], 75)]
        rig = {"camera_mast", "phone_cradle", "phone", "cup", "baseplate"}
        for tx, ty, tz in targets:
            sol = None
            for p in range(-90, 1, 5):
                j = K.ik(tx, ty, tz, math.radians(p))
                if j and K.clear_of_floor(j) and K.to_servo(j):
                    sol = j
                    break
            if sol is None:
                print((tx, ty, tz), "UNREACHABLE")
                continue
            b, sh, el, wr = (math.degrees(a) for a in sol)
            # software angles -> CAD horn angles (CAD: horn 0 = link along +x,
            # positive horn = link turns down)
            parts = scene(build(yaw=b, shoulder=-sh, elbow=-el, wrist=-wr))
            hits = [c for c in collisions(parts)
                    if (set(c[:2]) & rig) and not set(c[:2]) <= rig
                    and not ({"cup"} & set(c[:2]) and tz < 50)]
            print((tx, ty, tz), "ok" if not hits else hits)
