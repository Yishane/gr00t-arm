"""
Parametric 4-axis + gripper robot arm for SG90 micro servos.

Joints: base yaw, shoulder pitch, elbow pitch, wrist pitch, gripper.
Every servo mounts the same way: it drops through a rectangular hole in a
3 mm plate and its mounting tabs rest on the plate (screwed with the 2 tab
screws that come with the servo). Each link grabs the next servo's horn
with a slot that the double-arm horn is glued into (CA / superglue).

Build:  uv run --python 3.12 --with cadquery python arm.py
Output: stl/*.stl  (print every part twice: one follower arm, one leader arm)

All dimensions in mm. If the fit test is loose/tight, change the SG90 block
below and rebuild.
"""
import math
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).parent / "stl"

# ---------------------------------------------------------------- SG90 ----
BODY_L = 22.8          # body length (along tab direction)
BODY_W = 12.3          # body width
SHAFT_FROM_END = 5.9   # output shaft centre measured from the near end of body
TAB_SPAN = 32.3        # tab tip to tab tip
TAB_HOLE_SPACING = 27.8
TAB_TO_TOP = 6.8       # tab underside to top face of body (gear side)
TAB_TO_BOTTOM = 15.9   # tab underside to bottom of body
STACK = 11.5           # tab underside to outer face of a fitted horn

CLEAR = 0.4            # clearance around the body in its pocket
PILOT = 1.7            # pilot hole for the tab screws

# --------------------------------------------------------------- horn -----
HORN_SLOT_L = 38.0     # double-arm horn, generous so any horn variant fits
HORN_SLOT_W = 6.6
HORN_SLOT_D = 1.8
HORN_ACCESS = 6.0      # hole to reach the horn screw / clear the spline hub

# -------------------------------------------------------------- general ---
PLATE = 3.0            # thickness of every servo mounting plate
LINK = 4.0             # thickness of link plates (horn slot is cut into these)
L_UPPER = 65.0         # shoulder axis -> elbow axis
L_FORE = 60.0          # elbow axis -> wrist axis
BALL = 40.0            # ball the gripper is sized for (foam or ping-pong)
BLOCK = 25.0           # second object: printed cube, in a colour that contrasts with the ball
GRIP_SERVO = (40.0, 12.0)  # gripper servo shaft, in the gripper body frame
FINGER_T = 12.0        # finger thickness
# fingertips, measured from the gripper servo shaft (10 mm past the ball
# centre, so the tips clear the desk when grabbing a resting ball)
FINGER_END = 45.0
TT_T = 4.0             # turntable disc thickness

# body centre sits this far from the shaft, along the body's long axis
BODY_OFFSET = BODY_L / 2 - SHAFT_FROM_END
POCKET_L = BODY_L + 2 * CLEAR
POCKET_W = BODY_W + 2 * CLEAR
# shoulder axis above the turntable bottom; high enough that the shoulder
# horn clears the disc when the arm points straight up
TT_SHAFT_Z = TT_T + POCKET_W / 2 + 11.0


def servo_cut(wp_plate: cq.Workplane, x: float, y: float, along_x: bool = True,
              flip: bool = False) -> cq.Workplane:
    """Cut an SG90 pocket + 2 pilot holes through a plate lying in the XY
    plane of its own workplane. (x, y) is where the output shaft goes.
    The body extends from the shaft toward -x (or +x if flip)."""
    s = 1 if flip else -1
    cx = x + s * BODY_OFFSET if along_x else x
    cy = y if along_x else y + s * BODY_OFFSET
    lx, ly = (POCKET_L, POCKET_W) if along_x else (POCKET_W, POCKET_L)
    h = TAB_HOLE_SPACING / 2
    holes = [(cx - h, cy), (cx + h, cy)] if along_x else [(cx, cy - h), (cx, cy + h)]
    cutter = cq.Workplane("XY").center(cx, cy).rect(lx, ly).extrude(200, both=True)
    for hx, hy in holes:
        cutter = cutter.union(
            cq.Workplane("XY").center(hx, hy).circle(PILOT / 2).extrude(200, both=True))
    return wp_plate.cut(cutter)


def horn_cut(part: cq.Workplane, x: float, y: float, z_face: float,
             angle: float = 0.0) -> cq.Workplane:
    """Cut a horn slot into the face at height z_face (slot goes downward into
    the part from that face) plus a through access hole, centred on (x, y)."""
    slot = (cq.Workplane("XY").workplane(offset=z_face - HORN_SLOT_D)
            .center(x, y).rect(HORN_SLOT_L, HORN_SLOT_W).extrude(HORN_SLOT_D + 1)
            .rotate((x, y, 0), (x, y, 1), angle))
    hub = (cq.Workplane("XY").workplane(offset=z_face - HORN_SLOT_D)
           .center(x, y).circle(4.2).extrude(HORN_SLOT_D + 1))
    hole = cq.Workplane("XY").center(x, y).circle(HORN_ACCESS / 2).extrude(200, both=True)
    return part.cut(slot).cut(hub).cut(hole)


# ===================================================================== parts

def base() -> cq.Workplane:
    """Box that holds the base (yaw) servo vertically, shaft up.
    Tabs rest on the top plate. Side flanges are for clamping to the desk."""
    W, D, H = 64.0, 52.0, 24.0
    wall = 2.4
    outer = cq.Workplane("XY").box(W, D, H, centered=(True, True, False))
    inner = (cq.Workplane("XY").box(W - 2 * wall, D - 2 * wall, H - PLATE,
                                    centered=(True, True, False)))
    body = outer.cut(inner)
    # clamp feet on both sides. Printed upside down (top plate on the bed), so
    # each foot is a 45 deg wedge that needs no supports.
    for sx in (-1, 1):
        x0 = sx * W / 2
        foot = (cq.Workplane("XZ", origin=(0, 12, 0))
                .polyline([(x0, 0), (x0 + sx * 22, 0), (x0 + sx * 22, 4),
                           (x0, 24)]).close().extrude(24))
        foot = foot.cut(cq.Workplane("XY").center(x0 + sx * 15, 0)
                        .circle(2.2).extrude(30))
        body = body.union(foot)
    body = servo_cut(body, 0, 0, along_x=True)
    # cable exit
    body = body.cut(cq.Workplane("XY").center(0, -D / 2).box(12, 8, 10,
                                                              centered=(True, True, False)))
    return body


def turntable() -> cq.Workplane:
    """Disc on the base servo horn carrying the shoulder bracket.
    Shoulder shaft is horizontal (along +Y), the arm swings in the XZ plane."""
    R, T = 30.0, TT_T
    disc = cq.Workplane("XY").circle(R).extrude(T)
    # horn slot on the UNDERSIDE: cut from z=0 upward
    slot = (cq.Workplane("XY").rect(HORN_SLOT_L, HORN_SLOT_W).extrude(HORN_SLOT_D))
    hub = cq.Workplane("XY").circle(4.2).extrude(HORN_SLOT_D)
    hole = cq.Workplane("XY").circle(HORN_ACCESS / 2).extrude(T)
    disc = disc.cut(slot).cut(hub).cut(hole)

    # vertical bracket plate in the XZ plane, its +Y face is where tabs rest
    z_shaft = TT_SHAFT_Z                       # shoulder axis height above disc bottom
    plate_h = z_shaft + POCKET_W / 2 + 6.0
    plate_w = TAB_SPAN + 8.0
    y_face = -4.0                              # +Y face of the bracket plate
    x_shaft = 0.0
    bracket = (cq.Workplane("XZ", origin=(0, y_face, 0))
               .center(x_shaft - BODY_OFFSET, plate_h / 2)
               .rect(plate_w, plate_h).extrude(PLATE))   # XZ extrudes toward -Y
    # pocket through the bracket (work in XZ coordinates)
    pocket = (cq.Workplane("XZ", origin=(0, y_face + 1, 0))
              .center(x_shaft - BODY_OFFSET, z_shaft).rect(POCKET_L, POCKET_W).extrude(PLATE + 2))
    pilots = cq.Workplane("XZ", origin=(0, y_face + 1, 0))
    for hx in (-TAB_HOLE_SPACING / 2, TAB_HOLE_SPACING / 2):
        pilots = pilots.union(cq.Workplane("XZ", origin=(0, y_face + 1, 0))
                              .center(x_shaft - BODY_OFFSET + hx, z_shaft)
                              .circle(PILOT / 2).extrude(PLATE + 2))
    bracket = bracket.cut(pocket).cut(pilots)
    # gussets behind the plate (servo body sticks out the back, so gussets sit
    # outboard of the tabs)
    for sx in (-1, 1):
        gx = x_shaft - BODY_OFFSET + sx * (plate_w / 2 - 1.5)
        g = (cq.Workplane("YZ", origin=(gx - 1.5, 0, 0))
             .polyline([(y_face - PLATE, T), (y_face - PLATE - 18, T),
                        (y_face - PLATE, plate_h - 4)]).close().extrude(3))
        bracket = bracket.union(g)
    return disc.union(bracket).cut(hole)


def link(length: float, width: float = 20.0) -> cq.Workplane:
    """Flat link. Proximal end (x=0) has the horn slot on its underside
    (z=0 face, faces the driving servo). Distal end (x=length) has a pocket
    for the next servo, inserted from the top (+Z) so its tabs rest on top
    and its shaft points +Z as well."""
    body = (cq.Workplane("XY").center(length / 2, 0)
            .rect(length, width).extrude(LINK)
            .edges("|Z").fillet(width / 2 - 0.5))
    # widen the distal end to carry the next servo + its tab screws
    pad_len = TAB_SPAN + 6
    pad = (cq.Workplane("XY").center(length - BODY_OFFSET, 0)
           .rect(pad_len, POCKET_W + 9).extrude(LINK)
           .edges("|Z").fillet(3))
    body = body.union(pad)
    # proximal horn slot cut up from the underside
    slot = cq.Workplane("XY").rect(HORN_SLOT_L, HORN_SLOT_W).extrude(HORN_SLOT_D)
    hub = cq.Workplane("XY").circle(4.2).extrude(HORN_SLOT_D)
    # slot runs along the link; the hub boss needs room in the rounded end
    body = body.union(cq.Workplane("XY").circle(width / 2 + 2).extrude(LINK))
    body = body.cut(slot).cut(hub)
    body = body.cut(cq.Workplane("XY").circle(HORN_ACCESS / 2).extrude(LINK))
    # distal servo pocket: body extends back toward the proximal end
    body = servo_cut(body, length, 0, along_x=True, flip=False)
    # lightening slot in the middle
    mid_l = length - HORN_SLOT_L / 2 - (POCKET_L - SHAFT_FROM_END) - 10
    if mid_l > 8:
        x0 = HORN_SLOT_L / 2 + 4
        body = body.cut(cq.Workplane("XY").center(x0 + mid_l / 2, 0)
                        .slot2D(mid_l, 7).extrude(LINK))
    return body


def gripper_body() -> cq.Workplane:
    """Attaches to the wrist horn (horn slot at origin, underside).
    Carries the gripper servo (shaft at G) and the fixed finger, which is
    raised by STACK so it lines up with the moving finger on the servo horn."""
    G = GRIP_SERVO
    inner_v = G[1] - 4 - (BALL - 3)     # inner face of the fixed finger (y)
    plate = (cq.Workplane("XY")
             .polyline([(-14, -12), (40, inner_v - 8), (66, inner_v - 8),
                        (66, inner_v + 6), (56, 26), (-14, 14)]).close()
             .extrude(LINK).edges("|Z").fillet(3))
    plate = plate.union(cq.Workplane("XY").circle(14).extrude(LINK))
    slot = cq.Workplane("XY").rect(HORN_SLOT_L, HORN_SLOT_W).extrude(HORN_SLOT_D)
    hub = cq.Workplane("XY").circle(4.2).extrude(HORN_SLOT_D)
    plate = plate.cut(slot).cut(hub)
    plate = plate.cut(cq.Workplane("XY").circle(HORN_ACCESS / 2).extrude(LINK))
    # gripper servo: shaft at G, body runs back toward the wrist (-x)
    plate = servo_cut(plate, G[0], G[1], along_x=True, flip=False)

    # fixed finger sits in the same plane as the moving jaw on the servo horn
    z0 = LINK
    # underside of the jaw plane: the jaw's horn slot floor sits on the horn
    # face, so the jaw starts HORN_SLOT_D below STACK
    z_f = LINK + STACK - HORN_SLOT_D
    post = (cq.Workplane("XY").workplane(offset=z0)
            .center(56, inner_v - 4).rect(20, 8).extrude(z_f - z0))
    finger = (cq.Workplane("XY").workplane(offset=z_f)
              .polyline([(46, inner_v - 8), (G[0] + FINGER_END, inner_v - 6),
                         (G[0] + FINGER_END, inner_v), (46, inner_v)]).close()
              .extrude(FINGER_T).edges("|Z").fillet(1.5))
    # shallow groove on the inner face so the ball self-centres
    cup = (cq.Workplane("YZ", origin=(0, 0, 0))
           .center(inner_v + BALL / 2 - 1.5, z_f + FINGER_T / 2)
           .circle(BALL / 2).extrude(200))
    return plate.union(post).union(finger.cut(cup))


def gripper_jaw() -> cq.Workplane:
    """Moving finger, glued to the gripper servo horn (slot on its underside at
    the origin). Closed position: finger parallel to the fixed one."""
    t = FINGER_T
    hub = cq.Workplane("XY").circle(11).extrude(LINK)
    slot = cq.Workplane("XY").rect(HORN_SLOT_L, HORN_SLOT_W).extrude(HORN_SLOT_D)
    hhub = cq.Workplane("XY").circle(4.2).extrude(HORN_SLOT_D)
    arm = (cq.Workplane("XY")
           .polyline([(0, -7), (FINGER_END, -4), (FINGER_END, 4), (0, 7)]).close().extrude(LINK))
    finger = (cq.Workplane("XY")
              .polyline([(18, -4), (FINGER_END, -4), (FINGER_END, 4), (18, 4)]).close().extrude(t)
              .edges("|Z").fillet(1.5))
    cup = (cq.Workplane("YZ", origin=(0, 0, 0))
           .center(-4 - BALL / 2 + 1.5, t / 2).circle(BALL / 2).extrude(200))
    jaw = hub.union(arm).union(finger.cut(cup))
    jaw = jaw.cut(slot).cut(hhub)
    jaw = jaw.cut(cq.Workplane("XY").circle(HORN_ACCESS / 2).extrude(t))
    return jaw


def fit_test() -> cq.Workplane:
    """Small coupon: one servo pocket + one horn slot. Print this first."""
    p = (cq.Workplane("XY").center(0, 0).rect(TAB_SPAN + 10, POCKET_W + 10)
         .extrude(PLATE))
    p = servo_cut(p, BODY_OFFSET, 0, along_x=True)
    h = (cq.Workplane("XY").center(0, POCKET_W / 2 + 14).rect(HORN_SLOT_L + 8, 16)
         .extrude(LINK))
    slot = (cq.Workplane("XY").center(0, POCKET_W / 2 + 14)
            .rect(HORN_SLOT_L, HORN_SLOT_W).extrude(HORN_SLOT_D))
    hub = cq.Workplane("XY").center(0, POCKET_W / 2 + 14).circle(4.2).extrude(HORN_SLOT_D)
    hole = cq.Workplane("XY").center(0, POCKET_W / 2 + 14).circle(HORN_ACCESS / 2).extrude(LINK)
    return p.union(h.cut(slot).cut(hub).cut(hole))


# ============================================================ work cell
# Everything below is positioned in the arm frame: origin on the baseplate
# top surface under the base servo shaft, +x forward, z up. The arm is
# mobiobi's Easy Robot Arm (cad/stl_opensource/easy_robot_arm); its base
# drum stands in a ring on the baseplate (the 3 BaseLeg feet aren't used).

BP_X = (-58.0, 192.0)          # baseplate extent (250 x 200, fits a 256 bed)
BP_Y = (-45.0, 155.0)
BP_T = 4.0
DRUM_D = 74.6                  # Easy Robot Arm Base.stl outer diameter
RING_GAP = 0.5                 # clearance around the drum in its ring
RING_W, RING_H = 2.5, 3.0      # ring wall and height
# ball start spots: 3x3 grid, all reachable gripper-down (software/kinematics)
SPOT_X = (85.0, 110.0, 135.0)
SPOT_Y = (0.0, 40.0, 80.0)
CUP_XY = (20.0, 110.0)         # at x = 0 the base would need just over 90 deg of yaw
CUP_R = 30.0
MAST_XY = (178.0, 55.0)
MAST_W = 16.0                  # square mast section
SOCKET_D = 18.0                # mast depth in the baseplate socket
MAST_H = 185.0                 # baseplate top -> centre of the mast's top face (no riser)
MAST_RISER = 80.0              # 15_mast_riser lifts the mast (and camera) by this much
MAST_TOP = MAST_H + MAST_RISER  # centre of the mast's top face as built
CAM_DOWN = 65.0                # phone camera looks this far below horizontal
PIN = 12.0                     # square pin joining mast and phone cradle
PIN_L = 26.0


def _engrave_cross(plate, x, y, size=10.0, w=1.2, depth=0.8):
    for dx, dy in ((size, w), (w, size)):
        plate = plate.cut(cq.Workplane("XY").workplane(offset=BP_T - depth)
                          .center(x, y).rect(dx, dy).extrude(depth + 1))
    return plate


def baseplate() -> cq.Workplane:
    """Desk plate the arm base and the camera mast both locate on, so the
    camera sees the same scene every session. Engraved: ball start spots
    A1..C3 and the cup ring. Clamp its back edge to the desk (or tape it).
    The arm's base drum drops into the raised ring; tape or hot-glue it so
    it can't turn."""
    w, d = BP_X[1] - BP_X[0], BP_Y[1] - BP_Y[0]
    cx, cy = sum(BP_X) / 2, sum(BP_Y) / 2
    plate = (cq.Workplane("XY").center(cx, cy).rect(w, d).extrude(BP_T)
             .edges("|Z").fillet(8))
    # ring that locates the arm's base drum
    r_in = DRUM_D / 2 + RING_GAP
    plate = plate.union(cq.Workplane("XY").workplane(offset=BP_T)
                        .circle(r_in + RING_W).circle(r_in).extrude(RING_H))
    # mast socket
    boss = (cq.Workplane("XY").workplane(offset=BP_T).center(*MAST_XY)
            .rect(MAST_W + 12, MAST_W + 12).extrude(SOCKET_D - 2)
            .edges("|Z").fillet(3))
    plate = plate.union(boss).cut(
        cq.Workplane("XY").workplane(offset=2).center(*MAST_XY)
        .rect(MAST_W + 0.4, MAST_W + 0.4).extrude(SOCKET_D + 2))
    # ball spots
    for i, x in enumerate(SPOT_X):
        for j, y in enumerate(SPOT_Y):
            plate = _engrave_cross(plate, x, y)
            label = (cq.Workplane("XY").workplane(offset=BP_T - 0.8)
                     .center(x + 9, y - 9)
                     .text("ABC"[i] + str(j + 1), 6, 1.8, kind="bold",
                           halign="center", valign="center"))
            plate = plate.cut(label)
    # cup ring
    ring = (cq.Workplane("XY").workplane(offset=BP_T - 0.8).center(*CUP_XY)
            .circle(CUP_R + 1.5).circle(CUP_R + 0.3).extrude(2))
    return plate.cut(ring)


def _rot_y(deg):
    return cq.Location(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), deg)


def camera_mast() -> cq.Workplane:
    """Square mast. Bottom drops into the baseplate socket, the top is cut
    square to the phone cradle's tilt and has a socket for the pin. Print
    standing up (no supports). Modelled with its foot at z = -SOCKET_D + 2,
    i.e. already in its place relative to the baseplate top."""
    beta = 90.0 - CAM_DOWN                       # cradle tilt from horizontal
    z0 = 2.0 - BP_T                               # socket floor, relative to plate top
    top = MAST_H + 20
    mast = cq.Workplane("XY").workplane(offset=z0).rect(MAST_W, MAST_W).extrude(top - z0)
    head = (cq.Workplane("XY").workplane(offset=MAST_H - 40).rect(MAST_W, MAST_W)
            .workplane(offset=8).rect(30, 30).loft()
            .union(cq.Workplane("XY").workplane(offset=MAST_H - 32).rect(30, 30)
                   .extrude(60)))
    mast = mast.union(head)
    # cut everything above the tilted top face through (0, 0, MAST_H)
    cutter = (cq.Workplane("XY").box(300, 300, 300).translate((0, 0, 150))
              .val().moved(cq.Location(cq.Vector(0, 0, MAST_H)) * _rot_y(beta)))
    mast = mast.cut(cq.Workplane().add(cutter))
    hole = (cq.Workplane("XY").box(PIN + 0.4, PIN + 0.4, PIN_L)
            .val().moved(cq.Location(cq.Vector(0, 0, MAST_H)) * _rot_y(beta)))
    mast = mast.cut(cq.Workplane().add(hole))
    return mast.edges("|Z").fillet(1.0)


def mast_riser() -> cq.Workplane:
    """Raises the camera mast by MAST_RISER (so the camera sees the whole arm):
    its foot drops into the baseplate socket like the mast's, and the mast
    drops into the socket on top. Same 0.4 mm clearance as the baseplate.
    Print upside down (the wide socket end on the bed, no supports: the
    socket's ceiling is a short bridge). Modelled in place: foot at
    z = 2 - BP_T, like camera_mast()."""
    z0 = 2.0 - BP_T                               # socket floor in the baseplate
    floor = z0 + MAST_RISER                       # where the mast's foot now sits
    top = floor + SOCKET_D
    w = MAST_W + 12                               # same boss as the baseplate's
    bar = cq.Workplane("XY").workplane(offset=z0).rect(MAST_W, MAST_W).extrude(floor - 8 - z0)
    flare = (cq.Workplane("XY").workplane(offset=floor - 8).rect(MAST_W, MAST_W)
             .workplane(offset=8).rect(w, w).loft())
    boss = cq.Workplane("XY").workplane(offset=floor).rect(w, w).extrude(SOCKET_D)
    riser = bar.union(flare).union(boss).edges("|Z").fillet(1.0)
    return riser.cut(cq.Workplane("XY").workplane(offset=floor)
                     .rect(MAST_W + 0.4, MAST_W + 0.4).extrude(SOCKET_D + 1))


def phone_cradle() -> cq.Workplane:
    """Tilted shelf for a phone used as the camera, held in LANDSCAPE with
    its rear camera near the lip (bottom). Big window so any phone's camera
    sees through. Local frame: back face on z=0 (print this way up, no
    supports), +x points down the slope toward the lip and the pin boss."""
    t, lip_h = 3.0, 10.0
    half_y = 92.0
    frame = cq.Workplane("XY").center(-45, 0).rect(90, 2 * half_y).extrude(t)
    frame = frame.cut(cq.Workplane("XY").center(-45, 0).rect(66, 2 * half_y - 24)
                      .extrude(t))
    # cross ribs so the frame doesn't flex
    for y in (-30.0, 30.0):
        frame = frame.union(cq.Workplane("XY").center(-45, y).rect(66, 6).extrude(t))
    lip = cq.Workplane("XY").center(2, 0).rect(4, 2 * half_y).extrude(t + lip_h)
    boss = (cq.Workplane("XY").center(4 + 15, 0).rect(30, 34).extrude(16)
            .edges("|Z").fillet(3))
    cradle = frame.union(lip).union(boss)
    cradle = cradle.cut(cq.Workplane("XY").center(19, 0)
                        .rect(PIN + 0.4, PIN + 0.4).extrude(PIN_L / 2 + 0.5))
    return cradle


# USB webcam (KitGody 1080p, monitor clip folded shut, screwed to the mount
# through the clip's 1/4"-20 hole). Measured by Yishane: head 91 x 28 x 25,
# lens centred; clip bar 61 long x 42 wide; hole on the folding leg, centred,
# ~17 mm from its hinge (back) end. Folded-clip thickness and the head's
# height above it are estimates; the head's own +-20 deg hinge covers them.
WEBCAM_TILT = 65.0             # camera looks this far below horizontal
CLIP_L, CLIP_W, CLIP_T = 61.0, 42.0, 14.0
HOLE_FROM_BACK = 17.0
LENS_UP, LENS_FWD = 34.0, 13.0  # lens above the plate / ahead of the hole
SCREW = 6.8                    # slot width for the 1/4"-20 screw


def webcam_mount() -> cq.Workplane:
    """Plate the folded webcam clip screws onto, on the camera mast's pin.
    Local frame = the mast's tilted top face (z = its normal, pin at the
    origin), so print it with that face on the bed (no supports needed).
    With the mast face at 90 - CAM_DOWN = 25 deg, a plate standing square to
    it points the webcam 65 deg down at the arm. The plate's camera side faces
    -x; the camera looks along -z; gravity pulls the clip toward the lip."""
    t = 5.0
    face_x = -11.8                                   # camera side of the plate
    z_front = 11.05                                  # front end of the clip
    z_top = z_front + CLIP_L + 8
    # everything starts on the bed (z = 0, the mast face), so no supports
    plate = (cq.Workplane("XY").box(t, 60, z_top, centered=(False, True, False))
             .translate((face_x, 0, 0)))
    lip = (cq.Workplane("XY").box(4, CLIP_W + 4, z_front, centered=(False, True, False))
           .translate((face_x - 4, 0, 0)))
    body = (cq.Workplane("XY").box(16 - face_x - t, 32, 40, centered=(False, True, False))
            .translate((face_x + t, 0, 0)).edges("|Z").fillet(2))
    m = plate.union(lip).union(body)
    # pin socket
    m = m.cut(cq.Workplane("XY").box(PIN + 0.4, PIN + 0.4, PIN_L / 2 + 0.5,
                                     centered=(True, True, False)))
    # screw slot: the hole sits HOLE_FROM_BACK from the clip's back end; +-10 mm of play
    z_hole = z_front + CLIP_L - HOLE_FROM_BACK
    m = m.cut(cq.Workplane("YZ").workplane(offset=face_x - 1).center(0, z_hole)
              .slot2D(20 + SCREW, SCREW, angle=90).extrude(t + 2))
    # zip-tie slots either side of the clip, as a backup
    for y in (-25.5, 25.5):
        for z in (22.0, z_top - 12):
            m = m.cut(cq.Workplane("XY").box(t + 2, 3, 6).translate((face_x + t / 2, y, z)))
    return m


def webcam_dummy() -> cq.Workplane:
    """Rough webcam (folded clip + head) in webcam_mount's frame, for checks."""
    face_x, z_front = -11.8, 11.05
    clip = (cq.Workplane("XY").box(CLIP_T, CLIP_W, CLIP_L, centered=(False, True, False))
            .translate((face_x - CLIP_T, 0, z_front)))
    z_lens = z_front + CLIP_L - HOLE_FROM_BACK - LENS_FWD
    head = cq.Workplane("XY").box(28, 91, 25).translate((face_x - LENS_UP, 0, z_lens))
    return clip.union(head)


def pin() -> cq.Workplane:
    """Square pin joining the mast top and the phone cradle. Print lying flat."""
    return (cq.Workplane("XY").box(PIN_L, PIN, PIN, centered=(True, True, False))
            .edges().chamfer(0.6))


def cup() -> cq.Workplane:
    """Drop target. Sits in the engraved ring on the baseplate."""
    h, wall = 35.0, 1.6
    c = cq.Workplane("XY").circle(CUP_R).extrude(h)
    c = c.cut(cq.Workplane("XY").workplane(offset=1.6).circle(CUP_R - wall).extrude(h))
    return c.faces(">Z").edges().fillet(0.6)


def ball_half() -> cq.Workplane:
    """Half of a 40 mm ball, only if you don't have a ping-pong ball. Print 2,
    open side down, glue the rims together with CA."""
    r, wall = BALL / 2, 1.2
    shell = cq.Workplane("XY").sphere(r).cut(cq.Workplane("XY").sphere(r - wall))
    return shell.cut(cq.Workplane("XY").box(2 * r + 2, 2 * r + 2, r + 1,
                                            centered=(True, True, False))
                     .translate((0, 0, -r - 1)))


def block() -> cq.Workplane:
    """The second object for "pick up the block". Print 1 in a colour that
    stands out from the ball and the baseplate (red or blue next to a white
    ball). Edges chamfered so it doesn't catch the fingers. Place it with one
    face pointing at the arm's base, since the jaws close sideways."""
    return (cq.Workplane("XY").box(BLOCK, BLOCK, BLOCK, centered=(True, True, False))
            .edges().chamfer(1.0))


PARTS = {
    "00_fit_test": fit_test,
    "01_base": base,
    "02_turntable": turntable,
    "03_upper_arm": lambda: link(L_UPPER),
    "04_forearm": lambda: link(L_FORE),
    "05_gripper_body": gripper_body,
    "06_gripper_jaw": gripper_jaw,
    "07_baseplate": baseplate,
    "08_camera_mast": camera_mast,
    "09_phone_cradle": phone_cradle,
    "10_pin": pin,
    "11_cup": cup,
    "12_ball_half": ball_half,
    "13_block": block,
    "14_webcam_mount": webcam_mount,
    "15_mast_riser": mast_riser,
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in PARTS.items():
        shape = fn()
        bb = shape.val().BoundingBox()
        cq.exporters.export(shape, str(OUT / f"{name}.stl"), tolerance=0.02,
                            angularTolerance=0.1)
        print(f"{name:18s} {bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f} mm"
              f"  vol {shape.val().Volume() / 1000:5.1f} cm3")
