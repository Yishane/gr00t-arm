"""MuJoCo simulation of the arm, the webcam and the work cell (plate, cup,
ball, block). Every script that takes --mujoco runs in it instead of on the
real arm: autodemo.py records demos, run_policy.py tests a trained policy,
teleop.py drives it from the keyboard.

    uv run sim.py --view                  # what the simulated webcam sees -> data/sim_view.png
    uv run sim.py --compare <photo.jpg>   # next to (and over) a real webcam frame
    uv run sim.py --viewer                # MuJoCo's 3D viewer, to look around

How it matches the real arm:
- The arm's parts are the author's 3D model (meshes in sim_assets/, made by
  cad/sim_meshes.py), on joints placed with the lengths in config.py.
- Commands are the same 5 servo angles the Nano gets. They go through the
  calibration (kinematics.from_servo) to joint angles, so an episode recorded
  here has the same numbers as one recorded on the real arm.
- Each servo: like the firmware, its target moves at most 180 deg/s, in 10 ms
  ticks; then an SG90-like position servo with limited torque follows it, so
  the arm sags a little under gravity and the gripper squeezes with limited
  force.
- The two geared fingers turn on their pins, mirrored. The calibrated
  gripper "open" angle opens each finger FINGER_OPEN_DEG; "closed" closes it to
  FINGER_CLOSED_DEG, a little narrower than the block, so it squeezes.
- The camera sits where the webcam mount puts it, 65 deg down. Its exact
  place and field of view are estimates: line them up with --compare.
- With randomize on, every new layout also changes the light, the colours and
  the camera pose a little, so a model trained here doesn't rely on the
  simulator's exact look.
"""
import argparse
import colorsys
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

import autodemo as A
import config as C
import kinematics as K
from arm_io import HOME

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")     # no screen (a cloud machine): render off-screen

ASSETS = Path(__file__).parent / "sim_assets"
DATA = Path(__file__).resolve().parent.parent / "data"

# Gripper fingers: vertical pivot pins (x, y) and the fingers' mid-height z,
# mm, in the home pose (measured from the author's STEP model).
FINGER_PIVOTS = {"finger_l": (103.5, 6.04), "finger_r": (103.5, -9.64)}
FINGER_Z = 89.25
FINGER_OPEN_DEG = 30.0      # each finger, at C.GRIPPER_OPEN
FINGER_CLOSED_DEG = 6.0     # at C.GRIPPER_CLOSED: a little narrower than the block

# webcam: fitted to a real webcam frame (2026-10-29) from where the plate's 9
# spot crosses appear (0.6 px rms), with the lens's x held at 154 mm, in
# front of the mast where the mount puts it; the other numbers come from the fit.
CAMERA_POS = (154.0, 57.9, 307.9)
CAMERA_DOWN = 70.7
CAMERA_YAW = -2.3        # deg about the vertical, + = counter-clockwise from above
CAMERA_FOVY = 39.6       # 640x480 is the middle of the webcam's sensor: narrower than its 90 deg
WIDTH, HEIGHT = 640, 480

# work cell (mm), as in cad/arm.py
PLATE_X, PLATE_Y, PLATE_T = (-58.0, 192.0), (-45.0, 155.0), 4.0
CUP_R, CUP_H, CUP_WALL = 30.0, 35.0, 1.6
BALL_R, BLOCK = 20.0, 25.0
MAST_XY, MAST_W, MAST_H = (178.0, 55.0), 16.0, 185.0
MAST_HEAD = (145.0, 272.0)  # the mast's 30 mm head + 15_mast_extension, z range (mm)
MAST_SOCKET = 28.0          # the baseplate's square boss round the mast's foot (z 0-16)
# --anywhere: objects go anywhere in this box (mm) that is clear of the fixed
# parts and that autodemo can plan a pick for
ANYWHERE_X, ANYWHERE_Y = (50.0, 180.0), (-30.0, 130.0)
# no object nearer the base axis than spot A1 (Yishane, 2026-10-30)
NEAREST = math.hypot(*A.SPOTS["A1"])
CLEAR = 22.0        # object footprint radius (ball 20, block's half diagonal 17.7) + margin

# SG90: about 0.18 N m stall torque
SERVO_KP, SERVO_TORQUE, ARMATURE = 3.0, 0.18, 5e-4
TIMESTEP, TICK = 0.002, 0.010          # physics step; firmware servo update
KNOCK_MM = 15.0      # the other object moved further than this: knocked
# Shadows: on (clean on the laptop GPU), as the simulated demos were recorded.
# A model must be tested with the same look it was trained on: SIM_SHADOWS=0
# turns them off, for a model trained on demos recorded without shadows. Always
# off with software rendering on a screenless cloud machine (MUJOCO_GL=egl),
# where the shadow map speckles.
SHADOWS = os.environ.get("SIM_SHADOWS") != "0" and os.environ.get("MUJOCO_GL") != "egl"

BLACK = "0.06 0.06 0.06 1"
# --ball-colour: one ball colour for every layout (default: a new one each time)
BALL_COLOURS = {"orange": (1.0, 0.42, 0.08), "red": (0.85, 0.08, 0.08),
                "yellow": (1.0, 0.85, 0.1), "blue": (0.1, 0.3, 0.9),
                "pink": (1.0, 0.4, 0.7), "purple": (0.5, 0.2, 0.75)}
COLOURS = {"plate": (0.78, 0.78, 0.76), "ball": (1.0, 0.42, 0.08),
           "block": (0.10, 0.55, 0.20), "desk": (0.55, 0.45, 0.35)}


def _m(*v):
    """mm -> 'x y z' in metres."""
    return " ".join(f"{a / 1000:.6g}" for a in v)


# collision groups: arm parts collide with the plate, cup, mast and objects but
# not with each other; the gripper (bit 4) also hits the base drum, which the
# upper arm's convex shape would wrongly rub against
ARM, GRIPPER = 1, 5


def _arm_mesh(name, contype=ARM):
    return (f'<geom type="mesh" mesh="{name}" material="pla_black" contype="{contype}" '
            f'conaffinity="0" condim="4" friction="1 0.01 0.001"/>')


def model_xml():
    sh = C.SHOULDER_HEIGHT
    wl = FINGER_PIVOTS["finger_l"]
    wr = FINGER_PIVOTS["finger_r"]
    gx, gz = C.L_FORE, sh + C.L_UPPER                # wrist axis in the home pose
    meshes = ["drum", "base_cap", "turntable", "upper_arm", "forearm", "gripper_trunk",
              "gripper_mount", "gripper_cap", "finger_pin_l", "finger_pin_r",
              "finger_l", "finger_r", "baseplate", "cup"]
    n, cx, cy, r = 32, *A.CUP, CUP_R - CUP_WALL / 2      # cup wall: n boxes round a circle
    cup_walls = "\n".join(
        f'<geom type="box" size="{_m(CUP_WALL / 2, math.pi * CUP_R / n + 0.4, CUP_H / 2)}" '
        f'pos="{_m(cx + r * math.cos(a), cy + r * math.sin(a), CUP_H / 2)}" '
        f'euler="0 0 {a:.5f}" group="3" contype="2" conaffinity="1"/>'
        for a in (2 * math.pi * i / n for i in range(n)))
    crosses = "\n".join(
        f'<geom type="box" size="{_m(5, 0.6, 0.05)}" pos="{_m(x, y, 0.05)}" euler="0 0 {e}" '
        f'material="engraving" contype="0" conaffinity="0"/>'
        for x, y in A.SPOTS.values() for e in ("0", "1.5708"))
    pw, pd = PLATE_X[1] - PLATE_X[0], PLATE_Y[1] - PLATE_Y[0]
    px, py = (PLATE_X[0] + PLATE_X[1]) / 2, (PLATE_Y[0] + PLATE_Y[1]) / 2
    # camera axes: x = image right (+y), y = image up; it looks along its -z
    # turned CAMERA_YAW about the vertical (positive = counter-clockwise from above)
    down, yaw = math.radians(CAMERA_DOWN), math.radians(CAMERA_YAW)
    cam_x = f"{-math.sin(yaw):.5f} {math.cos(yaw):.5f} 0"
    cam_y = (f"{-math.sin(down) * math.cos(yaw):.5f} {-math.sin(down) * math.sin(yaw):.5f} "
             f"{math.cos(down):.5f}")
    return f"""
<mujoco model="gr00t_arm">
  <compiler angle="radian" meshdir="{ASSETS.as_posix()}" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    {"".join(f'<mesh name="{m}" file="{m}.stl" maxhullvert="32"/>' for m in meshes)}
    <material name="pla_black" rgba="{BLACK}" specular="0.3" shininess="0.3"/>
    <material name="plate" rgba="{' '.join(map(str, COLOURS['plate']))} 1" specular="0.1"/>
    <material name="engraving" rgba="0.62 0.62 0.6 1"/>
    <material name="ball" rgba="{' '.join(map(str, COLOURS['ball']))} 1" specular="0.05"/>
    <material name="block" rgba="{' '.join(map(str, COLOURS['block']))} 1" specular="0.2"/>
    <material name="desk" rgba="{' '.join(map(str, COLOURS['desk']))} 1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0.05 0.05 0.6" dir="0.1 0 -1" diffuse="0.7 0.7 0.7"
           castshadow="{str(SHADOWS).lower()}"/>
    <camera name="front" pos="{_m(*CAMERA_POS)}" xyaxes="{cam_x} {cam_y}" fovy="{CAMERA_FOVY}"/>
    <geom name="desk" type="plane" size="1 1 0.01" pos="{_m(0, 0, -PLATE_T)}" material="desk"
          contype="2" conaffinity="1"/>
    <geom name="plate_look" type="mesh" mesh="baseplate" material="plate" contype="0" conaffinity="0"/>
    <geom name="plate" type="box" size="{_m(pw / 2, pd / 2, PLATE_T / 2)}" pos="{_m(px, py, -PLATE_T / 2)}"
          group="3" contype="2" conaffinity="1" friction="0.8 0.01 0.001"/>
    {crosses}
    <geom name="cup_look" type="mesh" mesh="cup" pos="{_m(*A.CUP, 0)}" material="pla_black" contype="0" conaffinity="0"/>
    {cup_walls}
    <geom name="cup_floor" type="cylinder" size="{_m(CUP_R, 0.8)}" pos="{_m(*A.CUP, 0.8)}" group="3"
          contype="2" conaffinity="1"/>
    <geom name="mast" type="box" size="{_m(MAST_W / 2, MAST_W / 2, MAST_H / 2)}" pos="{_m(*MAST_XY, MAST_H / 2)}"
          material="pla_black" contype="2" conaffinity="1"/>
    <geom name="mast_head" type="box" size="{_m(15, 15, (MAST_HEAD[1] - MAST_HEAD[0]) / 2)}"
          pos="{_m(*MAST_XY, sum(MAST_HEAD) / 2)}" material="pla_black" contype="2" conaffinity="1"/>
    <geom name="mast_socket" type="box" size="{_m(MAST_SOCKET / 2, MAST_SOCKET / 2, 8)}"
          pos="{_m(*MAST_XY, 8)}" group="3" contype="2" conaffinity="1"/>
    <geom type="mesh" mesh="drum" material="pla_black" contype="0" conaffinity="0"/>
    <geom type="mesh" mesh="base_cap" material="pla_black" contype="0" conaffinity="0"/>
    <geom name="drum" type="cylinder" size="{_m(39, 20.4)}" pos="{_m(0, 0, 20.4)}" group="3"
          contype="2" conaffinity="6"/>
    <body name="turntable">
      <inertial pos="0 0 0.045" mass="0.025" diaginertia="1e-5 1e-5 1e-5"/>
      <joint name="base" type="hinge" axis="0 0 1" armature="{ARMATURE}" damping="0.002"/>
      {_arm_mesh("turntable")}
      <body name="upper_arm" pos="{_m(0, 0, sh)}">
        <inertial pos="{_m(C.L_UPPER / 2, 0, 0)}" mass="0.020" diaginertia="2e-6 7e-6 7e-6"/>
        <joint name="shoulder" type="hinge" axis="0 -1 0" armature="{ARMATURE}" damping="0.002"/>
        {_arm_mesh("upper_arm")}
        <body name="forearm" pos="{_m(C.L_UPPER, 0, 0)}">
          <inertial pos="{_m(C.L_FORE / 2, 0, 0)}" mass="0.020" diaginertia="2e-6 7e-6 7e-6"/>
          <joint name="elbow" type="hinge" axis="0 -1 0" armature="{ARMATURE}" damping="0.002"/>
          {_arm_mesh("forearm")}
          <body name="gripper" pos="{_m(C.L_FORE, 0, 0)}">
            <inertial pos="{_m(35, 0, -15)}" mass="0.025" diaginertia="3e-6 9e-6 9e-6"/>
            <joint name="wrist" type="hinge" axis="0 -1 0" armature="{ARMATURE}" damping="0.002"/>
            {_arm_mesh("gripper_trunk", GRIPPER)}
            {_arm_mesh("gripper_mount", GRIPPER)}
            {_arm_mesh("gripper_cap", GRIPPER)}
            {_arm_mesh("finger_pin_l", 0)}
            {_arm_mesh("finger_pin_r", 0)}
            <site name="tool" pos="{_m(C.L_TOOL, C.LATERAL, C.TOOL_UP)}" size="0.002" group="5"/>
            <body name="finger_l" pos="{_m(wl[0] - gx, wl[1], FINGER_Z - gz)}">
              <inertial pos="0.02 -0.003 0" mass="0.003" diaginertia="2e-7 1e-6 1e-6"/>
              <joint name="finger_l" type="hinge" axis="0 0 1" range="0 0.7" armature="2e-4" damping="0.001"/>
              {_arm_mesh("finger_l", GRIPPER)}
            </body>
            <body name="finger_r" pos="{_m(wr[0] - gx, wr[1], FINGER_Z - gz)}">
              <inertial pos="0.02 0.003 0" mass="0.003" diaginertia="2e-7 1e-6 1e-6"/>
              <joint name="finger_r" type="hinge" axis="0 0 -1" range="0 0.7" armature="2e-4" damping="0.001"/>
              {_arm_mesh("finger_r", GRIPPER)}
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="ball" pos="{_m(*A.SPOTS["C1"], BALL_R)}">
      <freejoint name="ball"/>
      <geom type="sphere" size="{_m(BALL_R)}" mass="0.006" material="ball" contype="3" conaffinity="3"
            condim="6" friction="1 0.01 0.003"/>
    </body>
    <body name="block" pos="{_m(*A.SPOTS["A3"], BLOCK / 2)}">
      <freejoint name="block"/>
      <geom type="box" size="{_m(BLOCK / 2, BLOCK / 2, BLOCK / 2)}" mass="0.006" material="block"
            contype="3" conaffinity="3" condim="4" friction="1 0.01 0.001"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="world" body2="turntable"/>
    <exclude body1="finger_l" body2="finger_r"/>
  </contact>
  <equality>
    <joint joint1="finger_r" joint2="finger_l" polycoef="0 1 0 0 0"/>
  </equality>
  <actuator>
    {"".join(f'<position name="{j}" joint="{j}" kp="{SERVO_KP}" dampratio="1" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>'
             for j in ("base", "shoulder", "elbow", "wrist"))}
    <position name="gripper" joint="finger_l" kp="1.5" dampratio="1" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>
  </actuator>
</mujoco>
"""


def clear_of_fixed_parts(xy):
    """True if an object at xy (mm) is on the plate and doesn't touch the base
    drum, the cup or the mast's socket."""
    x, y = xy
    mx, my = MAST_XY
    dx = max(abs(x - mx) - MAST_SOCKET / 2, 0.0)
    dy = max(abs(y - my) - MAST_SOCKET / 2, 0.0)
    return (PLATE_X[0] + CLEAR <= x <= PLATE_X[1] - CLEAR
            and PLATE_Y[0] + CLEAR <= y <= PLATE_Y[1] - CLEAR
            and math.hypot(x, y) >= 39 + CLEAR
            and math.dist(xy, A.CUP) >= CUP_R + CLEAR
            and math.hypot(dx, dy) >= CLEAR)


def ball_rgb(colour):
    """--ball-colour: a BALL_COLOURS name or "#rrggbb" -> (r, g, b) 0..1; None -> None."""
    if colour is None:
        return None
    c = colour.strip().lower()
    if c in BALL_COLOURS:
        return BALL_COLOURS[c]
    if len(c) == 7 and c.startswith("#"):
        try:
            return tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
        except ValueError:
            pass
    raise SystemExit(f"--ball-colour {colour!r}: use one of {', '.join(BALL_COLOURS)} or #rrggbb")


def finger_angle(servo_deg):
    """Gripper servo degrees -> each finger's opening (degrees), through the
    calibrated open / closed angles."""
    f = (servo_deg - C.GRIPPER_CLOSED) / (C.GRIPPER_OPEN - C.GRIPPER_CLOSED)
    return FINGER_CLOSED_DEG + f * (FINGER_OPEN_DEG - FINGER_CLOSED_DEG)


def controls(servo):
    """5 servo degrees -> the 5 actuator targets (joint radians)."""
    return [*K.from_servo(servo[:4]), math.radians(finger_angle(servo[4]))]


class World:
    """The simulated work cell. realtime: physics runs in a background thread
    at the speed of the wall clock (like the real arm). Otherwise it only
    moves when told to (advance(), or every arm.send() advances 1/30 s), which
    is faster when nothing has to wait for a model."""

    def __init__(self, realtime=True, randomize=False, seed=None, ball_colour=None):
        """ball_colour: a BALL_COLOURS name or "#rrggbb" to keep the ball that
        colour; None = a new colour every layout (with randomize)."""
        import mujoco
        self.ball_rgb = ball_rgb(ball_colour)
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(model_xml())
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.rng = random.Random(seed)
        self.randomize = randomize
        self.speed = 180.0                       # deg/s, the firmware's default
        m = self.model
        self.cam = m.camera("front").id
        self._nominal = {"cam_pos": m.cam_pos[self.cam].copy(),
                         "cam_quat": m.cam_quat[self.cam].copy(),
                         "light_dir": m.light_dir[0].copy(),
                         "rgba": {k: m.mat(k).rgba.copy() for k in COLOURS}}
        if self.ball_rgb is not None:
            self._nominal["rgba"]["ball"][:3] = self.ball_rgb
            m.mat("ball").rgba[:3] = self.ball_rgb
        self._qadr = {j: m.joint(j).qposadr[0] for j in
                      ("base", "shoulder", "elbow", "wrist", "finger_l", "finger_r", "ball", "block")}
        self._vadr = {j: m.joint(j).dofadr[0] for j in ("ball", "block")}
        self.place(A.SPOTS["C1"], A.SPOTS["A3"])
        self.arm = MujocoArm(self, lockstep=not realtime)
        self.camera = MujocoCamera(self)
        self._stop = threading.Event()
        self._thread = None
        self._checker = None                     # hidden world for trying layouts
        if realtime:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------ physics
    def _servo_tick(self):
        """What the firmware does every 10 ms: move each output toward its
        target by at most speed * 10 ms."""
        d = self.speed * TICK
        self.out = [o + max(-d, min(d, t - o)) for o, t in zip(self.out, self.target)]
        self.data.ctrl[:] = controls(self.out)

    def advance(self, seconds):
        """Run the physics for this long (sim time)."""
        with self.lock:
            for _ in range(max(1, round(seconds / TIMESTEP))):
                if self.data.time >= self._next_tick:
                    self._next_tick += TICK
                    self._servo_tick()
                self.mj.mj_step(self.model, self.data)

    def _run(self):
        while not self._stop.is_set():
            with self.lock:
                behind = self._sim0 + time.perf_counter() - self._wall0 - self.data.time
                if behind > 0.1:      # can't keep up: let the clock slip, don't race
                    self._sim0 -= behind - TIMESTEP
                    behind = TIMESTEP
                if behind > 0:
                    self.advance(behind)
            time.sleep(0.002)

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.camera.close()
        if self._checker:
            self._checker.close()

    # ------------------------------------------------------------ layout
    def place(self, ball_xy, block_xy, block_yaw=None):
        """Arm home, gripper open, ball and block (x, y in mm) on the plate.
        block_yaw (deg): default turns a face toward the arm's base, as the
        guide says to place it."""
        with self.lock:
            if self.randomize:
                self._randomize()
            d, q = self.data, self._qadr
            self.mj.mj_resetData(self.model, d)
            self.target = list(HOME) + [C.GRIPPER_OPEN]
            self.out = list(self.target)
            for j, a in zip(("base", "shoulder", "elbow", "wrist"), K.from_servo(HOME)):
                d.qpos[q[j]] = a
            d.qpos[q["finger_l"]] = d.qpos[q["finger_r"]] = math.radians(FINGER_OPEN_DEG)
            if block_yaw is None:
                block_yaw = math.degrees(math.atan2(block_xy[1], block_xy[0]))
                if self.randomize:
                    block_yaw += self.rng.uniform(-10, 10)
            yaw = math.radians(block_yaw)
            d.qpos[q["ball"]:q["ball"] + 7] = [ball_xy[0] / 1000, ball_xy[1] / 1000, BALL_R / 1000, 1, 0, 0, 0]
            d.qpos[q["block"]:q["block"] + 7] = [block_xy[0] / 1000, block_xy[1] / 1000, BLOCK / 2000,
                                                 math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            d.ctrl[:] = controls(self.out)
            self._next_tick = 0.0
            self.mj.mj_forward(self.model, d)
            self.advance(0.3)                    # let everything settle
            self.start = {k: self.pos(k) for k in ("ball", "block")}
            self._sim0, self._wall0 = self.data.time, time.perf_counter()

    def random_layout(self, offset=0.0, target=None, anywhere=False):
        """(target, target xy, other xy) for one episode. Default: spots as
        autodemo.layout() picks them, each moved up to `offset` mm in a random
        direction. anywhere: both objects anywhere in ANYWHERE_X/Y. Always: at
        least MIN_GAP apart, no nearer the base than A1 (NEAREST), clear of the drum, cup, mast and plate edge, in
        the camera's picture, and the scripted pick works there (tried first
        in a hidden copy of the world), so every layout is one the arm can
        really do and the camera can see."""
        for _ in range(400):
            t, spot, other_spot = A.layout()
            t = target or t
            if anywhere:
                xy = [(self.rng.uniform(*ANYWHERE_X), self.rng.uniform(*ANYWHERE_Y))
                      for _ in range(2)]
            else:
                xy = [self._nudge(A.SPOTS[s], offset) for s in (spot, other_spot)]
            if (math.dist(*xy) >= A.MIN_GAP and all(map(clear_of_fixed_parts, xy))
                    and all(math.hypot(*q) >= NEAREST for q in xy)
                    and all(self.in_view((*q, 15.0)) for q in xy)
                    and self._pick_works(t, *xy)):
                return t, xy[0], xy[1]
        raise SystemExit(f"no reachable layout with --offset {offset:g}: try a smaller one")

    def in_view(self, p, margin=20):
        """Is the point p (x, y, z in mm) inside the camera's picture, at
        least `margin` pixels in from the edge? (The camera without the
        random changes.)"""
        rot = np.zeros(9)
        self.mj.mju_quat2Mat(rot, self._nominal["cam_quat"])
        x, y, z = rot.reshape(3, 3).T @ (np.array(p) / 1000 - self._nominal["cam_pos"])
        if z >= 0:                                   # behind the camera (it looks along -z)
            return False
        f = (HEIGHT / 2) / math.tan(math.radians(CAMERA_FOVY) / 2)
        u, v = WIDTH / 2 + f * x / -z, HEIGHT / 2 - f * y / -z
        return margin <= u <= WIDTH - margin and margin <= v <= HEIGHT - margin

    def _pick_works(self, target, xy, other_xy):
        """Does autodemo's pick work for this layout? Tried in a second,
        hidden world (physics only, ~0.1 s), so this one isn't disturbed."""
        try:
            A.plan(xy, target)
        except ValueError:
            return False
        if self._checker is None:
            self._checker = World(realtime=False)
        c = self._checker
        c.place_for(target, xy, other_xy)
        A.run(c.arm, _NoCamera(), xy, target)
        return c.success(target)

    def place_for(self, target, target_xy, other_xy):
        other = next(n for n in A.OBJECTS if n != target)
        xy = {target: target_xy, other: other_xy}
        self.place(xy["ball"], xy["block"])

    def _nudge(self, xy, offset):
        r = offset * math.sqrt(self.rng.random())
        a = self.rng.uniform(0, 2 * math.pi)
        return (xy[0] + r * math.cos(a), xy[1] + r * math.sin(a))

    def _randomize(self):
        """A little different light, colours and camera every layout."""
        m, n, u = self.model, self._nominal, self.rng.uniform
        m.cam_pos[self.cam] = n["cam_pos"] + [u(-0.004, 0.004) for _ in range(3)]
        tilt = math.radians(u(-1.5, 1.5)) / 2
        q = np.zeros(4)
        self.mj.mju_mulQuat(q, n["cam_quat"], np.array([math.cos(tilt), math.sin(tilt), 0, 0]))
        m.cam_quat[self.cam] = q
        m.cam_fovy[self.cam] = CAMERA_FOVY + u(-1.5, 1.5)
        m.light_dir[0] = n["light_dir"] + [u(-0.3, 0.3), u(-0.3, 0.3), 0]
        m.light_diffuse[0] = [u(0.5, 0.9)] * 3
        for k, rgba in n["rgba"].items():
            j = 0.15 if k == "desk" else 0.04
            m.mat(k).rgba[:3] = np.clip(rgba[:3] + [u(-j, j) for _ in range(3)], 0, 1)
        if self.ball_rgb is not None:            # one colour, shaded a little (above)
            return
        # a new ball colour every layout, as on the real arm: any strong colour
        # except green (the block's) - hue 170..440 deg wraps past red to 80
        hue = (u(170, 440) % 360) / 360
        m.mat("ball").rgba[:3] = colorsys.hsv_to_rgb(hue, u(0.65, 1.0), u(0.7, 1.0))

    # ------------------------------------------------------------ checks
    def pos(self, name):
        """(x, y, z) of a body in mm."""
        return tuple(1000 * self.data.body(name).xpos)

    def tool(self):
        return tuple(1000 * self.data.site("tool").xpos)

    def in_cup(self, name):
        x, y, z = self.pos(name)
        return math.dist((x, y), A.CUP) < CUP_R - CUP_WALL and z < CUP_H

    def result(self, target):
        """'ok' (the right object in the cup, the other where it was),
        'knocked' (right object, but the other was pushed), 'wrong object'
        or 'missed'."""
        other = next(n for n in A.OBJECTS if n != target)
        with self.lock:
            if self.in_cup(other):
                return "wrong object"
            if not self.in_cup(target):
                return "missed"
            moved = math.dist(self.pos(other)[:2], self.start[other][:2])
            return "knocked" if moved > KNOCK_MM else "ok"

    def success(self, target):
        return self.result(target) == "ok"


class _NoCamera:
    """For trying layouts: autodemo.run reads a frame each step; nothing looks."""

    def read(self, fresh=False):
        return None


class MujocoArm:
    """Same interface as arm_io.Arm, driving the simulated servos."""

    def __init__(self, world, lockstep=False):
        self.world = world
        self.lockstep = lockstep
        self.last = list(world.target)

    def send(self, angles):
        angles = [float(a) for a in angles]
        with self.world.lock:
            self.world.target = [min(max(a, 0.0), 180.0) for a in angles]   # as the firmware clamps
        self.last = angles
        if self.lockstep:
            self.world.advance(1.0 / C.FPS)

    def speed(self, deg_per_s):
        self.world.speed = float(deg_per_s)

    def wait(self, seconds):
        """Let this much time pass (a real sleep, or sim time in lockstep)."""
        if self.lockstep:
            self.world.advance(seconds)
        else:
            time.sleep(seconds)

    def close(self):
        self.world.close()


class MujocoCamera:
    """Same interface as arm_io.Camera: RGB 640x480 frames, rendered."""

    def __init__(self, world):
        self.world = world
        self.renderer = None

    def read(self, fresh=False):
        w = self.world
        if self.renderer is None:          # made on first use, in the reading thread
            self.renderer = w.mj.Renderer(w.model, HEIGHT, WIDTH)
        with w.lock:
            self.renderer.update_scene(w.data, camera="front")
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", action="store_true", help="save the simulated webcam view")
    ap.add_argument("--compare", type=Path, metavar="PHOTO",
                    help="a real webcam frame (e.g. frame 0 of a recorded episode)")
    ap.add_argument("--ball", default="C1", choices=sorted(A.SPOTS))
    ap.add_argument("--block", default="A3", choices=sorted(A.SPOTS))
    ap.add_argument("--viewer", action="store_true", help="open MuJoCo's 3D viewer")
    a = ap.parse_args()
    world = World(realtime=False)
    world.place(A.SPOTS[a.ball], A.SPOTS[a.block])
    if a.viewer:
        import mujoco.viewer
        mujoco.viewer.launch(world.model, world.data)
        return
    img = cv2.cvtColor(world.camera.read(), cv2.COLOR_RGB2BGR)
    DATA.mkdir(exist_ok=True)
    if a.compare:
        real = cv2.imread(str(a.compare))
        if real is None:
            raise SystemExit(f"can't read {a.compare}")
        real = cv2.resize(real, (WIDTH, HEIGHT))
        out = np.vstack([np.hstack([real, img]),
                         np.hstack([cv2.addWeighted(real, 0.5, img, 0.5, 0),
                                    np.zeros_like(img)])])
        path = DATA / "sim_compare.png"
        cv2.imwrite(str(path), out)
        print(f"wrote {path}: real | sim on top, 50/50 blend below. Put the ball on "
              f"{a.ball} and the block on {a.block} in the real photo, arm at home.")
    else:
        path = DATA / "sim_view.png"
        cv2.imwrite(str(path), img)
        print(f"wrote {path}")
    world.close()


if __name__ == "__main__":
    main()
