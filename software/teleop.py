"""Drive the arm with the keyboard and record demonstrations.

    uv run teleop.py                 # phone camera (index 1), arm auto-detected
    uv run teleop.py --camera 0      # laptop webcam
    uv run teleop.py --sim           # no hardware
    uv run teleop.py --mujoco        # drive the simulated arm (sim.py)

Keys (click the window first):
    W / S      forward / back            A / D   left / right
    R / F      up / down                 Q / E   tilt gripper up / down
    SPACE      open / close gripper      H       go home
    ENTER      start / stop recording    BACKSPACE  discard current recording
    T          switch the instruction saved with the next recording
               ("pick up the ball" / "pick up the block")
    ESC        quit
A gamepad works too if one is plugged in (left stick xy, triggers z, A = gripper).
"""
import argparse
import math
import time

import numpy as np
import pygame

import config as C
import kinematics as K
from arm_io import HOME, add_hw_args, open_hardware
from recorder import TASKS, Recorder

SPEED = 90.0          # mm/s
PITCH_SPEED = 60.0    # deg/s


def home_target():
    x, y, z, pitch = K.fk(*K.from_servo(HOME))
    return [x, y, z, pitch]


STATUS_H = 40   # px under the picture for the status line


def fit(window):
    """Size to draw the 640x480 picture in a window of this size, keeping its
    shape and leaving room for the status line. Only the display is scaled;
    recorded frames stay 640x480."""
    s = min(window[0] / 640, (window[1] - STATUS_H) / 480)
    return max(1, round(640 * s)), max(1, round(480 * s))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hw_args(ap)
    ap.add_argument("--dataset", help="where recordings go: data/raw/DATASET "
                    "(default pick_ball, or pick_ball_mujoco with --mujoco)")
    ap.add_argument("--task", default=TASKS["ball"],
                    help="instruction stored with demos at the start (T switches it)")
    args = ap.parse_args()

    arm, cam = open_hardware(args)
    dataset = args.dataset or ("pick_ball_mujoco" if args.mujoco else "pick_ball")
    rec = Recorder(dataset, args.task, C.FPS, "teleop", sim=args.mujoco)
    pygame.init()
    screen = pygame.display.set_mode((640, 480 + STATUS_H), pygame.RESIZABLE)
    pygame.display.set_caption("gr00t-arm teleop")
    font = pygame.font.SysFont("consolas", 16)
    pad = pygame.joystick.Joystick(0) if pygame.joystick.get_count() else None

    target = home_target()
    grip_closed = False
    servo = HOME + [C.GRIPPER_OPEN]
    arm.send(servo)
    msg = ""
    dt = 1.0 / C.FPS
    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    grip_closed = not grip_closed
                elif ev.key == pygame.K_h:
                    target = home_target()
                elif ev.key == pygame.K_t and not rec.recording:
                    names = list(TASKS.values())
                    rec.task = names[(names.index(rec.task) + 1) % len(names)
                                     if rec.task in names else 0]
                elif ev.key == pygame.K_RETURN:
                    rec.stop() if rec.recording else rec.start()
                elif ev.key == pygame.K_BACKSPACE and rec.recording:
                    rec.stop(keep=False)
            elif ev.type == pygame.JOYBUTTONDOWN and ev.button == 0:
                grip_closed = not grip_closed

        keys = pygame.key.get_pressed()
        v = np.zeros(4)
        v[0] = keys[pygame.K_w] - keys[pygame.K_s]
        v[1] = keys[pygame.K_a] - keys[pygame.K_d]
        v[2] = keys[pygame.K_r] - keys[pygame.K_f]
        v[3] = keys[pygame.K_q] - keys[pygame.K_e]
        if pad:
            dead = lambda a: a if abs(a) > 0.15 else 0.0
            v[0] += -dead(pad.get_axis(1))
            v[1] += -dead(pad.get_axis(0))
            if pad.get_numaxes() > 5:
                v[2] += (pad.get_axis(5) - pad.get_axis(4)) / 2
        new = list(target)
        for i in range(3):
            new[i] += v[i] * SPEED * dt
        new[3] += math.radians(v[3] * PITCH_SPEED * dt)
        sol = K.solve(*new)
        if sol is not None:
            target = new
            msg = ""
        elif v.any():
            msg = "can't reach that"
            sol = K.solve(*target)
        if sol is not None:
            grip = C.GRIPPER_CLOSED if grip_closed else C.GRIPPER_OPEN
            action = sol + [grip]
            frame = cam.read()
            if rec.recording:
                rec.add(frame, servo, action)      # state = what the arm was told last step
            servo = action
            arm.send(servo)
        else:
            frame = cam.read()

        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.fill((20, 20, 20))
        win_w = screen.get_width()
        pic_w, pic_h = fit(screen.get_size())
        left = (win_w - pic_w) // 2
        screen.blit(pygame.transform.scale(surf, (pic_w, pic_h)), (left, 0))
        x, y, z, p = target
        status = (f"x{x:6.0f} y{y:6.0f} z{z:5.0f} pitch{math.degrees(p):5.0f}  "
                  f"grip {'closed' if grip_closed else 'open'}  {msg}")
        screen.blit(font.render(status, True, (230, 230, 230)), (8, pic_h + 12))
        screen.blit(font.render(f'task: "{rec.task}"', True, (230, 230, 230)), (left + 8, 8))
        if rec.recording:
            pygame.draw.circle(screen, (230, 40, 40), (left + pic_w - 20, 20), 10)
            screen.blit(font.render(f"REC {len(rec.rows)}", True, (230, 40, 40)),
                        (left + pic_w - 110, 12))
        pygame.display.flip()
        clock.tick(C.FPS)

    rec.stop()
    arm.close()
    cam.close()
    pygame.quit()


if __name__ == "__main__":
    main()
