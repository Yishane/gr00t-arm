"""Talking to the hardware: the arm (over USB serial) and the camera.

Both have a simulated stand-in (--sim) so every script can run without the
arm plugged in.
"""
import sys
import time

import numpy as np

import config as C

HOME = [90.0, 90.0, 90.0, 90.0]   # servo degrees for base, shoulder, elbow, wrist
FLUSH = 2   # frames dropped by read(fresh=True)


class Arm:
    """5 servo angles in degrees: base, shoulder, elbow, wrist, gripper."""

    def __init__(self, port=None):
        import serial
        from serial.tools import list_ports
        if port is None:
            ports = [p.device for p in list_ports.comports()
                     if any(k in (p.description or "") for k in ("Arduino", "CH340", "USB"))]
            if not ports:
                raise SystemExit("No Arduino found. Plug it in or pass --port COM3.")
            port = ports[0]
        self.ser = serial.Serial(port, C.BAUD, timeout=1)
        time.sleep(2.0)                         # Uno resets when the port opens
        self.ser.reset_input_buffer()
        self.last = HOME + [C.GRIPPER_OPEN]
        print(f"arm on {port}")

    def send(self, angles):
        angles = [float(a) for a in angles]
        self.ser.write((",".join(f"{a:.1f}" for a in angles) + "\n").encode())
        self.last = angles

    def speed(self, deg_per_s):
        self.ser.write(f"S {deg_per_s:.0f}\n".encode())

    def close(self):
        self.ser.close()


class SimArm:
    """Stand-in that just remembers what it was told."""

    def __init__(self, port=None):
        self.last = HOME + [C.GRIPPER_OPEN]
        print("simulated arm")

    def send(self, angles):
        self.last = [float(a) for a in angles]

    def speed(self, deg_per_s):
        pass

    def close(self):
        pass


def camera_backend(cv2, platform=sys.platform):
    """DirectShow on Windows (opens phone-webcam apps reliably); OpenCV's
    default everywhere else, where DirectShow doesn't exist."""
    return cv2.CAP_DSHOW if platform == "win32" else cv2.CAP_ANY


class Camera:
    def __init__(self, index=0, width=640, height=480):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index, camera_backend(cv2))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep frames current (not every backend obeys)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.size = (width, height)
        if not self.cap.isOpened():
            raise SystemExit(f"Camera {index} didn't open. Try --camera 1 (or 2).")

    def read(self, fresh=False):
        """RGB uint8 frame, height x width x 3. fresh=True first drops frames
        the driver buffered while nobody was reading."""
        if fresh:
            for _ in range(FLUSH):
                self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = self.cv2.resize(frame, self.size)
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class FakeCamera:
    """Draws a moving square so recordings/exports can be tested."""

    def __init__(self, index=0, width=640, height=480):
        self.size = (width, height)
        self.t = 0

    def read(self, fresh=False):
        w, h = self.size
        img = np.full((h, w, 3), 40, np.uint8)
        x = int((np.sin(self.t / 15) * 0.4 + 0.5) * (w - 60))
        img[200:260, x:x + 60] = (220, 120, 30)
        self.t += 1
        return img

    def close(self):
        pass


def open_hardware(args):
    """Arm + camera from the common command-line flags."""
    arm = SimArm() if args.sim else Arm(args.port)
    cam = FakeCamera() if (args.sim or args.camera < 0) else Camera(args.camera)
    return arm, cam


def add_hw_args(parser):
    parser.add_argument("--port", help="serial port, e.g. COM3 (auto-detected)")
    parser.add_argument("--camera", type=int, default=1,
                        help="camera index: 0 = laptop webcam, 1 = phone usually")
    parser.add_argument("--sim", action="store_true",
                        help="no hardware: simulated arm and fake camera")
