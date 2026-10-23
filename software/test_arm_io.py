import cv2
import numpy as np

import arm_io


class StubCap:
    """Stands in for cv2.VideoCapture and remembers what was called."""

    def __init__(self):
        self.calls = []

    def grab(self):
        self.calls.append("grab")
        return True

    def read(self):
        self.calls.append("read")
        return True, np.zeros((6, 8, 3), np.uint8)


def test_backend_is_dshow_only_on_windows():
    assert arm_io.camera_backend(cv2, "win32") == cv2.CAP_DSHOW
    assert arm_io.camera_backend(cv2, "linux") == cv2.CAP_ANY
    assert arm_io.camera_backend(cv2, "darwin") == cv2.CAP_ANY


def test_fresh_read_flushes_buffered_frames():
    cam = arm_io.Camera.__new__(arm_io.Camera)        # skip opening a device
    cam.cv2, cam.size = cv2, (8, 6)
    cam.cap = StubCap()
    cam.read(fresh=True)
    assert cam.cap.calls == ["grab"] * arm_io.FLUSH + ["read"]
    cam.cap.calls.clear()
    cam.read()
    assert cam.cap.calls == ["read"]


def test_fake_camera_accepts_fresh():
    assert arm_io.FakeCamera().read(fresh=True).shape == (480, 640, 3)
