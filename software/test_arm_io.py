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


class FakeSerial:
    """Stands in for serial.Serial: records the DTR state when the port
    opens, and answers "?" like the firmware does."""
    opened_with_dtr = None

    def __init__(self, *a, **k):
        self.port, self.baudrate, self.timeout = None, None, None
        self.dtr = self.rts = True
        self.replies = []
        self.written = []

    def open(self):
        FakeSerial.opened_with_dtr = self.dtr

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.written.append(data)
        if data == b"?\n":
            self.replies.append(b"P 90.0,90.0,90.0,90.0,90.0\n")

    def readline(self):
        return self.replies.pop(0) if self.replies else b""

    def close(self):
        pass


def test_arm_connects_without_resetting_the_nano(monkeypatch):
    import serial
    monkeypatch.setattr(serial, "Serial", FakeSerial)
    t0 = __import__("time").perf_counter()
    arm = arm_io.Arm("COM3")
    assert FakeSerial.opened_with_dtr is False          # DTR low = no reset pulse
    assert __import__("time").perf_counter() - t0 < 1.0  # no blind 2 s wait
    arm.close()


def test_connecting_resets_the_slew_speed(monkeypatch):
    # without a reset, a slow speed set by calibrate.py would carry over
    import serial
    monkeypatch.setattr(serial, "Serial", FakeSerial)
    arm = arm_io.Arm("COM3")
    assert arm.ser.written[-1] == b"S 180\n"
    arm.close()


class ResettingSerial(FakeSerial):
    """A port whose driver resets the Nano anyway. Like the old Nano
    bootloader, it never starts the firmware if it keeps getting bytes
    while it boots; otherwise it prints READY and answers "?"."""
    BOOT = 0.3

    def open(self):
        import time
        self.boot_until = time.time() + self.BOOT
        self.boot_bytes = 0
        self.ready_sent = False

    def write(self, data):
        import time
        self.written.append(data)
        if time.time() < self.boot_until:
            self.boot_bytes += 1
        elif data == b"?\n" and self.boot_bytes <= 1:
            self.replies.append(b"P 90.0,90.0,90.0,90.0,90.0\n")

    def readline(self):
        import time
        if time.time() < self.boot_until:
            time.sleep(0.01)
            return b""
        if not self.ready_sent and self.boot_bytes <= 1:
            self.ready_sent = True
            return b"READY\n"
        return self.replies.pop(0) if self.replies else b""


def test_arm_connects_when_the_driver_resets_the_nano_anyway(monkeypatch):
    import serial
    monkeypatch.setattr(serial, "Serial", ResettingSerial)
    monkeypatch.setattr(arm_io.Arm, "BOOT_S", 1.0)
    arm = arm_io.Arm("COM3")          # SystemExit here = it never got an answer
    assert arm.ser.boot_bytes <= 1    # stayed quiet while the Nano booted
    arm.close()


class StuckSerial(ResettingSerial):
    """A Nano whose bootloader is stuck until DTR resets it again."""

    def open(self):
        self.boot_until, self.boot_bytes, self.ready_sent = 0.0, 99, True
        self.stuck = True

    @property
    def dtr(self):
        return getattr(self, "_dtr", True)

    @dtr.setter
    def dtr(self, v):
        import time
        if getattr(self, "stuck", False) and v:
            self.stuck = False                 # a DTR pulse resets it
            self.boot_until, self.boot_bytes, self.ready_sent = time.time() + self.BOOT, 0, False
        self._dtr = v


def test_arm_resets_a_stuck_nano_as_a_last_resort(monkeypatch):
    import serial
    monkeypatch.setattr(serial, "Serial", StuckSerial)
    monkeypatch.setattr(arm_io.Arm, "BOOT_S", 1.0)
    arm = arm_io.Arm("COM3")
    assert arm.ser.stuck is False
    arm.close()
