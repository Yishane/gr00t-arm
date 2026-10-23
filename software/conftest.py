import os

# before anything imports config: the tests use the default calibration,
# not the real arm's calibration.json
os.environ["GR00T_ARM_DEFAULT_CALIBRATION"] = "1"

from arm_io import SimArm  # noqa: E402


class RecordingSimArm(SimArm):
    """SimArm that keeps every command it was sent, in .sent."""

    def __init__(self, port=None):
        super().__init__(port)
        self.sent = []

    def send(self, angles):
        super().send(angles)
        self.sent.append(self.last)
