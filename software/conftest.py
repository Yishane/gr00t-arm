from arm_io import SimArm


class RecordingSimArm(SimArm):
    """SimArm that keeps every command it was sent, in .sent."""

    def __init__(self, port=None):
        super().__init__(port)
        self.sent = []

    def send(self, angles):
        super().send(angles)
        self.sent.append(self.last)
