import os

import pytest

import calibrate
import config as C


def test_flip_keeps_an_earlier_trim(monkeypatch):
    monkeypatch.setitem(C.OFFSET, "shoulder", C.OFFSET["shoulder"] + 4)   # trim 1 4
    monkeypatch.setitem(C.SIGN, "shoulder", C.SIGN["shoulder"])
    before = C.OFFSET["shoulder"] + C.SIGN["shoulder"] * 90
    calibrate.flip("shoulder")
    assert C.OFFSET["shoulder"] + C.SIGN["shoulder"] * 90 == pytest.approx(before)
    assert C.SIGN["shoulder"] == 1


def test_tests_use_default_calibration():
    # a calibration.json from the user's arm must not leak into the tests
    assert os.environ["GR00T_ARM_DEFAULT_CALIBRATION"] == "1"
    assert C.OFFSET == {"base": 90.0, "shoulder": 180.0, "elbow": 0.0, "wrist": 90.0}
