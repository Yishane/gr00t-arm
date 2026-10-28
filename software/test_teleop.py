import teleop


def test_picture_fills_a_resized_window_without_stretching():
    assert teleop.fit((640, 520)) == (640, 480)             # the starting window
    assert teleop.fit((1280, 1000)) == (1280, 960)          # twice as big
    assert teleop.fit((1920, 1050)) == (1347, 1010)         # wide window: height decides
    assert teleop.fit((100, 30)) == (1, 1)                  # tiny window doesn't crash
