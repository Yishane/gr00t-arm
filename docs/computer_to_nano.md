# How the computer talks to the Nano

The short version: the laptop does all the thinking, and the Nano only moves
servos. The laptop sends the Nano a line of text over USB, about 30 times a
second, with five angles in it. The Nano moves the five servos toward those
angles. That's the whole link.

GR00T never talks to the Nano directly. It runs on a cloud GPU and talks to
the laptop over the internet. The laptop then passes its answer on to the Nano.

```
 phone camera ──(webcam app)──▶ ┌────────────┐ ◀──(internet, port 5555)──▶ ┌──────────────┐
                                │   LAPTOP   │                             │ cloud GPU    │
 you type "pick up the ball" ─▶ │ run_policy │                             │ GR00T server │
                                └─────┬──────┘                             └──────────────┘
                                      │ USB cable (serial, 115200 baud)
                                      │ "90.0,90.0,90.0,90.0,125.0\n"  ×30 per second
                                      ▼
                                ┌────────────┐  5 PWM signal wires  ┌──────────────────────┐
                                │ Arduino    │ ───────────────────▶ │ servos (power from   │
                                │ Nano       │  D3 D5 D6 D9 D10     │ the 5.5 V supply)    │
                                └────────────┘                      └──────────────────────┘
```

With ACT instead of GR00T, the cloud box goes away. The model runs on the
laptop's GPU, and everything from the laptop down stays the same.

---

## 1. The USB cable: a serial port

When the Nano is plugged in, the laptop sees it as a serial port (`COM3`,
`COM5`, … on Windows). The Python code finds it on its own (`arm_io.py` looks
for a port named "Arduino", "CH340" or "USB"). Pass `--port COM5` if it picks
the wrong one.

- Speed: **115200 baud** (`BAUD` in `software/config.py`, `Serial.begin` in
  `firmware/arm/arm.ino`; both must match).
- Opening a serial port normally **resets the Nano** (the servos twitch, then
  snap to 90°). The Python scripts open it with the reset line (DTR) held off,
  so the Nano keeps running, then send `?` until it answers. If a driver resets
  it anyway, that just takes ~2 s longer. They also set the speed back to the
  default 180°/s, since a run that isn't reset keeps the last speed it was given.
- When the Nano does start (power-up, reset button, or the Arduino IDE's Serial
  Monitor), it prints `READY` and moves every servo to 90°.
- Only one program can hold the port at a time. Close the Arduino IDE's
  Serial Monitor before running the Python scripts.

## 2. The messages

Everything is plain text, one command per line:

| Laptop sends | What the Nano does |
|---|---|
| `90.0,120.5,45.0,90.0,125.0` | New target angles in degrees, in the order **base, shoulder, elbow, wrist, gripper**. |
| `S 120` | Sets the top servo speed to 120°/s (the default is 180°/s). |
| `?` | Replies `P a0,a1,a2,a3,a4` with the angles it's at right now. Handy for debugging. The Python code doesn't use it. |

A line without exactly 5 numbers is ignored. You can type these lines into
the Arduino Serial Monitor yourself (115200 baud, "Newline" line ending) to
test the arm with no Python at all.

## 3. What the Nano does with them (`firmware/arm/arm.ino`)

1. It reads characters until it gets a newline, then splits the line on commas.
2. It clamps each angle to 0–180° and stores it as that servo's **target**.
3. Every 10 ms it moves each servo a small step toward its target, at most
   180°/s (the "slew" limit). So if a bad command asks for a big jump, the
   arm swings over smoothly instead of slamming.
4. It turns each angle into a pulse width, 0° = 500 µs and 180° = 2400 µs,
   which the servo reads on its signal wire (pins D3, D5, D6, D9, D10).

The Nano doesn't know about kinematics, the camera, the ball or GR00T. It
also can't tell where the servos really are. SG90s don't report their
position, so the Nano just remembers the last angle it sent.

## 4. What the laptop does before it sends

The numbers on the wire are always **servo angles** (0–180°), not joint angles.
Where they come from depends on which script is running:

- **`teleop.py` / `autodemo.py`** (you driving, or scripted demos): these work
  in "move the gripper to this x, y, z" terms. `kinematics.py` turns that
  position into joint angles, and `kinematics.to_servo` turns those into
  servo angles using your calibration (`calibration.json`: which way each
  servo turns, and where its zero is).
- **`run_policy.py`** (GR00T or ACT driving): the model was trained on servo
  angles, so its answer already *is* servo angles. The laptop just clips each
  one to 0–180° and sends it on.

## 5. The full loop when GR00T drives

This is `run(...)` in `software/run_policy.py`:

1. Grab a photo from the phone camera (640×480).
2. Package the photo, the 5 angles it last sent, and your sentence
   ("pick up the ball").
3. Send that to the GR00T server on the cloud GPU (`policy_client.py`: a ZeroMQ
   request on port 5555, through the SSH tunnel). GR00T replies with **16
   future steps** of 5 angles each (an "action chunk").
4. Send the first **8** of those steps to the Nano, one line every 1/30 s
   (about 0.27 s total).
5. Go back to step 1 with a new photo.

It stops after 20 s (`--seconds`) or when you press Ctrl+C, then sends the
home pose and asks for the next instruction.

The "state" GR00T sees is the angles the laptop last **sent**, not measured
angles, because the servos can't report back. The recordings were made the
same way, so the model learned with the same kind of state it gets now.

Recording (`recorder.py`) uses this same link: every 1/30 s it saves the photo
and the 5 angles it just sent. Those pairs are what GR00T and ACT learn from.

## 6. Power is a separate path

The USB cable only powers the Nano's chip and carries data. The servos are
powered from the **bench supply (5.5 V, 3 A limit)**, never the Nano's 5V pin.
Five SG90s pull more current than USB can give, and the Nano would reset in
the middle of a move. The **ground of the supply must be wired to the Nano's
GND**. Without that common ground the signal pulses have no shared reference,
and the servos jitter or don't move at all.

## 7. If it doesn't move

| Symptom | Likely cause |
|---|---|
| `No Arduino found` | Cable is charge-only, or the CH340 driver is missing. Try another cable, or pass `--port`. |
| Port opens but nothing moves | Supply off, no common ground, or the Serial Monitor is holding the port. |
| Moves once, then the Nano resets | Servos are powered from the Nano. Move them to the bench supply. |
| Servos twitch or buzz | Weak supply or no common ground. On a single joint, the servo is pushing against something (check the limits). |
| Arm goes the wrong way | Calibration: rerun `calibrate.py` and fix that joint's direction. |
| Test it by hand | Serial Monitor at 115200, type `90,90,90,90,125`, then `?`. |
