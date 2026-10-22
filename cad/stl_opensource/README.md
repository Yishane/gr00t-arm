# Open-source gripper STLs

Third-party gripper downloaded as-is (not designed by us). The original
gripper in `cad/stl/05_gripper_body.stl` + `06_gripper_jaw.stl` is untouched.

## simple_sg90_claw/

- **Model:** "Simple SG90 servo/micro servo Claw" by **CtrlPlusAlt**
- **Source:** https://www.printables.com/model/1778845-simple-sg90-servomicro-servo-claw
- **License:** CC BY-NC, as listed on Printables (credit the author, no commercial use)
- **Downloaded:** 2026-10-22, unmodified

| File | Print | What it is |
|---|---|---|
| `Claw_connector_V4.stl` | 1 | frame: the SG90 sits in the window, gear peg on the side |
| `Claw_finger_V2_A.stl` | 1 | finger A (geared) |
| `Claw_finger_V3_B.stl` | 1 | finger B (geared, meshes with A) |
| `Claw_holder_V2.stl` | 1 | small clip (10 x 2 x 5 mm) |

Size: frame 31 x 19 x 23 mm, fingers ~58 mm long. One finger goes on the servo
output, the other on the frame's side peg; the photos on the Printables page
show which is which. The author lists no extra screws. Author's tip: wrap a
rubber band around the fingers for more grip. `preview.png` shows each part from
four sides.

### Fitting it to our arm
- Glue the wrist servo's horn (CA) to the flat back face of `Claw_connector_V4`
  so the fingers point forward, in line with the forearm. This replaces parts 05 + 06.
- The gripper servo stays on pin D10. Re-run `calibrate.py` to set the open/closed
  angles, because this claw's travel is different from our printed jaw.
- The fingers are ~58 mm long overall, vs 45 mm (`FINGER_END`) in our design, so
  the real fingertip sits further out than the kinematics assume. Check it during calibration.
