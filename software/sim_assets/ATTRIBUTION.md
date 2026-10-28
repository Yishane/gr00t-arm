# Simulation meshes

Made by `cad/sim_meshes.py` for the MuJoCo simulation (`software/sim.py`).

**The arm meshes** (`drum`, `base_cap`, `turntable`, `upper_arm`, `forearm`,
`gripper_*`, `finger_*`) are converted from the author's assembled model
`E-Z_RoboArm_v23.step` of **"Easy Robot Arm (SG90 servos & now steppers too)"**
by [mobiobi](https://www.printables.com/@mobiobi),
[Printables 97370](https://www.printables.com/model/97370-easy-robot-arm-sg90-servos-now-steppers-too-update),
licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/).
They are the author's shapes, tessellated into triangles and each moved into
the frame of the joint that turns it (the upper arm is the author's
`Arm60mm` turned upright, as in `cad/assembly.py`). Because they are derived
from the author's files, they keep the same license: credit mobiobi, no
commercial use, share alike.

**`baseplate.stl` and `cup.stl`** are this project's work-cell parts
(`cad/arm.py`).
