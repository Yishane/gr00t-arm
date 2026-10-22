# gr00t-arm

A 3D-printed 4-DOF desk arm (5x SG90 servos, Arduino) used to demonstrate how
NVIDIA GR00T learns robot skills: record keyboard-teleoperated demos, export them
as a GR00T/LeRobot dataset, train ACT locally or fine-tune GR00T N1.7 on a cloud GPU,
and run the learned policy on the arm.

- `cad/` — parametric CadQuery source, printable STLs, renders, collision checks
- `firmware/` — Arduino servo firmware (serial protocol)
- `software/` — kinematics, calibration, teleop/recording, dataset export, policy runners
- `docs/` — the build + GR00T guide (LaTeX/PDF)
- `context.md` — running project notes / where work left off
