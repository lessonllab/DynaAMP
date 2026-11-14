# LoopMimic Deploy

Unitree G1 29-DoF deployment with LoopMimic's **14,000-update EMA checkpoint** and NVIDIA GEAR-SONIC. LoopMimic tracks the full **131.48 s** LAFAN1 `dance1_subject2` motion. SONIC handles standby, joystick locomotion, and the return to Idle.

The exported ONNX and policy configuration are part of this repository. Setup downloads the pinned SONIC models and the original motion reference from their publishers and prepares the reference locally.

- [安装与快速开始](deploy/README.md)
- [Sim2Sim · MuJoCo 与遥控器回环复验](deploy/sim2sim/README.md)
- [Sim2Real · Unitree SDK2、原装遥控器和里程计](deploy/sim2real/README.md)
- [14,000-update EMA deployment artifact](deploy/shared/models/loopmimic_014000/)

The Sim2Real input and handoff can be exercised using CRC-valid Unitree gamepad packets in MuJoCo. Hardware deployment requires a live pelvis odometry source on the target G1 and hardware commissioning. See the Sim2Real guide for the state contract.
