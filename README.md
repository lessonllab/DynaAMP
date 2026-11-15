# DynaAMP

**DynaAMP: Learning Humanoid Control through Differentiable Dynamics and Adversarial Motion Priors**

DynaAMP optimizes humanoid policies with an adversarial motion prior through a learned differentiable world model and 32-step BPTT.

- **DynaAMP** denotes the variant without additional AMP data feedback.
- **DynaAMP + Feedback** denotes the paired-intervention, priority-replay and world-model calibration variant, previously called LoopMimic.

The training configuration and **14,000-update EMA policy released here belong to DynaAMP + Feedback** (formerly LoopMimic v2). The policy completed 160/160 episodes under the fixed LAFAN1 G1 evaluation protocol. Its exported ONNX is included in this repository. NVIDIA GEAR-SONIC provides standby and joystick locomotion; the released DynaAMP + Feedback policy provides motion tracking. The default deployment clip is the full **131.48 s** `dance1_subject2`.

Existing model directories and internal IDs retain their original names for compatibility. See [the name mapping](METHOD_NAMES.json).

- [训练、数据准备和复现](training/README.md) · [方法](training/METHOD.md) · [14,000 次更新结果](training/config/result_014000.json)
- [部署安装](deploy/README.md) · [Sim2Sim](deploy/sim2sim/README.md) · [Sim2Real](deploy/sim2real/README.md)
- [14,000-update EMA ONNX and configuration](deploy/shared/models/loopmimic_014000/)

The Sim2Real controller's gamepad and handoff path has been exercised in MuJoCo with CRC-valid Unitree packets. See the hardware guide for the pelvis odometry requirements.
