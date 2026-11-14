# LoopMimic

LoopMimic v2 is a differentiable world-model method for Unitree G1 whole-body imitation. Its AMP loss trains the policy through 32-step BPTT. Paired action interventions in MuJoCo supply real data to calibrate the world model's prediction of style changes.

The **14,000-update EMA** policy completed 160/160 episodes under the fixed LAFAN1 G1 evaluation protocol. Its exported ONNX is included in this repository. The deployment stack uses NVIDIA GEAR-SONIC for standby and joystick locomotion and LoopMimic for motion tracking. The default deployment clip is the full **131.48 s** `dance1_subject2`.

- [训练、数据准备和复现](training/README.md) · [方法](training/METHOD.md) · [14,000 次更新结果](training/config/result_014000.json)
- [部署安装](deploy/README.md) · [Sim2Sim](deploy/sim2sim/README.md) · [Sim2Real](deploy/sim2real/README.md)
- [14,000-update EMA ONNX and configuration](deploy/shared/models/loopmimic_014000/)

The Sim2Real controller's gamepad and handoff path has been exercised in MuJoCo with CRC-valid Unitree packets. See the hardware guide for the pelvis odometry requirements.
