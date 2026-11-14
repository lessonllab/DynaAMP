# Sim2Sim · SONIC 与 LoopMimic

完成 [公共安装](../README.md#安装) 后，从仓库根目录运行。

```bash
bash deploy/sim2sim/sonic.sh --view --output /tmp/loopmimic-view
```

在终端操作。站稳后按 **Y** 触发完整的 131.48 秒舞蹈，结束后自动交回 SONIC。**1/2/3** 选慢走、快走、跑步，**W/S** 前后移动，**A/D** 调整移动方向，**Q/E** 转向，**9/0** 调速，**R** 回待机，**O** 退出。由慢走经快走进入跑步。 [SONIC 键盘文档](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html)

## 自动复验

```bash
bash deploy/sim2sim/sonic.sh --dance-demo --output /tmp/loopmimic-sim
bash deploy/sim2sim/sonic.sh --gamepad-demo --output /tmp/loopmimic-gamepad
```

第一个命令复验键盘模式的完整舞蹈、待机、走跑和停车。第二个命令把原装遥控器格式的按键、摇杆和 CRC 有效的状态包经回环 DDS 送给真机入口。它按 **Start → F1 → A** 触发舞蹈，检查完整参考播放，之后推摇杆走路、按 **B** 停车并按 **Select** 退出。

仿真采用 500 Hz MuJoCo，LoopMimic 和 SONIC 原生跟踪器均按 50 Hz 更新。仿真全过程使用同一个 G1 29 自由度的物理状态。输出目录含 `results.json`、`trace.npz` 和 SONIC 日志。回环手柄复验增加 `loopmimic.log`。可通过 `--dance` 传入其他 50 Hz 且具备相同 29 个关节的 NPZ。

[完整舞蹈的回环验证结果](validation_014000.json)

## 录像

```bash
MUJOCO_GL=egl deploy/.runtime/venv/bin/python deploy/sim2sim/render.py \
  --trace /tmp/loopmimic-gamepad/trace.npz --output /tmp/loopmimic-gamepad.mp4
```
