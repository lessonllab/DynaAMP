# Sim2Real · Unitree G1 29 自由度

SONIC 原生 C++ 程序是身体电机命令的唯一发布者。站立、走跑和停车沿用 SONIC。遥控器 A 在 Planner Idle 时请求播放 LoopMimic 的完整舞蹈。部署包含 29 路身体关节，关闭 Dex3。

此入口已在 MuJoCo 回环上测试。实物 G1 厂固件是否在低层控制期间持续发布可用里程计，以及真机的落地跟踪与舞后交接，需要在目标 G1 上验证。

## 先读真机状态

运行 [公共安装](../README.md#安装) 后，把 `enp3s0` 换成连到 G1 的有线网卡。让 G1 在平地静止站立，再检查状态。

```bash
bash deploy/sim2real/run.sh --interface enp3s0 --check-state
```

此命令只订阅 DDS。它检查 `rt/lowstate` 的 CRC、关节和 IMU 输入，以及持续更新的骨盆位置和速度。默认读取 Unitree `rt/odommodestate`，类型为 `unitree_go::msg::dds_::SportModeState_`。也支持 SDK2 `OdoState_`，通过 `--odom-type hg --odom-topic rt/odostate` 选择。可以用 `--odom-topic` 改成现场估计器发布的话题。

骨盆位置必须是同一世界系中的米制坐标、Z 向上且地面为 Z=0，线速度必须是同一世界系的 m/s。IMU 姿态来自 `LowState`，里程计的水平坐标轴必须与它的世界轴对齐。若里程计只是高度零点不同，`--odom-z-offset` 可填写实测的常量偏移。若输出跟踪的是传感器或设备而非骨盆，先在估计器里做刚体变换。诸如始终输出零速度的占位里程计不能用于此策略。 [Unitree G1 SDK2](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py)

## 运行与遥控

```bash
bash deploy/sim2real/run.sh --interface enp3s0
```

1. 按 SONIC 原厂流程让机器人进入低层控制。原装遥控器按 **Start** 启动控制，再按 **F1** 打开 Planner。
2. G1 站稳且摇杆回中后按一下 **A**。默认播放完整 `dance1_subject2`，进入和退出均平滑混合关节目标与 PD。
3. 舞蹈播放完自动回到 **SONIC Idle**。按 **B** 可提前收回到 Idle。之后左摇杆移动、右摇杆转向、L1/R1 换走跑档、L2/R2 调速。**Select** 保留 SONIC 原生的停止控制并退出行为。

首次真机复验时用 SONIC 原厂的吊挂启动方式，先看检查输出和小范围起步，再验证完整舞蹈的空间与片尾交接。Sim2Sim 中整段动作的骨盆最远移动约 4 米。 [SONIC 遥控器文档](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/gamepad.html) · [SONIC 上机流程](https://nvlabs.github.io/GR00T-WholeBodyControl/best_practices/teleoperation.html)

## 状态与交接

舞蹈 ONNX 每步读取 71 维状态和 10 帧参考。关节位置、速度和骨盆 IMU 来自 CRC 验证后的 `LowState`，骨盆世界坐标和线速度来自里程计。不新鲜或高度异常的状态会取消舞蹈。舞蹈和交接时 SONIC 抑制走跑命令，收回控制时用当下的关节和朝向重新生成待机轨迹。真机仍由 SONIC 计算 SDK2 低层命令 CRC。副进程只发布内部舞蹈关节目标。
