# LoopMimic G1 部署

面向 Unitree G1 29 自由度。NVIDIA SONIC 负责站立待机、遥控走跑和停车，LoopMimic 负责舞蹈。默认动作是 LAFAN1 的完整 `dance1_subject2`，共 **6,574 帧、131.48 秒**。起舞前另有 **1.2 秒**的姿态过渡。

- [Sim2Sim](sim2sim/README.md) · MuJoCo、键盘控制和遥控器 DDS 回环复验
- [Sim2Real](sim2real/README.md) · Unitree 原装遥控器、SDK2 和真机骨盆状态输入
- [14,000 次更新的 EMA ONNX](shared/models/loopmimic_014000/motion.onnx) · 约 28 MB，Git clone 直接包含推理权重

## 安装

需要 Linux、NVIDIA GPU、CUDA 和匹配平台的 TensorRT。NVIDIA 对桌面 x86_64 要求 TensorRT 10.13，对 G1 机载 Orin 要求 JetPack 6 和 TensorRT 10.7，对 Thor 要求 JetPack 7.1 和 TensorRT 10.13。安装完 TensorRT 后，按实机路径设置 `TensorRT_ROOT`。[NVIDIA 部署安装手册](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)

Ubuntu 上还需要编译工具、Python 虚拟环境和 CycloneDDS 0.10.x 开发库。Python 3.11 和 Ubuntu 24.04 用于本次桌面验证。

```bash
sudo apt-get install build-essential cmake git curl python3-dev python3-venv \
  cyclonedds-dev libeigen3-dev libmsgpack-dev libzmq3-dev cppzmq-dev \
  libyaml-cpp-dev nlohmann-json3-dev libasound2-dev zlib1g-dev libgl1 fonts-noto-cjk
export CYCLONEDDS_HOME=/usr
bash deploy/setup.sh
```

`setup.sh` 下载固定版本的 SONIC 源码、官方三个 ONNX 权重、SDK2、G1 模型和参考 CSV。它会验证权重及参考文件的 SHA-256，在本地把完整参考转换到 50 Hz，并编译原生 C++ 控制器。建议留出 5 GB 磁盘空间，首次启动由 TensorRT 为当前 GPU 构建引擎。已有 Python 3.11 时，可用 `LOOPMIMIC_PYTHON=python3.11 bash deploy/setup.sh` 指定解释器。系统 CycloneDDS 版本与 0.10.x 不匹配时，可先按 SDK2 手册构建 0.10.x，并把 `CYCLONEDDS_HOME` 指向其安装目录。[Unitree SDK2 Python 安装](https://github.com/unitreerobotics/unitree_sdk2_python)

## 运行仿真

```bash
bash deploy/sim2sim/sonic.sh --view --output /tmp/loopmimic-view
```

站稳后在启动终端按 **Y** 开始跳舞。结束后由 SONIC 待机。可用 **W/S** 控制前后走，**Q/E** 转向，**R** 回待机。

下面的自动验证把遥控器字节和带 CRC 的 `LowState` 送给同一个 C++ 控制器。它用 MuJoCo 产生的 DDS 骨盆状态运行 `sim2real/controller.py`，检查完整舞蹈、接回待机、摇杆移动和停车。

```bash
bash deploy/sim2sim/sonic.sh --gamepad-demo --output /tmp/loopmimic-gamepad
```

本次回环复验播放了 6,574/6,574 帧，舞后待机 200/200 帧双脚接地，继续用摇杆走出约 2.17 米后停车。 [验证记录](sim2sim/validation_014000.json)

## 来源

LoopMimic 推理权重来自 LAFAN1 G1 训练的第 14,000 次策略更新，导出 EMA actor 和训练归一化。克隆后可以直接加载 ONNX，无需训练环境或训练 checkpoint。参考轨迹从原数据源下载，只在本地转换。[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) · [Unitree G1 重定向数据](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) · [SONIC 源码](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda) · [SONIC 发布权重](https://huggingface.co/nvidia/GEAR-SONIC/tree/6733128a3d8a523b1418b06bca3cdf61c8b0987f)
