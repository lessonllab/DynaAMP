# DynaAMP + Feedback 训练

此处复现历史反馈版本（原 LoopMimic v2）；14,000 次更新的部署模型不代表关闭反馈组的结果。

本目录包含生成部署版 14,000 次更新 EMA 策略的训练源码。算法基于 HybridRobotics SuperMimic，保留 32 步 BPTT，并加入可微 AMP、从同一物理状态采集的配对动作干预，以及用真实干预校准世界模型的风格改善预测。详细计算见 [方法说明](METHOD.md)。

## 安装和数据

在仓库根目录运行。训练需要 Linux、带 CUDA 的 NVIDIA GPU、C++ 编译器、Git 和 [uv](https://docs.astral.sh/uv/)。`setup.sh` 固定 MjLab Research 提交并用锁文件安装 Python 3.13 训练环境。实测使用 RTX 5080 16 GB、PyTorch 2.15.0、MuJoCo 与 MuJoCo Warp 3.16.0、Warp 1.19.0。训练与部署使用各自的环境。

```bash
git clone https://github.com/lessonllab/DynaAMP.git
cd DynaAMP
bash training/setup.sh
uv run --project training --locked python training/prepare_data.py
```

数据准备脚本下载 [Unitree 重定向后的 LAFAN1 G1 CSV](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/ce1572906efe6157840e8474d5a0d7aa87481e74)，核验 [40 条原始轨迹的 SHA-256](config/lafan1_g1.json)，并用固定版本的 MjLab 转成 50 Hz NPZ。发布时验证了 40 个输出文件与原训练数据逐字节一致。轨迹保存在 `training/data/`，按 [LAFAN1 源数据条款](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 使用。无需 W&B 登录，训练默认写离线日志。

## 从随机初始化训练

```bash
uv run --project training --locked python training/train.py \
  --output training/runs/lafan40_seed0 --target 14000 --every 1000 \
  --stop-on-full-coverage --skip-videos
```

[训练参数](config/runner_014000.json) 对应 G1 的 4096 个并行环境、策略 batch 4096、32 步 BPTT，世界模型 batch 2048、每次 10 批，原实验网络宽度、跟踪奖励和域随机化。每 1000 次实际策略更新保存完整 PyTorch checkpoint，评估 40 条完整轨迹 × 4 副本，使用 EMA 策略与固定种子 `20261113`。首次评估 160/160 或到达更新上限时结束。去掉 `--skip-videos` 可在关键节点生成录像。首次运行要编译 MuJoCo Warp 和 PyTorch CUDA 内核。

短时入口检查调用同一套网络、batch 和仿真环境，临时提高补采频率。正式训练的 AMP 权重在前 100 次更新为零，短训会单独审计 AMP 的 32 步梯度。

```bash
uv run --project training --locked python training/train.py \
  --output training/runs/smoke --smoke --target 24 --skip-videos
```

### 续训和评估

从本次运行已保存的 `updates_*.pt` 或 `training_state.pt` 续训，`--target` 为累计更新数。

```bash
uv run --project training --locked python training/train.py \
  --output training/runs/lafan40_seed0 \
  --resume training/runs/lafan40_seed0/updates_010000.pt \
  --target 14000 --every 1000 --stop-on-full-coverage --skip-videos
```

选取的 checkpoint 必须覆盖该运行已记录的最新评估。续训恢复模型、优化器、EMA、采样器和随机状态，并重新采集仿真回放。可以对本地 checkpoint 单独运行评估。

```bash
uv run --project training --locked python -m supermimic.scripts.evaluate_matched_amp evaluate \
  --checkpoint training/runs/lafan40_seed0/updates_014000.pt \
  --data training/data/lafan1_g1_50hz --output training/runs/eval_014000.json
```

[原实验记录](config/result_014000.json) 显示 14,000 次时首次达到 **160/160**，沿保存分支的训练交互量为 **163,385,344**。历史实验在 4,000、10,000、11,000 次恢复过训练，第一次和第二次评估分别在 2,000、4,000 次进行。上面的连续训练命令使用同样的模型和数据，从 1,000 次开始评估。需要逐项核对原固定种子的评估初态时，可给训练命令加 `--verify-reference-init`。不同 GPU 或驱动的接触计算可能改变数值结果。

### 导出到部署

用训练时保存的 checkpoint 导出 EMA 策略和 29 关节控制参数。Sim2Sim 和 Sim2Real 可用同一个导出目录。

```bash
uv run --project training --locked python training/export_ema.py \
  --checkpoint training/runs/lafan40_seed0/updates_014000.pt \
  --output training/runs/ema_014000
bash deploy/sim2sim/sonic.sh --dance-demo --bundle training/runs/ema_014000 \
  --output /tmp/loopmimic-export
```

运行仿真前先完成 [部署安装](../deploy/README.md)。[随仓库提供的 14,000 次 EMA ONNX](../deploy/shared/models/loopmimic_014000/) 可以直接用默认部署命令加载。

## 发布验证

使用固定依赖完成 4096 环境、32 步 BPTT 的 24 次更新检查，执行 3 轮闭环补采并生成 160 条 EMA 评估记录。独立梯度审计确认末帧 AMP 损失能回传到第一个动作。原始 14,000 次 checkpoint 经上面的导出入口生成了与部署包逐字节一致的 ONNX。

## 来源与许可证

- [HybridRobotics SuperMimic](https://github.com/HybridRobotics/supermimic/tree/abc124c1bd5976341034668af3d12c087f2f1f55)，Copyright 2026 The SuperMimic Authors，Apache-2.0。
- [HybridRobotics MjLab Research](https://github.com/HybridRobotics/mjlab_research/tree/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd)，Apache-2.0。
- `src/supermimic/vendor/` 保留 AMP_mjlab 和 supermimic_AMP 实验比较代码的来源、版权和许可证。运行 DynaAMP + Feedback 使用自身的可微 AMP 与配对干预实现。
