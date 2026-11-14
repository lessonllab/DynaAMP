# LoopMimic v2 方法

正式实验从随机初始化开始。第 14,000 次策略更新的 EMA 权重对应部署模型。[训练和复现说明](README.md)

## 改动与来源

- **Root 位置反馈。** 吸收 [BlinkMimic-plus 的 mdp.py](https://github.com/rascal-yang/BlinkMimic-plus/blob/main/src/blinkmimic/mdp.py) 中机器人坐标系位置误差的思想。输入为 `R_root^T (p_ref-p_root)`，直接加入共享 decoder，新增 3 列零初始化；encoder、latent 和所有隐藏层宽度保留。新增参数可学习，初始策略输出与原网络相同。
- **EMA 与困难片段。** 借鉴 [BlinkMimic-plus 的 train.py](https://github.com/rascal-yang/BlinkMimic-plus/blob/main/src/blinkmimic/train.py) 的 EMA 与失败前重置思路。EMA 系数 0.995，只用于确定性评估；训练始终用 live actor。归一化统计直接复制。恢复 SuperMimic 的 return sampler，加入 20% 失败前 3 秒的采样分布，限制在同一 clip 内，并保留均匀探索。
- **真实终止转移。** 参考 [JonnyDing 的 supermimic 分支](https://github.com/JonnyDing/supermimic_AMP/tree/supermimic) 的终止前状态捕获与有效前缀训练：在 reward 计算时保存自动 reset 之前的真实 next state。世界模型仍预测 7 次转移，评分包含倒下那一步，mask 掉 reset 之后的片段。判别器也可以看到真实失败转移。
- **Grounded AMP。** 在现有 Body195 上增加 root 高度和投影重力 4 维，得到 199D / 相邻对 398D。Body195 本身对整体刚体变换不敏感，这个扩展使判别器可感知高度、倾斜。真实状态、专家和模型预测共用一套编码；不加入参考帧或任务误差。D 隐藏层仍是 1024/512/256。
- **配对动作改善校准。** 延伸 LoopMimic 原有的 AMP 补采闭环：同一物理快照下执行原动作序列与 AMP 调整后的动作序列，固定 D，对齐“模型预计的风格变化”与“真实物理中的风格变化”。原来只有状态监督与优先回放；现在这对真实干预直接监督世界模型的动作响应。
- **共用补采数据。** 补采真实窗口同时进入 WM 数据、25% 的 BPTT 初始状态以及 25% 的 D 负样本。优先级逐渐回落，减少旧判别器产生的历史优先级长期主导训练。

JonnyDing 的实际 `supermimic` 分支已经含 WM、AMP 和解析 BPTT，LoopMimic v2 在此类可微路径上加入配对真实干预校准与数据闭环。

## 更新规则

更新策略时冻结 WM 与 D 参数，保留输入的 Autograd 路径。

`L_actor = L_tracking + w_amp L_style + L_bootstrap + L_aux`

原有跟踪奖励、AMP warmup/ramp、优化器保持。AMP 正 BCE 使用 detached survival 权重，使策略无法通过预测存活权重的梯度直接降低这项正代价。跟踪和 bootstrap 继续对生存项求导。

更新世界模型时冻结 D 参数，保留 `D(F_theta(...))` 对 WM 的导数。`S_D` 是 7 个相邻转移的平均负 BCE。

`gain_model = S_D(WM(s,a+delta)) - S_D(WM(s,a))`

`gain_real = stopgrad(S_D(MuJoCo(s,a+delta)) - S_D(MuJoCo(s,a)))`

`L_WM = L_state + 0.01 Huber((gain_model-gain_real) / stopgrad(max(RMS(gain_real),0.001)))`

该项拟合真实的有符号改善量。每次训练使用当前冻结的 D 重新计算真实目标。先更新 WM 并检查独立 holdout，再更新 D，最后更新策略与 EMA。

## 固定配置与验证

- LAFAN1 重定向到 G1 的全部 40 条 50Hz 轨迹；seed 0；4096 个 MuJoCo 环境。
- 官方 actor 隐藏层 2048/1024/512/256；WM 2048/2048/2048；policy batch 4096，32 步 BPTT。
- WM batch 2048、每次 10 个 batch、每窗口 8 个状态 / 7 次转移。每个 batch 为 128 原动作配对窗口 + 128 调整动作配对窗口 + 256 补采窗口 + 1536 常规窗口。补采池未就绪时全部使用常规窗口并将校准 mask 置零。
- MLP 使用 BF16，状态和损失使用 FP32。反向传播时重算部分激活，在上述网络宽度、batch 与 BPTT 时域下节约显存。
- 评估使用 40 条完整动作 × 4 副本与固定种子 20261113。最佳及首次满覆盖节点为 14,000 次，EMA 和 live 权重分别保存。[原实验结果](config/result_014000.json)
- 训练预算累计正常采样、配对干预、重复性检查、额外扰动与补采 holdout 的物理交互。评估另计。
- 首次 2,000、首次满覆盖和最终节点录制走、跑、舞蹈、跳跃各 20 秒；真实 MuJoCo，无自动重置，终止后冻结画面并标注。

训练入口为 [train.py](train.py)，训练参数在 [runner_014000.json](config/runner_014000.json)。

行为检查覆盖新位置反馈、终止前 next state、跨 reset mask、配对改善梯度方向、冻结 D、失败采样的 clip 边界以及原有 32 步路径审计。GPU 冒烟检查仍采用完整网络/batch/4096 环境，临时缩短补采间隔，以在 24 次更新内触发闭环。AMP 梯度的连通性由独立审计测试。
