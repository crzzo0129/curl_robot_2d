# 从 737280 快速推进速度跟踪

这轮直接从 `rolling_heading_handoff_20260913_111545/actor` 的 PPO **737280** 检查点继续 actor。复用现有 bank，不重新采集，不再进行 critic-only 预热。actor、critic 正常联合更新。

## 云端直接运行

在 `curl_robot_2d` 仓库根目录解压新源码包后运行：

```bash
unzip -o results/rolling_speed_fast_v1.zip -d .

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.run_rolling_speed_fast \
  --handoff-bank results/rolling_handoff_bank_20260913_090129.npz
```

默认源目录：`results/rolling_heading_handoff_20260913_111545/actor`。默认源 step：737280。输出自动使用新目录 `results/rolling_speed_fast_<时间>`，已有目录不会覆盖。

默认运行 **491520 环境步、5 次评估（含 step=0）**，4 张 GPU、2048 环境。训练量为上一轮的四分之一，固定评估仍保留 256 个初态和 10 秒时长，便于比较。不会跳过 step=0。

训练入口先运行云端轻量契约和数值检查，不做测试用物理 rollout。程序完成或触发评估停止条件时自动生成同名 ZIP，例如 `results/rolling_speed_fast_<时间>.zip`；发回这个包即可。代码在本地仅进行 AST 语法和差异检查，没有执行单元测试、仿真或训练。

## 本轮改变什么

| 项目 | 设置 |
|---|---|
| 学习率 | 初始 **1e-5**，adaptive KL 上限 **3e-5**，下限 1e-7；原来最高仅 3e-6 |
| KL 目标 | 保留 0.01；PPO clip 保留 0.05 |
| 初态、命令、奖励、PD、DR | 从源 run 配置继承，保留 0.45–0.75 m/s、接管斜坡、P=5/D=0.1、DR=0.25、lateral=0.50 m、vx/yaw 权重 6/3 |
| anchor/归一化 | 保留源训练使用的原始 student 和 frozen normalizer；实际 actor/critic 由 737280 的完整 PPO `params` 恢复 |
| 探索 | 恢复原策略方差，不重新扩大 std；优化器状态重新初始化 |
| 选模 | 在存活约束内，优先接管组 **1 秒平均速度误差**，再看接管组逐步稳态误差、yaw 误差 |
| 停止条件 | 总成功率或任一 reset 组完整 10 秒比例，相对本轮 step=0 下降超过 **3 个百分点**，保存后停止 |

目标是快速确认上一轮学习率上限是否限制了速度学习。本轮不同时更改奖励、PD 或 DR，以免失去判断依据。没有预先承诺提高学习率一定有效。

## 新增指标

在 `fixed_eval_history.json` 的 `tracking_by_reset_source.handoff` 中查看：

- `windowed_forward_mae_m_s`：1 秒滑动平均的实际速度与平均指令之差；窗口必须完整，且全窗口命令斜坡均已完成。用它观察滚动周期波动之外的持续偏快/偏慢，不替代原始逐步误差。
- `ever_sustained_tracking_rate`：1 秒平均速度进入目标 ±0.05 m/s，并连续保持 2 秒的 episode 比例。之后仍可能离开目标带或失败，因此须同时看 full_horizon_rate。
- `speed_settling_time_mean_s` / `median_s`：满足上述条件的 episode 中，首次合格区间的起点时间；从 snapshot reset 开始计时，需再经过 2 秒才能确认。不是仿真瞬时速度的严格 settling time。
- `windowed_in_band_fraction`：有效窗口中健康且处于误差带内的比例。
- `per_episode.speed_settling_time_s`：未观察到合格区间记为 **-1**，不当成 0 秒；均值仅针对达到条件的样本。

所有这些量来自同一轮固定评估，不添加额外 rollout。原始逐步速度和 yaw MAE、过渡段/稳态段误差仍保留。

PPO 日志新增 `actor_grad_norm`、`critic_grad_norm`、`global_grad_norm`、`grad_clip_scale`、`grad_clip_fraction`。使用现有反向传播得到的、跨设备归约后的梯度，不再求一次梯度。只在当前训练进程中接入诊断，结束后恢复，不编辑安装的 Brax 文件。critic 梯度含原 value-loss 权重；不能仅根据 loss 大小或裁剪系数断言 actor 更新被压制。

## 如何判断继续

`speed_result.json` 会给出本轮 step=0 和所选检查点的接管组窗口误差及改善比例。若最优 step 仍为 0，说明本轮没有找到满足存活约束且更好的速度候选，不要导出最后一轮替换它。

优先寻找接管组窗口误差有可辨识的下降（例如至少约 5%），同时原始稳态速度误差不恶化、持续跟踪比例提高。相同验证面板上的小幅变化还需独立样本验证。看梯度和 KL 后再决定是否提高探索或修改奖励，不沿着无提升结果直接加长训练。

需要原学习率对照时，在同一源和 bank 上另开一个输出目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.run_rolling_speed_fast \
  --handoff-bank results/rolling_handoff_bank_20260913_090129.npz \
  --learning-rate 0.000003 --max-learning-rate 0.000003
```

默认先运行加大学习率的一轮，节省时间。新源码包不含新权重，也不会部署机器人；本轮产出的策略仍需遵循 heading_handoff_v3 接管契约。
