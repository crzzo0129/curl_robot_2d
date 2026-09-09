# 多指令 CEM 滚动蒸馏

这条路径只训练独立滚动策略，不训练 stand-to-roll。学生接口保持 720 维历史观测、
12 维完整关节位置动作；每个 36 维帧中的 command 顺序固定为
`[vy_m_s, vx_m_s, yaw_rate_rad_s]`，当前训练固定 `vy=0`。

`--teacher-source cem` 使用零 residual，让环境中已验证的 CEM reference、
前进速度查表和转向差分先验直接产生完整动作标签，不需要 privileged PPO checkpoint。
前进速度默认在查表有效区间 0.4111–0.8110 m/s 连续均匀采样；
每回合 40% 直行、30% 左转、30% 右转，转向幅值 0.02–0.08 rad/s。
可以用参数收窄到实际已验收的可行域。

命令在一个 10 秒回合内保持不变。每个并行环境采样不同 command；
BC 因而一次学习多个速度和转向，而不是依次训练固定速度。

`--random-cem-snapshots` 首先让 CEM 从 compact phase-0 向前运行随机 20–300 个
控制步（0.4–6.0 秒），然后把当时完整 MJX state、CEM oscillator phase、
20 帧部署观测历史和上一完整动作作为训练起点。每段继续采集 100 步后重新采样。
如果教师在预滚过程中终止，则保留终止前最后一个有效状态。

BC 在 CEM 控制的真实轨迹上学习完整动作。DAgger 由学生访问状态，
在完全相同的 PRE-step state 和 command 上查询 CEM，标签仍是 CEM 实际应用的
完整动作；教师介入概率按原脚本从 25% 降到 0。最终闭环评估也使用随机 CEM snapshot。

先做云端烟雾检查（只检查能否完整运行，步数不足以判断效果）：

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  --teacher-source cem \
  --geometry rollingquad_2_abd10 \
  --command-conditioned --random-cem-snapshots \
  --preset smoke \
  --out results/rolling_command_distill_smoke
```

正式 H200 训练：

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  --teacher-source cem \
  --geometry rollingquad_2_abd10 \
  --command-conditioned --random-cem-snapshots \
  --preset h200 \
  --out results/rolling_command_distill_h200
```

如已验证范围比默认查表范围更窄，追加：

```text
--forward-command-min-m-s <最小vx> --forward-command-max-m-s <最大vx>
--turn-command-min-rad-s <最小非零角速度> --turn-command-max-rad-s <最大角速度>
```

输出 `distillation.json` 会保存完整命令分布、teacher/closed-loop task、
BC/DAgger loss 和闭环评估。闭环摘要新增平均 `vx` 绝对跟踪误差和 yaw-rate
绝对跟踪误差。controller_config.json 明确记录 command 顺序。

当前第一版不在回合内切换 command，也不叠加 terrain/deploy DR/stand reset；
这些与随机 CEM snapshot 同时使用会被参数检查拒绝。先验收 nominal 多指令接管，
再单独增加指令切换和鲁棒化，避免混淆失败原因。

按约定，代码未在本地运行、编译、仿真或训练；云端先跑 smoke。
