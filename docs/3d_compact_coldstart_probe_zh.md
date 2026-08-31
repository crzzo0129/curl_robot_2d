# 低速 compact 冷启动验证

入口：`python -m scripts.probe_3d_compact_coldstart`。
使用实际冻结滚动教师，验证 compact 附近初态带有余速时，能否从教师年龄 0 开始起滚。
这不是训练 stand→compact，也不证明 compact 是静态平衡姿态或能从 stand 到达。

## 协议

- 从教师 compact reset 出发，原始参考/残差 ramp、相位初始化、力矩限幅、失败条件不变。
- 清掉底层随机 reset，再显式注入可记录、可复现的初态扰动；只在 episode 起点注入一次。
- 扰动后重新计算物理派生量及实际观测；不伪造教师输入，不在执行途中清零速度。
- 关节扰动只作用于 8 个 hip/knee；4 个 abduction 的初始位置/速度不加噪声。
- 保留 compact 控制目标和零上一动作、零相位、零时钟，不混入前次启动策略控制历史。
- 各分量独立均匀分布于 ±表中幅度。幅度是分量上限，不是向量模长上限。
- 各组使用相同随机方向、改变幅度，帮助区分速度因素。NumPy RNG 与训练 JAX RNG
  不同，`training_noise` 是相同分布，不是相同 seed 下逐样本一致。
- 默认每组 16 个随机初态；exact 只算 1 个独立样本，即使计算时为了复用编译做了复制。
- 固定名义物理参数，无地面摩擦、质量或 gain 的重新随机化。

| 组 | q 误差 rad | 关节余速 rad/s | 机体线速度 m/s | 机体角速度 rad/s | 轴倾斜扰动 rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact | 0 | 0 | 0 | 0 | 0 |
| training_noise | 教师配置 | 教师配置 | 教师配置 | 教师配置 | 教师配置 |
| joint_002 / 005 / 010 | 教师配置 | .02 / .05 / .10 | 0 | 0 | 0 |
| linear_001 / 003 | 教师配置 | 教师配置 | .01 / .03 | 0 | 0 |
| angular_005 / 010 | 教师配置 | 教师配置 | 0 | .05 / .10 | 0 |
| combined_low | 教师配置 | .02 | .01 | .05 | 0 |
| combined_medium | 教师配置 | .05 | .03 | .10 | 0 |
| combined_high | 教师配置 | .10 | .05 | .20 | 0 |
| nearcompact_low | .01 | .02 | .01 | .05 | .02 |

针对机体负 X 速度，另有 `vx_neg_001/003/005/010` 和 `vx_pos_001/003/005/010`。
它们分别固定世界 X 初速度为 −/+0.01、0.03、0.05、0.10 m/s；其它 5 个根部速度分量
严格为零。正负组使用相同关节位置/速度噪声，隔离方向效应。固定值记录在
`fixed_root_vx_m_s`，覆盖随机根部速度采样，不是另叠加一个随机幅度。

本次提供的 gain 教师配置中，关节位置噪声 ±.005 rad、关节速度噪声 ±.005 rad/s，
根部速度和轴倾斜噪声为零。两个记录的 gain 阶段相同；仅凭这份配置不能证明更早续训
阶段从未使用过其它范围。
同时核对当前 `INDEPENDENT_RESET_V4_STAGES_3D`：其 6 阶段关节速度幅度从 .0005
增至 .005 rad/s，根部速度噪声始终为零；因此该课程本身不覆盖机体滑动/转动余速。

## 评价与输出

默认运行 10 秒，成功要求完整窗口无配置内失败、有效圈数至少 5 圈且净相位向前。
有效圈数取累计转动圈数与前向位移折算圈数的较小值。episode 一旦终止即冻结，不自动
reset 后混入另一局。前 3、6 秒只记录进度，不套用动态接管时的 3 秒/1.5 圈标准。

- `experiment.json`：原配置、两种模型哈希、教师哈希、噪声审计、版本与运行设备。
- `summary.json`：全部完成后生成的正式汇总；`partial_summary.json` 仅是已完成组。
- `trials.csv`：每个初态的最终成功/失败、圈数、有符号 y 和最大横移。
- `checkpoints.csv`：3、6、10 秒状态；提前失败者时间保留在实际终止时刻。
- 每组 NPZ：全部初态偏移、起止状态、逐控制帧 qpos/qvel/ctrl/时间/相位。

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_3d_compact_coldstart \
  --teacher results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --teacher-config results/rollingquad2_floor_mass_gain_v3_h200_seed0/training_config.json \
  --trials 16 --seed 31 \
  --out results/compact_coldstart_gain_seed31
```

加 `--dry-run` 只输出试验协议，不加载教师运行；更大复测可改 `--trials 64 --seed 32`
并使用新输出目录。`--groups combined_low nearcompact_low` 可选择组，仍保留 exact 和
training_noise 对照。

有限样本通过不等于整个噪声区间保证成功，尤其不能证明速度越小成功率一定单调提高。
从 stand 接管还需验证实际到达状态的高度、接触、关节残余伺服误差和上一动作，不能只
看 q/qvel 接近。此入口不修改当前启动 PPO、候选库或部署策略。
