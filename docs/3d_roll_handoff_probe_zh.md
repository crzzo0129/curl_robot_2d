# 前 3 秒滚动教师接管实验

目的：为独立的 stand→起滚技能寻找**可被冻结滚动教师继续控制的候选状态**。
本实验不是启动 PPO，不执行 stand→compact 插值，也不直接设置部署接管门槛。

## 实验究竟验证什么

1. 用真实 checkpoint 从原 compact reset 运行教师，采集最初 0–3 s 的完整状态。
2. 原轨迹至少运行 10 s；只有无配置内失败且达到 5 圈的轨迹才标为合格来源。
3. 每个候选点分叉，交还给同一个冻结教师，继续运行 3 s。
4. 分别测精确回放、物理状态扰动、CEM 相位扰动、上一动作记忆扰动、三者联合扰动。

不是只复制 qpos：快照包含 qvel、ctrl、求解器状态、环境计数、全局位置原点、
振荡器相位、实际滚动相位、上一动作、观测等全部动态叶节点。精确回放需要与
原轨迹相同终点的 qpos/qvel 一致；最大差异超过 `1e-5` 就中止结论。
不同 MuJoCo/MJX 版本之间不能直接假定快照格式可互换。

候选时刻描述它来自教师轨迹的哪个位置，**不是训练启动技能时必须定时接管的时刻**。
此处只证明小邻域内的局部延续能力；并未证明能从 stand 到达，也没有测出完整吸引域。

## 默认扰动（独立均匀分布的半宽）

| 项目 | 半宽 |
| --- | ---: |
| 8 个 hip/knee 关节位置 | 0.01 rad |
| 8 个 hip/knee 关节速度 | 0.10 rad/s |
| 根部三轴线速度 | 0.02 m/s |
| 根部三轴角速度 | 0.10 rad/s |
| 滚动轴倾斜扰动（world x/z） | 0.02 rad |
| CEM 振荡器相位 | 0.20 rad |
| 上一动作记忆（归一化） | 0.05 |

关节位置受原限位约束，四元数归一化；不瞬移根部位置。
history 扰动只改变网络记忆输入，不直接修改电机 ctrl；因此纯 reference 对它不敏感。
不同 case 使用独立随机样本，小样本下联合扰动偶尔优于单项扰动不说明因果关系。

保留 `_2` 完整碰撞几何、1 ms 物理步长、20 子步/动作、CG20、原电机参数及
±3 Nm 限制。`|y| > 0.20 m` 等失败标准不放宽，接管时不清零世界位置或时间。
本实验不随机化模型摩擦、质量、电机增益；读取的是配置的基础 `task`，而非
`curriculum.stages[*].task`。后者的 `floor_contact_friction_override=true` 及
gain/floor/mass 采样应在后续独立复测，不能视作已经覆盖。

## 成功与失败必须分开看

`success` 同时要求：完整续滚窗口、没有配置内失败、窗口内保守圈数达到
`minimum_turn_rate × continuation_s`，且有正向净转动。
默认即 **3 s 内至少 1.5 圈**，这只是候选筛选阈值，不是原训练成功率定义。

保守圈数沿用环境：累计绝对滚动角/2π 与前进距离/(2πR) 取小者，R=0.1275 m。
只累计接管之后的新进度，同时另存有符号净转角，避免把接管前的圈数算进去。

- `failure_free_rate`：完整运行且没有物理/数值/任务失败，不要求圈数。
- `slow_but_failure_free_rate`：完整无失败，但未达上述进度要求。
- `failure_rates`：分别列 lateral drift、轴倾斜、高度、禁止接触和非有限值等。
- `qualified_source_success_rate`：仅从完整 10 s 合格原轨迹取样的结果。

原点附近仍在启动的状态，即使 `success=0`，也可能全部无物理失败。
不可仅凭这个数断言教师“接不住”。多次扰动同一轨迹不是多个独立初态。

`max_control_sample_torque_nm` 只在控制边界采样，不是每个物理子步的真实峰值。
`first_command_jump_rad` 是接管后第一个控制周期末 ctrl 与接管前 ctrl 的最大关节差，
不是周期内瞬时最大跳变。快照保存了实际命令和状态，便于进一步分析。

## 运行

配置和 checkpoint 必须对应。参数文件不包含可完整恢复的环境配置，默认要求
同目录 `training_config.json`，也可显式传入。缺少教师时不会静默退化成 reference。

Linux / 原训练环境的小规模复现实验：

```bash
python -m scripts.probe_3d_roll_handoff \
  --backend mjx-teacher \
  --teacher '/path/to/params_best (3)' \
  --teacher-config /path/to/training_config.json \
  --window-s 3 --sample-every-s 0.5 --continuation-s 3 \
  --source-duration-s 10 --donors 2 --trials 4 --seed 1 \
  --out results/handoff_teacher_3s_seed1
```

扩大验证时可用 `--donors 8 --trials 32 --sample-every-s 0.2`；先确认显存能承受
256 个并行扰动分支。初次 MJX 编译可能较久，日志会显示 source 和各分支进度。
换 seed / 更大噪声使用新输出目录，禁止覆盖已有非空实验目录。

独立 CPU reference 对照需显式指定 `--backend cpu-reference`；它不加载教师，
使用 NumPy reset RNG，不能与 MJX 同 seed 按逐状态相等比较，更不能代替教师结果。

## 输出和事后核对

- `experiment.json`：配置、来源、checkpoint/MJCF 哈希、运行版本和协议。
- `sources.json`：原始 10 s 教师轨迹资格与最终圈数、y、失败项。
- `snapshot_*.npz`：候选完整动态状态。
- `candidate_features.npz`：按时刻、来源排列的状态与控制器特征。
- `probe_*_*.npz`：每次扰动数值，以及分支首末状态特征。
- `trials.csv`：逐分支数据、圈数、失败项、精确回放误差。
- `summary.json/csv`：分时刻/扰动类型的汇总。

为旧输出增加“无失败/进度不足”分项，或核对后来补充的训练配置：

```bash
python -m scripts.analyze_3d_roll_handoff results/handoff_teacher_3s_seed1 \
  --teacher-config /path/to/training_config.json \
  --out results/handoff_teacher_3s_seed1/analysis.json
```

分析器新建报告，不修改原始实验来源声明。跨平台 reference 路径可以不同，
但数值系数必须相同；缺失的新增配置项按兼容默认值展开后比较。
数值配置相符不代表 JAX/物理引擎版本、设备后端和 curriculum 随机化已相符。

## 对后续启动学习的约束

候选是“物理状态 + 控制器上下文”的组合，不是一张姿态图。
还需要训练启动技能从真实 stand 连续到达候选邻域，并用**实际到达状态**接入教师，
计入之后数秒的失败和偏移；不能把到达状态替换为银行内快照。
接管判定应检查实际状态、相位、速度和动作连续性，多帧确认且有超时失败。

当前记忆/相位噪声范围很小，还没验证“相位归零、历史全零”能否接管；更不能
把教师轨迹的仿真时间直接当实机启动时间。未来要明确初始化相位和 reference/ramp
时间上下文，同时保持真实物理时间、位姿、速度和失败计数连续。
成功到达后才将启动教师和滚动教师一起蒸馏到实机 obs，并做无教师干预的学生评估。
50 Hz 教师仿真与 52 Hz 实机控制的差异仍需验证；260 Hz 只是 IMU 发布频率。
