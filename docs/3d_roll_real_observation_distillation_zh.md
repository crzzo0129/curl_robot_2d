# 3D rolling 实机观测蒸馏

站立起滚的新训练入口见 [从站立启动 3D rolling](3d_stand_to_roll_startup_zh.md)。
启用时先验证带启动段的教师，再把同样的 `--reset-pose stand` 等参数传给蒸馏；
学生闭环没有隐藏的启动控制器。

## Primitive mass-gain 教师与横漂诊断模式

当教师由 `rollingquad_2_primitive` 的 `floor_mass_gain_v3` 训练得到时，蒸馏和
后续 reward-driven Student DR 必须显式传入同一几何。入口会据此自动选择
primitive CEM reference，避免把 primitive checkpoint 静默放进完整 mesh 环境：

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  results/primitive_stiff_mass_gain_v3_smoke_seed0/params_best \
  --geometry rollingquad_2_primitive \
  --lateral-drift-diagnostic-only \
  --preset h200 \
  --out results/rollingquad2_primitive_roll_student_seed0 \
  --mujoco-gl disable --memory-fraction 0.80
```

`--lateral-drift-diagnostic-only` 仍以 0.20 m 生成横漂越界统计，但不因该项终止
episode。输出同时区分 `strict_success_rate` 与 `non_lateral_success_rate`；前者仍将
横漂越界视为不通过，后者只统计数值、失高、轴倾斜和禁止接触等物理失败。

第一级 deploy DR：

```bash
python -m scripts.train_mjx_3d_roll_student_dr_ppo \
  results/rollingquad2_primitive_roll_student_seed0/student_params \
  --geometry rollingquad_2_primitive \
  --lateral-drift-diagnostic-only \
  --dr-strength 0.25 --student-anchor-weight 0.02 \
  --preset h200 --max-devices 4 \
  --out results/rollingquad2_primitive_roll_student_dr025_seed0 \
  --mujoco-gl disable --memory-fraction 0.80
```

后续使用上一阶段的 `params_final` 通过 `--restore-ppo` 依次继续到
`--dr-strength 0.50` 和 `1.00`。横漂仍应报告 p50/p95/max，诊断模式只改变验收
语义，不代表横漂已经消失或不受场地尺寸约束。

## 频率约定

实机 `controller_manager` 运行在 520 Hz，`neural_controller.repeat_action=10`，
所以策略实际更新频率是 **52 Hz**（周期约 19.23 ms）。配置中的 260 Hz 是
IMU 和 joint-state broadcaster 的发布频率，不是策略控制频率。

当前 MJX rolling teacher 仍使用已经完成 curriculum 的物理和控制周期：
`1 ms × action_repeat 20 = 20 ms`，即 50 Hz。蒸馏不修改教师已经通过验证的
动力学时间步。部署到 52 Hz 后只是把同一学生策略以实机控制器的实际周期运行，
两者周期相差约 3.85%。如果后续实机验证显示这个差异不可忽略，再单独加入控制周期
随机化；不要把策略误改成 260 Hz。

## 蒸馏契约

教师继续使用当前 65 维仿真观测和 8 维 residual action。监督标签不是 residual
本身，而是环境真正施加的 `CEM reference + residual` 最终 8 维归一化 hip/knee
命令。

训练分为两段：先在教师轨迹上做 behavior cloning，再执行 online DAgger。DAgger
让学生用完整动作直接控制模型，并在学生实际访问的状态上调用拥有 65 维 privileged
observation 的教师重新标注。行为策略中的教师干预概率默认从 25% 线性降至 0%；每个
终止环境独立 reset。H200 preset 为 `20000` 步 BC 加 `10000` 步 DAgger，DAgger
使用较小的 `1e-4` 学习率。

学生输入严格匹配 `neural_controller.cpp`：单帧 36 维、最新帧在前、历史 20 帧，
总计 720 维：

1. body-frame angular velocity：3
2. projected gravity：3
3. command：3（当前 rolling 固定为零）
4. desired world z：3
5. 12 个关节相对 compact 的位置：12
6. 上一次 12 维策略动作：12

学生输出为控制器顺序的 12 维完整动作：每条腿均为
`abduction, hip, knee`，腿序为 `FL, FR, RL, RR`。实机 URDF 的 `_1/_2/_3`
分别代表 hip/abduction/knee，因此控制器 `joint_names` 使用 `_2,_1,_3` 完成物理
映射。滚动教师没有控制 abduction，导出配置把四个 abduction action scale 固定为
0，并把网络对应的四个输出硬锁为精确 0；实机以 KP/KD 保持 compact 零位。只把
action scale 设为 0 还不够，因为 C++ 会把原始网络输出写回下一帧的 last-action
观测，非零的无效输出仍会造成历史分布漂移。

## 先跑 smoke

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --preset smoke \
  --out results/rollingquad2_roll_distill_smoke_seed0 \
  --mujoco-gl disable \
  --memory-fraction 0.50
```

## H200 正式蒸馏

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --preset h200 \
  --out results/rollingquad2_roll_distill_h200_seed0 \
  --mujoco-gl disable \
  --memory-fraction 0.80
```

如需显式覆盖 DAgger 预算和退火，可以追加：

```bash
--dagger-steps 10000 \
--dagger-learning-rate 1e-4 \
--dagger-teacher-start-probability 0.25 \
--dagger-teacher-end-probability 0.0
```

如果已经完成旧版 20000 步 BC，可以复用其 checkpoint，只补 DAgger。该模式会读取
旧 checkpoint 的 720 维归一化统计和学生参数、把 abduction 最终层硬投影为 0，并
跳过统计与 BC：

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --restore-student results/rollingquad2_roll_distill_h200_seed0_v2/student_params \
  --preset h200 \
  --out results/rollingquad2_roll_dagger_h200_seed0 \
  --mujoco-gl disable \
  --memory-fraction 0.80
```

训练结束会自动做一次无教师闭环验证：学生只接收 720 维实机 ABI 观测，并直接输出
完整动作。终端会分别报告 `failure_free` 和 `success`：前者只表示没有触发故障终止，
后者还要求在 10 秒内达到默认 5 圈，因此“没倒但也没滚”不再计为成功。平均/最小圈数
和 abduction 输出误差也会记录在 `distillation.json`。

输出目录包含：

- `student_params`：JAX/Flax 学生 checkpoint；
- `student_rtneural.json`：供实机 RTNeural 控制器加载的模型；
- `controller_config.json`：20 帧历史、compact、KP/KD、关节限制和 action scale；
- `distillation.json`：训练参数、损失和无教师闭环评估。

smoke 的目标只是验证数据流、显存和导出，不代表策略已蒸馏成功。正式模型至少要确认
闭环 `success_rate` 和圈数接近教师，再进入实机吊绳/急停保护测试。

## 在现有 Student 上做 reward-driven deploy DR PPO（推荐）

现有 Student 已经完成模仿学习后，后续 DR 应由环境奖励驱动，而不是继续让随机化后的
Student 模仿教师。新入口将现有 Student 权重直接复制为 PPO Actor 初值：Actor 只读取
实机可用的 720 维历史观测，Critic 读取 65 维特权状态。PPO 只输出 8 个 hip/knee
动作，四个 abduction 通道在结构上锁为零；导出时再扩回实机控制器要求的 12 通道。

建议按 25% → 50% → 100% 三档推进，并逐档减小原 Student 的行为约束：

```bash
# 第一级：现有 Student 初始化 Actor，25% DR
python -m scripts.train_mjx_3d_roll_student_dr_ppo \
  results/rollingquad2_roll_dagger_h200_seed0/student_params \
  --dr-strength 0.25 --student-anchor-weight 0.02 \
  --preset h200 --max-devices 4 \
  --out results/rollingquad2_roll_student_reward_dr025_seed0 \
  --mujoco-gl disable --memory-fraction 0.80

# 第二级：恢复完整 PPO（Actor、特权 Critic 和归一化统计），50% DR
python -m scripts.train_mjx_3d_roll_student_dr_ppo \
  results/rollingquad2_roll_dagger_h200_seed0/student_params \
  --restore-ppo results/rollingquad2_roll_student_reward_dr025_seed0/params_final \
  --dr-strength 0.50 --student-anchor-weight 0.005 \
  --preset h200 --max-devices 4 \
  --out results/rollingquad2_roll_student_reward_dr050_seed0 \
  --mujoco-gl disable --memory-fraction 0.80

# 第三级：完整 DR，解除行为约束
python -m scripts.train_mjx_3d_roll_student_dr_ppo \
  results/rollingquad2_roll_dagger_h200_seed0/student_params \
  --restore-ppo results/rollingquad2_roll_student_reward_dr050_seed0/params_final \
  --dr-strength 1.00 --student-anchor-weight 0 \
  --preset h200 --max-devices 4 \
  --out results/rollingquad2_roll_student_reward_dr100_seed0 \
  --mujoco-gl disable --memory-fraction 0.80
```

每次评估会打印 `turns`、`success`、`failed`、横向漂移失败率和相对原 Student 的
逐步动作 RMSE。这里 `success` 表示无故障且至少完成 5 圈。每一级输出的
`params_final` 用于下一档继续 PPO；`student_rtneural.json` 是已扩为 12 通道且锁定
abduction 的实机模型。

## 在现有 Student 上继续做 deploy-DR DAgger（保留的旧方案）

下面的入口仍然可用于实验对照，但它本质上还是模仿学习，不是推荐的 Student DR
微调路径。

`--deploy-dr` 只允许和 `--restore-student` 一起使用：它保留现有 Student 的 720 维
归一化统计，跳过 BC，并在随机化后的学生闭环状态上继续 DAgger。随机化类型与
`train_ppo_deploy.py` 对齐：滑动摩擦、躯干/腿质量、惯量、躯干 COM、逐电机
KP/KD/力矩上限、0/20/40 ms 动作延迟、5% 控制 deadline miss、电机零偏、编码器
固定偏置，以及原有的 gyro/重力/关节观测噪声。没有加入 random shove 或动作低通。

不要直接从 nominal Student 跳到完整范围。建议每一级都从上一级输出继续：

```bash
# 第一级：25% deploy DR
python -m scripts.train_mjx_3d_roll_distillation \
  results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --restore-student results/rollingquad2_roll_dagger_h200_seed0/student_params \
  --deploy-dr --deploy-dr-strength 0.25 \
  --preset h200 \
  --out results/rollingquad2_roll_student_dr025_seed0 \
  --mujoco-gl disable --memory-fraction 0.80

# 第二级把 restore-student 换成 dr025 的输出，strength 改为 0.50；
# 第三级再换成 dr050 的输出，strength 改为 1.00。
```

强度会把所有模型范围、COM/零偏幅值、deadline miss 概率以及非零延迟概率一起从
nominal 线性扩展到完整 deploy 范围。每个并行环境拥有独立的固定随机模型；零偏、
编码器偏置和延迟在 episode reset 时重新采样。DR 评估也使用一批独立随机模型，并在
`closed_loop_evaluation` 中记录 `mean_deadline_miss_rate`。由于教师本身没有在完整
deploy 范围上训练，升档依据应同时看圈数、非横向故障和教师标签误差，不能只看当前
较严格的 lateral-drift success gate。

## 只评估现有 DAgger checkpoint：横向漂移诊断

Actuator-gain 教师的平均圈数约为 8.8 圈；不要再用早期 reference 约 6.7 圈作为这个
教师的性能基准。要判断学生横向漂移的来源，需要记录逐步数据，而不只是最终成功率。

以下命令跳过统计、BC、DAgger，不覆盖 checkpoint、不重新导出模型：

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --restore-student results/rollingquad2_roll_dagger_h200_seed0/student_params \
  --eval-only --preset h200 --eval-envs 256 --episode-length 500 \
  --eval-seed 0 \
  --out results/rollingquad2_roll_dagger_lateral_eval_seed0 \
  --mujoco-gl disable --memory-fraction 0.80
```

记录内容：

- `evaluation.json`：成功率、圈数、任务配置、评估 seed 和实际 reset keys。
- `lateral_diagnostics.json`：正/负 y 越界数、共同/差动误差 RMSE 与有符号偏差，以及按
  无故障、横向失败、正向失败、负向失败分组的误差。
- `lateral_episodes.csv`：每个 episode 的终止时刻、最终 y/vy/航向、越过5/10/15 cm的
  时刻及动作误差。
- `lateral_timeseries.csv`：每个控制时刻的 y/vy/航向和动作误差统计；明确列出仍活动的
  episode 数量，避免忽略提前终止导致的样本变化。
- `lateral_trace.npz`：完整 `(time, env, ...)` 轨迹、active 掩码、学生与教师8维动作、
  共同/差动分量及误差。终止后的冻结状态不参与汇总。

`y` 是相对起点的世界系横向位移，`vy` 是世界系横向速度。航向采用环境原有的
`atan2(-body_y_axis.x, body_y_axis.y)`，避免滚动翻转导致普通 Euler 角解释混乱。
共同/差动定义分别为 `(L+R)/2`、`(L-R)/2`；误差为学生减教师，单位为归一化动作，
四个通道依次是前 hip、前 knee、后 hip、后 knee。

教师标签来自**同一学生 pre-step 状态**的教师查询，沿用 DAgger 的最后子步有效动作
定义；不是另外一条教师轨迹。教师查询得到的物理状态会丢弃，学生始终独立控制，所以
记录动作对照不等于教师接管。查询需要额外物理计算，诊断会比只评估学生更慢。

这里固定 `--eval-seed 0` 便于后续复现；旧版评估没有保存 reset keys，因此本次是新的
评估样本，不保证逐 episode 重现旧的31.6%。如需在后续训练的最终评估同时记录，追加
`--record-diagnostics --eval-seed 0` 即可。
