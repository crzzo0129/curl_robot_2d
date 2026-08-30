# 3D rolling 实机观测蒸馏

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
