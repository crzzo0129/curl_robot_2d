# 从站立启动 3D rolling

> 下文是早期固定插值启动原型，尚未通过直线启动验证；不是已经学会自主起滚的策略。
> 当前独立启动技能的前置验证见 [滚动教师接管实验](3d_roll_handoff_probe_zh.md)。

本功能让 reset 真正使用 `_2` 模型的 `stand` keyframe，再通过电机位置目标完成
“站立 → 收腿 → 起滚”。`compact` 仍是所有动作的零点，不是改成站立零点。
新开关默认关闭，已有 compact 起始的训练/评估保持原行为。

## 启动定义

各入口使用相同参数：

```text
--reset-pose stand --stand-hold-s 0.2 --stand-to-compact-s 1.0
```

- 0–0.2 s：保持 stand；CEM 相位冻结，残差尚未介入。
- 0.2–1.2 s：stand 到 compact 的关节目标用 smoothstep 插值。
  残差在收腿开始后的 0.25 s 内渐入，PPO 可以学习收腿中的纠偏。
- 1.2 s 起：开始推进 CEM 相位，滚动 reference 按原来的 0.25 s ramp 渐入。
  残差不在这个边界重新归零，避免突然移除已有修正。

只有 reset 会设置 qpos/qvel；后续，包括收腿结束时，都不重置状态。机身高度、
姿态、速度、接触与真实跟踪误差连续传递。重力、完整 CAD 碰撞、原力矩限制均保留。
目前这是一条“可学习残差修正的启动先验”，不是已经训练好的任意站姿恢复策略。

stand 的 hip/knee 相对 compact 归一化动作约为 `(0.986465, 0.200618)`，
仍在原来的 ±1 动作范围内。按关节名称读取 keyframe，不按 body 遍历顺序硬拷贝。
四个外摆仍锁定零位；不改变教师 65 维观测、8 维动作和学生 720/12 维接口。

## 时间与评估

启动包含在 episode 内，也包含在失败检查、奖励和实际位移/转角统计内。
50 Hz 下，560 steps = 11.2 s，包含 1.2 s 启动和约 10 s 滚动；如果继续使用
500 steps，则滚动段只有约 8.8 s，不能直接与旧模型的“10 s 滚动”比较圈数。
不修改 lateral drift 等失败阈值，也不在切换时清空初始位置。

环境新增 `stand_startup_active` 和 `rolling_elapsed_s` 指标；完整配置保存于原有
训练/评估 JSON 的 `task` 字段。`startup_action_ramp` 表示残差渐入，不是收腿进度。

## 本地先看实际运动

```bash
python -m scripts.view_3d_cem_reference \
  --geometry rollingquad_2 --physics-profile cg20 \
  --reset-pose stand --stand-hold-s 0.2 --stand-to-compact-s 1.0 \
  --duration 11.2
```

加 `--headless` 可关闭窗口；这是 CPU MuJoCo 的纯 reference，不包含教师网络。
它输出真实 `startup_handoff` qpos/qvel、关节误差和峰值力矩，不把运行完毕称为成功。

2026-08-31 本地 CPU MuJoCo 3.12.0 的单条无扰动测试：上述设置能收腿并起滚，
峰值力矩约 1.91 Nm、无 MuJoCo 数值警告；但 11.2 s 末横向位移约 -2.44 m，
明显不满足训练的 ±0.20 m 直线标准。相同 cg20 的旧 compact 起始 10 s 对照，
横向位移约 -0.148 m。收腿时长 0.3、0.5、2、3 s 的试跑也未消除航向偏差。
这些是无失败截断的单条 CPU 诊断，不是 MJX 成功率，更不是实机验证。

因此“启动过程接通”不等于“启动策略通过”。需要先测教师，再微调并蒸馏。

## 先评估已有教师

以下路径沿用现有 gain 教师；如果服务器上的 checkpoint 名称不同，替换该路径。

```bash
python -m scripts.evaluate_mjx_3d_policy \
  results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --geometry rollingquad_2 --physics-profile cg20 --initial-policy-std 0.10 \
  --reset-pose stand --stand-hold-s 0.2 --stand-to-compact-s 1.0 \
  --episode-length 560 --batch-size 256 --chunk-size 256 \
  --save-rollout --diagnostic-rollouts 4 \
  --out results/rollingquad2_stand_start_teacher_eval_seed0 \
  --mujoco-gl disable --memory-fraction 0.80
```

查看输出目录的 `deterministic_eval.json`、`eval_arrays.npz` 和诊断轨迹。
评估时必须传入相同启动参数；评估器不会自动从 checkpoint 猜测 reset 配置。

## 从已通过的教师微调启动

先只加入新的初始状态，不同时扩大已有 domain randomization。此示例为额外的
名义启动微调，不是重新从零训练；它尚未在本地执行，也不保证当前残差权限足够纠偏。

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 --recipe robust_recovery_v15 \
  --geometry rollingquad_2 --physics-profile cg20 --curriculum none \
  --restore-params results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --reset-pose stand --stand-hold-s 0.2 --stand-to-compact-s 1.0 \
  --episode-length 560 --steps 10000000 --num-evals 20 --eval-envs 256 \
  --phase-rate-scale 1.0 --selection-target-turns 6 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --out results/rollingquad2_stand_start_h200_seed0
```

启动稳定后，再用相同启动配置复测此前的 reset/floor/mass/gain 条件。
新结果单独保存，不覆盖已通过的旧教师。

## 学生与实机边界

`train_mjx_3d_roll_distillation.py` 也接受上述三个开关以及 `--episode-length 560`。
先确认新教师能稳定完成整个 episode，再用新教师重新 BC + DAgger。
旧学生从未学过站立初态，不应直接当作能站立起滚的策略。

教师标签包含收腿先验和残差，学生输出完整电机命令。direct student 环境只在
reset 放置 stand，step 不叠加先验、不做插值、不强制保持站立；学生闭环评估必须
凭自己的动作完成整个启动。DAgger 的混合教师干预仍只发生在训练中。
这是为网络自主学习启动准备的接口，不是已向实机 C++ 控制器安装启动 supervisor。

策略频率仍为仿真 50 Hz / 实机 52 Hz；没有改成 IMU 的 260 Hz。
实机切换还需验证实际站立状态分布、历史观测初始化和首帧命令连续性。
从一个 stand keyframe 加小扰动训练，不等于已覆盖行走中直接切换等任意初态。
