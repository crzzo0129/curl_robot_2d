# 新版真实几何 3D nominal 避碰训练

## 目标

本阶段把新版 180 mm、腿径 50 mm、足端直径 60 mm 的 2D CEM 复制到左右两条 3D 侧轨，并训练一个小幅 residual policy，使其在保持滚动速度的同时减少足端与膝电机、大小腿之间的碰撞。

本阶段不是 domain-randomization 训练。通过后保存的 `params_best` 才作为新版模型 `reset -> friction -> mass/inertia -> actuator` 鲁棒性课程的起点。

训练固定使用：

- `geometry=real`；
- `curl_robot_3d_real_geometry.xml`；
- 新版 2D CEM controller；
- 足部目标间距 2 mm、tracking margin 12 mm；
- `phase_locked_coupled` 对称 residual，左右差分通道缩放为 0.25；
- nominal friction、mass/inertia 和 actuator gain；
- 仅保留 0.005 rad / 0.005 的轻微对称 reset noise；
- checkpoint 必须达到至少 5 个 conservative turns，随后才按内部接触质量排序。

## 云端 smoke

在 `curl_robot_2d` 目录运行：

```bash
python -m scripts.train_mjx_3d_real_geometry_nominal \
  --preset smoke \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.50 \
  --out results/mjx_3d_real_geometry_contact_v1_smoke_seed0
```

smoke 必须确认：新版 XML 能被 MJX 编译、训练配置打印 `geometry=real`，并且能够写出 `training_config.json`、`params_best` 和 `params_final`。如果所有 evaluation 都没有达到 5 圈，`params_best` 可能无法通过 contact gate；smoke 的主要目标是验证训练链路。

## H200 正式训练

```bash
python -m scripts.train_mjx_3d_real_geometry_nominal \
  --preset h200 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.80 \
  --out results/mjx_3d_real_geometry_contact_v1_h200_seed0
```

`h200` 默认使用现有 20M-step preset。若先进行 10M 探索，可增加 `--steps 10000000`，但不要写入同一个正式输出目录。

默认从零 residual policy 开始：零动作恰好等于新版 CEM reference。只有做显式 A/B 实验时才使用 `--restore-params` 热启动旧模型 checkpoint。

## Reference 基线评估

训练前先在云端保存一份 zero-residual reference 基线：

```bash
python -m scripts.evaluate_mjx_3d_policy \
  --out results/mjx_3d_real_geometry_reference_eval_seed0 \
  --evaluation-mode reference \
  --geometry real \
  --controller results/staged_cem_real_geometry_180_d50_foot60/03_foot_gap_2mm/best_phase_controller.json \
  --minimum-foot-gap-mm 2 \
  --foot-gap-tracking-margin-mm 12 \
  --batch-size 1024 \
  --chunk-size 128 \
  --episode-length 500 \
  --seed 0 \
  --reset-joint-noise-rad 0.005 \
  --reset-velocity-noise 0.005 \
  --reset-pair-differential-scale 0.0 \
  --reset-axis-tilt-noise-rad 0.0 \
  --reference-ramp-start-scale 0.25 \
  --reference-ramp-duration-s 0.25 \
  --physics-profile cg20 \
  --mujoco-gl disable \
  --memory-fraction 0.20
```

## Policy 验收评估

使用完全相同的 1024 个 reset 评估 `params_best`：

```bash
python -m scripts.evaluate_mjx_3d_policy \
  results/mjx_3d_real_geometry_contact_v1_h200_seed0/params_best \
  --out results/mjx_3d_real_geometry_contact_v1_h200_seed0/eval_nominal_reset1024 \
  --geometry real \
  --controller results/staged_cem_real_geometry_180_d50_foot60/03_foot_gap_2mm/best_phase_controller.json \
  --minimum-foot-gap-mm 2 \
  --foot-gap-tracking-margin-mm 12 \
  --reference-weight 1.0 \
  --minimum-residual-gain 0.20 \
  --residual-pair-differential-scale 0.25 \
  --initial-policy-std 0.15 \
  --batch-size 1024 \
  --chunk-size 128 \
  --episode-length 500 \
  --seed 0 \
  --reset-joint-noise-rad 0.005 \
  --reset-velocity-noise 0.005 \
  --reset-pair-differential-scale 0.0 \
  --reset-axis-tilt-noise-rad 0.0 \
  --reference-ramp-start-scale 0.25 \
  --reference-ramp-duration-s 0.25 \
  --physics-profile cg20 \
  --mujoco-gl disable \
  --memory-fraction 0.20
```

`deterministic_eval.json` 现在额外报告：

- `forbidden_contact_fraction`；
- `first_turn_forbidden_contact_fraction`；
- `maximum_forbidden_penetration_m`；
- `actuator_torque_rms_Nm`；
- `maximum_actuator_torque_Nm`；
- `maximum_actuator_torque_per_joint_Nm`；
- `torque_saturation_fraction`。

建议验收条件：

- failure rate 不高于 5%；
- conservative turns 的中位数至少为 5；
- 第一圈足端撞腿显著低于 reference 基线，目标为 0；
- 总 forbidden-contact fraction 显著低于 reference；
- 最大穿透不恶化；
- torque saturation fraction 为 0；
- 横向漂移和 rolling-axis tilt 不恶化。

## 渲染诊断 rollout

评估时增加 `--diagnostic-rollouts 4`，然后渲染保存的 rollout：

```bash
python -m scripts.render_mjx_3d_policy \
  results/mjx_3d_real_geometry_contact_v1_h200_seed0/eval_nominal_reset1024/diagnostic_rollouts \
  --geometry real \
  --physics-profile cg20 \
  --mujoco-gl egl
```

不要只依据训练过程的平均 reward 验收；必须比较同 seed 的 reference 与 policy deterministic evaluation。
