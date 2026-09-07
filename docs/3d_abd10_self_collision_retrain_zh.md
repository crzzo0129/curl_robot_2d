# 前 ABD −10° / 后 ABD +10° 自碰撞重训与重新蒸馏（primitive 训练）

## 背景

之前的 teacher/student 蒸馏在未开启模型自碰撞下完成。用 `rollingquad.xml`
（前 −15°/后 +15°）加选择性自碰撞白名单回放时，前后同侧 shank 会互相碰撞
（`results/rolling_student_5s_self_collision_whitelist`：active 36.5%、最大穿透
约 1.9 mm），滚动几乎无法建立。

已确认：前腿 ABD **−10°**、后腿 ABD **+10°** 时不会发生自接触，且力矩不饱和。
结论见 [`pupper_reference_mesh_abduction_energy_20260907_zh.md`](pupper_reference_mesh_abduction_energy_20260907_zh.md)。

**训练只用 primitive 碰撞模型**（mesh-vs-mesh 接触对 MJX 太贵）。已烘焙模型：
`assets/rollingquad_description_2/mjcf/rollingquad_primitive_abd10.xml`（compact
前 −10°/后 +10°，保留 primitive 选择性自碰撞白名单与 foot/shank 显式 `<pair>`）。
完整 CAD mesh 版本 `rollingquad_abd10.xml` 只用于最终 CPU 自接触验证。

## 新几何

- `rollingquad_2_primitive_abd10`：训练用，模型 `rollingquad_primitive_abd10.xml`，
  CEM reference 复用 `rollingquad_primitive_stiff_cem/03_strict_10s/best_phase_controller.json`
  （CEM 只命令 hip/knee，ABD 完全来自模型 compact keyframe，因此无需重新 CEM）。
- `rollingquad_2_abd10`：验证用，完整 CAD mesh `rollingquad_abd10.xml`，
  复用 `rollingquad_2_3d_self_collision_cem_reference.json`。

自碰撞 contract 测试覆盖两个几何：
`test_rollingquad_abd10_uses_same_self_collision_and_baked_abduction` 与
`test_primitive_abd10_is_training_geometry_with_baked_abduction`。

## 1. 重训 teacher 最后阶段 `floor_mass_gain_v3`（primitive）

只重跑最后一个课程阶段，从 primitive 的 `floor_mass_v2` 热启动，几何换成 abd10：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe robust_recovery_v15 \
  --geometry rollingquad_2_primitive_abd10 \
  --physics-profile cg20 \
  --curriculum floor_mass_gain_v3 \
  --restore-params results/primitive_stiff_mass_v2_smoke_seed0/params_best \
  --episode-length 500 --num-evals 12 --eval-envs 64 \
  --phase-rate-scale 1.0 --selection-target-turns 6.0 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.50 \
  --out results/primitive_abd10_mass_gain_v3_smoke_seed0
```

smoke 通过后跑 H200（restore 换成 smoke 输出或已有 primitive `floor_mass_v2`
`params_best`）：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe robust_recovery_v15 \
  --geometry rollingquad_2_primitive_abd10 \
  --physics-profile cg20 \
  --curriculum floor_mass_gain_v3 \
  --restore-params results/primitive_stiff_mass_v2_h200_seed0/params_best \
  --episode-length 500 --num-evals 30 --eval-envs 256 \
  --phase-rate-scale 1.0 --selection-target-turns 6.0 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.80 \
  --out results/primitive_abd10_mass_gain_v3_h200_seed0
```

> `--restore-params` 是 primitive teacher 课程上一阶段 `floor_mass_v2` 的
> `params_best`，按你云端的实际目录名填写（`*_smoke_*` 或 `*_h200_*`）。

验收沿用 actuator-gain 阶段标准（三个 kp 端点成功率 ≥95%、nominal 圈数无回退、
物理失败为 0），并确认 rollout 中 `failure_forbidden_contact/depth` 为 0、无
front/rear shank 自接触。最终 checkpoint 为
`results/primitive_abd10_mass_gain_v3_h200_seed0/params_best`。

## 2. 重新蒸馏（primitive）

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  results/primitive_abd10_mass_gain_v3_h200_seed0/params_best \
  --geometry rollingquad_2_primitive_abd10 \
  --preset h200 \
  --out results/primitive_abd10_roll_distill_h200_seed0 \
  --mujoco-gl disable --memory-fraction 0.80
```

闭环评估应确认 `velocity_estimation_rmse`、`success_rate`、`failure_rates`（尤其
`failure_forbidden_contact`/`failure_forbidden_depth` 为 0）。导出
`student_rtneural.json` / `controller_config.json` 的 `default_joint_pos` 自动带 ±10°。

## 3. 验证（本地 CPU，完整 CAD mesh 自接触检查）

用完整 mesh 模型回放导出的 RTNeural JSON，确认自接触为 0：

```powershell
python -m scripts.analyze_rolling_student_5s \
  --mode simulate \
  --policy results/primitive_abd10_roll_distill_h200_seed0/student_rtneural.json \
  --model assets/rollingquad_description_2/mjcf/rollingquad_abd10.xml \
  --out results/primitive_abd10_roll_distill_h200_seed0/analyze_5s_mesh
```

`rolling_student_5s_summary.json` 的 `self_collision.active_step_fraction` 应接近 0。
（若只需验证 primitive 自身，把 `--model` 换成 `rollingquad_primitive_abd10.xml`。）
