# 3D rolling terrain-adaptive（坡度）实施

## 目标

在平地教师稳定后，让机器人能适应上坡、下坡和坡度变化。第一阶段只做**小坡度**
（±2°）的 teacher curriculum，性能稳定后再扩坡；蒸馏随后在同一坡度分布上继续
DAgger。

## 地形表示：flat → slope → flat 高度场

- 滚动方向为世界 +x，地形高度只随 x 变化、横向（y）恒定。
- 剖面是 smoothstep 进入/退出过渡 + 恒定坡度段，避免机器人撞上尖角。
- 用 MuJoCo **hfield** 表示：`surface = base + scale * data`，`hfield_data` 列主序
  （x 外层），且必须 ≥0（负值会丢失碰撞——已实证）。
- 下坡地形整体上移 `|total_rise|`，机器人从高处平地起步；重置 z 加
  `terrain_surface_offset`，终止判断用 `root_z - terrain_surface_z_at(root_x)`。

## 新增模块与开关

- `curl_robot_2d_mjx/terrain_3d.py`：`SlopeTerrainConfig`、剖面、hfield 数据、
  表面高度、候选（flat/uphill/downhill）、`inject_hfield_into_mjcf`（烘焙）。
- `config_3d.py`：`Rolling3DConfig` 增加 `terrain_enabled` 等 7 个字段，默认关闭，
  平地流程不受影响。
- `environment_3d.py`：`terrain_enabled` 时加载 hfield 模型并写 `hfield_data`；
  reset 加地表偏移；root_z 终止与观测改为地形相对。
- `randomization_3d.py`：`make_terrain_randomization_fn_3d`（纯地形）与
  `make_domain_randomization_with_terrain_fn_3d`（物理 DR + 地形合并）。
- `curriculum_3d.py`：`slope_v1` 课程（±2°、70% 平地 / 15% 上坡 / 15% 下坡）。
- 烘焙地形模型：`rollingquad_primitive_abd10_terrain.xml`（训练）、
  `rollingquad_abd10_terrain.xml`（完整 mesh 验证）。

## `slope_v1` 课程

单阶段 `slope_02`：保留平地 `floor_mass_gain_v3` 的 reset（八关节 0.005 独立噪声）
与物理 DR（floor 摩擦 U(0.90,1.10)、质量/惯量 U(0.95,1.05)、kp U(0.95,1.05)），
只新增地形：70% 环境为平地，其余 30% 均分为 +2° 上坡 / −2° 下坡。

## 重训 teacher（从平地 `floor_mass_gain_v3` 热启动）

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 --recipe robust_recovery_v15 \
  --geometry rollingquad_2_primitive_abd10 --physics-profile cg20 \
  --curriculum slope_v1 \
  --restore-params results/primitive_abd10_mass_gain_v3_h200_seed0/params_best \
  --episode-length 500 --num-evals 30 --eval-envs 256 \
  --phase-rate-scale 1.0 --selection-target-turns 6.0 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.80 \
  --out results/primitive_abd10_slope02_h200_seed0
```

验收：上坡/下坡/平地三者分别的成功率，`failure_forbidden_contact/depth` 为 0，
nominal 圈数无明显回退。上坡常见失败是失速/姿态崩溃，下坡是过速/振荡，需要分开
统计（当前评估指标里 `failure_*` 已能区分，后续补 per-terrain 分组）。

## 坡度蒸馏

`train_mjx_3d_roll_distillation.py` 已接入 `--terrain-enabled` 开关：BC 阶段用平地
教师数据热启动，DAgger 阶段用每环境 flat/uphill/downhill 地形混合
（`--terrain-slope-probability` 默认 0.30、`--terrain-max-angle-deg` 默认 2.0）让
学生在自己访问的坡度状态上继续学习教师标签。教师标签沿用 DAgger 的「同状态查询」
近似（`teacher_env` 以平地模型步进，标签经 65 维地形感知观测得到）。

```bash
python -m scripts.train_mjx_3d_roll_distillation \
  results/primitive_abd10_slope02_h200_seed0/params_best \
  --geometry rollingquad_2_primitive_abd10 \
  --terrain-enabled --terrain-slope-probability 0.30 --terrain-max-angle-deg 2.0 \
  --preset h200 \
  --out results/primitive_abd10_slope02_roll_distill_h200_seed0 \
  --mujoco-gl disable --memory-fraction 0.80
```

导出 `student_rtneural.json` / `controller_config.json` 的 `default_joint_pos` 仍带
±10° ABD；闭环评估的 `velocity_estimation_rmse`、`success_rate`、`failure_rates`
（尤其 forbidden contact/depth）按平地蒸馏同一套指标输出。

## 扩坡（Stage 5）

`slope_v2` 课程按 ±2° → ±4° → ±6° → ±8° → ±10° 五阶段扩坡，每阶段仍保持 70% 平地 /
30% 坡度（15% 上坡 / 15% 下坡）。**逐阶段推进**：先跑 `slope_02`，验证上/下/平地
成功率达标后再 `--curriculum-stage slope_04` 续训（`--restore-params` 用上一阶段
`params_best`），不要在成功率不足时直接跳到大坡。

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 --recipe robust_recovery_v15 \
  --geometry rollingquad_2_primitive_abd10 --physics-profile cg20 \
  --curriculum slope_v2 --curriculum-stage slope_02 \
  --restore-params results/primitive_abd10_slope02_h200_seed0/params_best \
  --episode-length 500 --num-evals 30 --eval-envs 256 \
  --phase-rate-scale 1.0 --selection-target-turns 6.0 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.80 \
  --out results/primitive_abd10_slope04_h200_seed0
```

（自动化的「成功率阈值 > 85% 才升级」运行时门控尚未实现，当前用逐阶段手动验证替代。）

## 尚未完成

1. **地形相对朝向奖励**：当前沿用 `axis_tilt`（滚动轴对齐），`z_body · n_terrain`
   型朝向奖励留作后续细化。
2. **蒸馏的 per-terrain 分组统计**：上坡/下坡/平地分开报成功率（当前只报总的
   `failure_*` 指标）。
3. **自动运行时扩坡门控**：`slope_v2` 目前是逐阶段手动推进。
