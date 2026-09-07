# 3D 直线速度 command tracking（v_cmd）实现记录

更新时间：2026-09-07

## 目标

让 `train_mjx_3d_residual_ppo.py` 接收直线速度 command `v_cmd`，用主高速 reference
的 `target-scale` 查表生成 `q_ref(v_cmd, phase)`，训练第一版 8 维 residual action 的
直线速度 tracking 策略。速度奖励使用 command-tracking Gaussian，而不是单纯奖励高速。

## 数据基础：v_cmd → target_scale 查表

来自 `results/rollingquad_abd10_target_scale_probe/`（10 s 扫描，kp=5 kd=0.1
torque_limit=3，前 −10° / 后 +10°，cg20）：

- 单调可训练区间：`target_scale ∈ [0.36, 1.00]` → `v ∈ [0.41, 0.81] m/s`。
- `≤ 0.35` 失锁倒滚；`≥ 1.05` 自碰撞。
- 查表为 `FORWARD_COMMAND_LOOKUP_SPEEDS_M_S` / `FORWARD_COMMAND_LOOKUP_SCALES`
  （`curl_robot_2d_mjx/environment_3d.py`），函数
  `forward_command_to_target_scale_3d(xp, v_cmd)` 用分段线性插值反查 target_scale。

参考-only 端到端复核（CPU physics 10 s）：

| v_cmd | scale | 实际速度 | 误差 | 自接触 |
|---:|---:|---:|---:|---:|
| 0.45 | 0.387 | 0.450 | 0.000 | 0 |
| 0.50 | 0.428 | 0.498 | −0.002 | 0 |
| 0.55 | 0.473 | 0.555 | +0.005 | 0 |
| 0.60 | 0.520 | 0.603 | +0.003 | 0 |
| 0.65 | 0.570 | 0.646 | −0.004 | 0 |
| 0.70 | 0.633 | 0.702 | +0.002 | 0 |
| 0.75 | 0.710 | 0.751 | +0.001 | 0 |
| 0.80 | 0.885 | 0.800 | 0.000 | 0 |

误差 ≤ 0.005 m/s，全部零自接触。

## 代码改动

### `curl_robot_2d_mjx/reward_3d.py`
- `REWARD_3D_TERM_NAMES` 新增 `forward_velocity`。
- `Rolling3DRewardConfig` 新增 `forward_velocity`（权重）、
  `forward_velocity_sigma_m_s`（默认 0.10）。
- `reward_terms_3d` 新增：
  `forward_velocity * exp(-(v_x - v_cmd)^2 / sigma^2)`。

### `curl_robot_2d_mjx/config_3d.py`
- `Rolling3DConfig` 新增 `forward_command_enabled`（默认 False）、
  `forward_command_min_m_s`（0.47）、`forward_command_max_m_s`（0.80）、
  `forward_command_fixed_m_s`（None）。
- `validate_3d_config` 增加校验（min/max 正且 min ≤ max，fixed 有限）。

### `curl_robot_2d_mjx/environment_3d.py`
- `OBSERVATION_SIZE_3D` 61 → 62（新增 `forward_velocity_command`，反射偶）。
- 新增查表常量 + `forward_command_to_target_scale_3d`。
- `reference_startup_scale_3d` 增加可选 `target_scale` 参数（向后兼容）。
- `reset`：采样 `forward_velocity_command`，算 `forward_command_scale`，写入 info。
- `_scaled_reference_action_8d` 增加可选 `command_scale`，作为 effective target scale。
- `step`：算 `forward_velocity_m_s`，加入 reward inputs 与 metrics
  （`forward_velocity_error_m_s` / `forward_velocity_error_abs_m_s`）。
- 观测新增 `forward_velocity_command` 标量。

### `curl_robot_2d_mjx/environment_autonomous_startup_3d.py` / `scripts/probe_3d_roll_handoff.py`
- 适配 `_observation` 新参数 `forward_velocity_command`。

### `scripts/train_mjx_3d_residual_ppo.py`
- 新增 CLI：`--forward-command-enabled`、`--forward-command-min-m-s`、
  `--forward-command-max-m-s`、`--forward-command-fixed-m-s`。
- 新 recipe `command_tracking_v1`（roll_progress 降为 1.0，forward_velocity=8.0）。
- `Rolling3DConfig` 接入 forward command 字段。
- `PER_STEP_EVAL_METRICS_3D`、eval 报告、checkpoint selection 增加
  forward-velocity 误差（`command_quality` 权重 0.10，进入 balanced objective）。

### `scripts/evaluate_mjx_3d_policy.py`
- 新增 CLI：`--forward-command-enabled`、`--forward-command-min-m-s`、
  `--forward-command-max-m-s`、`--forward-command-fixed-m-s`（评估时固定 v_cmd）。
- summary 新增 `forward_velocity_command_m_s`、`average_forward_velocity_m_s`、
  `average_forward_velocity_error_abs_m_s`（速度误差/自碰撞报告）。

## 训练命令（Linux GPU 实例；本 Windows 机器无 jax/brax/flax）

```bash
cd curl_robot_2d

# smoke（最小验证）
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe command_tracking_v1 \
  --geometry rollingquad_2_primitive_abd10 \
  --physics-profile cg20 \
  --controller results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json \
  --out results/mjx_3d_forward_command_smoke

# 正式训练（4090 preset）
python -m scripts.train_mjx_3d_residual_ppo \
  --preset 4090 \
  --recipe command_tracking_v1 \
  --geometry rollingquad_2_primitive_abd10 \
  --physics-profile cg20 \
  --controller results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json \
  --out results/mjx_3d_forward_command_v1
```

注意：训练用 `rollingquad_2_primitive_abd10`（解析碰撞近似，速度最快），
`--controller` 必须显式指向主高速 reference，否则会用 geometry 默认 reference。

训练后评估（固定 v_cmd，输出速度误差/自碰撞）：

```bash
python -m scripts.evaluate_mjx_3d_policy \
  params_best \
  --out results/mjx_3d_forward_command_v1/eval_vcmd_060 \
  --geometry rollingquad_2_primitive_abd10 \
  --physics-profile cg20 \
  --controller results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json \
  --forward-command-fixed-m-s 0.60 \
  --zero-residual-policy-init
```

## 评估指标

- `eval/avg_forward_velocity_command`：平均 v_cmd。
- `eval/avg_forward_velocity_m_s`：平均实际前向速度。
- `eval/avg_forward_velocity_error_m_s`：平均速度误差（带符号）。
- `eval/avg_forward_velocity_error_abs_m_s`：平均绝对速度误差（用于 checkpoint 选择）。

## 单元测试

新增/更新测试（numpy-only，全部通过）：

- `tests/test_mjx_3d_reward.py`：forward_velocity Gaussian 奖励；修复 zero_inputs 缺
  `same_side_foot_gap`（既有缺口）。
- `tests/test_mjx_3d_contract.py`：forward command 默认/校验/查表单调与往返。
- `tests/test_mjx_3d_training.py`：`command_tracking_v1` recipe 默认值。

## 注意事项

- 第一版只做直线 v_cmd；yaw_rate_cmd 后续再加（§14 已有 8 维差分转向能力）。
- 低速（< 0.41 m/s）与停止不靠 reference 幅值缩放，属于独立 curriculum（§18）。
- 不要用 full 高速 CEM 结果（自碰撞换速度）。
