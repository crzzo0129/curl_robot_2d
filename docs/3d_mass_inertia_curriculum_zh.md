# 3D rolling mass/inertia curriculum

## 设计边界

`mass_v1` 从已经通过独立端点评估的 `friction_v1/params_best` 热启动。本阶段不修改 torso 碰撞体、reference、reward、termination、observation 或 actuator gain。

训练期间始终保留：

- reset joint noise：`0.015 rad`；
- reset qvel noise：`0.030`；
- reset pair differential：`0.25`；
- reset axis tilt：`0.030 rad`；
- geom friction：`U(0.90, 1.10)`。

每个非 world body 独立采样一个质量缩放值，并使用同一个值同时缩放该 body 的质量和主惯量：

```text
body_mass    *= scale
body_inertia *= scale
```

这样保留该 body 的回转半径，避免第一版课程产生质量与惯量互相矛盾的刚体参数。由于各 body 独立采样，课程也包含左右质量差和重心偏移。

## 阶段

| stage | 权重 | friction | 每个 body 的 mass/inertia |
|---|---:|---:|---:|
| `mass_02` | 0.30 | U(0.90, 1.10) | U(0.98, 1.02) |
| `mass_05` | 0.70 | U(0.90, 1.10) | U(0.95, 1.05) |

H200 的名义预算是20M steps。受 PPO batch 粒度向上取整影响，当前实际计划为 `mass_02=6,553,600`、`mass_05=15,728,640` effective steps。

`params_best` 只从最终 `mass_05` 阶段选择。±10% mass/inertia 不属于 `mass_v1` 的强制目标；通过 ±5% 后再根据失败模式决定是否建立 `mass_v2`。

## Smoke

在 `curl_robot_2d` 目录运行：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe phase_locked_coupled_v6 \
  --physics-profile cg20 \
  --curriculum mass_v1 \
  --restore-params results/mjx_3d_friction_v1_h200_seed0/params_best \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.50 \
  --out results/mjx_3d_mass_v1_smoke_seed0
```

## H200 正式训练

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe phase_locked_coupled_v6 \
  --physics-profile cg20 \
  --curriculum mass_v1 \
  --restore-params results/mjx_3d_friction_v1_h200_seed0/params_best \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.80 \
  --out results/mjx_3d_mass_v1_h200_seed0
```

## 固定端点 paired evaluation

以下 Bash 函数让所有 case 使用相同的1024个 reset。全体质量缩放和左右附加缩放都会同时作用于质量与惯量。

```bash
run_mass_eval () {
  local tag="$1"
  local friction="$2"
  local mass="$3"
  local left="$4"
  local right="$5"

  python -m scripts.evaluate_mjx_3d_policy \
    results/mjx_3d_mass_v1_h200_seed0/params_best \
    --out "results/mjx_3d_mass_v1_h200_seed0/eval_${tag}_reset1024" \
    --batch-size 1024 \
    --chunk-size 128 \
    --episode-length 500 \
    --seed 0 \
    --physics-profile cg20 \
    --geom-friction-scale "$friction" \
    --body-mass-scale "$mass" \
    --body-mass-left-scale "$left" \
    --body-mass-right-scale "$right" \
    --reset-joint-noise-rad 0.015 \
    --reset-velocity-noise 0.030 \
    --reset-pair-differential-scale 0.25 \
    --reset-axis-tilt-noise-rad 0.030 \
    --reference-ramp-start-scale 0.50 \
    --reference-ramp-duration-s 0.10 \
    --mujoco-gl disable \
    --memory-fraction 0.20
}

run_mass_eval nominal       1.00 1.00 1.00 1.00
run_mass_eval f090_m095     0.90 0.95 1.00 1.00
run_mass_eval f090_m105     0.90 1.05 1.00 1.00
run_mass_eval f110_m095     1.10 0.95 1.00 1.00
run_mass_eval f110_m105     1.10 1.05 1.00 1.00
run_mass_eval left_heavy    1.00 1.00 1.05 0.95
run_mass_eval right_heavy   1.00 1.00 0.95 1.05
```

相同七个 case 还应把 checkpoint 换回 `friction_v1/params_best` 重跑一次，输出到单独目录。保持相同 `--seed 0`，才能逐个 reset 比较新旧策略。

## 验收标准

- 七个固定 case 的 failure rate 均不高于5%；
- nonfinite、root-height、axis-tilt、forbidden contact/depth failure 均为0；
- nominal median turns 相比 `friction_v1` 基线下降不超过5%；
- 四个 friction/mass 组合端点的 median turns 至少达到 nominal 的90%；
- left-heavy 与 right-heavy 的失败率和横漂分布不能出现明显单侧偏置；
- `params_best` 必须来自 `mass_05`。

训练内部的随机模型评估与上述固定端点评估都通过后，才算完成 `mass_v1`。随后冻结 mass/inertia 分布，再单独建立 actuator-gain curriculum。
