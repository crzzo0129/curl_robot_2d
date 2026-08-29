# 3D rolling actuator gain curriculum

## `floor_mass_gain_v3`

该课程从已通过的
`results/rollingquad2_floor_mass_v2_h200_seed0/params_best` 热启动，并固定保留：

- 八关节 q/qdot 独立 reset 噪声 `0.005 / 0.005`；
- root velocity 和 axis tilt 为 0；
- floor 接触摩擦 `U(0.90, 1.10)`；
- 每个 body 独立且质量/惯量耦合的 `U(0.95, 1.05)`；
- 全局 geom friction scale 为 1。

唯一新增变量是每个 actuator 独立采样的 position gain，也就是 `kp` 缩放。
`kd=0.1` 和 ±3 Nm 力矩限制保持不变：

| stage | 权重 | kp scale | floor friction | body mass/inertia |
|---|---:|---:|---:|---:|
| `floor_mass_gain_02` | 0.30 | U(0.98, 1.02) | U(0.90, 1.10) | U(0.95, 1.05) |
| `floor_mass_gain_05` | 0.70 | U(0.95, 1.05) | U(0.90, 1.10) | U(0.95, 1.05) |

## Smoke

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe robust_recovery_v15 \
  --geometry rollingquad_2 \
  --physics-profile cg20 \
  --curriculum floor_mass_gain_v3 \
  --restore-params results/rollingquad2_floor_mass_v2_h200_seed0/params_best \
  --episode-length 500 --num-evals 12 --eval-envs 64 \
  --phase-rate-scale 1.0 --selection-target-turns 6.0 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.50 \
  --out results/rollingquad2_floor_mass_gain_v3_smoke_seed0
```

## H200

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe robust_recovery_v15 \
  --geometry rollingquad_2 \
  --physics-profile cg20 \
  --curriculum floor_mass_gain_v3 \
  --restore-params results/rollingquad2_floor_mass_v2_h200_seed0/params_best \
  --episode-length 500 --num-evals 30 --eval-envs 256 \
  --phase-rate-scale 1.0 --selection-target-turns 6.0 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.80 \
  --out results/rollingquad2_floor_mass_gain_v3_h200_seed0
```

训练后用 `evaluate_mjx_3d_policy --actuator-gain-scale` 固定测试 0.95、1.00、
1.05 三个 kp 端点，并继续使用相同 reset 和 environment seed 做 paired comparison。
三个端点成功率均达到 95%、nominal 圈数无明显下降，并且所有物理失败项为 0，
才进入下一类扰动。
