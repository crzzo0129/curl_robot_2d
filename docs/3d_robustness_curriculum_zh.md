# 3D rolling 鲁棒性 curriculum

## 结论

不要一次性加入全部 randomization。当前 nominal reference 已经能稳定滚动，但 residual policy 在小扰动下反而增加横向失败；如果同时加入 reset、摩擦、质量和执行器误差，失败率变化将无法归因。

训练入口现在提供两个显式课程：

- `reset_v1`：只逐步增加左右非对称 reset 扰动；
- `robustness_v1`：先完整执行 `reset_v1`，再依次加入摩擦和动力学随机化。

默认仍为 `--curriculum none`，旧命令行为不变。

## 阶段

| stage | joint noise | qvel noise | 左右 differential | 轴倾斜 | 物理随机化 |
|---|---:|---:|---:|---:|---|
| `symmetric_reset` | 0.005 rad | 0.005 | 0 | 0 | 无 |
| `differential_005` | 0.005 rad | 0.005 | 0.05 | 0 | 无 |
| `differential_010` | 0.0075 rad | 0.010 | 0.10 | 0.005 rad | 无 |
| `differential_025` | 0.015 rad | 0.030 | 0.25 | 0.030 rad | 无 |
| `friction` | 同上 | 同上 | 同上 | 同上 | 所有 geom 摩擦乘以 U(0.90, 1.10) |
| `dynamics` | 同上 | 同上 | 同上 | 同上 | 摩擦 + body mass/inertia U(0.95, 1.05) + actuator gain U(0.95, 1.05) |

左右 joint 和 joint velocity 使用 common/differential 分解。`differential=0` 时左右对应关节的 reset 完全对称；之后只逐步放大差分分量。root 的六维初速度仍独立采样，所以 qvel noise 同时包含有效的侧向初速度与轴角速度扰动。初始 y 位置保持 0，因为无限平面在 y 方向平移不变，随机位置不会形成新的物理困难。

物理参数在每个并行环境中独立采样，并在一个 stage 内保持固定。reset 会在 stage 开始时采样，并在每个 eval interval 后重新采样；Brax 的 episode auto-reset 在 interval 内仍复用该环境的初始状态。

## 推荐顺序

先从现有 `params_best` 运行 reset-only smoke：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe phase_locked_coupled_v6 \
  --physics-profile cg20 \
  --curriculum reset_v1 \
  --restore-params results/mjx_3d_residual_cg20_ramp050_010_h200_seed0/params_best \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --mujoco-gl disable \
  --memory-fraction 0.50 \
  --out results/mjx_3d_reset_curriculum_smoke_seed0
```

smoke 能编译完成、各 stage 指标没有断崖式下降后，再运行 reset-only 长训练：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe phase_locked_coupled_v6 \
  --physics-profile cg20 \
  --curriculum reset_v1 \
  --restore-params results/mjx_3d_residual_cg20_ramp050_010_h200_seed0/params_best \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --mujoco-gl disable \
  --memory-fraction 0.80 \
  --out results/mjx_3d_reset_curriculum_h200_seed0
```

训练过程中的不同 stage 使用不同扰动分布，所以 stage 间的 failure rate 不能直接作因果比较。训练完成后，应让旧 checkpoint 和新 checkpoint 使用相同 seed、相同 1024 个最终 reset 分布分别评估：

```bash
python -m scripts.evaluate_mjx_3d_policy \
  results/mjx_3d_reset_curriculum_h200_seed0/params_best \
  --out results/mjx_3d_reset_curriculum_h200_seed0/eval_final_reset1024 \
  --batch-size 1024 \
  --chunk-size 128 \
  --episode-length 500 \
  --seed 0 \
  --physics-profile cg20 \
  --reset-joint-noise-rad 0.015 \
  --reset-velocity-noise 0.030 \
  --reset-pair-differential-scale 0.25 \
  --reset-axis-tilt-noise-rad 0.030 \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --mujoco-gl disable \
  --memory-fraction 0.20
```

把 checkpoint 和 `--out` 换成旧策略后再跑一次。只有新策略在同一批 reset 上降低 failure rate，且 turns 没有明显下降，才进入完整物理随机化：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe phase_locked_coupled_v6 \
  --physics-profile cg20 \
  --curriculum robustness_v1 \
  --restore-params results/mjx_3d_reset_curriculum_h200_seed0/params_best \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --mujoco-gl disable \
  --memory-fraction 0.80 \
  --out results/mjx_3d_robustness_curriculum_h200_seed0
```

## 消融

`--curriculum-stage` 可只运行一个 stage。例如从同一 checkpoint 分别运行 `differential_025` 和 `friction`，可以隔离摩擦随机化的影响：

```bash
--curriculum robustness_v1 --curriculum-stage differential_025
--curriculum robustness_v1 --curriculum-stage friction
```

每个 stage 保存 `params_stage_<index>_<name>_{best,final}`。总输出的 `params_best` 只从最后一个、也就是最难的 stage 中选择，避免较容易 stage 的分数覆盖鲁棒策略。完整记录位于 `training_config.json` 和 `curriculum_history.json`。

课程内部不会把 failure rate 超过 5% 的 checkpoint 全部丢弃，因为困难 stage 可能先经历 `20% -> 8%` 但尚未过线；这时仍按 survival、turns、横向漂移和接触安全选更好的 warm-start。5% 仍保留为最终 acceptance gate，训练末尾会明确打印 `PASS` 或 `NOT_YET`。训练脚本自带的 post-training rollout 不含新的物理参数随机化；完整物理 randomization 的独立 1024 eval 需要后续在 evaluator 中接入。
