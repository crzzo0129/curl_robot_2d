# 3D rolling 鲁棒性 curriculum

## 当前结论

不要一次加入 reset、摩擦、质量和执行器误差。当前已通过的 `reset_v2` 负责初始状态鲁棒性；下一步使用独立的 `friction_v1`，从 `reset_v2/params_best` 热启动，只扩大摩擦系数范围。这样失败率变化可以归因于摩擦，而不是混杂的模型变化。

训练入口提供四个显式课程：

- `reset_v1`：旧版 reset 课程，用于复现；
- `reset_v2`：固定 joint/qvel/differential 扰动，逐步增加 axis tilt；
- `friction_v1`：保持 `reset_v2` 最终 reset 难度，只逐步扩大摩擦随机化；
- `robustness_v1`：旧版 `reset_v1 + friction + dynamics`，保留兼容，不用于当前 checkpoint 的续训。

默认仍为 `--curriculum none`。

## `reset_v2 -> friction_v1` 的边界

本阶段冻结以下项目：

- torso 和其他几何碰撞体；
- 质量、惯量和 actuator gain；
- reference/controller、reward、termination 和 observation；
- reset 分布：joint noise `0.015 rad`、qvel noise `0.030`、左右 differential `0.25`、axis tilt `0.030 rad`。

唯一训练变量是所有 geom 的摩擦三元组统一乘以一个标量。每个并行环境独立采样该标量；同一训练 interval 内保持固定，随后重采样。因此这是“全局摩擦不确定性”，同时影响地面接触和机器人自接触，不是只改变 floor。若以后只想随机地面摩擦，应新建课程版本，不能沿用 `friction_v1` 的结果含义。

## `friction_v1` 阶段

| stage | 训练权重 | 摩擦缩放 | joint | qvel | differential | axis tilt |
|---|---:|---:|---:|---:|---:|---:|
| `friction_02` | 0.20 | U(0.98, 1.02) | 0.015 rad | 0.030 | 0.25 | 0.030 rad |
| `friction_05` | 0.30 | U(0.95, 1.05) | 0.015 rad | 0.030 | 0.25 | 0.030 rad |
| `friction_10` | 0.50 | U(0.90, 1.10) | 0.015 rad | 0.030 | 0.25 | 0.030 rad |

H200 preset 的名义预算为 20M steps，按 `0.20/0.30/0.50` 分配。受 PPO batch 粒度向上取整影响，当前配置打印的实际 effective steps 分别为 4,587,520、6,553,600、10,485,760。`params_best` 只从最后的 `friction_10` stage 中选取；每个阶段另外保存 `params_stage_<index>_<name>_{best,final}`。

## 训练命令

以下命令从 `curl_robot_2d` 目录运行。先跑 smoke，确认三个 stage 都能编译和保存：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe phase_locked_coupled_v6 \
  --physics-profile cg20 \
  --curriculum friction_v1 \
  --restore-params results/mjx_3d_reset_v2_h200_seed0/params_best \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.50 \
  --out results/mjx_3d_friction_v1_smoke_seed0
```

smoke 正常后运行正式训练：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe phase_locked_coupled_v6 \
  --physics-profile cg20 \
  --curriculum friction_v1 \
  --restore-params results/mjx_3d_reset_v2_h200_seed0/params_best \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.80 \
  --out results/mjx_3d_friction_v1_h200_seed0
```

`--restore-params` 是必要的课程衔接条件：`friction_v1` 自己不会重复训练 `reset_v2`。输出目录中的 `seed0` 只是命名；真正控制随机数的是 `--seed 0`。

## 固定摩擦端点评估

独立 evaluator 现在支持 `--geom-friction-scale`。它使用固定物理端点，不在一个 batch 内连续随机摩擦；这适合做可复现的 0.90、1.00、1.10 三点验收。三次都使用相同 seed 和相同 1024 个 reset，属于 paired evaluation。

```bash
python -m scripts.evaluate_mjx_3d_policy \
  results/mjx_3d_friction_v1_h200_seed0/params_best \
  --out results/mjx_3d_friction_v1_h200_seed0/eval_friction_090_reset1024 \
  --batch-size 1024 --chunk-size 128 --episode-length 500 --seed 0 \
  --physics-profile cg20 --geom-friction-scale 0.90 \
  --reset-joint-noise-rad 0.015 --reset-velocity-noise 0.030 \
  --reset-pair-differential-scale 0.25 --reset-axis-tilt-noise-rad 0.030 \
  --reference-ramp-start-scale 0.50 --reference-ramp-duration-s 0.10 \
  --mujoco-gl disable --memory-fraction 0.20

python -m scripts.evaluate_mjx_3d_policy \
  results/mjx_3d_friction_v1_h200_seed0/params_best \
  --out results/mjx_3d_friction_v1_h200_seed0/eval_friction_100_reset1024 \
  --batch-size 1024 --chunk-size 128 --episode-length 500 --seed 0 \
  --physics-profile cg20 --geom-friction-scale 1.00 \
  --reset-joint-noise-rad 0.015 --reset-velocity-noise 0.030 \
  --reset-pair-differential-scale 0.25 --reset-axis-tilt-noise-rad 0.030 \
  --reference-ramp-start-scale 0.50 --reference-ramp-duration-s 0.10 \
  --mujoco-gl disable --memory-fraction 0.20

python -m scripts.evaluate_mjx_3d_policy \
  results/mjx_3d_friction_v1_h200_seed0/params_best \
  --out results/mjx_3d_friction_v1_h200_seed0/eval_friction_110_reset1024 \
  --batch-size 1024 --chunk-size 128 --episode-length 500 --seed 0 \
  --physics-profile cg20 --geom-friction-scale 1.10 \
  --reset-joint-noise-rad 0.015 --reset-velocity-noise 0.030 \
  --reset-pair-differential-scale 0.25 --reset-axis-tilt-noise-rad 0.030 \
  --reference-ramp-start-scale 0.50 --reference-ramp-duration-s 0.10 \
  --mujoco-gl disable --memory-fraction 0.20
```

为了判断训练是否真的带来提升，把上面三个命令中的 checkpoint 换成 `results/mjx_3d_reset_v2_h200_seed0/params_best`，输出到另一个目录，再以完全相同的 `--seed 0` 重跑。不要用不同 seed 的两个比例直接比较；相同 reset key 才能定位哪些样本由失败变成功。

## 验收标准

建议把以下条件同时作为通过标准：

- 0.90、1.00、1.10 三个端点各自 failure rate 不高于 5%；
- nonfinite、forbidden contact/depth、root-height 和 axis-tilt 类失败均为 0；
- nominal（1.00）median turns 相比 `reset_v2` 基线下降不超过 5%；
- 0.90 和 1.10 的 median turns 均至少达到 nominal 的 90%；
- lateral drift 的中位数及高分位没有明显恶化。

若只在某一个端点失败，不要立即加入 mass 或 gain 随机化；先检查失败样本是否集中在同一种接触模式，再决定调整训练权重、摩擦范围或 reward。通过本阶段后，下一阶段才是独立的 mass/inertia curriculum，随后是 actuator gain，最后才合并 torso 碰撞体分支并做一次统一回归。
