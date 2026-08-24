# 3D rolling 鲁棒性 curriculum

## 当前结论

Reference 已经提供名义滚动能力。当前 residual PPO 的目标不是重新学习滚动，而是在八个关节受到真实独立初态扰动后恢复 reference 轨道，并在不需要恢复时保持 residual 接近零。

当前推荐的新实验组合是：

- `--recipe robust_recovery_v15`：使用有界 Huber 状态代价、误差下降恢复奖励和更强的 residual reference 锚定；
- 已完成的 reset 基线使用 `--curriculum independent_reset_v4`：从第一阶段开始八个活动关节始终独立采样，只增加 q/qdot 噪声幅度；
- 当前下一阶段使用 `--curriculum floor_friction_v2`：从已通过的 v4 `params_best` 热启动，保持目标独立 reset 不变，只扩展地面接触摩擦；
- `--physics-profile cg20`：训练与 reference 验证使用相同求解器。

旧 `reset_v1/reset_v2/nominal_reset_v3` 和旧 recipe 全部保留用于复现实验，但不再作为当前独立关节鲁棒性实验的推荐入口。`friction_v1` 也保留原来的全局 geom 摩擦语义。质量、执行器和 root 扰动仍不与当前摩擦实验同时加入。

默认仍为 `--curriculum none`。

## `independent_reset_v4`

v4 不再使用 `symmetric -> differential -> independent` 的相关结构 curriculum。六个阶段均令 `reset_pair_differential_scale=None`，即八个活动关节的位置与速度分别独立均匀采样。课程只改变幅度，最终阶段 100% 等于当前目标独立噪声分布。各阶段在代码中显式固定 `reset_root_velocity_noise=0` 和 `reset_axis_tilt_noise_rad=0`，不会继承基础配置中的非零值。

| stage | 训练权重 | joint q noise | joint qdot noise | 结构 | root velocity | axis tilt |
|---|---:|---:|---:|---|---:|---:|
| `reset4_independent_0005` | 0.05 | 0.0005 rad | 0.0005 | independent | 0 | 0 |
| `reset4_independent_0010` | 0.10 | 0.001 rad | 0.001 | independent | 0 | 0 |
| `reset4_independent_0020` | 0.15 | 0.002 rad | 0.002 | independent | 0 | 0 |
| `reset4_independent_0030` | 0.20 | 0.003 rad | 0.003 | independent | 0 | 0 |
| `reset4_independent_0040` | 0.20 | 0.004 rad | 0.004 | independent | 0 | 0 |
| `reset4_independent_0050` | 0.30 | 0.005 rad | 0.005 | independent | 0 | 0 |

阶段间仍传递当前阶段 `params_best`，不是 `params_final`。每个阶段至少包含 step 0 和训练后的 eval；正式实验应增加 eval 数和 eval batch，不能用 8 个样本的 12.5% 跳变判断鲁棒性。

## `robust_recovery_v15` reward

旧正高斯稳定奖励在 v15 中全部置零。定义归一化有界 Huber：

```text
z = abs(error) / sigma
Huber(z) = 0.5*z^2                 z <= 1
           z - 0.5                 z > 1
bounded_huber = min(Huber(z), 1)
```

综合稳定性代价为：

```text
E = 0.25 * Huber(vy / 0.20 m/s)
  + 1.00 * Huber((y-y0) / 0.10 m)
  + 0.25 * Huber(yaw_rate / 0.30 rad/s)
  + 0.50 * Huber(yaw / 0.10 rad)
```

每一步新增：

```text
state_cost = -E
recovery = 4.0 * clip(E_previous - E, -0.25, 0.25)
residual_anchor = -0.05 * mean(residual_action^2)
```

因此误差下降得到正 recovery，误差扩大得到负 recovery，保持偏移仍持续承担 state cost。`roll_progress`、`roll_mismatch`、碰撞、failure progress clawback 和 termination 保留。训练日志会分别输出四个负 `*_cost` reward、标为 `delta` 的已加权 recovery reward，以及正值 `stability_error_cost` metric。

## 当前训练命令

先从零运行 smoke，不恢复旧 PPO 参数：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe robust_recovery_v15 \
  --geometry pupper_open60 \
  --controller results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json \
  --physics-profile cg20 \
  --curriculum independent_reset_v4 \
  --episode-length 500 \
  --num-evals 24 \
  --eval-envs 64 \
  --reset-root-velocity-noise 0 \
  --reset-axis-tilt-noise-rad 0 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.50 \
  --out results/independent_reset_v4_recovery_v15_cg20_smoke
```

smoke 确认 reward 符号、恢复项和六阶段 checkpoint 都正确后，再运行 H200：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe robust_recovery_v15 \
  --geometry pupper_open60 \
  --controller results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json \
  --physics-profile cg20 \
  --curriculum independent_reset_v4 \
  --episode-length 500 \
  --num-evals 30 \
  --eval-envs 256 \
  --reset-root-velocity-noise 0 \
  --reset-axis-tilt-noise-rad 0 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.80 \
  --out results/independent_reset_v4_recovery_v15_cg20_h200_seed0
```

最终验收以 `reset4_independent_0050` 的 paired evaluation 为主，同时单独跑 zero-noise reference retention。训练通过后才建立 root velocity curriculum；root x/y 的物理平移在无限平地上不构成有效扰动，定位误差应作为 observation noise 单独处理。

## `floor_friction_v2`

`independent_reset_v4 + robust_recovery_v15` 已在新 environment seed 的目标关节噪声和零关节噪声上达到 100% 成功率，因此摩擦阶段不重新爬 reset curriculum。三个阶段都固定：

- 八关节 q/qdot 独立均匀噪声均为 `0.005`；
- root velocity 为 0，axis tilt 为 0；
- mass/inertia、actuator gain、reference、reward、termination 和 observation 不变；
- 全局 `geom_friction_scale=(1, 1)`，只采样 floor 接触摩擦。

| stage | 训练权重 | floor friction scale | joint q/qdot | reset 结构 | root/tilt |
|---|---:|---:|---:|---|---:|
| `floor_friction_02` | 0.20 | U(0.98, 1.02) | 0.005 / 0.005 | independent | 0 / 0 |
| `floor_friction_05` | 0.30 | U(0.95, 1.05) | 0.005 / 0.005 | independent | 0 / 0 |
| `floor_friction_10` | 0.50 | U(0.90, 1.10) | 0.005 / 0.005 | independent | 0 / 0 |

### 为什么不能直接缩放 floor geom

MuJoCo 对同优先级的动态 geom 接触逐项取两侧摩擦系数的最大值。名义 floor 和机器人壳体都为 `0.8 0.02 0.01`；若只把 floor 乘 0.90，接触仍会取壳体的名义值，低摩擦端不会生效。

v2 在启用 floor override 时，先按当前 `priority/solmix` 计算原 floor 与所有可接触机器人 geom 的等效 `condim/solref/solimp/friction`，再把等效值写到高优先级 floor。这样 `--floor-friction-scale 1.0` 产生与旧动态混合相同的名义接触参数，而 0.90/1.10 真正只改变 floor 接触；其他 geom 数组及机器人自接触摩擦保持不变。若模型以后出现显式 floor pair、adhesion 或互不相同的 floor 接触模型，代码会直接拒绝运行，避免静默改变物理含义。

### 训练前固定端点

先用已验收基线在相同 256 个 reset 上跑三个端点，得到训练前 paired baseline。三次只修改 scale、label，其他参数完全相同：

```bash
eval_floor () {
  scale="$1"
  label="$2"
  python -m scripts.evaluate_mjx_3d_policy \
    results/independent_reset_v4_recovery_v15_cg20_h200_seed0/params_best \
    --evaluation-mode policy \
    --geometry pupper_open60 \
    --controller results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json \
    --physics-profile cg20 \
    --floor-friction-scale "$scale" \
    --episode-length 500 --batch-size 256 --chunk-size 64 \
    --seed 0 --environment-seed 50000 \
    --reset-joint-noise-rad 0.005 --reset-velocity-noise 0.005 \
    --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
    --reference-weight 1.0 --minimum-residual-gain 0.15 \
    --phase-rate-scale 1.0 --reference-action-scale 1.0 \
    --reference-ramp-start-scale 0.0 --reference-ramp-duration-s 0.25 \
    --residual-pair-differential-scale 0.25 \
    --explicit-phase-observation --zero-residual-policy-init \
    --initial-policy-std 0.10 --mujoco-gl disable --memory-fraction 0.50 \
    --out "results/floor_friction_v2_pretest_${label}_seed50000"
}
eval_floor 0.90 090
eval_floor 1.00 100
eval_floor 1.10 110
```

### smoke 与正式训练

smoke 使用 12 个总 eval 点，按当前整数分配规则为 3/4/5；三个阶段都会包含 step 0 和训练后评估：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset smoke \
  --recipe robust_recovery_v15 \
  --geometry pupper_open60 \
  --controller results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json \
  --physics-profile cg20 \
  --curriculum floor_friction_v2 \
  --restore-params results/independent_reset_v4_recovery_v15_cg20_h200_seed0/params_best \
  --episode-length 500 --num-evals 12 --eval-envs 64 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.50 \
  --out results/floor_friction_v2_recovery_v15_cg20_smoke_seed0
```

smoke 的三阶段编译、日志和 checkpoint 正常后，再运行 H200。30 个总 eval 点在每阶段先保留 step 0 和训练后两个点，再按权重分配剩余点，当前精确结果为 7/9/14，而不是每个阶段只有一次训练后观察：

```bash
python -m scripts.train_mjx_3d_residual_ppo \
  --preset h200 \
  --recipe robust_recovery_v15 \
  --geometry pupper_open60 \
  --controller results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json \
  --physics-profile cg20 \
  --curriculum floor_friction_v2 \
  --restore-params results/independent_reset_v4_recovery_v15_cg20_h200_seed0/params_best \
  --episode-length 500 --num-evals 30 --eval-envs 256 \
  --reset-root-velocity-noise 0 --reset-axis-tilt-noise-rad 0 \
  --seed 0 --mujoco-gl disable --memory-fraction 0.80 \
  --out results/floor_friction_v2_recovery_v15_cg20_h200_seed0
```

训练后把上面 `eval_floor` 函数里的 checkpoint 换成新目录的 `params_best`，environment seed 仍保持 50000，并输出到独立 posttest 目录。0.90、1.00、1.10 各自成功率达到 95% 即通过；同时要求 nominal 圈数无明显回退，且 forbidden contact/depth、nonfinite、root-height 和 axis-tilt 类失败为 0。必须使用 `params_best`，不要用 `params_final`。

## 旧 `reset_v2 -> friction_v1` 实验记录

以下内容保留旧实验的含义与复现命令，不代表当前 v15/v4 的直接续训路径。

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
