# CEM Reference Residual RL 课程

## 1. 控制形式

Residual PPO 使用冻结的碰撞约束 CEM 控制器：

`results/collision_constrained_cem/best_phase_controller.json`

CEM reference 保留原来的相位锁定振荡器，而不是按时间播放一条动作轨迹：

```text
oscillator_rate =
    nominal_rate + coupling * sin(root_pitch - oscillator_phase)
```

每个 1 ms MJX 物理子步都会更新 oscillator 和 CEM 目标。PPO 动作在一个
20 ms 控制周期内保持不变。

实际归一化动作是：

```text
effective_action =
    reference_weight * cem_action + residual_gain * policy_action

residual_gain =
    minimum_residual_gain
    + (1 - reference_weight) * (1 - minimum_residual_gain)
```

默认 `minimum_residual_gain=0.05`。第一阶段只开放很小的 residual，优先保留
CEM 已验证的滚动；reference 降低后再快速增加策略控制权：

| 阶段 | reference weight | residual gain |
|---|---:|---:|
| 1 | 1.0 | 0.050 |
| 2 | 0.5 | 0.525 |
| 3 | 0.0 | 1.000 |

最后阶段的 CEM action 和 oscillator 特征在 observation 中也会乘以零。因此
最终策略既没有 CEM 动作控制权，也不能依赖 CEM oscillator 作为隐藏 teacher。

## 2. 性能门控

每个非零 reference 阶段至少训练约 500k 步。达到以下全部条件后才进入下一
阶段：

- 平均 episode 长度达到上限的 80%；
- 平均净滚动达到 3 圈；
- episode failure rate 不高于 20%；
- 没有 action 或 physics NaN。

默认 H200 配置的 PPO rollout quantum 为 131,072 步，所以 500k 门槛实际在
524,288、1,048,576、1,572,864 等位置检查。

总预算只在训练开始时取整一次：

```text
requested budget = 2,000,000
effective budget = 2,097,152
```

课程切换不会增加预算。如果某阶段未达标，就保持当前 reference 权重继续训练。
如果预算耗尽前没有进入并实际训练 `reference_weight=0` 阶段，结果会明确标记
`curriculum_success=false`，不会把带 reference 的策略作为纯 RL 成果。
即使已经进入零 reference 阶段，也只有零 reference 的确定性 eval 通过同一
物理 gate 后才会标记成功并保存 `params_best_zero_reference`。

固定预算、严格达标后降权和保证最终零 reference 总能训练足够长，在阶段始终
不达标时无法同时满足。当前实现优先遵守固定预算和性能门槛，并把未完成课程
作为失败结果暴露出来。

## 3. 训练前的纯 CEM 对照

Residual 训练前先在 MJX 中令 `policy_action=0`、`residual_gain=0`，验证
reference 本身。该诊断不训练 PPO，可在本地 CPU 上运行：

```bash
python -m scripts.compare_mjx_cem_reference \
  --physics-profile cg12 \
  --controller results/collision_constrained_cem/best_phase_controller.json \
  --noise-seeds 32 \
  --mujoco-gl disable \
  --output results/mjx_cem_reference_ablation/summary.json
```

脚本同时运行 CPU MuJoCo reference 和三组 MJX 对照：

- A：保留训练使用的 reset noise 和 XML root damping；
- B：从精确 compact、零速度启动，保留 XML root damping；
- C：从精确 compact、零速度启动，并像 CPU CEM 一样清零 root damping。

A 到 B 的改善表示启动噪声敏感；B 到 C 的改善表示 root damping 不一致；
C 仍明显落后 CPU reference 则表示 MJX 接触或积分轨迹尚未对齐。在 CEM 的
MJX 结果通过课程 gate 前，不应开始 residual curriculum。

## 4. 云端命令

```bash
python -m scripts.train_mjx_residual_ppo \
  --preset h200 \
  --physics-profile cg12 \
  --controller results/collision_constrained_cem/best_phase_controller.json \
  --reference-weights 1.0 0.5 0.0 \
  --minimum-residual-gain 0.05 \
  --minimum-stage-steps 500000 \
  --gate-check-steps 500000 \
  --gate-min-survival 0.80 \
  --gate-min-turns 3.0 \
  --gate-max-failure-rate 0.20 \
  --learning-rate 3e-5 \
  --entropy-cost 1e-3 \
  --discounting 0.995 \
  --reward-termination 10 \
  --reward-early-termination-scale 1 \
  --seed 0 \
  --mujoco-gl egl \
  --out results/mjx_residual_cem_curriculum_seed0
```

该入口独立于 `scripts/train_mjx_ppo.py`。纯 PPO 训练命令和 action 路径保持
不变，只复用日志、评估、参数保存和 GIF 渲染工具。

## 5. 输出

- `training_config.json`：CEM 参数、课程、gate、PPO 和 runtime 快照；
- `curriculum_history.json`：每次 eval 的 reference 权重、gate 和完整指标；
- `training_summary.json`：课程是否完成、各阶段占用步数和最终权重；
- `params_stage_*_final`：每个实际训练阶段结束时的参数；
- `params_final`：预算结束时的策略；
- `params_best_zero_reference`：仅从 reference 为零的 eval 中选择；
- `eval_visualizations/`：每个 eval 独立保存当时的参数、确定性 rollout、
  summary 和 GIF，目录名包含全局 eval 编号、step 与 reference 权重；
- `eval_visualizations.json`：全部 eval 可视化的路径、结果或失败原因索引；
- `evaluation_zero_reference/`：强制使用 reference 为零的确定性回放与 GIF。

阶段切换保留 observation normalizer、policy 和 value 参数。Brax 的公开
`restore_params` 接口不保存 Adam optimizer state，因此每次成功降权时会重新
初始化 optimizer；默认只有两次切换。
