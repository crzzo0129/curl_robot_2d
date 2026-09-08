# 3D unified stand-to-roll：BC + reset curriculum

> v2 更新：当前入口先训练 rolling_orbit -> mixed_75 -> mixed_25 -> compact，
> 后续再进入站姿课程。BC 标签时序和归一化已经修改，必须重新训练 BC，
> 不兼容旧 checkpoint。请按 [v2 云端验证说明](stand_to_roll_v2_cloud.md) 执行；
> 下文直接从 BC 开始 compact 的旧命令已被替代。v2 未在本地运行验证。

当前目标截止到 full-stand 能稳定进入持续滚动。CEM、compact 和 capture
shaping 在五个 PPO 阶段保持不变；teacher shaping annealing 不在本阶段执行。

## 固定契约

- 从 BC 到 full-stand 始终是同一个 12-action policy。
- 环境执行的动作始终是 policy action；不存在 CEM 接管或 action blending。
- Actor observation 对齐 `scripts/train_ppo_deploy.py`：36 维单帧、20 帧历史、
  newest-first，共 720 维；不包含 CEM phase、`D_min`、root linear velocity 或
  joint velocity。
- 动作范围不复用 walking deploy 范围。中心为
  `[ABD=0, hip=0.1108283, knee=0.9092587] x 4`，范围为
  `[0.30, 0.80, 1.20] x 4` rad。
- full-stand reset 的 hip/knee 来自 `stand` keyframe，四个 ABD 强制为 0。
- Capture 仅是 `D_min` 与正向角速度连续满足 80 ms 后触发的一次性奖励和指标。

## 训练顺序

先做 CEM behavior cloning：

```powershell
python -m scripts.train_mjx_3d_stand_to_roll `
  --stage bc `
  --cem-data results/cem_cycle_data/cem_cycles.npz `
  --out results/mjx_3d_stand_to_roll
```

然后依次训练五个 PPO stage。`compact` 从 BC 初始化；后续阶段必须恢复前一阶段
的完整 PPO checkpoint：

```powershell
python -m scripts.train_mjx_3d_stand_to_roll `
  --stage compact --preset 4090 `
  --bc-params results/mjx_3d_stand_to_roll/bc/bc_params `
  --out results/mjx_3d_stand_to_roll

python -m scripts.train_mjx_3d_stand_to_roll `
  --stage slightly_open --preset 4090 `
  --bc-params results/mjx_3d_stand_to_roll/bc/bc_params `
  --restore-checkpoint results/mjx_3d_stand_to_roll/compact/ppo_checkpoint `
  --out results/mjx_3d_stand_to_roll
```

其余顺序为：

```text
crouch → semi_stand → full_stand
```

每一级都应把 `--restore-checkpoint` 指向前一级的 `ppo_checkpoint`。不要跳级，
也不要重新初始化 Actor。所有阶段继续传入同一个 `bc_params`，用于冻结 BC 得到的
720 维 observation mean/std，避免课程切换时 normalizer 漂移破坏已有 rolling skill。

## Reset 范围

| Stage | alpha 范围 |
|---|---:|
| compact | 0.00–0.10 |
| slightly_open | 0.00–0.30 |
| crouch | 0.20–0.60 |
| semi_stand | 0.50–1.00 |
| full_stand | 1.00 |

只有当前 stage 的独立评估 capture rate 至少 80%、failure rate 不高于 20% 时，
才进入下一阶段。最终验收还必须检查 full-stand episode 的净滚动进度和持续时间，
不能只看 capture bonus。

## 当前暂不包含

- CEM/compact/capture shaping 退火；
- domain randomization；
- teacher action 在线介入；
- RL/CEM action blending；
- full-stand 之后的 pure-task reward fine-tuning。
