# Primitive 高动态 Roll to Stand

## 实现

新入口 `scripts.train_mjx_3d_roll_to_stand` 强制开启动态恢复模式，默认模型为 `rollingquad_2_primitive`。新任务训练及训练评估拒绝 mesh；模型加载后还检查所有有接触权限的几何，发现 mesh 碰撞几何即报错。primitive XML 中保留的 CAD 外观不参与碰撞。

默认使用 CG12、1 ms 物理步长和 50 Hz 控制。CPU MuJoCo 对比中，同一 stiff reference 在 Newton4、Newton20、CG12 下的 10 s 净滚动分别约 0.65、5.58、6.78 圈，因此不沿用旧 Newton4 默认值。这只是 CPU 来源诊断；MJX 采集仍需实际通过覆盖检查。可用 `python -m scripts.probe_primitive_roll_reference` 重跑并保存原始汇总。

复用旧过渡环境的 12 关节动作映射、36×20 历史观测、PPO 包装器和滚动相位采样。旧 `brake_*` 名称在新入口中仅表示快照范围，**不表示执行制动阶段**：新任务始终直接进入自由恢复，取消制动门槛、展开门槛及其超时，保留全局高度、数值有效性和总时长检查。

成功要求连续满足站立条件共 3 s（1 s 站稳 + 2 s 验证），任何一次不满足即重新计时。站立条件包括现有姿态、关节误差、高度、低速、至少三个足端接触，且无非足端地面接触。训练 episode 为 10 s。恢复初期允许壳体接地。奖励中低速收益与足端站立质量绑定，没有单独奖励先制动。

切换保留 qpos/qvel/ctrl/time，首次 previous-action 由实际 ctrl 换算。快照 reset 不抬高机器人、不修改速度；离线快照没有精确求解器 warm-start，需与 live takeover 区分。观测历史使用明确的冷启动方式，不伪造切换前的传感器历史。

## 使用

以下命令在 `curl_robot_2d` 目录、具备项目 MJX/Brax 依赖的 Python 环境中运行。正式 PPO 建议使用 Linux GPU；本地 CPU 仅进行接口与小规模运行检查。

先分别采集训练与独立验证轨迹（输出不得覆盖已有文件）：

```bash
python -m scripts.collect_reference_roll_to_stand --out results/rts_reference/train.npz --episodes 8 --seed 0
python -m scripts.collect_reference_roll_to_stand --out results/rts_reference/eval.npz --episodes 4 --seed 1000
```

采集器使用 primitive 对应的 CEM reference、完整 reference 权重和严格零 residual，没有 RL 权重。采集实际滚动周期并检查八个相位桶覆盖；不足即退出失败，不能补造快照。相邻周期不被假定为不同速度，汇总保留实际速度范围。

先做站立附近的恢复预训练：

```bash
python -m scripts.train_mjx_3d_roll_to_stand --stage deploy_near_stand --preset 4090 --out results/rts_near_stand
```

再从真实滚动状态训练。使用预训练保存的完整 PPO checkpoint 路径作为 `--restore-checkpoint`；路径必须实际存在，不能把 `params_final` 当作训练 checkpoint。

```bash
python -m scripts.train_mjx_3d_roll_to_stand --stage brake_full --preset 4090 --roll-snapshots results/rts_reference/train.npz --eval-roll-snapshots results/rts_reference/eval.npz --out results/rts_dynamic
```

上面未传恢复参数时是从零训练。训练前验证两组快照的 reference 哈希、采集 seed、物理步长、控制周期、求解器、摩擦、质量惯性缩放及伺服缩放；不允许两组使用相同 seed。原始快照还检查模型 XML 哈希和关节/执行器顺序。

本地运行检查：

```bash
python -m scripts.smoke_dynamic_roll_to_stand
python -m scripts.train_mjx_3d_roll_to_stand --stage walking_start --preset cpu_smoke --hidden-layers 32 32 --out results/rts_cpu_smoke
```

`cpu_smoke` 仅为 1536 步、2 环境的运行检查，不可据此评价恢复成功率。正式训练入口支持 `--dry-run` 检查展开后的完整配置。

## 当前边界

- 当前 MJX 奖励未测接触冲击力，新任务明确把 impact 权重设为零。物理子步冲击、力矩峰值及视频评估仍需补充，不能用当前输出声称已验证。
- primitive 验证通过后，再补全从 reset 开始的 mesh reference → 同一恢复策略 → 持续站立链路。不得在接管时切换碰撞模型；mesh 不进入训练或训练期间评估。
- 现有独立 `--eval-only` 是快照评估，并不等于完整 reference 链路的 mesh 验证。
- 运行检查、短 PPO smoke、完整技能训练、mesh 验证必须分别报告；没有成功率证据时不宣布技能训练完成。
