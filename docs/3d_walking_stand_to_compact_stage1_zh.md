# 第一阶段：行走名义初始站姿 → 低速 compact

本阶段只验证固定站姿到 compact 的可达性。入口已实现，尚未完成 PPO 训练或收敛验收；不代表已能起滚或从行走中切换。

## v2：持续姿态奖励

当前 contract 为 `walking_stand_to_compact_only_v2_dense_pose`。针对 v1 的 best 全程输出 stand 的轨迹，新增每控制步的状态奖励：

```text
reward_pose = -0.10 × (1 - pose_quality)
```

`pose_quality` 是已有 compact 势函数的姿态部分（0–1），同时考虑全部关节位置、根部高度和机身姿态。它不乘速度门槛。该项每步直接计入，包括最后一步，不是势函数差分，因此中途收拢的收益不会在超时时被抵消。越接近目标扣分越少，目标处为零，避免为延迟成功退出提供正的存活收益。

保留原有势函数塑形、余速、动作变化、力矩、拖地及防弹跳项，保留原严格成功门槛。物理失败提前结束时，追加 `剩余步数 × (0.10 + time_cost)` 的负奖励，防止提前结束逃掉新增的姿态代价和时间代价。正常超时与成功不追加此项。权重可用 `--pose-reward-weight` 调整，仅适用于 compact-only。

日志新增 `reward_pose`、`terminal_pose_quality` 和七项终点 gate 比值（位置、关节速度、高度、线速度、角速度、姿态、相位）；比值大于 1 表示该项不达标。评估 JSON 同时保存平均姿态质量及各终点 gate 比值。best 仍先按成功率选取；均失败时总 reward 现在包含持续姿态代价。

请使用新目录从头训练，不传旧 v1 的 `--restore-startup`；版本校验会拒绝将旧奖励任务当作新任务直接恢复。新增负奖励后，总 reward 的数值会下降，不能把 v1 的 −16 当作 v2 的对照门槛。应关注姿态质量上升、各 gate 误差下降以及最终成功率。

```bash
python -m scripts.train_mjx_3d_startup_ppo --compact-only \
  --preset h200 --max-devices 1 --pose-reward-weight 0.10 \
  --out results/walking_stand_compact_stage1_v2_seed0
```

本次验证：4 项新增奖励回归测试及 2 项已有配置测试通过，语法与 diff 检查通过。测试覆盖部分收拢比站立得分更高、同终点超时不抵消中途姿态收益、运动中仍有姿态收益及版本/门槛隔离；使用的是奖励数值测试，不是动力学可达性证明。尚未执行本版 MJX stepping 或 PPO 收敛验证。

## 起点、模型与动作

- 使用 `rollingquad_2_primitive`（`rollingquad_primitive.xml`）的 stand keyframe，根部额外抬高 5 mm，关节速度和根部速度为零。沿用行走名义站姿的关节定义，不包含行走 reset 的随机偏航、关节噪声与额外高度噪声。
- 默认采用现有滚动 primitive 的解析碰撞几何，保留外壳接触及已有自碰撞白名单；保留的 mesh 资源不参与碰撞。采用 CG20、1 ms 物理步长、50 Hz 控制及原模型伺服/力矩限幅。行走入口关闭外壳接触，因此本阶段不声称逐项复用了行走物理配置。compact 目标直接读取 primitive 模型的 keyframe，不复用 mesh 模型的根部高度。
- 先复用 8 维髋膝动作，4 个外摆保持零位伺服；观测为现有 53 维特权观测。它是固定站姿基线，并非 12 维行走策略的直接续训或可部署学生。
- 不加载行走或滚动权重。模型不会在 episode 中切换；reset 之后不改写姿态或清零速度。不播放固定收腿轨迹。

## 目标与终止

最多 3 s 内，连续 5 个控制帧（0.10 s）进入 compact 状态窗口：

| 指标 | 上限 |
| --- | ---: |
| 全部 12 关节最大角度误差 | 0.02 rad |
| 全部 12 关节最大速度绝对值 | 0.05 rad/s |
| 根部高度误差 | 0.01 m |
| 根部各轴线速度绝对值 | 0.02 m/s |
| 根部各轴角速度绝对值 | 0.10 rad/s |
| 姿态角距离及滚动相位误差 | 各 0.05 rad |
| 横向偏移 | 0.05 m |
| 滚动轴倾斜 | 0.10 rad |

同时要求没有当前禁止接触或物理失败。越过任一门槛会清除连续确认计数。达标即结束，不启动教师，不要求长期静止。原有物理失败检查、足端接触滑动惩罚、收拢势函数、余速惩罚和防弹跳项保留。

本模式不检查滚动教师首帧命令差，因为没有加载教师；这一项留给第二阶段。`--continuation-s` 和 `--minimum-turns` 在此模式不参与验收。

## 运行

以下命令在 `curl_robot_2d` 目录及已有 MJX/Brax 训练环境执行。配置检查不训练：

```bash
python -m scripts.train_mjx_3d_startup_ppo --compact-only \
  --preset smoke --dry-run --out results/walking_stand_compact_stage1_check
```

短接口测试（只保持初始站姿，不是成功演示）：

```bash
python -m scripts.train_mjx_3d_startup_ppo --compact-only \
  --preset smoke --smoke-steps 2 --out results/walking_stand_compact_stage1_smoke
```

通过接口测试后训练：

```bash
python -m scripts.train_mjx_3d_startup_ppo --compact-only \
  --preset h200 --max-devices 1 --out results/walking_stand_compact_stage1_seed0
```

使用新输出目录。此模式拒绝 `--teacher` / `--teacher-config`，并使用独立 contract 阻止误续训旧启动任务。默认 5 帧确认，可显式调整 `--confirmation-steps`，但验收报告必须注明实际时长。

## 结果解释与后续

独立评估输出 `compact_reach_rate`、拖地指标、到达时速度以及原始轨迹。共享日志中的 `handoff` 仅表示 compact 窗口已确认，`success` 仅表示本阶段成功；报告明确标注 `rolling_continuation_evaluated=false`。当前固定且无噪声的初态配合确定性评估会产生重复轨迹，批量 100% 不能解释为鲁棒成功率。

先检查动作轨迹和真实状态，确认不是弹跳、碰撞或短暂穿透造成的表面达标。然后增加站姿小扰动与多训练种子测试；若 8 维基线难以保持平衡，再扩展 12 维动作。

第二阶段使用实际到达状态，在同一 primitive 模型中连续接入冻结滚动策略，保留物理状态、上一命令和观测历史，检查首帧命令连续性并续滚验收。此入口暂不实现 primitive 教师接管；原有带教师任务的模型限制保持不变。完整 mesh 留作训练后的几何复核，不能通过 episode 中切换碰撞几何宣称接管成功。最后才扩展为行走各步态相位起步。

2026-09-06 起 `--compact-only` 默认模型已由完整 mesh 改为 primitive，运行命令不变。使用新输出目录；模型指纹和目标指纹校验仍保留，旧 mesh 权重不能作为相同模型直接续训。

此次修改通过 XML 静态核对：没有 mesh 碰撞 geom，100 个外壳 geom 保留地面碰撞，显式自碰撞 pair 保留，stand/compact 的关节控制目标与完整模型一致。2 项无需 MuJoCo 的配置测试、语法检查及改动文件的 diff 空白检查通过。当前解释器仍缺少 MuJoCo，未执行动力学 stepping、速度对比或 PPO；不据此宣称已经验证转换可达性或实际加速倍数。

## 本次本地检查（2026-09-05）

Python 语法检查及本次改动的 diff 空白检查通过。相关三个测试模块共 32 项：23 通过、5 项可选 MJX 集成测试跳过、4 项出错。其中 3 项因当前解释器没有 MuJoCo，另 1 项为未修改的历史候选库与当前模型指纹不一致。未修改历史指纹或放宽校验。依赖下载未完成，已停止安装；模型配置 dry-run、MJX stepping 与 PPO 均尚未验证。上述命令是待在完整训练环境执行的入口，不是已跑通的训练记录。
