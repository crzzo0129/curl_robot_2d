# 二维滚动 CEM / Residual RL 到 3D 双模态 Handoff

更新时间：2026-07-29

项目目录：

```text
C:/Users/12481/Desktop/OH-WorkSpace/robot_description/curl_robot_2d
```

云端目录：

```text
/inspire/hdd/project/leverage-robot/ky26210/curl_robot_2d
```

当前 Git 提交：

```text
04d0937
```

## 1. 用户最终目标

目标不是继续无限优化二维滚动，而是开发一个滚动-行走双模态机器人，并在滚动
策略与最终结构选择中引入 structure-motion co-design。

当前推荐总体架构：

```text
CEM                    名义滚动极限环、碰撞约束、可解释周期动作
Residual RL            CEM 之上的小幅反馈修正
3D residual            侧倾、偏航、左右非对称接触与模型误差
walking controller     独立完成稳态行走
transition controller  roll-to-walk / walk-to-roll
mode manager           管理双模态与切换条件
co-design              联合评价滚动、行走和切换可行性
```

当前决定：

- 暂时不做速度 tracking；先完成双模态比尽善尽美更重要。
- 不再为了展示 RL 而加入不现实的大幅向后速度冲击。
- 不要求纯 RL 从零替代 CEM。
- 二维平地部署与二维 co-design 快速评价优先使用纯 CEM。
- 保留 Residual RL 框架，预计它在 3D 侧向稳定和 sim-to-real 中更有价值。

## 2. 二维任务与运行配置

主要入口：

```text
scripts/optimize_phase_controller.py
scripts/compare_mjx_cem_reference.py
scripts/train_mjx_ppo.py
scripts/train_mjx_residual_ppo.py
scripts/render_mjx_policy.py
```

主要模块：

```text
curl_robot_2d_mjx/config.py
curl_robot_2d_mjx/environment.py
curl_robot_2d_mjx/cem_reference.py
curl_robot_2d_mjx/reward.py
curl_robot_2d_mjx/reward_config.py
```

物理与 episode：

```text
physics profile      cg12
physics timestep     0.001 s
action repeat        20
control timestep     0.02 s
episode length       500 control steps = 10 s
root damping         disabled, 与 CPU CEM 一致
rendering            EGL 可用；无渲染评估可使用 disable
```

Residual policy：

```text
action size          4
action               front/rear hip/knee normalized residual
observation size     30
reference weight     永久保留为 1.0
residual scale       课程式 0.01 -> 0.03 -> 0.05
```

`reference_weight` 和 `residual_gain` 已从 observation 删除，因为它们在 stage
内是常量，经过 observation normalization 后切换 stage 会成为离群输入。旧的
32 维 residual PPO checkpoint 与当前 30 维 observation 不兼容；CEM JSON 不受
影响。

## 3. 当前正式 CEM Reference

不要再使用旧 reference：

```text
results/collision_constrained_cem/best_phase_controller.json
```

当前新 reference：

```text
results/collision_constrained_cem_foot_gap_2mm_short_contact/best_phase_controller.json
```

关键配置：

```text
minimum foot surface gap     0.002 m
foot gap tracking margin     0.004 m
oscillator rate              3.2936865141 rad/s
oscillator coupling          4.4804737479 /s
```

新旧 reference 在当前模型上的公平 CPU 接触诊断：

| 指标 | 旧 reference | 新 reference |
| --- | ---: | ---: |
| 10 s 圈数 | 9.049 | 8.712 |
| 总自碰撞时间 | 1.145 s | 0.145 s |
| 足-足接触时间 | 1.048 s | 0.077 s |
| 最长足-足接触 | 36 ms | 12 ms |
| 最大自碰撞穿透 | 1.662 mm | 0.782 mm |

新 reference 仍有很短的接触：主要为 `rear_foot` 擦到 `front_shank`，其次为
`rear_foot` 擦到 `front_thigh`。但旧 reference 中导致持续失速的长时间足-足
卡碰已经显著减少。

本地诊断目录：

```text
results/collision_constrained_cem/contact_diagnostic/
results/collision_constrained_cem_foot_gap_2mm_short_contact/contact_diagnostic/
```

## 4. 新旧 CEM 鲁棒性结果

固定 medium 瞬时速度冲击：

```text
root x delta velocity       uniform within +/-1.0 m/s
root pitch delta velocity   uniform within +/-3.0 rad/s
disturbance step            100..400，即 reset 后 2..8 s
128 samples
```

结果：

| 指标 | 旧 CEM | 新 CEM |
| --- | ---: | ---: |
| mean | 7.316 | 8.050 |
| p10 | 3.438 | 6.074 |
| median | 8.814 | 8.710 |
| min | 1.452 | 1.442 |
| under3 | 9/128 | 5/128 |
| under5 | 26/128 | 12/128 |

结论：新 CEM 牺牲约 1% 的中位圈数，显著改善低分位和低圈比例，适合作为正式
reference。

## 5. 扰动混合采样实现

当前代码支持：

```text
--disturbance-probability
--disturbance-level-scales
--disturbance-level-probabilities
--disturbance-backward-probability
--environment-seed
--rollout-seed
```

最后两个参数位于 `compare_mjx_cem_reference.py`，用于让 CEM 和 RL 经历完全
相同的 reset 与扰动，执行逐样本配对评价。

现实混合训练分布：

```text
50% no push
30% mild    scale 0.5
15% medium  scale 1.0
5%  strong  scale 1.5

conditional backward probability = 20%
```

对应参数：

```bash
--disturbance-root-x-velocity 1.00 \
--disturbance-root-pitch-velocity 3.00 \
--disturbance-probability 0.50 \
--disturbance-level-scales 0.50 1.00 1.50 \
--disturbance-level-probabilities 0.60 0.30 0.10 \
--disturbance-backward-probability 0.20 \
--disturbance-min-step 100 \
--disturbance-max-step 400
```

注意：当前 disturbance 是直接执行 `qvel += delta_velocity` 的瞬时速度冲击，
不是有限时间外力。它适合作为压力测试，但不应被描述为完整的现实扰动模型。

## 6. 最终 Residual PPO 训练

云端输出：

```text
results/mjx_cem_foot_gap_residual_mix_2m_seed0
```

训练配置：

```text
preset                         h200
requested steps                2,000,000
effective steps                2,097,152
reference weight               1.0
residual scales                0.01, 0.03, 0.05
minimum stage steps            524,288
gate check steps               262,144
learning rate                  3e-6
entropy cost                   1e-4
discount                       0.995
reward roll progress           15
reward residual action         0.02
termination penalty            10
early termination scale        1
final robust eval              512 samples
```

训练结果：

```text
status                         SUCCESS
consumed                       2,097,152 / 2,097,152
final residual scale           0.05
steps at final scale           1,048,576
elapsed                        30.8 min

single rollout turns           8.765
single rollout x               8.478 m
single rollout failure         none

robust mean                    8.576
robust min                     1.447
robust p10                     8.608
robust median                  8.737
robust p90                     8.761
robust max                     8.822
robust termination failure     0%
```

实际采样比例：

```text
no push      49.4%
scale 0.5    29.7%
scale 1.0    16.4%
scale 1.5     4.5%
```

`status=SUCCESS` 只表示通过绝对圈数、生存率和 failure gate，不表示 RL 优于
CEM。

## 7. 最重要的逐样本配对结论

CEM 与 RL 使用完全相同的 512 个 reset 和 disturbance 后：

```text
CEM mean                     8.622
RL mean                      8.576
mean delta                  -0.046
median delta                -0.043

CEM under5                  10
RL under5                   10
CEM under3                   3
RL under3                    3

improved > 0.25 turns        1
worsened < -0.25 turns       5
rescue5                      0
regression5                  0
rescue3                      0
regression3                  0
```

必须保留的科学结论：

> 当前二维 Residual PPO 没有学到可测量的恢复能力。它基本保持了 CEM，但平均
> 性能略降约 0.5%，没有救回任何 CEM 的 under5 或 under3 样本。

之前非配对评价中 `under5 14 -> 10`、`under3 5 -> 3` 的变化来自两批随机样本
差异，不能归因于 PPO。

因此：

- 不要声称 Residual RL 显著提高了二维平地鲁棒性。
- 可以声称小尺度 residual 基本不破坏高质量 CEM reference。
- 不要继续盲目放大 residual scale。
- 不要为了让 RL 显得有效而增加不现实的强向后扰动。
- 如果未来专门研究二维恢复，需要使用真实困难状态重采样、相位条件采样或
  tail/CVaR 目标；这属于新的研究分支，不是当前主线。

## 8. Reward、termination 与评价注意事项

当前训练 reward 以滚动进度为主，同时包含：

```text
roll progress
roll/translation mismatch
backward motion
action rate
residual action
torque
airborne
foot gap
collision
termination
early termination
```

termination 主要覆盖：

```text
root high
foot gap excessive
leg crossing
nonfinite / NaN
```

没有固定 root-low termination，因为合法二维滚动的 root z 会周期性降得很低。

评价时必须区分：

```text
termination failure          数值或硬约束失败
turns < 5 functional failure 功能性低速失锁
turns < 3 severe failure     严重恢复失败
```

`failure=0%` 不代表所有 rollout 都完成了有效滚动。

## 9. 当前不做的事项

### 9.1 暂不做速度 Tracking

CEM 已经解决固定速度平地滚动。速度 tracking 会引入 command-conditioned CEM、
宽速度 reference、停车/加速/减速等额外问题，暂时不属于双模态最小可行目标。

### 9.2 暂不继续二维纯 RL

纯 RL 从零探索周期接触滚动困难是预期结果。CEM 是合理结构先验，不需要为了
方法纯粹性强制替换。

### 9.3 暂不继续优化极端向后冲击

现实中大幅向后瞬时速度跳变很少。未来若做鲁棒性，应优先考虑电机误差、接触
变化、左右不对称、有限时间外力和地形，而不是扩大 `delta qvel`。

## 10. 进入 3D 的阶段标准

建议的二维 closure 条件：

```text
CPU / MJX nominal difference <= about 5%
10 s nominal turns >= 8
no numerical failure
no persistent foot-locking collision
realistic-mix p10 >= 8
realistic-mix under5 <= 3%
residual nominal degradation <= 2..3%
reproducible configuration and artifacts
```

当前系统已经基本满足这些条件。无需等待二维尽善尽美，可以开始 3D 原型；二维
配对实验已经完成并给出了清晰结论。

## 11. 推荐的 3D 路线

不要从 3D 纯 RL 开始。先把二维 CEM 提升为左右对称 reference：

```text
left sagittal joints   <- 2D CEM
right sagittal joints  <- 2D CEM
```

初始 3D residual 只解决新增问题：

```text
body roll stabilization
yaw suppression
lateral velocity suppression
left/right asymmetric contact correction
```

推荐阶段：

1. 3D 平地直线滚动，只验证左右对称 CEM，不加扰动。
2. 小 residual 稳定 body roll 和 yaw。
3. 加轻微侧向初始误差和侧向有限时间外力。
4. 加左右摩擦差异和执行器误差。
5. 再做转向。
6. 独立完成稳态行走。
7. 分别实现 roll-to-walk 与 walk-to-roll transition。
8. 最后加入 mode manager。

推荐拆分 residual：

```text
symmetric residual      前进/滚动修正
antisymmetric residual  侧倾/偏航修正
```

在 3D 中，零 residual 无法处理横向自由度，RL 的价值会比二维平地更明确。

## 12. Structure-Motion Co-design 建议

不要对每个结构候选完整训练一次 PPO。推荐分层：

```text
outer loop          结构参数搜索
inner loop          每个结构重新优化 CEM
fast 2D screening   圈数、碰撞、能耗、扭矩、足部间隙
3D shortlist        侧向稳定、偏航、行走支撑与切换可行性
final candidates    再训练 Residual RL
```

候选结构指标至少包括：

```text
rolling stability
foot collision clearance
energy and peak torque
walking workspace and support stability
roll/walk transition feasibility
robustness to realistic parameter error
```

二维用于筛选 sagittal rollability，最终结构必须在 3D 决定，因为二维无法评价
横向稳定、左右宽度、偏航和行走支撑多边形。

## 13. 当前代码状态与验证

当前仓库在 handoff 生成前为 clean，提交：

```text
04d0937
```

已实现：

- root damping 与 CPU CEM 对齐。
- CEM foot-gap target 在 CPU/MJX reference 中一致应用。
- retained-CEM residual scale curriculum。
- stage gate、best/last/final checkpoint 与 rollback。
- 每个 eval 生成 GIF。
- reward 明细和可读终端输出。
- final robust distribution evaluation。
- disturbance probability、severity level 和 backward probability。
- paired environment/reset seed。
- 每个 robust 样本保存实际 disturbance 参数。

本地测试：

```text
73 tests passed
```

测试命令：

```powershell
& 'C:\Users\12481\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m unittest discover -s tests -p 'test_*.py'
```

本地没有 JAX/Brax GPU 运行时，真实 MJX rollout 与 PPO 在 Linux H200 云端运行。

## 14. 新对话协作偏好

- 使用中文沟通。
- 用户要求每次推荐实验时都给出完整可运行命令。
- EGL 可用；需要渲染时使用 `--mujoco-gl egl`。
- 用户说“只是讨论”时不要修改代码。
- 不要覆盖纯 PPO 脚本；Residual RL 使用独立入口，公共能力可以复用。
- 先读代码和本 handoff，再决定修改，不要重新猜测已经确认的 reward/termination。
- 对实验结论保持严格：配对结果优先于非配对均值，绝对 gate success 不等于
  相对 CEM 改善。

## 15. 新对话启动提示词

把下面内容作为新对话第一条消息：

```text
请先完整阅读：
C:/Users/12481/Desktop/OH-WorkSpace/robot_description/curl_robot_2d/docs/handoff_2026-07-29_2d_rolling_to_3d_zh.md

这是当前项目的权威 handoff。不要重新假设二维 Residual RL 已经提高鲁棒性；
逐样本配对结果表明 rescue5=0、regression5=0，RL 平均比 CEM 少 0.046 圈。

当前主线是：冻结二维 collision-constrained CEM baseline，开始规划/实现 3D
左右对称滚动 reference，再用小尺度 residual 解决 roll/yaw/左右非对称接触。
暂时不做速度 tracking，不继续扩大二维 residual scale，也不使用不现实的大幅
向后瞬时速度冲击。

请先检查当前 git status、现有 3D 模型资产和关节/执行器映射，然后向我总结：
1. 哪个 3D 模型最适合作为下一阶段基线；
2. 如何把二维 CEM 映射到左右腿；
3. 最小 3D smoke test 应验证哪些指标。

如果建议我运行实验，请始终给出完整命令。除非我说“只是讨论”，否则在确认
现有代码结构后直接实施并测试。
```

