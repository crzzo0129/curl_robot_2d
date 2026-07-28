# 双模态机器人统一评估协议

更新日期：2026-07-28

## 1. 目的

本协议用于 CEM、RL、规则控制、二维和未来三维策略的统一评价。核心原则是：

- reward 是训练信号，不是最终指标；
- 硬失败、任务表现、控制代价和鲁棒性分别报告；
- 筛选实验、里程碑验收和最终结论使用不同强度的评价；
- 不同 reward 权重的策略仍可通过同一组物理指标直接比较。

## 2. 固定评价条件

每次可比较实验必须记录：

- git commit 或工作区快照说明；
- XML/model hash、物理 profile 和 MuJoCo/MJX 版本；
- 控制周期、物理步长、episode 时长和 action repeat；
- 初始 keyframe、reset 噪声、命令序列和随机种子；
- 终止条件及阈值；
- 策略 checkpoint、是否 deterministic、训练步数；
- 所有与父实验不同的参数。

当前二维名义基准使用 1 ms 物理步长、20 ms 控制周期和 10 s episode。MJX
训练使用 `cg12` 时，最终策略仍须在未修改的 CPU MuJoCo 参考模型中回放。

## 3. 指标分层

### 3.1 硬约束

任一硬失败都必须单独记录，不能只汇总成 `done`：

| 指标 | 当前二维定义 | 当前阈值 |
|---|---|---:|
| non-finite | action 或 physics 出现 NaN/Inf | 必须为 0 |
| leg crossing | 前后腿中心线拓扑交叉 | 必须为 0 |
| root high | Torso root z 过高 | 0.70 m |
| foot gap | 前后足中心距离过大 | 0.28 m |
| root low | 低 root z | 禁用 |

`root_low` 在二维滚动中不是“侧向倒下”。已知有效 CEM 控制器每圈会到达约
0.0437 m，因此不能使用固定下限淘汰正常滚动。未来三维中应改用 base roll、
pitch、支撑状态和不可恢复性判断真实跌倒。

接触穿透当前作为质量约束连续报告，而不是新的硬终止：

- 非允许接触比例与总时长；
- 非允许穿透积分；
- 最大非允许穿透；
- 允许足端接触超过 0.5 mm 容差的积分与最大值。

这些量对 solver 设置敏感。只有完成 CPU/MJX 和不同 solver 的一致性检查后，
才能将某个穿透值升级为跨后端硬门槛。

### 3.2 滚动任务表现

| 指标 | 定义 |
|---|---|
| net turns | Torso pitch 净变化除以 \(2\pi\) |
| displacement | root x 的净位移 |
| translation-equivalent turns | 位移除以名义滚动周长 |
| conservative progress | 相位进展与位移等效进展的较小值 |
| rolling mismatch | 相位进展与位移等效进展之差，用于识别滑动或原地自转 |
| backward travel | 负方向相位和位移累计 |
| survival fraction | 实际步数除以 episode 最大步数 |
| start latency | 从命令开始到达到最小持续滚动速度的时间 |

圈数必须和位移、mismatch、地面接触一起解释。只增加 pitch 而没有相应水平
位移，不算有效滚动。

### 3.3 控制代价与接触质量

- 电机正向、吸收和净做功；
- 单位距离正向做功；
- 最大执行器力矩和饱和比例；
- normalized torque RMS；
- action RMS 和 action-rate RMS；
- 腾空比例、总时长和最长连续腾空；
- 足端最大间距、root z 最小值和最大值；
- 非允许接触与穿透指标。

当前 MJX 训练已经直接记录大部分逐步指标，但最终确定性评价还需要从
`evaluation_rollout.npz` 补算 min/max、连续腾空和做功等轨迹统计，才能与
CEM 完全对齐。

### 3.4 速度、制动和切换指标

M2 以后新增：

| 技能 | 必须报告 |
|---|---|
| speed tracking | 稳态速度误差、超调、上升时间、命令切换后恢复时间 |
| brake | 停止时间、停止距离、残余角速度、回滚量 |
| align | 进入可展开相位窗口的时间、最终相位误差、窗口内保持时间 |
| curl/uncurl | 完成时间、关节裕度、碰撞、足端可支撑性 |
| recover stand | 成功率、恢复时间、最终 base 姿态和支撑稳定性 |
| complete chain | 整链成功率、首个失败状态、每个状态停留时间 |

### 3.5 三维新增指标

- base roll/pitch/yaw 误差和角速度；
- 横向漂移及每米滚动的横向漂移；
- 左右关节、接触相位和地面冲击不对称；
- 行走速度跟踪、足端滑移、离地间隙和跌倒率；
- mode switch 前后速度连续性；
- 弧壳和配重对行走工作空间及稳定裕度的影响。

## 4. 评价强度

| 级别 | 用途 | 最低设置 |
|---|---|---|
| S0 smoke | 检查能否运行和编译 | 1 seed，短训练/短 rollout，不作性能结论 |
| S1 screen | 判断是否值得继续训练 | 固定 seed；最近评估存活 >=20%，估算净滚动 >=0.25 圈，non-finite=0 |
| V1 nominal | 单一名义设置验收 | 5 个未训练种子，每个 10 s，deterministic |
| V2 cross-backend | 排除 MJX 数值捷径 | V1 候选在 CPU MuJoCo 参考模型重放 |
| R1 robustness | 判断是否可进入下一系统阶段 | 质量、COM、摩擦、力矩、reset 和传感噪声的未见组合 |
| H1 hardware | 实体逐级验证 | 限流、低速、低高度/系绳，逐项解除保护 |

S1 是计算预算门槛，不是任务成功标准。当前 M1 的 V1 验收要求为：

- 5/5 rollout 无硬失败且完整存活 10 s；
- 最差净滚动至少 1 圈；
- 净滚动中位数至少 5 圈；
- 位移方向正确，且不存在明显原地自转；
- V2 CPU 回放仍满足硬约束和最低 1 圈要求。

这些条件刻意低于 CEM 的名义速度，因为 M1 只验证 RL 是否学会有效滚动。
后续价值主要由调速、制动、恢复和鲁棒性评价。

## 5. 当前参考基准

碰撞约束 CEM 控制器在修改前 2 mm 外壳 CPU MuJoCo 模型、10 s 名义评价
中的参考值：

| 指标 | 数值 |
|---|---:|
| net turns | 9.914 |
| root x displacement | 9.595 m |
| translation-equivalent turns | 10.350 |
| positive actuator work | 35.980 J |
| maximum torque | 3.830 N m |
| airborne fraction | 12.91% |
| longest airborne | 15 ms |
| maximum foot gap | 0.208 m |
| forbidden contact fraction | 1.36% |
| maximum forbidden penetration | 0.459 mm |
| maximum allowed foot penetration | 0.602 mm |
| leg crossing | 0 |

2026-07-28 的独立终止条件回放还测得：

- 1 kHz 控制更新：root z 最小 0.043703 m，最大 0.273141 m；
- 50 Hz 动作保持：root z 最小 0.043694 m，最大 0.273141 m；
- 两种回放均未触发 root_high 或 foot_gap。

当前 28 mm 外壳模型上，同一控制器不重训直接回放为 9.049 圈、8.831 m，
腿部交叉为 0。该结果证明缩短外壳后滚动仍可行，不意味着 RL 必须复制其周期
或达到相同名义速度。当前模型的正式 CEM 基准仍需重新搜索。

## 6. 比较和决策规则

1. 先比较硬失败，再比较任务表现，最后比较能耗和动作平滑度。
2. 不用不同 reward 配置的 total reward 排名。
3. 每次参数实验尽量只改变一个假设组；同时改变多项时必须明确无法单独归因。
4. 报告全部种子和分位数，不只报告最好 checkpoint 或最好视频。
5. 未通过 S1 时不投入无界长训；先诊断失败类型并做有界对照。
6. 通过名义评价但未通过 CPU 回放或扰动评价的策略，不进入结构结论。
7. COM 比较分为“固定策略扫 COM”和“每个 COM 重训”，两类结果不得混用。

## 7. 当前输出与待补项

PPO 训练目录目前已保存：

- `training_config.json` 和 `reward_config.json`；
- `metrics_history.json` 和 `reward_history.json`；
- `training_summary.json`；
- `params_best` 和 `params_final`；
- `evaluation_rollout.npz` 和 `evaluation_summary.json`。

近期应补一个统一离线 evaluator，使 CEM CSV 和 PPO NPZ 产生相同字段的
`benchmark_summary.json`。在该工具完成前，跨方法对比需明确哪些量来自训练
指标、哪些量来自独立 CPU 回放。
