# 第一阶段:0.4 m/s 行走 → compact(walk-to-roll 起始段,v1)

> 2026-09-08 设计定稿。从行走切换而非站姿切换:episode 从真实 0.4 m/s
> 行走状态开始,12 自由度 actor 边减速边收拢到 compact 姿态;**终点只认
> compact 姿态门,不看速度;不加载 rolling policy 接管**(后续阶段再接)。

## 1. 设计决策摘要

| 项 | 决定 |
| --- | --- |
| episode 起点 | 行走策略(实机 deploy 接口)0.4 m/s 稳态轨迹的**真实状态快照**(CPU 采集),reset 到快照,保留速度/步态相位 |
| 行走来源 | 上级目录 `rollingquad_2_deploy_robust_dr_policy_stable.json`(36 维×20 帧观测、12 维动作的 deploy 行走策略) |
| 模型/几何 | mesh `rollingquad_abd10.xml`(geometry 名 `rollingquad_2_abd10`);**compact keyframe = 前腿 abd −10°/后腿 +10°**,root z 0.1663 m |
| 接触口径 | 保持 XML 默认:外壳/Torso 等与地面接触开,**自碰撞不开**(default geom contype=0 conaffinity=1,无 pair/exclude) |
| 物理 | 运行 XML 只替换 `<option>`:0.002 s implicitfast、pyramidal、Newton 20/10、impratio 10、关 eulerdamp —— 与 CPU 快照回放完全一致 |
| actor 观测/动作 | 与 deploy 控制器同接口:36×20=720 维历史观测,12 维绝对位置目标 `pose + scale × action`;obs 指令字段全程固定 [0.4, 0, 0] |
| 终点门 | **纯姿态门**:12 关节 ≤0.02 rad、root z 误差 ≤0.01 m、姿态四元数距离 ≤0.05 rad、横向偏移 ≤0.05 m;连续 5 帧(0.10 s)达标即成功;速度/余速不参与判定 |
| episode 预算 | 5 s(250 × 20 ms);成功/超时/非有限数终止;超时无额外惩罚(dense pose 奖励已覆盖进度) |
| 奖励(v1) | 每步 dense pose `−0.10×(1−quality)` + 成功 +20;时间/动作变化/力矩代价;轻量防跳项(向上 vz、超出 stand/compact 高度包络、三轴角速度) |
| 不做的事 | 无足端拖滑罚、无固定收腿轨迹、无 trajectory 插值、无 episode 内碰撞几何切换、无 rolling teacher |

为什么这样搭:

- 与 roll→walk 方向的快照课程同思路:训练 reset 来自真实轨迹状态,不靠
  冻结策略在线跑,收敛稳定、可复现。
- "边走边转"体现在:起点带有 0.4 m/s 前向速度与真实步态相位,actor 在减速
  过程中收拢;终点不要求停稳,因此不再需要 stand→compact 那种低速窗口。
- obs 用 deploy 接口是刻意的:与行走策略同一观测合同,后续可训练可导出的
  实机 actor,历史帧让策略自己推断速度与时机(实机没有状态估计器)。
- compact 目标的 −10°/+10° 以 `rollingquad_abd10.xml` 的 keyframe 为准;
  基础 `rollingquad.xml` 的 compact 是 ±15°,不要混用。

## 2. 代码结构

| 文件 | 内容 |
| --- | --- |
| `curl_robot_2d_mjx/walk_compact_3d.py` | contract(`walking_0p4_to_compact_v1_pose_gate_mesh_abd10`)、`WalkCompactConfig`、纯姿态门/势函数/防跳项(xp=numpy|jax.numpy 双端)、快照 bank 校验、运行 XML 生成、fingerprint |
| `curl_robot_2d_mjx/environment_walk_compact_3d.py` | MJX `WalkCompactEnv`(快照 reset、36×20 历史帧 obs、12 维绝对位置目标、姿态门终止、dense pose+防跳奖励)与自动 reset 包装器 |
| `scripts/collect_walking_start_snapshots.py` | CPU(mujoco)采集脚本:deploy 行走策略固定 0.4 m/s 回放,热身后采样并过滤,输出 npz+meta |
| `scripts/train_walk_compact_ppo.py` | PPO 训练入口(仿 `train_mjx_3d_startup_ppo`:smoke/dry-run/eval-only/best 选取/报告) |

## 3. 运行方法

### 3.1 采集行走快照(本地 CPU,需 mujoco + numpy)

```powershell
python -m scripts.collect_walking_start_snapshots ^
  --policy ..\rollingquad_2_deploy_robust_dr_policy_stable.json ^
  --out results\walk_start_snapshots_0p4
```

默认:1.5 s 热身、随后 5 s 内每控制步采样、要求 |vx−0.4|≤0.08 m/s、
|vy|≤0.12、倾角 ≤20°、无非足地面接触/自穿透;默认单 episode
(确定性回放即可覆盖全部步态相位),也可 `--episodes N --reset-noise 0.005`
做多扰动采集。产出 `walk_start_snapshots.npz`(qpos/qvel/ctrl/hist/
last_action/time)与 `walk_start_snapshots_meta.json`(模型/策略指纹、动作
元数据、观测统计)。meta 的 `action.default/scale/lower/upper` 是训练 env
的 ctrl 语义来源,与策略 JSON 一致。

### 3.2 训练(云端 MJX/JAX)

先做合同检查(不训练,不需要 JAX):

```powershell
python -m scripts.train_walk_compact_ppo --dry-run ^
  --snapshots results\walk_start_snapshots_0p4 --out results\walk_compact_check
```

接口 smoke(真实 MJX 编译步进,不做 PPO):

```powershell
python -m scripts.train_walk_compact_ppo --preset smoke --smoke-steps 40 ^
  --snapshots results\walk_start_snapshots_0p4 --out results\walk_compact_smoke
```

正式训练(H200 预设,新目录):

```bash
python -m scripts.train_walk_compact_ppo --snapshots results/walk_start_snapshots_0p4 \
  --preset h200 --max-devices 1 --out results/walk_compact_stage1_seed0
```

产物:运行 XML + `training_config.json`(含快照/模型指纹与全部 gate/奖励
参数)、`metrics_history.json`、`params_best/params_final`、
`evaluation_best.json`(独立确定性评估:success/timeout/failed 率、姿态质量、
终点 gate 误差)、`summary.json`。

验收参考:独立评估 `success_rate ≥ 0.95` 才算名义通过;文档明示
`rolling_continuation_evaluated=false`、`deployable_actor=false`
(特权观测定义已按 deploy 合同,但本轮 actor 仍是仿真网络,未导出)。

### 3.3 本地合同测试(无需 mujoco/jax)

```powershell
python -m unittest tests.test_walk_compact_3d -v
```

覆盖:观测/动作合同尺寸、策略关节顺序、运行 XML 只改 option、姿态门公式
(含四元数符号不变性)、dense pose 奖励、防跳项、快照 bank 校验、采集/训练
入口参数、以及 `--dry-run` 端到端;有 mujoco 时附加验证 abd10 compact
keyframe 确为 ±10° 与执行器顺序。

## 4. 已知边界与后续

- 姿态门含与 compact keyframe 的四元数距离:若实际收拢终点带残余滚动相位
  (绕 y 轴转了非整圈),该门会拒绝;先按"停在近相位"的保守口径,参数
  `--orientation-rad` 可按需放宽。这是**临时训练门,不是验收标准**。
- 无自碰撞、无足滑罚:收拢过程允许腿/壳互相接近,拖地也不罚;若训练出现
  明显利用(如腿部穿插、跳起),再加回 compact startup 的自碰撞白名单/足滑项。
- 快照来自 CPU 回放,训练在 MJX:两侧物理选项已逐项对齐(0.002 s、
  Newton 20/10、pyramidal、impratio 10),但求解器实现仍有数值差异,
  快照 reset 的接触一致性需在 smoke 中观察首帧是否跳变。
- 第二阶段:同一 primitive/mesh 模型从"实际到达状态"连续接入冻结滚动
  策略(不改物理状态、上一命令与观测历史),再做完整 mesh 几何复核;
  deploy 化(36×20 特权→实机观测、RTNeural 导出)与"从行走各步态相位
  起步"都排在 v1 收敛之后。
