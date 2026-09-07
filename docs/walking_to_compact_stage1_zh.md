# 第一阶段:0.4 m/s 行走 → compact(walk-to-roll 起始段,v1)

> 2026-09-08 设计定稿。从行走切换而非站姿切换:episode 从真实 0.4 m/s
> 行走状态开始,12 自由度 actor 边减速边收拢到 compact 姿态;**终点只认
> compact 姿态门,不看速度;不加载 rolling policy 接管**(后续阶段再接)。

## 1. 设计决策摘要

| 项 | 决定 |
| --- | --- |
| episode 起点 | 行走策略(实机 deploy 接口)0.4 m/s 稳态轨迹的**真实状态快照**(CPU 采集),reset 到快照,保留速度/步态相位 |
| 行走来源 | 上级目录 `rollingquad_2_deploy_robust_dr_policy_stable.json`(36 维×20 帧观测、12 维动作的 deploy 行走策略) |
| 模型/几何 | mesh `rollingquad_abd10_no_self_collision.xml`(由 `rollingquad_abd10.xml` 生成的专用版本,geometry 名 `rollingquad_2_abd10`);**compact keyframe = 前腿 abd −10°/后腿 +10°**,root z 0.1663 m |
| 接触口径 | **自碰撞关闭**:所有机器人 geom 改为 `contype=0 conaffinity=1`(只对地面接触),`floor` 保持 `contype=1 conaffinity=0`;源模型里滚动自碰撞白名单的位掩码(torso 16/7、前腿 2/29、后腿 4/27、足端 8/15)全部去除 |
| 物理 | 运行 XML 替换 `<option>`:0.002 s implicitfast、pyramidal、Newton 20/10、impratio 10、关 eulerdamp;并给 `<compiler>` 注入 `meshdir` 指向源 mjcf 目录(源 XML 的 mesh 是 `../meshes/*.stl` 相对路径)—— 与 CPU 快照回放完全一致 |
| actor 观测/动作 | 与 deploy 控制器同接口:36×20=720 维历史观测,12 维绝对位置目标;**nominal=行走默认姿态,scale=逐关节非对称全范围 `max(high−nominal, nominal−low)`**(与 roll→walk transition policy 同一约定),保证 compact 的 hip/knee 目标落在动作 `[-1,1]` 内;obs 指令字段全程固定 [0.4, 0, 0] |
| 终点门 | **纯姿态门**:12 关节 ≤0.02 rad、root z ≤0.01 m、**滚动轴侧倾(axis tilt,绕 body-X 侧翻)≤0.10 rad**、横向偏移 ≤0.05 m;连续 5 帧(0.10 s)达标即成功;速度/余速不参与判定;**不锁身体四元数**——收拢会绕横轴翻转,任何前向滚动相位都合法 |
| episode 预算 | 5 s(250 × 20 ms);成功/超时/非有限数终止;超时无额外惩罚(dense pose 奖励已覆盖进度) |
| 奖励(v1) | 每步 dense pose `−0.10×(1−quality)` + 成功 +20;`quality` 用**宽松 settling sigma**(关节 0.20 rad、root z 0.03 m、axis tilt 0.20 rad)计算,保证从行走姿态起仍有梯度;时间/动作变化/力矩代价;轻量防跳项(向上 vz、超出 stand/compact 高度包络、三轴角速度) |
| 不做的事 | 无足端拖滑罚、无固定收腿轨迹、无 trajectory 插值、无 episode 内碰撞几何切换、无 rolling teacher |

为什么这样搭:

- 与 roll→walk 方向的快照课程同思路:训练 reset 来自真实轨迹状态,不靠
  冻结策略在线跑,收敛稳定、可复现。
- "边走边转"体现在:起点带有 0.4 m/s 前向速度与真实步态相位,actor 在减速
  过程中收拢;终点不要求停稳,因此不再需要 stand→compact 那种低速窗口。
- obs 用 deploy 接口是刻意的:与行走策略同一观测合同,后续可训练可导出的
  实机 actor,历史帧让策略自己推断速度与时机(实机没有状态估计器)。
- compact 目标的 −10°/+10° 以 `rollingquad_abd10.xml` 的 keyframe 为准;
  基础 `rollingquad.xml` 的 compact 是 ±15°,不要混用。自碰撞专用版由
  `curl_robot_2d_mjx/walk_compact_3d.py` 的 `write_no_self_collision_variant()`
  从 `rollingquad_abd10.xml` 生成,改动只有碰撞掩码,mesh/keyframe/actuator 不变。

## 2. 代码结构

| 文件 | 内容 |
| --- | --- |
| `curl_robot_2d_mjx/walk_compact_3d.py` | contract(`walking_0p4_to_compact_v1_pose_gate_mesh_abd10`)、`WalkCompactConfig`、纯姿态门/势函数/防跳项(xp=numpy|jax.numpy 双端)、快照 bank 校验、运行 XML 生成、`disable_self_collision_xml`/`write_no_self_collision_variant`、fingerprint |
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
|vy|≤0.12、倾角 ≤20°、无非足地面接触/自穿透;默认单 episode(确定性回放
已覆盖全部步态相位,约 250 帧,可用作 smoke)。实测 0.4 m/s 指令下稳态
实际 vx 在 0.34–0.44 m/s 振荡、均值 ≈0.39 m/s。生成规模更大的正式
bank(多扰动、覆盖相位更密)用:

```powershell
python -m scripts.collect_walking_start_snapshots ^
  --policy ..\rollingquad_2_deploy_robust_dr_policy_stable.json ^
  --episodes 70 --reset-noise 0.005 --max-snapshots 16384 ^
  --out results\walk_start_snapshots_0p4
```

产出 `walk_start_snapshots.npz`(qpos/qvel/ctrl/hist/last_action/time)与
`walk_start_snapshots_meta.json`(模型/策略指纹、动作元数据、观测统计)。
meta 的 `action.default`(行走默认姿态)作为 transition actor 的 nominal;
`scale/low/high` 由训练 env 从模型 actuator ctrlrange 计算(逐关节非对称
全范围),不沿用行走策略的固定 scale。

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
(含 axis-tilt 对前向滚动的旋转不变性)、dense pose 奖励(含"远处梯度非平坦"
回归)、防跳项、快照 bank 校验、compact 目标在 transition 动作空间内可达、
采集/训练入口参数、以及 `--dry-run` 端到端;有 mujoco 时附加验证 abd10
compact keyframe 确为 ±10°、执行器顺序与无自碰撞变体。

## 4. 已知边界与后续

- 姿态门不锁四元数,只用滚动轴侧倾(axis tilt)防侧翻:绕 body-Y 的前向滚动
  相位任意都合法,但绕 body-X 侧翻超过 0.10 rad 会拒。这是**临时训练门,
  不是验收标准**,`--axis-tilt-rad` 可按需放宽。
- 动作映射已从行走策略的固定 scale 换成 transition 的非对称全范围 scale,
  否则 compact hip(0.11 rad)在行走 scale(0.5、默认 0.9)下根本够不到,
  会导致 success 恒为 0。
- 无自碰撞、无足滑罚:收拢过程允许腿/壳互相接近,拖地也不罚;若训练出现
  明显利用(如腿部穿插、跳起),再加回 compact startup 的自碰撞白名单/足滑项。
- 快照来自 CPU 回放,训练在 MJX:两侧物理选项已逐项对齐(0.002 s、
  Newton 20/10、pyramidal、impratio 10),但求解器实现仍有数值差异,
  快照 reset 的接触一致性需在 smoke 中观察首帧是否跳变。
- 第二阶段:同一 primitive/mesh 模型从"实际到达状态"连续接入冻结滚动
  策略(不改物理状态、上一命令与观测历史),再做完整 mesh 几何复核;
  deploy 化(36×20 特权→实机观测、RTNeural 导出)与"从行走各步态相位
  起步"都排在 v1 收敛之后。
