# 3D Rolling Handoff

更新日期：2026-08-04

项目目录：

```text
C:/Users/12481/Desktop/OH-WorkSpace/robot_description/curl_robot_2d
```

Linux 云端目录以实际 clone 位置为准。本文只讨论 **3D rolling**，不讨论 walking。

本文基于提交：

```text
5c9835b8a3161e626ced228a90a640e970c48340
```

## 1. 当前目标与总决策

3D rolling 不是从零开始的 3D locomotion。当前主线是：

```text
2D collision-constrained CEM reference
-> 左右复制到 3D 对称滚动 reference
-> Residual RL 修正 2D 到 3D 的共同模型误差
-> 左右差动 residual 稳定横漂、侧倾和非对称接触
```

正式 reference：

```text
results/collision_constrained_cem_foot_gap_2mm_short_contact/best_phase_controller.json
```

当前不建议：

- 在 3D 中冷启动重做完整 CEM；
- 在 reference parity 未确认前继续长时间 PPO；
- 为了适配一个未收敛求解器而把 reference 重新优化成求解器伪影；
- 把 `exact` 失败和 `noisy` 失败解释成同一个问题。

## 2. 3D 模型确实由 2D 模型提升

主要文件：

```text
curl_robot_2d/model_3d.py
assets/curl_robot_3d.xml
curl_robot_2d/parameters.py
```

提升关系：

- 二维前/后 sagittal chain 分别复制到左、右 side rail；
- 每侧 thigh/shank 质量是二维聚合质量的一半；
- 每侧执行器的力矩上限、`kp`、`kd`、armature 和 damping 是二维成对参数的一半；
- 左右两侧使用同一二维动作时，总质量和总驱动能力保持二维量级；
- torso 质量不变，绕滚动轴的惯量使用二维 planar inertia；
- 3D 根节点使用 freejoint，因此新增 lateral、yaw、axis tilt 等自由度。

二维动作到三维动作的当前映射：

```text
front_hip, front_knee, rear_hip, rear_knee
->
front_left_hip,  front_left_knee,
front_right_hip, front_right_knee,
rear_left_hip,   rear_left_knee,
rear_right_hip,  rear_right_knee
```

结论：3D 的对称子空间应近似复现 2D 动力学，但不会天然完全等价。差异来自完整
6-DoF 根节点、左右分离的空间惯量、重复且空间分离的接触点，以及 3D 摩擦/接触求解。

更重要的是，当前 reset 对 8 个关节和全部 `qvel` 独立加噪。一旦启用 noisy reset，
状态立即离开左右对称子空间；二维 reference 没有专门控制这些反对称模态。

## 3. 当前 3D MJX 环境

主要文件：

```text
curl_robot_2d_mjx/config_3d.py
curl_robot_2d_mjx/environment_3d.py
curl_robot_2d_mjx/reward_3d.py
curl_robot_2d_mjx/cem_reference.py
```

基本配置：

```text
physics timestep        0.001 s
action repeat           20
control timestep        0.020 s
episode length          500 steps = 10 s
action size             8
base observation size   59
explicit phase obs      +4, total 63
startup action ramp     0.25 s
joint reset noise       independent +/-0.005 rad
velocity reset noise    independent +/-0.005
```

注意：上面的 noisy reset 还没有 common/differential 分解，也没有课程调度。

## 4. 相位锁定已经修复

当前 reference 不是固定时钟直接线性推进。

定义：

```text
theta   机器人实际累计滚动相位
phi     CEM oscillator/reference 相位
```

每个 1 ms physics step：

1. 用当前实现中的 `qvel[4]` 积分更新 `theta`；
2. 用 `advance_oscillator(theta, phi, ...)` 更新 `phi`；
3. 用新的 `phi` 计算 CEM reference action；
4. reference 与 residual 混合后执行物理步。

因此，物理滚动变慢时，相位反馈会降低 reference 相位速度；这不是简单的线性计时器。

`phase_locked_coupled_v6` 还显式向 policy 提供：

```text
sin(theta), cos(theta), sin(theta - phi), cos(theta - phi)
```

这使策略既知道机器人实际滚到哪里，也知道它相对 reference 相位差多少。旧的
`phase_locked_v3` 没有这 4 维显式反馈。

## 5. 当前 Residual Action

零 residual 时，动作严格回到左右对称的 2D CEM reference。当前代码支持两种模式：

```text
independent: policy 直接输出 8 个左右独立 residual
coupled:     policy 输出 4 个 common + 4 个 differential residual
```

coupled 模式的公式：

```text
a_left  = a_ref + gain * (a_common + d * a_diff)
a_right = a_ref + gain * (a_common - d * a_diff)
```

`phase_locked_coupled_v6` 当前参数：

```text
reference weight          1.00
residual gain             0.15
differential scale d      0.25
zero residual init        true
initial policy std        0.20
learning rate             5e-5
entropy cost              1e-3
```

这意味着动作分解已经实现，但 residual gain 和 reset noise 还不是分阶段课程。

## 6. Reward 与 Termination

`phase_locked_coupled_v6` 使用的主要 reward 权重：

```text
roll progress                 +8.0
roll/translation mismatch     -0.8
backward progress             -1.0
lateral velocity squared      -4.0
lateral drift absolute        -6.0
axis tilt squared             -10.0
action rate                   -0.02
raw residual action cost      -0.01
failed progress clawback      -2.0 * positive accumulated progress
termination                   -40
severe extra termination      -40
```

滚动进度使用 conservative potential：

```text
P = min(cumulative rotation, cumulative translation)
```

它用于防止只有躯干自转但没有真实平移的策略拿到高分。日志中的 `net_rotation`、
`abs_rotation`、`translation` 和 `mismatch` 必须一起看，不能只看 rotation。

足部接触惩罚包括：

- 同侧前后足接触开始事件；
- 接触持续时间；
- 超出允许深度后的积分和最大深度增量；
- cross-side foot contact；
- 其他 forbidden contact 的持续时间和穿透深度。

当前主要终止条件：

```text
root z < 0.025 m continuously for 0.20 s
root z > 0.80 m
abs lateral drift > 0.20 m
rolling axis tilt > 0.50 rad continuously for 0.10 s
forbidden penetration > 0.004 m
forbidden contact continuously for 0.20 s
nonfinite physics/action
```

横漂终止和所谓“撞线”是同一阈值事件：机器人累计横向位置越过 `+/-0.20 m`；不是
与墙或实体边界发生接触。旧 reward 会允许策略先滚出大量 progress，再横漂终止仍
保留较高总回报，因此后来增加了 failed-progress clawback 和更强 terminal penalty。

## 7. 已观察到的 PPO 现象

一次 `phase_locked_v3 + cg12` smoke 的周期评估如下：

| step | mean length | failed | turns/episode | lateral failure |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 223.4 | 75.0% | 1.433 | 75.0% |
| 25,600 | 345.1 | 50.0% | 1.694 | 50.0% |
| 51,200 | 467.4 | 12.5% | 0.879 | 12.5% |
| 76,800 | 387.4 | 37.5% | 1.696 | 37.5% |

这次训练降低横漂失败时也明显降低滚动速度，随后又发生回退。它不能证明 Residual
RL 已经解决 3D rolling，只说明 reward、reference 物理一致性和探索结构需要先稳定。

相关修复已经加入：

- deterministic evaluation；
- failed-progress clawback；
- best/final PPO 参数选择修复；
- common/differential residual exploration；
- 显式 phase-lock observation；
- solver/backend parity 工具。

目前没有一份可以宣布成功的 3D rolling PPO checkpoint。

## 8. 求解器排查结论

用户在新 parity 代码上的 10 s exact 结果：

| case | conservative | rotation | translation | 结论 |
| --- | ---: | ---: | ---: | --- |
| CPU Newton exact | 7.936 | 7.936 | 8.246 | reference 在收敛物理下可滚 |
| CPU CG12 exact | 0.448 | 0.487 | 0.448 | CG12 明显失速 |
| MJX CG12 exact | 0.448 | 旧指标 3.464 | 0.448 | 平移与 CPU CG12 一致 |
| MJX CG12 noisy | median 0.439 | 旧指标 3.510 | mean 0.558 | 大多数 seed 失速 |

这里旧 `rotation` 是累计绝对旋转，后来已拆成 `net_rotation` 和 `abs_rotation`。CG12
下 abs rotation 很高但 translation 很低，说明接触滑移/往复运动被旧指标放大。

本地求解器 sweep：

```text
Newton 8 iterations / 8 line-search       about 7.94 turns
Newton 12 / 6                             about 0.28 turns
Newton 12 / 10                            about 7.93 turns
CG12 / 6                                  about 0.46 turns
CG20 / 10                                 about 8.01 turns
CG40 / 20                                 about 7.94 turns
CG80 / 40                                 about 7.94 turns
```

结论不是“CG 算法不适合”，而是 CG12/6 在 reference 的临界接触段没有收敛。line-search
预算同样关键。当前可选 profile：

```text
reference   Newton 20 / line-search 10
newton4     Newton 4 / 4
newton8     Newton 8 / 8
cg12        CG 12 / 6
cg20        CG 20 / 10
```

2026-08-04 云端 H200 已补跑 `cg20_seed0` parity。完整四组结果：

| case | conservative | net_rotation | abs_rotation | translation | mismatch | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU Newton exact | 7.944 | 7.936 | 7.944 | 8.246 | -0.302 | 收敛物理下可滚 |
| CPU CG20 exact | 8.005 | 7.997 | 8.005 | 8.359 | -0.354 | CPU CG20/10 可滚 |
| MJX CG20 exact | 7.960 | 7.937 | 7.960 | 8.251 | -0.292 | MJX exact 与 CPU 基本一致 |
| MJX CG20 noisy | mean 0.677, median 0.442, range 0.434--8.002 | 0.703 | 3.587 | 0.688 | 2.900 | 大多数 noisy seed 失速，少数 seed 可滚 |

因此主判断已经落到：`mjx_cg20_exact` 正常，主要问题是 noisy reset 把系统立即推出
左右对称子空间。下一步优先实现 reset common/differential curriculum，而不是重做
3D CEM 或直接长训 PPO。

但同一脚本另一次只运行 `mjx_cg20_exact mjx_cg20_noisy` 时曾出现异常：

| case | conservative | net_rotation | abs_rotation | translation | mismatch | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MJX CG20 exact | 0.435 | 0.449 | 3.411 | 0.435 | 2.976 | exact 冷启动异常失速 |
| MJX CG20 noisy | mean 0.557, median 0.442, range 0.432--7.861 | 0.583 | 3.516 | 0.559 | 2.957 | 与 noisy 失速分布一致 |

这次 exact 异常不应直接解释为 reference/backend parity 失败；它与后续完整 parity
相矛盾，更像 MJX 冷启动、编译缓存、导入顺序或运行环境版本差异导致的可复现性问题。
后续 parity 日志必须打印 `physics_profile`、solver iterations、line-search iterations
和 reset noise，以便确认异常时是否真的使用了 CG20/10 exact 配置。

原始诊断分叉仍然保留：

1. MJX CG20 exact 与 CPU CG20 不一致；还是
2. exact 正常、只有 noisy 分布失败。

这两种情况必须采用不同修复路线。

## 9. 当前最重要的诊断分叉

### A. `mjx_cg20_exact` 也失败

优先检查 backend/model parity，不先训练 PPO，也不立即重做完整 CEM：

- 对比 CPU Newton、CPU CG20、MJX CG20 的 first-circle trajectory；
- 定位第一圈首次 phase rate 降速、接触集合分叉和 slip 激增时刻；
- 核对 2D 到 3D 的惯量、接触几何、friction、actuator scaling；
- 必要时只调一个小型 transfer adapter，例如初始相位、target scale、oscillator
  coupling，而不是重新搜索全部 Fourier controller 参数。

### B. `mjx_cg20_exact` 正常，只有 `mjx_cg20_noisy` 失败

reference 的二维对称主运动仍然有效；失败来自 3D 新增反对称模态。下一步应实现
reset noise curriculum，而不是用 3D CEM 代替 RL。

建议把 reset 扰动拆成：

```text
q_left  = q_nominal + noise_common + alpha * noise_diff
q_right = q_nominal + noise_common - alpha * noise_diff
```

速度扰动也做相同分解。

## 10. 推荐的 2D 到 3D 课程

### Stage 0：reference parity

- zero residual；
- exact symmetric reset；
- 先要求第一圈一致，再看完整 10 s；
- CPU Newton、CPU CG20、MJX CG20 必须分别记录。

### Stage 1：共同模态修正

- 左右完全相同的 joint/qvel noise；
- `alpha=0`，没有 differential reset noise；
- residual 零初始化；
- common residual gain 从 `0.02` 或 `0.05` 开始；
- 目标是修正 2D 到 3D 的共同模型误差，不追求 lateral recovery。

### Stage 2：轻微反对称扰动

- 逐步把 `alpha` 从 `0.05 -> 0.10 -> 0.25`；
- 保留 common/differential action；
- differential action gain 小于 common gain；
- gate 同时要求 turns、survival、lateral drift 和 axis tilt，不以 reward 单独选模。

### Stage 3：完整 3D 稳定化

- 扩大 differential reset noise；
- 加侧向初速度、roll/yaw 初始误差；
- 最后再加摩擦、质量和执行器 domain randomization；
- 每一阶段都保留 exact/noisy reference-only baseline。

当前 Stage 1--3 的 reset 课程尚未实现。`phase_locked_coupled_v6` 只实现了动作分解和
显式相位 observation。

补充：`startup_action_ramp_s` 当前只作用于 residual action，不会放大 CEM reference。
`reference_startup_boost=0.20` 已在云端验证会让 CPU/MJX CG20 exact 同时掉到约
0.44 圈，因此不要继续用放大起步幅值作为主线。更合理的是测试 reference amplitude
从安全端逐步回到原始 CEM：

```text
reference_action_scale                 全程 reference normalized action 缩放
reference_ramp_start_scale             起步 reference scale；None 表示从最终 scale 开始
reference_ramp_duration_s              从起步 scale ramp 到最终 scale 的时间
reference_startup_boost                起步额外 boost，默认 0
reference_startup_boost_duration_s     boost 衰减回全程 scale 的时间
```

默认 `reference_ramp_start_scale=None`，不改变既有结果。建议先扫短 ramp：

```text
reference_action_scale=1.0
reference_ramp_start_scale=0.25 or 0.50
reference_ramp_duration_s=0.10, 0.25, 0.50
reference_startup_boost=0.0
```

验收仍看 exact/noisy 的 conservative、net_rotation、abs_rotation、translation 和 mismatch。

## 11. 下一次首先运行的实验

先用增强日志复现 MJX CG20 exact baseline 稳定性，不启动训练。同一命令建议独立运行两次：

```bash
python -m scripts.compare_mjx_3d_reference \
  --cases mjx_cg20_exact mjx_cg20_noisy \
  --episode-length 500 \
  --noise-seeds 64 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.50 \
  --output results/mjx_3d_reference_parity/cg20_seed0.json
```

如果 baseline exact 稳定，再做 reference startup boost sweep，例如：

```bash
python -m scripts.compare_mjx_3d_reference \
  --cases cpu_cg20_exact mjx_cg20_exact mjx_cg20_noisy \
  --episode-length 500 \
  --noise-seeds 64 \
  --seed 0 \
  --mujoco-gl disable \
  --memory-fraction 0.50 \
  --reference-ramp-start-scale 0.50 \
  --reference-ramp-duration-s 0.10 \
  --output results/mjx_3d_reference_parity/cg20_ramp050_010_seed0.json
```

必须保存并汇报：

```text
physics_profile
solver_iterations
solver_ls_iterations
reset_joint_noise_rad
reset_velocity_noise
conservative
net_rotation
abs_rotation
translation
mismatch
failure breakdown
```

如果两次独立运行的 `mjx_cg20_exact` 都稳定在约 8 圈，进入 reset common/differential
curriculum，并用 `phase_locked_coupled_v6` 做新 smoke。当前不建议直接发起长训练。
如果 exact 偶发掉到约 0.4 圈，先清理/禁用 JAX compilation cache 后重复 parity，并
记录导入顺序和 runtime 版本。

## 12. 渲染命令

CPU MuJoCo 下渲染 CG20 reference：

```bash
python -m scripts.view_3d_cem_reference \
  --physics-profile cg20 \
  --duration 10 \
  --headless \
  --gif results/3d_cg20_reference_10s.gif
```

注意该脚本是 CPU MuJoCo 渲染，不等于 MJX rollout。它用于看滚动和接触形态，不能
替代 `compare_mjx_3d_reference` 的 backend parity。

已有本地 CG12 视频：

```text
results/3d_cg12_reference_10s.gif
```

它约为：translation 0.455 圈、net rotation 0.501 圈、abs rotation 3.514 圈，显示
CG12 下机器人有明显运动，但没有形成持续有效滚动。

## 13. 关键代码入口

```text
curl_robot_2d/model_3d.py                    2D 到 3D 模型提升
curl_robot_2d_mjx/config_3d.py               任务和 physics profile
curl_robot_2d_mjx/environment_3d.py          phase/action/obs/contact/termination
curl_robot_2d_mjx/reward_3d.py               3D rolling reward
curl_robot_2d_mjx/cem_reference.py           phase-locked CEM controller
scripts/train_mjx_3d_residual_ppo.py          3D Residual PPO
scripts/compare_mjx_3d_reference.py           CPU/MJX parity
scripts/evaluate_3d_symmetric_cem_reference.py CPU reference 评估
scripts/view_3d_cem_reference.py              CPU reference 渲染
tests/test_mjx_3d_contract.py                 3D 环境合同
tests/test_mjx_3d_reward.py                   reward 单测
tests/test_mjx_3d_training.py                 recipe/训练合同
tests/test_compare_mjx_3d_reference.py        parity 工具单测
```

## 14. 相关提交

```text
2edd2ac  Penalize failed progress in 3D residual PPO
9b8ca1c  Balance failed progress penalty for 3D PPO
f1c94e1  Couple 3D residual exploration by side pairs
cb5dd19  Expose phase-lock feedback to 3D residual policy
71db8c8  Fix final 3D PPO best parameter selection
aa17c2d  Add 3D reference backend parity diagnostic
dd4414c  Avoid slow MJX Newton parity compilation
3b3b79c  Add efficient Newton 3D physics profile
bc321bd  Render 3D references with selectable physics
5c9835b  Add converged CG20 3D physics profile
```

## 15. 给下一次对话的启动摘要

```text
请先阅读 docs/handoff_2026-08-04_3d_rolling_zh.md。

3D rolling 的主 reference 来自 2D collision-constrained CEM，并左右对称复制到
3D。不要从零做 3D CEM。当前 phase lock、common/differential residual action、
显式 theta/phase-error observation、3D reward/termination 和 solver parity 工具均已
实现。CG12 已确认未收敛；本地 CPU CG20/10 可恢复约 8 圈，但用户报告云端 MJX
CG20 仍不行，完整 exact/noisy 指标尚待归档。

下一步先运行 CPU Newton exact、CPU CG20 exact、MJX CG20 exact/noisy 四组 parity。
如果 MJX exact 失败，查 backend/model/contact parity；如果 exact 正常但 noisy 失败，
实现 symmetric -> differential reset-noise curriculum，让 2D reference 负责 common
rolling，让 Residual RL 只解决 3D 新增的反对称稳定问题。不要在 parity 未确认前启动
长 PPO，也不要把 noisy failure 直接解释成 2D reference 无效。
```
