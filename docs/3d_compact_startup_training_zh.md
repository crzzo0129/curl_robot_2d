# stand → 低速 compact 启动 PPO（v3，防弹跳）

入口仍为 `python -m scripts.train_mjx_3d_startup_ppo`，但默认任务已替换。
不再追逐动态候选点，不再使用 `--candidate-bank`，不能续训旧 v1/v2 启动参数。
滚动教师参数保持冻结，作为接管后的实际验收；本次交付是训练代码，不是已训练成功的策略。

## 学习过程

1. 从 stand keyframe reset，保留教师配置的关节位置/速度 reset noise。
2. 启动 actor 自主输出 8 个 hip/knee 位置目标；没有 stand→compact 插值路径。
   4 个 abduction 维持原零位伺服。初始化网络均值为 stand，不是预设收腿动作。
3. 最多 3 s，学习靠近 compact，并降低关节、机体线速度和角速度；同时惩罚足端拖地。
4. 连续 3 个控制帧达到状态门槛后，冻结教师从参考时间/振荡相位 0 冷启动。
   不改变 qpos/qvel/ctrl、物理时间、接触状态、真实动作历史和失败计数。
   只为教师重置控制时钟，并对教师所见滚动相位设置 offset；全局累计圈数不清零。
5. 再真实模拟 10 s，沿用原物理失败条件，接管后至少新增 5 个保守圈数且净转角为正，
   才记成功。接管时并不直接奖励“成功”；后续结果属于同一 PPO episode。

模型、碰撞几何、±3 Nm、kp/kd、物理步长都不因此改变。当前教师配置是
1 ms × 20 子步 = 50 Hz；本次不把教师改为实机 52 Hz。
教师冷启动原有 reference/residual ramp 保留。最大 episode 为 13 s / 650 控制步。

compact 并非已验证可长期静态保持的平衡姿态：原生 MuJoCo 固定 compact 目标测试中，
机器人先落地，随后逐渐向负 X 倾倒。所以目标是主动控制到一个近 compact、低余速的
短暂接管窗口，不要求长期停住，也不通过清零速度制造成功。

## 防止“跳起—空中旋转”捷径

v2 的失败策略会突然给出大幅关节目标，利用 3 Nm 饱和力矩跳起。离地期间原本的
接触点滑动自然为零，而机体稳定惩罚在远离 compact 时太弱，于是128条 best eval
全部最终触发 `axis_tilt`，没有一次接管。v3 做以下修改：

- 启动阶段每个关节目标每20 ms最多改变 `0.05 rad`；8个动作仍完全独立。
- 全启动阶段密集惩罚正向根部 `vz`、超过姿态高度包络、三轴角速度和轴倾角。
- compact势函数除12个关节角外，也包含根部高度及完整四元数姿态。
- handoff不是episode终点，势函数在接管帧保持实际值，不再突然清零产生约`-5`的台阶。
- `axis_tilt=0.5 rad` 持续0.1 s的原终止条件不放宽。

按讨论，不增加足端接触数量/离地惩罚，不修改滑动平均定义，也不限制左右对称。
训练直接使用完整10 s教师验收，没有分阶段课程。

## 足端拖动惩罚

每个 1 ms 物理子步计算足端碰撞几何与地面接触点的相对速度，去除接触法向分量。
使用 `v_point = cvel_linear + omega × (point - root_subtree_COM)`，包含转动导致的
接触点速度，而不是直接把机体或足端质心速度当成滑动。

- 仅计入接地的足端几何，离地、自碰撞、纯法向运动不计；无滑动滚动也不计。
- 每只足对其接触点的切向速度平方取平均，再四足平均，避免 mesh 接触点越多罚得越重。
- 每控制步再平均 20 个子步，惩罚 `foot_slip_weight * mean_squared / foot_slip_sigma_m_s²`。
- 默认权重 0.05、速度尺度 0.10 m/s；只惩罚启动段，教师续滚段不罚。
- 现有 `*_foot_proxy` mesh 也包含小腿部分；这里计入这些几何的实际地面接触，
  不是仅鞋底小区域。没有新增或简化碰撞几何。

另有接近 compact 的势函数进度奖励，以及随接近目标增强的余速惩罚；保留时间、
动作变化和力矩代价。允许抬脚重新落脚，不强迫四足一直黏在地上。

## 暂定接管门槛

| 项目 | 默认上限 |
| --- | ---: |
| 12 关节位置最大误差 | 0.02 rad |
| 12 关节速度最大绝对值 | 0.05 rad/s |
| 根部高度误差 | 0.01 m |
| 根部各轴线速度最大绝对值（包括 ±X） | 0.02 m/s |
| 根部各轴角速度最大绝对值 | 0.10 rad/s |
| 姿态角距离、滚动相位误差 | 各 0.05 rad |
| 相对初始位置的横向偏移 | 0.05 m |
| 滚动轴倾斜 | 0.10 rad |
| 教师首子步目标与当前 ctrl 最大差 | 0.05 rad |

还须没有 forbidden contact、物理失败或非有限教师动作。X 位置不设目标。
这些是训练起始门槛，不代表其所有组合都已通过冷启动验证，也未证明从 stand 可稳定到达。
是否真正接得住，由后续 10 s 教师结果判断。

## 云端训练

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mjx_3d_startup_ppo \
  --teacher results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --preset h200 --max-devices 1 \
  --foot-slip-weight 0.05 \
  --out results/rollingquad2_stand_compact_v3_seed0
```

教师目录需有对应 `training_config.json`，否则另传 `--teacher-config`。
使用新的输出目录；不要传旧 candidate bank，也不要恢复旧 v1/v2 startup checkpoint。
默认 2000 万训练步、1024 环境、128 eval 环境，名义物理，不做模型域随机化。
启动 actor 仍使用 53 维特权观测，不是可以直接部署的实机学生。

关注训练输出的 `handoff`、`success`、`timeout`、`slip_distance`。
每次eval还会单独打印一行 `[startup failure]`，按发生率从高到低列出非零终止原因，
例如 `reasons=[axis_tilt=75.0%, startup_timeout=25.0%]`。原因包含轴倾、横漂、
根部过高/过低、非法接触/穿透、非有限数、启动超时和教师续滚不足；若总失败率非零
却没有匹配到已知指标，会显示 `unclassified_failure`，避免静默失败。
最终独立验收在 `evaluation_best.json`：

- `handoff_rate`：到达接管窗口的比例。
- `success_rate` / `conditional_success_after_handoff`：全流程成功率 / 已接管后的成功率。
- `mean_startup_foot_slip_distance_m`：四足各自接触点 RMS 滑速积分之和的均值，
  是拖地程度指标，不是机器人位移，也不是单个固定足端点的精确轨迹长度。
- `startup_foot_slip_rms_m_s` / `maximum_startup_foot_slip_m_s`：启动滑速 RMS / 峰值。
- `mean_handoff_*_speed_*`：每次接管各分量最大绝对速度的跨 episode 均值。

本地只执行静态和 NumPy/MuJoCo 合同测试；按用户要求，JAX 编译、PPO 与收敛验证留在云端。
