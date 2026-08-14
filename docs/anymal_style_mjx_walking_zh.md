# Curl Robot：ANYmal 风格 MJX 行走任务

本入口使用 `assets/curl_robot_3d_pupper_r127p5_open60_width120.xml`，目标是借鉴 ETH `legged_gym` 的 locomotion MDP，而不是复刻 ANYmal 的机构或步态。

## 策略接口

- 动作：12 个归一化关节位置增量，顺序为四条腿的 `abduction, hip, knee`，叠加到 `stand` 默认姿态后由 XML 中的 PD 执行器跟踪。
- 观测：严格采用 48 维本体感知结构：机体坐标线速度 3、机体坐标角速度 3、投影重力 3、速度命令 3、相对默认关节角 12、关节速度 12、上一动作 12。
- 观测噪声：训练时默认按 ETH 的类别设置加入均匀噪声；命令和上一动作不加噪声，可用 `--no-observation-noise` 做消融。
- 命令：`vx, vy, yaw_rate`，默认每 4 秒重采样；包含零命令样本以学习站立。

足端接触、世界坐标位置、绝对 yaw 和 episode 时钟不会进入 actor observation。它们只用于奖励、终止判断和日志，避免策略依赖仿真中难以在实机稳定获得的信息。

## 奖励结构

主任务项是平面速度命令和 yaw rate 的指数跟踪。正则项包括投影重力对应的直立姿态、机身高度、竖直速度、roll/pitch 角速度、足端腾空、摆动足净空、支撑足滑移、动作变化、动作幅值、关节速度、软关节限位、归一化力矩、非足端/自碰撞和失败终止。零命令时额外惩罚偏离默认站姿。

这套结构不使用参考轨迹、固定 gait phase 或规定接触序列；策略可自行形成步态。壳体与非足端接触仍被保留为安全代价，因为本机器人有滚动外壳，后续若要训练 walk/roll 混合策略，应另建任务而不是把行走任务的碰撞约束直接删除。

## 横向位移与终止

世界坐标系下相对初始位置的横向位移 `lateral_drift_m` 只作为诊断指标，不参与 episode 终止或 `episode_failed`。超过 `--diagnostic-lateral-drift`（默认 1.50 m）时会记录 `lateral_drift_exceeded`；旧参数名 `--terminate-lateral-drift` 仅作为兼容别名保留，同样不会触发终止。

训练质量应主要依据机体坐标系中的 `vx, vy, yaw_rate` 命令跟踪误差。固定直行 eval 的命令目标没有横向速度，此时 checkpoint 评分继续使用 `avg_lateral_drift_m` 作为辅助指标，以排除直行时持续侧滑的策略。

## PPO 策略分布与稳定性

行走 Actor 使用对角高斯策略：动作均值由 observation 决定，探索标准差是状态无关的全局可学习参数，默认初值为 `0.30`，随后动作仍由环境裁剪到 `[-1, 1]`。周期 eval 默认使用确定性动作均值；`--stochastic-eval` 仅用于额外检查带策略采样噪声的表现。

PPO 默认使用 `clipping_epsilon=0.20`、全局梯度范数上限 `1.0`、目标 KL `0.01` 和 `ADAPTIVE_KL` 学习率调度。eval 日志同时显示 `kl`、策略标准差 `std` 和实际学习率 `lr`，用于区分动作均值漂移与探索方差失控。

## 训练

```powershell
cd curl_robot_2d
python -m scripts.mjx_3d_walking_smoke --mujoco-gl disable
python -m scripts.train_mjx_3d_walking_ppo --preset 4090 --recipe anymal_v1 --out results/mjx_pupper_anymal_v1
```

默认启用批量 MJX 模型随机化：地面摩擦、各刚体质量/惯量、执行器增益、关节阻尼和 armature。可用 `--no-domain-randomization` 做 nominal baseline；各范围均有独立命令行参数。建议先通过 smoke，再做 nominal 短训，最后启用随机化长训，避免把模型或奖励错误误判为鲁棒性不足。
