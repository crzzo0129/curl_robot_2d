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

第一阶段只学习稳定直行，不加入无关难度：训练与 eval 都从无扰动 `stand` 开始，命令固定为 `vx=0.10 m/s, vy=0, yaw_rate=0`，并关闭 observation noise 与 domain randomization：

```bash
python -m scripts.train_mjx_3d_walking_ppo \
  --preset h200 \
  --recipe forward_stage1_v1 \
  --steps 3000000 \
  --num-evals 16 \
  --save-ppo-checkpoints \
  --ppo-checkpoint-dir results/mjx_pupper_forward_stage1_v2/ppo_checkpoint \
  --out results/mjx_pupper_forward_stage1_v2
```

当前实现对 48 维 observation 使用固定物理缩放，不再同时启用 PPO 的运行时 observation normalization。自由关节线速度由世界坐标旋转到机体坐标，MuJoCo 已经以机体局部坐标给出的角速度则不再重复旋转。动作从第一步起直接作为 `stand + scale * action` 的关节目标，不再使用启动 ramp。

Actor 均值输出层使用范围为 `1e-3` 的小随机初始化、bias 为框架默认的零；这不是固定零输出，参数会正常接收梯度。H200 preset 使用 `batch_size=256`、`num_minibatches=8`，rollout quantum 为 40,960；300 万请求步、16 次 eval 时约有 75 次 PPO 更新，而不是旧配置的约 5 次。stage-1 只保留速度命令跟踪，不再叠加与目标速度无关的原始 forward-progress 奖励。

时间上限通过环境的 `info["time_out"]` 明确交给 Brax PPO，并允许 value function bootstrap；真正跌倒仍是普通 termination，不做 bootstrap。最优 checkpoint 按“完整存活、存活时长、upright 失败率、命令跟踪、进度和接触质量”的字典序选择，后面的接触改善不能覆盖前面的存活退化。

启动日志应显示 `reset_noise joint=0 ... root_xy=0 ... root_yaw=0`、`observation=False` 和 `domain_randomization=False`。只有当策略能够持续 10 秒、速度接近 `0.10 m/s` 且失败率明显下降后，才进入速度范围、转向和随机化阶段。

```powershell
cd curl_robot_2d
python -m scripts.mjx_3d_walking_smoke --mujoco-gl disable
python -m scripts.train_mjx_3d_walking_ppo --preset 4090 --recipe anymal_v1 --out results/mjx_pupper_anymal_v1
```

默认启用批量 MJX 模型随机化：地面摩擦、各刚体质量/惯量、执行器增益、关节阻尼和 armature。可用 `--no-domain-randomization` 做 nominal baseline；各范围均有独立命令行参数。建议先通过 smoke，再做 nominal 短训，最后启用随机化长训，避免把模型或奖励错误误判为鲁棒性不足。
