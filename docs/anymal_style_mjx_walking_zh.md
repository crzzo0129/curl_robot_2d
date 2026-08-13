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

## 训练

```powershell
cd curl_robot_2d
python -m scripts.mjx_3d_walking_smoke --mujoco-gl disable
python -m scripts.train_mjx_3d_walking_ppo --preset 4090 --recipe anymal_v1 --out results/mjx_pupper_anymal_v1
```

默认启用批量 MJX 模型随机化：地面摩擦、各刚体质量/惯量、执行器增益、关节阻尼和 armature。可用 `--no-domain-randomization` 做 nominal baseline；各范围均有独立命令行参数。建议先通过 smoke，再做 nominal 短训，最后启用随机化长训，避免把模型或奖励错误误判为鲁棒性不足。
