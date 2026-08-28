# Curl Robot：ANYmal 风格 MJX 行走任务

> 当前主训练入口已经切换为 `scripts/train_ppo_walk3d.py` 和
> `scripts/train_ppo_deploy.py`。本文件后续关于
> `train_mjx_3d_walking_ppo.py` 的内容只作为旧实验记录，不再是推荐流程。

当前两个主入口都固定使用由正确 URDF 转换得到的真实结构模型：
`assets/rollingquad_description_2/mjcf/rollingquad.xml`。

## RollingQuad 2 重新训练

先执行 probe，再开始训练：

```bash
python -m scripts.train_ppo_walk3d probe
python -m scripts.train_ppo_walk3d

python -m scripts.train_ppo_deploy probe
python -m scripts.train_ppo_deploy
```

`train_ppo_walk3d.py` 使用48维仿真本体观测；`train_ppo_deploy.py` 使用与实机控制器一致的36维单帧、20帧历史观测。两者网络输入不同，需要分别训练，不能互相恢复 checkpoint。需要域随机化时在命令中加入 `dr`。

该适配不改变 URDF 的倾斜外摆轴、零位、范围、质量或惯量。模型内部 `qpos` 仍按导出层级排列，脚本会按关节名映射到统一的 `FL, FR, RL, RR` policy顺序，每腿均为 `abduction, hip, knee`。不要恢复旧 Pupper 或旧 MJX walking policy checkpoint。新输出使用独立文件名 `rollingquad_2_walk3d_policy.bin` / `rollingquad_2_deploy_policy.bin`，不会自动续训旧模型的 policy。

## 策略接口

- 动作：12 个归一化关节位置增量，顺序为四条腿的 `abduction, hip, knee`，叠加到 `stand` 默认姿态后由 XML 中的 PD 执行器跟踪。
- 观测：基础 recipe 采用 48 维本体感知结构：机体坐标线速度 3、机体坐标角速度 3、投影重力 3、速度命令 3、相对默认关节角 12、关节速度 12、上一动作 12。`forward_phase_bootstrap_v1` 额外追加2维 `sin/cos phase`，总计50维。
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

PPO 默认使用 `clipping_epsilon=0.20`、全局梯度范数上限 `1.0`、目标 KL `0.01` 和 `ADAPTIVE_KL` 学习率调度。每个 recipe 都显式限制自适应学习率的上下界，避免 Brax 在低 KL 时按 minibatch 连续放大学习率。eval 日志同时显示 `kl`、策略标准差 `std` 和实际学习率 `lr`，用于区分动作均值漂移与探索方差失控。

## 训练

第一阶段只学习稳定直行，不加入无关难度：训练与 eval 都从无扰动 `stand` 开始，命令固定为 `vx=0.10 m/s, vy=0, yaw_rate=0`，并关闭 observation noise 与 domain randomization：

```bash
python -m scripts.train_mjx_3d_walking_ppo \
  --preset h200 \
  --recipe forward_stage1_v1 \
  --steps 15000000 \
  --num-evals 32 \
  --save-ppo-checkpoints \
  --ppo-checkpoint-dir results/mjx_pupper_forward_stage1_v9/ppo_checkpoint \
  --out results/mjx_pupper_forward_stage1_v9
```

当前实现对 48 维 observation 使用固定物理缩放，不再同时启用 PPO 的运行时 observation normalization。自由关节线速度由世界坐标旋转到机体坐标，MuJoCo 已经以机体局部坐标给出的角速度则不再重复旋转。动作从第一步起直接作为 `stand + scale * action` 的关节目标，不再使用启动 ramp。

Actor 均值输出层使用范围为 `1e-3` 的小随机初始化、bias 为框架默认的零；这不是固定零输出，参数会正常接收梯度。H200 preset 使用 `batch_size=256`、`num_minibatches=8`。stage-1 的 `unroll_length=40`，rollout quantum 为 81,920；1500 万请求步、32 次 eval 时约有 186 次 PPO 更新。

stage-1 取消 progress reward，速度跟踪权重为 `4.0`、核宽为 `0.05 m/s`，并乘以 `upright_sigma=0.20 rad` 的姿态门控。奖励所用平面速度只取世界水平运动并旋转到 torso heading frame，roll/pitch 和世界竖直速度不会泄漏进前向速度。直行阶段的 yaw-rate 权重为 `0.25`，独立 upright 权重为 `0.2`。

stage-1 的归一化动作尺度为 `abduction/hip/knee = 0.06/0.50/0.65`，初始探索标准差为 `0.10`，entropy cost 为 `0.01`。摆动足目标高度为 `0.025 m`；脚离地、达到净空且在 torso 局部坐标中产生水平摆动时获得即时正奖励。局部足端位置差分同时消除机身刚性平移和旋转，因此整机前扑或静止举腿不会获得该项奖励。完整腾空时间仍在落地瞬间通过 air-time 项奖励。

速度 tracking 之外另有不乘 upright gate 的超速惩罚：沿命令方向超过 `command_speed + 0.05 m/s` 后开始扣分，在额外超速 `0.15 m/s` 时达到每步 `-1.0` 上限。正常 `0.10 m/s` 跟踪和小幅步态速度波动不受影响，`0.30~0.40 m/s` 的短暂前扑则会持续受到负反馈。

stage-1 还维护一个50个控制步（1秒）的根节点水平位置环形缓冲。沿当前平面命令方向的1秒位移低于 `0.05 m` 时，`upright` 正奖励按进度缺口线性衰减，并额外施加最多 `-0.2/step` 的 stagnation 惩罚；第一秒不处罚。达到或超过 `0.05 m/s` 的持续前进后，两项限制都自动解除。日志中的 `progress_1s` 和 `stagnation` 用于核对这一行为。

stage-1 保留 `ADAPTIVE_KL`，但把学习率硬限制在 `[2e-6, 2e-5]`，初值为 `2e-5`。这允许高 KL 时降低学习率，同时禁止低 KL 连续触发后把学习率放大到危险范围。

时间上限通过环境的 `info["time_out"]` 明确交给 Brax PPO，并允许 value function bootstrap；真正跌倒仍是普通 termination，不做 bootstrap。最优 checkpoint 按“完整存活、存活时长、upright 失败率、命令跟踪、进度和接触质量”的字典序选择，后面的接触改善不能覆盖前面的存活退化。

启动日志应显示 `reset_noise joint=0 ... root_xy=0 ... root_yaw=0`、`observation=False` 和 `domain_randomization=False`。只有当策略能够持续 10 秒、速度接近 `0.10 m/s` 且失败率明显下降后，才进入速度范围、转向和随机化阶段。

## 非学习的机构可行性验证

`demo_handcrafted_3d_walk` 对同一个12-DoF XML使用解析IK和对角小跑足端轨迹，不读取策略或reward。默认控制器在本地 nominal `cg12` 物理中连续6秒前进约0.534 m、平均0.089 m/s，最大tilt约0.093 rad，可用于区分机构问题和RL问题：

```bash
python -m scripts.demo_handcrafted_3d_walk \
  --duration 6 \
  --output results/handcrafted_3d_walk_default

python -m scripts.render_mjx_3d_policy \
  results/handcrafted_3d_walk_default/evaluation_rollout.npz \
  --model-xml assets/curl_robot_3d_pupper_r127p5_open60_width120.xml \
  --physics-profile cg12 \
  --output results/handcrafted_3d_walk_default/handcrafted_walk.gif
```

模型的前腿hip轴为 `-Y`、后腿为 `+Y`，这是前后镜像机构所需的符号。正hip增量使前足向 `+X`、后足向 `-X`，也就是都向各自外侧运动；世界坐标中同向摆腿时，前后hip增量本来就应异号。单独弯曲前hip会让前足前移并略微抬高、卸载前支撑，若knee不协同就可能绕后足向前俯倒，这不等于关节轴配置错误。

```powershell
cd curl_robot_2d
python -m scripts.mjx_3d_walking_smoke --mujoco-gl disable
python -m scripts.train_mjx_3d_walking_ppo --preset 4090 --recipe anymal_v1 --out results/mjx_pupper_anymal_v1
```

默认启用批量 MJX 模型随机化：地面摩擦、各刚体质量/惯量、执行器增益、关节阻尼和 armature。可用 `--no-domain-randomization` 做 nominal baseline；各范围均有独立命令行参数。建议先通过 smoke，再做 nominal 短训，最后启用随机化长训，避免把模型或奖励错误误判为鲁棒性不足。

## 推荐路线 B：Unitree/MjLab 风格非对称训练

路线 B 现在分成同构的 discovery 和 robust 两段。两段都使用47维 actor observation、74维 privileged critic observation、0.60秒对角步态 phase 和 observation normalization；都不使用手写关节轨迹或 phase-conditioned joint pose。`unitree_mjlab_velocity_discovery_v1` 先在 nominal 模型上发现0.10 m/s前进运动，`unitree_mjlab_velocity_v1` 再恢复命令范围、噪声、域随机化和足高正则。

本机器人前腿膝盖朝外，与 Unitree 前腿膝盖向后的机构不同。这里的 phase 只给步态时钟和期望接触组 `FL+RR / FR+RL`；actor 始终直接学习当前 MJCF 关节轴下的12维位置增量，不复制 Unitree 的 hip/knee 角度。

```bash
python -m scripts.train_mjx_3d_walking_ppo \
  --preset h200 \
  --recipe unitree_mjlab_velocity_discovery_v1 \
  --steps 3000000 \
  --num-evals 12 \
  --save-ppo-checkpoints \
  --ppo-checkpoint-dir results/mjx_pupper_unitree_discovery_v1/ppo_checkpoint \
  --out results/mjx_pupper_unitree_discovery_v1
```

discovery 固定 `vx=0.10 m/s`，关闭 observation noise、domain randomization、reset noise 和 `foot_clearance`，但保留对角接触节律；phase 仍然只给时钟和接触计划，不生成关节角。前2–3M步的验收线是10秒 eval 中 `velocity>0.05 m/s` 且 `distance>0.50 m`。未达到就停止排查，不继续盲跑20M；达到后可用相同47/74维网络的完整 PPO checkpoint 续训 robust recipe。

```bash
python -m scripts.train_mjx_3d_walking_ppo \
  --preset h200 \
  --recipe unitree_mjlab_velocity_v1 \
  --steps 20000000 \
  --num-evals 32 \
  --restore-checkpoint results/mjx_pupper_unitree_discovery_v1/ppo_checkpoint \
  --save-ppo-checkpoints \
  --ppo-checkpoint-dir results/mjx_pupper_unitree_robust_v1/ppo_checkpoint \
  --out results/mjx_pupper_unitree_robust_v1
```

该网络与旧48/50维 checkpoint 不兼容，只有新的 discovery/robust 两个 recipe 可以直接续训。`T=0.60 s`、`duty=0.56` 对应计划摆动窗0.264秒；新日志的 `air_time` 是 touchdown 事件的真实腾空秒数，而旧日志中的同名值实际是稀疏奖励均值。

首轮路线 B 暴露出两个适配问题：壳体的浅层接触会立即触发非法接触，且 `std=1.0/LR=1e-3` 在 Brax PPO 中把确定性动作均值推到接近饱和。当前版本已经让全部 `rolling_shell` 退出碰撞，保留内部结构代理；非法接触改为三维接触力模长超过1 N才累计，与 Unitree 判据一致。路线 B 现在使用 `std=0.5`、`LR∈[3e-5, 3e-4]` 和 `abduction/hip/knee=0.08/0.25/0.25 rad`。足高、软着陆和角动量也按本机尺度无量纲化。

robust 路线训练 `vx∈[0.10, 0.30] m/s`，`vy=yaw_rate=0`，eval 使用0.20 m/s。两个阶段的 PPO reward multiplier 都为控制周期0.02；这与 Unitree MjLab 默认把 reward rate 乘环境 step dt 的语义一致，避免 `termination=200` 直接制造巨大的 critic target。训练和周期 eval 都只因物理失败提前终止，并固定评估最多10秒。0.5秒低进展窗口只作为诊断，不再截断 eval；最低进展仍显示为 `min(0.05 m, 50% * vx_command * 0.5 s)`。checkpoint rank 要求实际位移，`meaningful_progress=0` 的站立策略不会显示 `new_best` 或更新 `params_best`。命令换挡发生在 transition 末尾，新命令只进入下一 observation。接触日志分别输出逐步均值 `nonfoot_force_avg` 和回合峰值 `nonfoot_force_peak`。

方向诊断不再只输出绝对yaw error和四脚平均值。周期eval同时记录有符号yaw rate、最终/峰值heading、最终lateral，以及FL/FR/RL/RR各自的接触率、touchdown air time、接触期slip和动作RMS。训练结束还会对 `params_best` 运行“训练最小速度/目标速度/训练最大速度 × phase 0/0.5”确定性网格；可用 `scripts.evaluate_mjx_3d_walking_policy` 对已有checkpoint独立重跑。网格结果是进入噪声和domain randomization前的验收依据，单个目标速度eval不再代表整个命令区间。
