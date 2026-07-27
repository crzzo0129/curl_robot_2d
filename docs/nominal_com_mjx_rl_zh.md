# 名义 COM 的 MJX 纯 RL 第一阶段

更新日期：2026-07-27

## 1. 目标

这一阶段只回答一个问题：

> 固定当前名义质量、COM、惯量、摩擦、伺服参数、力矩限制和碰撞模型，
> 不使用 CEM 动作示范，PPO 能否从随机策略学会持续正向滚动？

训练继续读取唯一的 `assets/curl_robot_2d.xml`，没有复制第二份机器人模型。
当前 Torso 局部 COM 仍为 `(0.025, 0, 0.015)` m；质量和惯量也完全来自当前
XML。第一阶段不做 COM 条件化，也不做 COM 或动力学随机化。

## 2. 训练接口

### 动作

策略以 50 Hz 输出四个归一化动作，映射为前髋、前膝、后髋和后膝的绝对
目标角。底层仍使用当前 MuJoCo 位置伺服、增益和 \(\pm6\ \mathrm{N\,m}\)
二维等效力矩限制。

策略没有相位振荡器、CEM 参数、参考轨迹或动作示范，因此属于从零开始的
纯 RL。动作均值初始化在 0 附近时，对应当前 compact 姿态，有利于初期避免
立即产生极端关节目标。

### 观测

当前 23 维观测包括：

- Torso 滚动相位的正弦和余弦；
- 根高度和 3 个根速度；
- 4 个关节角和 4 个关节速度；
- 上一步 4 维动作；
- 地面接触、允许足端接触、非允许自接触及两类穿透深度。

绝对水平位置不进入观测，避免策略记忆固定位置。

### 奖励和约束

主要正奖励是每个控制周期中的保守滚动进度：

$$
\Delta\phi_{\mathrm{roll}}
=
\min\left(
\Delta\phi_{\mathrm{torso}},
\frac{\Delta x}{R}
\right).
$$

同时惩罚：

- 转角和水平位移不一致；
- 回滚；
- 动作突变和归一化力矩；
- 腾空；
- 足端张开过大；
- 非允许接触的持续时间和穿透；
- 允许足端接触超过 0.5 mm 的穿透。

腿部中心线交叉会立即终止 episode 并施加强惩罚。自接触不会被误认为地面
支撑。任何非超时失败还会收到独立的 `termination` 惩罚，当前权重为 5。

奖励配置和实现不再混在环境文件中：

- `curl_robot_2d_mjx/reward_config.py`：经常调整的所有奖励权重和阈值；
- `curl_robot_2d_mjx/reward.py`：各奖励项的计算公式；
- `curl_robot_2d_mjx/environment.py`：只提供物理量、接触量和终止原因。

当前奖励固定拆成 `roll_progress`、`roll_mismatch`、`backward`、
`action_rate`、`torque`、`airborne`、`foot_gap`、`collision` 和
`termination` 九项。训练时每一项都独立记录，不再只看混合后的总 reward。

失败原因也分别记录为 `failure_nonfinite`、`failure_root_low`、
`failure_root_high`、`failure_foot_gap` 和 `failure_leg_crossing`。

## 3. MJX 与 CPU MuJoCo 的关系

MJX 使用和 CEM 相同的 XML、接触类别、关节限位和执行器。物理运行档位通过
加载 XML 后覆盖 MuJoCo option 实现，不复制第二份模型：

- `reference`：Newton `20/10`、1 ms、20 子步，作为最终物理真值；
- `newton4`：Newton `4/4`、1 ms、20 子步，是早期速度尝试；
- `cg12`：CG `12/6`、1 ms、20 子步，是当前训练候选。

冻结 CEM 控制器的 10 s CPU 对照中，`reference` 滚动 9.914 圈，最大非允许
穿透 0.459 mm；`newton4` 虽滚动 9.567 圈，但穿透增至 7.068 mm，不能作为
训练默认；`cg12` 滚动 9.906 圈，最大非允许穿透 0.410 mm，非允许接触比例
为 1.29%，与 reference 的 1.36% 接近。CG 的允许足端穿透仍偏大，因此最终
结论仍必须回到 reference。

可随时用冻结 CEM 控制器重做档位对照；脚本会自动选择新文件名，不覆盖历史
结果：

```bash
python -m scripts.compare_physics_profiles
```

因此 MJX 训练奖励只能用于选策略，最终结论必须把策略放回未修改的 CPU
MuJoCo `20/10` 模型，重新测量滚动圈数、接触、穿透、能耗和力矩。

## 4. 云端环境

基准解释器固定为 Python 3.12。推荐 Linux x86-64、RTX 4090 或 H200、
NVIDIA 驱动和 CUDA 12.8 环境。
项目使用 JAX 的 CUDA 12 pip 安装路线；pip 会安装匹配的 CUDA/cuDNN
运行库，不要求训练脚本直接依赖平台的 toolkit 文件。

在 `curl_robot_2d` 根目录运行：

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mjx.txt

python -c "import sys; print(sys.version)"
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Python 输出必须为 3.12.x，并且 JAX 必须看到 `gpu` 和 NVIDIA 设备；
若显示 `cpu`，不要开始正式训练。

## 5. 必须先运行的 MJX 冒烟测试

当前 curl 模型有 38 个 geom、外壳与腿部自碰撞，并且每个 20 ms 控制周期
包含 20 个物理子步；其 XLA 接触计算图明显重于 `disk_robot` 当前训练模型
（16 个 geom、每控制周期 5 个物理子步）。因此首次测试必须从单环境开始，
不能直接根据 GPU 显存把 batch 调到很大。

```bash
python -m scripts.mjx_smoke \
  --physics-profile cg12 \
  --batch-size 1 \
  --steps 1 \
  --mujoco-gl disable
```

支持 EGL 的节点可以使用 `--mujoco-gl egl`。脚本会依次打印
`environment_create`、`reset_compile`、`step_compile` 和 `cached_rollout`
阶段；首次出现 XLA `Very slow compile` 警告不等于失败，应等待当前阶段完成。
`step_signature_check` 用来确认 reset 与第一步产生的 JAX 状态类型一致；它和
随后 cached rollout 应显著快于首次 `step_compile`。

通过条件：

- `status` 为 `ok`；
- observation 形状为 `(1, 23)`；
- reward 和 observation 全部有限；
- `step_signature_check_s` 与 `cached_rollout_s` 不再接近首次编译时间；
- 没有 `mjx.put_model` 不支持当前模型特征的异常。

单环境通过后，再依次运行 `--batch-size 16 --steps 10` 和
`--batch-size 64 --steps 10`，记录两个 JSON 中的编译时间与
`cached_steps_per_second`。不同 batch shape 会触发新的编译。
若模型特征不支持，应修复模型/MJX 兼容问题，而不是直接开始 PPO。

## 6. 第一轮短训练

```bash
python -m scripts.train_mjx_ppo \
  --preset smoke \
  --physics-profile cg12 \
  --seed 0 \
  --mujoco-gl disable \
  --out results/mjx_ppo_nominal_smoke_cg12_terminal_v2_seed0
```

短训练使用 64 个并行环境和约 6.55 万个请求环境步。目的不是得到最终策略，
而是确认：

- PPO 能完成编译和更新；
- eval reward 不出现 NaN；
- 评价 rollout 的净滚动不是持续负值；
- 碰撞和交叉惩罚有正常数值。
- episode 长度是否开始接近完整 500 步；
- 终止惩罚能否消除“获得短期滚动奖励后主动失败”的策略。

## 7. 4090 与 H200 正式配置

4090：

```bash
python -m scripts.train_mjx_ppo \
  --preset 4090 \
  --physics-profile cg12 \
  --seed 0 \
  --mujoco-gl disable \
  --out results/mjx_ppo_nominal_4090_seed0
```

H200：

```bash
python -m scripts.train_mjx_ppo \
  --preset h200 \
  --physics-profile cg12 \
  --seed 0 \
  --mujoco-gl egl \
  --out results/mjx_ppo_nominal_h200_seed0
```

4090 preset 暂从 512 个并行环境和 2000 万步开始；H200 preset 暂从
2048 个并行环境和 5000 万步开始。这两个并行数是保守起点，不是固定结论。
必须根据前述 16/64 环境 smoke 的 `cached_steps_per_second` 和显存占用再决定
是否增大。由于当前模型接触 geom 较多，首次运行若显存不足，优先降低
`--envs` 和 `--eval-envs`，不要先改变物理或奖励。

训练脚本继承了 `disk_robot` 已验证的运行配置方式：headless GL 选择、XLA
latency-hiding/Triton 设置、编译缓存、运行时诊断、自动 PPO checkpoint 以及
`--restore-checkpoint` 续训入口。当前阶段没有复制 `disk_robot` 的 IK teacher、
teacher blend、命令条件化和 Student 蒸馏，因为这里要独立回答纯 RL 能否从零
学会滚动。

如训练被中断，可使用实际 checkpoint 目录续训：

```bash
python -m scripts.train_mjx_ppo \
  --preset 4090 \
  --restore-checkpoint results/mjx_ppo_nominal_4090_seed0/ppo_checkpoint \
  --out results/mjx_ppo_nominal_4090_seed0_resume
```

## 8. 自动参数扫描与晋级

一次 smoke 不能证明某个设置值得直接训练 2000 万或 5000 万步。可使用
`scripts.sweep_mjx_ppo` 在同一张 GPU 上顺序执行三阶段实验：

1. `screen`：比较当前基线以及终止惩罚、折扣率、学习率和 entropy cost；
2. `confirm`：前两名更换随机种子并使用更大预算复赛；
3. `final`：根据两轮结果自动选择一组参数，换第三个种子从零正式长训。

扫描保持 XML、物理 profile、初始状态、episode 长度和失败阈值不变。排序不使用
不同 reward 权重下不可直接比较的总 reward，而使用最近三次评估的 episode
寿命、估算净滚动圈数、失败率和非允许碰撞深度。任何 non-finite 结果直接淘汰。

4090 默认预算为每个候选约 52 万步、前两名各约 419 万步、胜者 2000 万步；
H200 默认预算为每个候选约 210 万步、前两名各约 1678 万步、胜者 5000
万步。所有候选顺序运行，避免同一 GPU 上多个训练进程争抢显存。

```bash
python -m scripts.sweep_mjx_ppo \
  --hardware h200 \
  --physics-profile cg12 \
  --seed 0 \
  --out results/mjx_ppo_sweep_terminal_v2_h200
```

中断后使用相同命令加 `--resume`。脚本会复用已完成的候选；不完整目录会保留，
重跑写入新的 `_retryN` 目录。若只想完成两轮选择、不启动最终长训，可添加
`--skip-final`。使用 `--screen-steps`、`--confirm-steps` 和 `--final-steps`
可以显式覆盖默认预算。

主要汇总文件：

- `leaderboard_screen.{json,csv}`：第一轮完整排名；
- `leaderboard_confirm.{json,csv}`：复赛排名及两轮加权分数；
- `selected_candidate.json`：最终参数及选择依据；
- `final_result.json`：正式长训的独立物理指标；
- `logs/`：每个候选的完整控制台日志。

## 9. 输出

每个训练目录包含：

- `training_config.json`：训练、任务、奖励和设备配置的完整快照；
- `reward_config.json`：本次训练独立使用的奖励权重；
- `reward_history.json`：总奖励、九个奖励分项及其每步平均值；
- `metrics_history.json`：不含奖励项的物理、失败和 PPO 指标；
- `training_summary.json`：耗时、最优步和最终指标；
- `params_best`、`params_final`：Brax 策略参数；
- `evaluation_rollout.npz`：确定性策略的 qpos、动作和奖励；
- `evaluation_summary.json`：净滚动、独立奖励分解、普通指标平均值和具体失败
  原因。

云端训练后应完整下载该目录。确认策略确实学习后，再实现 CPU MuJoCo
策略回放和与 CEM 的严格对照，不根据训练 reward 单独下结论。

训练入口默认拒绝写入非空输出目录，避免覆盖历史实验。只有明确需要时才使用
`--allow-existing-output`。
