# rollingquad_2：ROLL → Transition → READY TO WALK

## 当前实现范围

保留两个已有的 ROLL/WALK policy，新增一个 12 自由度 Transition policy，
统一学习减速、展开和站稳。高层只选择策略，不生成减速或展开轨迹。

模型固定为 `assets/rollingquad_description_2/mjcf/rollingquad.xml`。
不再读取或使用 `park`；它即使仍在原始 XML 中，也不参与任务。
课程锚点取自当前 `scripts/train_ppo_deploy.py` 的 Walking 启动状态：
关节按策略顺序为 `[0, 0.9, 1.15] × 4`，机身高度为
`0.1580029248 + 0.0005 m`，单位四元数，零速度。
新模型的 qpos 排列与 actuator 排列不同，代码全部按关节名称取索引。

本版已完成训练环境、课程、真实 ROLL 快照接口、观测/动作契约和 Actor 导出代码。
本地 CPU 检查通过不等于已训练出成功策略；MJX/JIT、PPO 收敛、真实 checkpoint
导出验证及三策略连续闭环仍需云端验证。没有修改实机控制器或安全配置。

### v4 训练修复：从 Walking 初态重新开始

首次 smoke 评估全部超时、READY 为零，不能进入下一课程。针对这一结果：

- PPO 和 smoke 都使用 `wrap_transition_3d`，结束回合后同时重置物理状态、
  步数、阶段、READY/失败计数、前一动作、足位置及完整 observation 历史。
  使用新随机种子重新采样课程初态；未结束的并行环境不受影响。
  终止步的 reward/done/metrics/time_out 仍保留，供 PPO 和评估统计读取。
- 新策略的最后一层初始 location 为零，pre-tanh 标准差默认 0.05；
  `--initial-policy-std` 可调整这个初始探索量。保留随机隐层和原来的
  24-logit 单输出层结构；不改变 720 维观察、关节 scale 或部署导出格式。
  恢复 checkpoint 时不覆盖已训练参数。熵系数默认改为 0.0003。
- STABILIZE 的直立/高度奖励必须有足部支撑且接近 Walking 姿态；增加
  有界姿态误差惩罚，稳定奖励同时考虑速度，避免机身朝上但趴地仍持续获利。
  DEPLOY 落足前的引导奖励保留。
- 只在 STABILIZE 阶段设置支撑/姿态失败判据：进入后先宽限 0.30 s，之后
  高度 <0.12 m、关节 RMS >0.65 rad、倾角 >0.65 rad、少于两足接触或
  出现非足部接地，任一异常持续 0.25 s 才失败。正常一帧会清零计数。
  BRAKE/DEPLOY 不使用这条规则；未修改实机急停或倾倒保护。
- 成功、失败和超时统计互斥。`curriculum_next_stage` 只表示课程顺序；
  `stage_passed` 需连续两次训练后的评估成功率 ≥90%、失败率 ≤5%、
  超时率 ≤5%，且统计完整有效。未通过时 `next_stage=null`，不建议升级。
  这是训练进度门限，不是实机安全认证。
- 物理诊断增加 `_per_step` 指标（如 `eval/episode_root_z_m_per_step`），
  Brax 会先按每个回合自己的步数归一化再求平均，避免把累计高度/速度误当作
  瞬时值。旧累计指标保留；成功/失败/超时仍是回合事件，不做逐步归一化。

原失败权重和日志应保留作对照。v4 首轮使用新输出目录、不恢复旧失败权重。
训练入口拒绝覆盖非空阶段目录。新增测试覆盖混合成功/失败/超时、连续回合
计数、历史重置、新随机种子、终止统计、零动作多回合 READY 和非零网络导出。

## Actor 与实机保持相同的 36 × 20 观察

接口依据工作区内 `pupperv3-monorepo/ros2_ws/src/neural_controller` 的实际
`neural_controller.cpp/.hpp` 和 `launch/config_rollingquad_2.yaml`，不是仅对齐总维度。
每帧均为原始值，顺序如下（零起点、左闭右开）：

| 索引 | 维度 | 内容 |
| --- | --- | --- |
| 0:3 | 3 | IMU 机身坐标系角速度，rad/s |
| 3:6 | 3 | 投影到机身坐标系的单位重力方向 |
| 6:9 | 3 | vx、vy、yaw 速度指令；Transition 固定为零 |
| 9:12 | 3 | 期望 world-z 在机身系的方向；本任务固定 `[0,0,1]` |
| 12:24 | 12 | 关节角减 Walking default_joint_pos，rad |
| 24:36 | 12 | 上一推理周期的原始 policy 输出 |

关节顺序为 FL/FR/RL/RR，每腿 abduction/hip/knee，对应硬件 `_2/_1/_3`。
没有关节 qvel、机身线速度、足接触、高度、阶段标志、计时器或仿真滚动相位。
角速度来自 IMU，不是新增关节速度传感器。20 帧历史能提供运动趋势信息，
但不能保证恢复所有真实速度，尤其不能把“角速度小”当作“平移已停止”。

Actor 采用 720 输入的历史 MLP。Critic 单独使用 86 维仿真特权状态，
包括速度、接触和课程阶段；两者经各自归一化，不共享输入拼接。
因此推荐先直接训练非对称 Actor/Critic PPO，不必先改实机 observation 或增加
速度估计器。若受限观测确实造成学习困难，再考虑特权 teacher → 720 维 student
蒸馏；不是当前第一版的额外依赖。

### 历史、时间和切换语义

- 最新帧在前。每次策略推理更新一帧，不在每个 ROS 控制循环都推进历史。
- 冷初始化的每帧全零，但 gravity-z 为 -1、desired-z 为 +1；插入当前帧后推理。
- 上次动作是原始输出，不是关节目标、裁剪后的目标或 fade-in 后的动作。
- 推理前 observation 裁剪到 ±100；只给新采集的传感器帧加噪，旧帧不重复加噪。
- 默认按配置名义频率 `520 / 10 = 52 Hz` 训练，物理子步为 `1/(52×20) s`。
  配置注释期望实际约 50 Hz；必须用实机时间戳测量后统一最终频率，不能把
  IMU 的 260 Hz 当作 policy 频率。20 帧在名义频率下覆盖约 0.38 s。
- Transition 接管期间速度指令必须归零、姿态指令固定；现有 C++ 不会因为换
  模型自动完成这件事，后续策略调度器需明确处理，不能沿用非零摇杆指令。

`reset_from_roll_state(data, rng, actor_history=None, last_action=None)` 保留完整
MJX data（含 qpos/qvel/ctrl），不执行归位、暖机或外部制动。
默认只冷初始化观测缓冲区，与快照课程一致，不重置物理状态或电机目标。
可选 actor_history 是**上次推理前的完整输入**，不是 C++ 推理后已旋转的内部
scratch buffer；接口会再插入当前帧。last_action 是相同动作约定下的原始输出。

同为 720 维不代表可无条件继承历史：当前 ROLL student 使用 compact 默认姿态，
Transition 使用 Walking 默认姿态，动作 scale 也不同。若希望热继承历史，
必须按目标策略重建关节偏差及动作通道，并在训练中覆盖这种初始化；尚未默认
启用该路线。最简第一版是在已激活控制器内切换模型、只重建观测缓冲区。

## 动作与导出

所有阶段使用同一个固定 Walking 中心，不随阶段/时间改变：

```text
scale = action_range_fraction * max(joint_high - default, default - joint_low)
target = clip(default + raw_action * scale, joint_low, joint_high)
```

这是 C++ 在 fade-in 完成后的固定逐关节 scale 映射。默认 fraction=1，使得
单个策略的动作范围能覆盖 compact、Walking 初态和关节限位；不是“已经减速后
只做小残差展开”。训练输出采用 tanh-normal，确定性推理为 tanh(location)。
已从旧的按正负方向分别缩放改成固定 scale；旧 66 维或旧映射的权重不能直接复用。

训练输出 `deployment_config.json`：36×20、默认姿态、逐关节 scale、限位、
均匀标量 kp/kd、硬件关节顺序、名义频率等。当前 C++ 只接受标量 kp/kd，
不接受每关节不同的增益数组；导出配置对此做显式检查。
部分字段（频率、joint_names、observation_limit、Transition 指令等）是契约记录，
**当前 C++ 不会全部从 JSON 自动应用**，部署前还必须核对 ROS 参数和调度逻辑。

Actor 隐层用 ELU，沿用已有 Walking RTNeural 导出格式。导出器只提取
`state` 的归一化统计，并折入首层；保留高斯 location 的 12 个通道，末层 tanh。
Critic、特权归一化和训练分布的 scale 输出均不进入实机文件。

## 从 Walking 初态向后扩展的课程

| 阶段 | 初始分布 | 学习任务 |
| --- | --- | --- |
| walking_start | 精确 Walking 初态 | 稳定保持并满足 READY |
| deploy_near_stand | 小幅关节/姿态/速度扰动，少量 compact 混合 | 回到 Walking 可接管状态 |
| deploy_capture | 扩大紧凑程度、倾角和初速度 | 展开、捕获支撑、稳定 |
| brake_early | 从 reset 起累计净翻滚满 1 圈、未满 2 圈的真实快照 | 从已滚起来的较早周期接管，自己制动展开 |
| brake_later | 累计净翻滚满 2 圈、未满 4 圈的真实快照 | 从后续周期接管，适应继续加速的 ROLL |
| brake_full | 累计净翻滚满 1 圈后的全部采集快照 | 覆盖采集库内早期与后期接管 |

这是手工分阶段扩展的反向初始状态课程，不是时间倒放或自动生成反向动力学。
compact 仅用于合成 reset 邻域插值，没有 compact→park→stand 动作播放器。
BRAKE/DEPLOY/STABILIZE 是奖励、诊断和 Critic 的内部阶段标签，Actor 不依赖标签。
减速奖励、直立/姿态捕获奖励、稳定/READY 奖励共同作用于同一网络。

冻结当前 ROLL policy 后可调用 `collect_roll_snapshots_3d(env, policy, path,
source_policy=...)`；policy 仍用自己的原生观察和动作接口。采集器只跑 ROLL，
不会主动刹车；保存完整 qpos、qvel、12 维 ctrl、时间、episode ID、来源标识、
模型 XML 摘要及关节/执行器顺序。也可由已有轨迹记录器调用 `save_roll_snapshots_3d`。
单有视频/姿态不足以生成合格的接管状态。

### v5：按真实滚动周期接管，取消低速起步筛选

`brake_low` 已删除，不再使用 0.35 m/s、3.5 rad/s 的初态筛选门槛；
内部 BRAKE→DEPLOY 的速度门控仍保留，那是判断策略是否已经减速，不是挑选初态。
上述圈数窗口是第一版默认课程，不保证更晚一定更快；应查看每圈实测速度统计。
三个 BRAKE 阶段都保留原始 qpos/qvel/ctrl，制动仍由同一个 Transition policy 完成。
720 维 actor、86 维 critic、动作映射与奖励不变，现有 v4 deploy_capture 权重可继续恢复。

圈数取 ROLL 环境逐物理步积分的有符号 `info['rolling_phase']`，减去本回合
reset 时的原点后除以 2π。它不是控制器的 oscillator phase，也不是角速度绝对值
累计；来回摆动相互抵消，不会因为摆了很多次就算完成翻滚。默认方向为机体 +Y，
反向 ROLL 使用 `--snapshot-roll-direction -1`。这是轴向旋转进度代理，不能单独证明
无滑移、地面接触安全或无腾空；仍需检查采集轨迹与连续闭环。

采集器现在默认 `warmup_steps=0, sample_every=1`，从 reset 开始记录原点。
即使显式跳过 warmup 或稀疏保存，也不重新计圈。新 NPZ schema v2 额外保存
`roll_phase_rad`、`roll_origin_phase_rad` 和进度来源。旧 v1 库不能直接用于新的
BRAKE 课程：需要重新采集，或从有完整旋转记录的原始日志重新导出，不能只凭
旧稀疏姿态、末帧四元数或经过时间补造圈数。

#### 一键生成快照库（已接入 residual ROLL checkpoint 加载）

在云端 `curl_robot_2d` 目录执行下列命令，使用已确认的 ROLL 权重和配套配置：

```bash
python -m scripts.collect_transition_roll_snapshots \
  --roll-checkpoint /inspire/qb-ilm2/project/leverage-robot/ky26210/curl_robot_2d/results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --roll-config /inspire/qb-ilm2/project/leverage-robot/ky26210/curl_robot_2d/results/rollingquad2_floor_mass_gain_v3_h200_seed0/training_config.json \
  --out results/roll_cycle_snapshots_v2.npz \
  --episodes 8 --sample-every 1
```

这条命令才会实际生成 `results/roll_cycle_snapshots_v2.npz`；文件不是仓库自带，
也不是 Transition 前三阶段的输出。可先附加 `--dry-run` 核对路径、配置和来源哈希，
该选项不会生成文件或导入 JAX。正常运行需云端 JAX/MJX，首次编译需等待。

入口读取保存的 task/reference/reward 与网络结构，恢复 ROLL 归一化统计和已训练参数，
使用确定性原生 ROLL 推理；不会用零动作、CPU 参考控制器或 Transition 替代它。
原 ROLL 若含残差参考控制器，其原有逻辑也完整保留。不会额外施加训练时的随机化
包装器，采集的是配置中名义物理模型下的轨迹，物理参数记录在报告中。

输出旁边还会生成 `roll_cycle_snapshots_v2.summary.json`，含权重/配置 SHA256、
实际采集配置、每阶段相位覆盖和速度范围。默认要求 brake_early 覆盖完整，否则
保留文件但以非零退出码结束；不能把“文件存在”当作采集通过。brake_later 是否可用
也会单独报告。若方向相反，采集检查使用 `--roll-direction -1`，后续训练相应使用
`--snapshot-roll-direction -1`。不足时检查报告；重采请换新输出名，绝不覆盖旧库。

每回合默认使用保存配置中的 episode_length；`--steps-per-episode` 可缩短，
不允许静默延长 ROLL 自身超时或忽略失败。是否采够后续圈数取决于实际轨迹，
不是只把采集步数设大就一定能解决。该入口针对这次 residual ROLL，不接受直接动作
student/Transition 权重冒充 ROLL；配置不匹配会报错。

如需接到其他已经正确加载的 ROLL 环境中，底层 API 仍可用：

```python
from curl_robot_2d_mjx.transition_initialization_3d import collect_roll_snapshots_3d

collect_roll_snapshots_3d(
    env, roll_policy, "results/roll_cycle_snapshots_v2.npz",
    source_policy="实际使用的ROLL权重路径或标识",
    episodes=8, steps_per_episode=500, warmup_steps=0, sample_every=1,
)
```

这里的 env、roll_policy 必须是你实际部署候选 ROLL 的加载结果；本修改不猜测
它的网络结构，不替换其启动流程，也不运行外部制动。500 步不是保证采够圈数，
还受 ROLL 自身 episode_length/终止条件影响，需用下方检查命令确认。

每圈分 8 个进度相位格，按“圈数×相位格”等概率采样，格内均匀选快照，
避免低速停留时间较长的部分占满训练批次。brake_early/later 要求指定窗口中的
每个完整圈都覆盖全部相位格；缺失就拒绝训练。brake_full 保留末尾不完整圈，
报告 `incomplete_cycles`，不暗中丢掉末段高速状态。它只覆盖实际采到的分布，
不是任意速度制动能力认证；采样覆盖也不等于每个相位都评估成功。

可以用整数 `--snapshot-min-turns` / `--snapshot-max-turns` 调整窗口，上界不含。
例如 brake_later 默认 [2,4)，也可先用 [2,3) 细化课程；这种覆盖应按实际窗口解释。
`--snapshot-tail-fraction` 仍可额外按每条轨迹的时间选择后段，默认 1.0，
不建议用于早期周期课程；若它删除了目标圈或相位，程序会报错，不回退到起步样本。
离线 NPZ 重建不保留求解器 warm-start；连续同模型接管应使用 full-data 接口。

## READY 和实机安全边界

仿真 READY 需连续满足约 0.40 s：线速度 ≤0.12 m/s、角速度 ≤0.45 rad/s、
倾角 ≤0.22 rad、关节 RMS 误差 ≤0.20 rad、高度 0.1280029248–0.1930029248 m、
至少三足接触、数值有限。任一条件失效清零保持时间。监督器只向前切换策略。

这套 READY 是**仿真验收条件**，不是已实现的实机门控。现有 36 维 observation
没有绝对线速度、高度和接触；不能把仿真真值传给实机，也不能静默删掉条件后
声称等价。后续需要经验证的状态估计器，或用仿真 READY 标签训练历史判别器，
结合可测姿态/关节保持条件做保守门控；必须单独验证滑移、冲击等误判情形。

实机当前激活流程会先归位 3 s、再淡入 2 s，倾角超过 1.5 rad 会急停。
不能通过停用/重新激活 neural_controller 在滚动中切策略；应在已激活控制器
内部管理三个模型。当前代码没有实现这个 ROS 热切换调度器，也没有关闭或修改
倾倒/急停保护。滚动工况的安全机制需要独立设计验证，不应直接提高阈值绕过。

## 本地检查与云端命令

以下从 `curl_robot_2d` 目录运行；本地无 JAX 时自动跳过真实 MJX 测试：

```powershell
python -m unittest tests.test_collect_transition_roll_snapshots tests.test_transition_3d tests.test_transition_deployment_3d tests.test_transition_training_3d tests.test_transition_roll_cycles_3d tests.test_transition_wrappers_3d tests.test_transition_mjx_3d -v
python -m scripts.train_mjx_3d_transition_ppo --stage walking_start --dry-run
```

云端按 `requirements-mjx.txt` 准备 Linux GPU 环境，先跑上述测试及 smoke：

```bash
python -m scripts.mjx_3d_transition_smoke --stage walking_start --physics-profile newton4 --steps 80 --require-ready
python -m scripts.train_mjx_3d_transition_ppo --stage walking_start --preset smoke \
  --out results/mjx_3d_transition_ppo_v4
```

`--require-ready` 检查零动作能够在每个并行环境站稳，而且没有失败；这不等于
学到完整转换策略。smoke 的 `success_fraction` 表示运行期间至少成功一次的
环境比例，而不是恰好最后一帧的成功脉冲。上述训练完成且 `stage_passed=true`
后才执行下一阶段：

```bash
python -m scripts.train_mjx_3d_transition_ppo --stage deploy_near_stand --preset h200 \
  --out results/mjx_3d_transition_ppo_v4 \
  --restore-checkpoint results/mjx_3d_transition_ppo_v4/walking_start/ppo_checkpoint
```

随后以同样方式训练 deploy_capture、brake_early、brake_later、brake_full。每阶段先评估通过，
再以最新恢复 checkpoint 热启动下一阶段；三个 BRAKE 阶段必须增加
`--roll-snapshots <真实采集的bank.npz>`。`params_final` 用于推理导出，
`ppo_checkpoint` 用于训练恢复，不要混用。阶段不会未经评估自动推进。

v5 恢复入口接受具体数字步数目录，也接受 `ppo_checkpoint` 父目录；后者自动
选择同时存在 `_METADATA` 和 `ppo_network_config.json` 的最大数字步数，
忽略未完成/临时目录。会打印实际恢复路径；没有合格存档则报错，不改用随机权重。

**已经训练完 deploy_capture 的用户不需要重训前面的阶段。** 按上一轮
`v4_retry` 输出位置，在云端先检查新库、跑 MJX smoke，再恢复训练：

```bash
python -m scripts.inspect_transition_roll_snapshots results/roll_cycle_snapshots_v2.npz --stage brake_early
python -m scripts.mjx_3d_transition_smoke --stage brake_early --roll-snapshots results/roll_cycle_snapshots_v2.npz --physics-profile newton4
python -m scripts.train_mjx_3d_transition_ppo --stage brake_early --preset h200 \
  --roll-snapshots results/roll_cycle_snapshots_v2.npz \
  --restore-checkpoint results/mjx_3d_transition_ppo_v4_retry/deploy_capture/ppo_checkpoint \
  --out results/mjx_3d_transition_ppo_v5
```

上面的 bank 路径必须是真实采集产物，若实际位置不同请相应替换。BRAKE smoke
只检验计算流程，不要求零动作能刹住；不要使用 `--require-ready`。
训练前会检查快照覆盖并在 `snapshot_selection.json` / `summary.json` 中保存
样本数、圈数窗口、每圈相位计数、实际速度范围。brake_early 通过后，使用同一库、
`--stage brake_later` 并恢复 v5/brake_early/ppo_checkpoint；随后同理进入 brake_full。

训练成功后导出（以下输入必须是真实产出的 checkpoint，而非占位文件）：

```bash
python -m scripts.export_transition_rtneural \
  results/mjx_3d_transition_ppo_v5/brake_full/params_final \
  results/mjx_3d_transition_ppo_v5/brake_full/transition.json \
  --config results/mjx_3d_transition_ppo_v5/brake_full/deployment_config.json
```

本地测试覆盖 Walking CPU 稳定性、关节映射、快照保真、逐帧 C++ 历史语义、
动作映射、合成权重导出数值一致性。另已在隔离的 CPU JAX 0.6.2 / Brax 0.14
环境通过三项完整重置测试，以及初始探索分布、非零随机 Actor 导出一致性测试；
这五项不运行 MJX 物理，也不证明 PPO 收敛。云端还需验证真实 Brax checkpoint、
批量 JIT、学习成功率、延迟/噪声鲁棒性和 WALK 接管后至少 1 s 的稳定性。
当前 `contact_force_peak_n` 奖励输入仍为零，未测量冲击峰值；不能据此声称
已验证实机接触安全或完整 ROLL→WALK 成功率。

### 导出一致性测试出现约 1e-3 误差时

旧测试直接比较 NumPy FP32 导出推理与未指定 matmul precision 的 JAX GPU
推理。float32 数组不保证底层矩阵乘法使用完整 FP32 精度；GPU 可能采用
TF32 等较低精度路径。仅凭误差幅值不能断定导出错误或精度问题。

测试现保留 `atol=rtol=2e-5`，在 `jax.default_matmul_precision("highest")`
作用域内编译并执行严格参考推理，同时打印原配置与最高精度输出的差异。
还分别比较归一化、未折叠的 NumPy 网络、归一化折入首层后的导出网络。
未修改导出权重、动作定义或训练/MJX 的全局精度设置。

只重跑这一项即可定位，无需重复编译物理环境：

```bash
python -m unittest tests.test_transition_mjx_3d.TransitionMJXTest.test_real_brax_actor_export_parity -v
```

如果 `configured_vs_fp32_max_abs` 约为原来的 1e-3，而四项严格比较通过，
说明这次误差来自推理计算精度。如果严格比较仍失败，保留完整的
`[transition export parity]` 诊断输出，继续检查对应计算环节，不能放宽阈值。
这项随机网络测试通过也不等于真实训练权重和闭环部署验收通过。

JAX 的精度控制说明见
[官方文档](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)。
