# 滚动机器人定相位制动与停车训练方案

## 1. 目标与边界

本任务不是简单地把滚动速度降为零，而是完成以下闭环动作：

```text
ROLL -> BRAKE_ALIGN -> PARK_DEPLOY -> HOLD
滚动      定相位制动       展开停车       稳定保持
```

最终停车状态要求足端朝下、线速度和角速度接近零、关节进入指定停车姿态，
并稳定保持至少 2 s。由于现有高速滚动控制器无法在保持 5 圈/10 s 的同时实现
全程零内部接触，零内部接触不作为滚动和制动阶段的全局硬约束；停车 HOLD 阶段
仍以零内部接触为目标。Torso 异常接触和腿部交叉在所有阶段禁止。

第一版采用独立 braking residual policy，冻结已验证的滚动 reference。完成后再把
ROLL 与 STOP 合并成 command-conditioned policy。

## 2. 状态机

### ROLL

- 使用现有 CEM/rolling policy 正常滚动。
- `stop_command=1` 时保持主动控制并进入 BRAKE_ALIGN，不立刻切换到 compact。
- 根据当前角速度、可实现减速度和安全余量，选择“来得及刹车”的下一次停车相位；
  最近一次相位窗口距离不足时，继续受控滚动一圈。

### BRAKE_ALIGN

- 目标相位为适合展开、足端朝下的 `park_phase`。
- 继续使用主动 rolling reference，逐渐降低其频率和幅值；compact 只是滚动参考中心，
  不是需要静态维持的制动终点。
- 第一版由 braking residual policy 调整基础滚动控制，RL 不负责选择停车周期或展开时机。
- 同时抑制线速度、pitch 角速度、倒滚和越过目标相位。
- 状态进入可展开集合后才进入 PARK_DEPLOY；若错过窗口则继续受控滚动至下一次窗口。

这里的“低速”不是一个需要长期维持或原地等待的独立模式。机器人从正常滚动速度
主动减速，短暂经过低速可展开区域，并在 gate 满足后立即展开。低速阶段的作用是
降低剩余动能、建立可重复的展开入口，而不是追求低速持续滚动圈数。

### PARK_DEPLOY

- 切换瞬间捕获真实 `q_capture/qdot_capture`、根节点速度、相位和接触状态。
- 使用满足初始位置和初始速度边界的五次多项式，从捕获状态平滑过渡至
  `park_pose`，而不是播放一条与初始状态无关的绝对轨迹。
- 只有相位、速度、姿态和接触同时落入可展开集合时才允许展开。
- 若直接插值产生足端撞击，则使用确定性的两段轨迹
  `q_capture -> q_mid -> q_park`；优先离线搜索 waypoint、时长和前后腿时序，
  固定轨迹覆盖不了制动结束状态分布时才增加 DEPLOY residual。
- 达到停车关节姿态且足端支撑建立后进入 HOLD。

### HOLD

- 直接使用固定 PD 位置控制跟踪 `park_pose`，第一版不训练 HOLD policy。
- 连续满足 settled 条件 2 s 后判定成功。
- 重新运动或姿态超差时退出 settled 计时，但不自动回到滚动模式。

## 3. 相位和 reference 设计

当前相位为 `phase`，前进方向为 `direction`，停车相位为 `park_phase`。
先计算同方向到某次停车相位的非负距离：

\[
\Delta\phi = ((\phi_{park}-\phi)\,direction) \bmod 2\pi.
\]

再估计最低制动相位距离：

\[
\Delta\phi_{required} \approx \frac{\omega^2}{2\alpha_{max}}+margin.
\]

选择首个满足 `delta >= delta_required` 的未包裹目标相位，避免在
`2*pi -> 0` 边界跳变，也避免为了追赶过近的窗口进行急刹。BRAKE_ALIGN 平滑调整
主动滚动 reference 的频率和幅值。PARK_DEPLOY 根据捕获的 `q/qdot` 在线生成满足
起止边界的确定性五次轨迹，不经过静态 compact 停留。

策略观察量增加：

- `stop_command`；
- 状态机 one-hot；
- `sin/cos(body_phase)`；
- `sin/cos(target_phase_error)`；
- `time_since_stop` 和 `time_in_mode`；
- 当前 reference 频率/幅值 scale、制动进度和各 deploy gate 的归一化余量；
- 根节点线速度、角速度、姿态和关节误差。

## 4. 停车姿态

第一候选使用 3-D 模型 `stand` keyframe，但它只是搜索起点，不直接视为最终设计。
静态停车姿态必须通过以下检查：

- 足端形成稳定地面支撑；
- Torso 不触地、不接触腿或电机；
- 无机器人内部穿透；
- 质心投影位于支撑区域；
- 自由根节点下保持 3 s；
- 最终线速度 <= 0.03 m/s；
- 最终角速度 <= 0.10 rad/s；
- Torso tilt <= 5 deg；
- 电机峰值力矩不超过额定上限的 70%。

静态验证失败时，先调整 park pose 或质心/支撑几何，不启动 PPO 掩盖问题。

## 5. 分阶段路线

| 阶段 | 方法 | Reset/输入 | 目标 | Gate |
|---|---|---|---|---|
| P0 HOLD | 固定 PD，无训练 | park pose | 静态保持 | 静态 gate 100% |
| P1 DEPLOY | 确定性轨迹搜索 | 低速滚动快照 | 从捕获状态展开至 park | success >= 95% |
| T1 BRAKE-LOW | RL braking residual | 低速、相位邻域 | 进入可展开集合 | success >= 90% |
| T2 BRAKE-FULL | RL braking residual | 正常速度、全相位 | 选择可达周期并定相位制动 | success >= 90% |
| T3 CHAIN | 联合评估/必要时微调 | 滚动快照 | 完整 stop state machine | success >= 90% |
| T4 ROBUST | curriculum | 随机扰动 | 鲁棒停车 | success >= 85% |

P0 已由当前 `park` keyframe 完成。P1 先不使用 RL；只有确定性展开模板无法覆盖
BRAKE 结束状态分布时，才在固定轨迹上增加小幅 DEPLOY residual。训练阶段只在 gate
通过后继续，并保存 stage best/final 和回退点。

### 5.1 低速阶段的具体意义

低速主动滚动只用于把“是否能够安全展开”和“如何从高速消除大量动能”拆成两个问题：

1. 在动能较小时扫描相位，确定适合展开的 `park_phase` 和相位窗口；
2. 采集真实 `qpos/qvel/phase/contact`，搜索状态条件展开轨迹；
3. 确定可展开集合的速度、姿态和接触边界；
4. 让 BRAKE-LOW 先学习精确进入该集合，再逐步提高初始滚动速度。

“持续低速滚动”和“刹车过程中短暂经过低速”是两个不同要求。前者要求控制器在很多
周期内持续注入恰当能量，存在最低维持频率；后者只要求机器人在倾倒或失速前进入
PARK_DEPLOY，不要求继续滚动一圈。因此，当前控制器低于某个频率不能维持 10 s，
不代表制动策略不能短暂经过更低角速度。三阶段 CEM 使用的“10 s 至少 5 圈”门槛
只评价持续滚动控制器，不作为停车切换 gate。

BRAKE-LOW 的训练成功条件也不是“低速滚了多少圈”，而是从低速滚动快照出发后，
在限定时间和距离内进入：

\[
\mathcal{S}_{deploy} =
\{q,\dot q,\phi,v,\omega,contact\mid
\text{phase/speed/pose/contact gates all pass}\}.
\]

进入集合后立即结束 BRAKE 阶段并执行确定性展开；不允许策略在集合内等待以刷 reward。

## 6. Reset 和滚动快照

停止策略不从静止重新学习滚动。使用现有滚动策略采集 snapshot dataset：

- 相位均匀划分为 32 个 bin；
- 每个 bin 至少 50 个快照；
- 保存 `qpos/qvel/oscillator_phase/reference_state`；
- 第一批覆盖可控低速区间，随后扩展至 70%--110% 名义速度；
- 保存摩擦、倾角、左右差分和 seed 元数据。

训练 episode 从快照随机 reset，停止命令在 0.5--3.0 s 后发出。

## 7. 奖励结构

ROLL 沿用现有滚动奖励。STOP 奖励按模式分开记录，不能只输出总 reward。

BRAKE_ALIGN：

- 奖励向目标相位前进并按计划减速；
- 惩罚线速度、pitch 角速度、倒滚、相位越过、动作变化和冲击力矩。

PARK_DEPLOY（仅在确定性模板失败、需要 residual 时）：

- 奖励 joint pose、Torso pose 和足端支撑；
- 惩罚高速展开、足端撞击、Torso 接触和关节突变。

HOLD 不训练，仅记录线速度、角速度、停车后位移、姿态误差、足端滑移和内部接触，
连续 settled 2 s 后由评估器判定成功。

终止条件包括倾倒、数值异常、root 高度越界、严重侧向漂移、腿部交叉和
持续 Torso 接触。普通目标误差不提前终止，避免策略通过早退规避任务。

## 8. Best 选择与评估

Best checkpoint 按以下字典序选择：

1. 数值安全和生存；
2. stop success rate；
3. 足端朝下和 HOLD success；
4. 停止时间、停止距离和相位误差；
5. reward。

每阶段使用 paired deterministic evaluation。开发 gate 至少 512 episodes，最终
候选使用 1024 episodes，并固定同一批 snapshot、命令时间和扰动 seed。

主要指标：success rate、stop time、extra distance、final linear/angular speed、
phase error、pose error、settled duration、Torso contact、leg crossing、internal
contact、torque、lateral drift 和 failure reason。

## 9. 代码架构

无需训练即可完成：

- `curl_robot_2d_mjx/stop_task.py`：配置、状态机、相位选择、reference 调度；
- `curl_robot_2d_mjx/stop_evaluation.py`：episode 指标、gate 和失败分类；
- `scripts/validate_3d_park_pose.py`：MuJoCo 静态停车验证；
- `deploy_trajectory.py`：根据捕获的 `q/qdot` 生成五次展开轨迹；
- `search_deploy_trajectory.py`：在低速快照集合上搜索相位窗口、时长和 waypoint；
- `tests/test_stop_task.py`：状态机、wrap、blend 和 gate 单元测试。

后续训练阶段新增：

- `environment_stopping_2d.py` / `environment_stopping_3d.py`；
- `reward_stopping.py`；
- `train_mjx_stopping_ppo.py`；
- `evaluate_mjx_stopping_policy.py`；
- `collect_rolling_snapshots.py`。

这些训练文件在 park pose 静态 gate 通过后再接入，避免过早复制现有环境代码。

## 10. 实施顺序

1. 固化任务配置、状态机、reference 调度和评估 gate。
2. 验证或搜索静态 `park` pose，并用固定 PD 通过 HOLD gate（已完成）。
3. 可视化并标定 2-D 低速主动滚动区间，采集按相位分箱的 `q/qdot` 快照。
4. 在低速快照集合上搜索状态条件展开模板和可展开集合，不训练策略。
5. 训练低速和正常速度 BRAKE residual，使状态进入可展开集合。
6. 用真实制动结束状态重新搜索/收紧展开模板，完成 2-D paired evaluation。
7. 在真实几何 3-D 模型建立滚动基线后，对称提升 2-D 停车 reference。
8. 加入 3-D residual、摩擦、tilt 和左右差分 curriculum。

## 11. 当前实现与基线结果

已完成的无需训练部分：

- `curl_robot_2d_mjx/stop_task.py`：四状态状态机、跨周相位目标和策略观察量；当前
  reference 混合原型仍含 compact 中间目标，接入训练前必须替换为主动制动与
  `q/qdot` 条件展开接口；
- `curl_robot_2d_mjx/stop_evaluation.py`：动态停车与静态停车的独立硬门槛和失败分类；
- `scripts/validate_3d_park_pose.py`：在重力下独立验证一个 keyframe，输出 JSON 与时间序列；
- `scripts/search_3d_park_pose.py`：对前后腿四个左右对称关节目标做 CEM 静态搜索，不训练策略；
- `tests/test_stop_task.py`：覆盖状态转移、相位换周、reference 混合、观察量和评估门槛。

原 `stand` keyframe 的 3 s 验证失败：机器人继续翻滚，最终四足均离地，说明它不能
作为停止训练的目标。静态搜索得到的新 `park` keyframe 已加入
`assets/curl_robot_3d_real_geometry.xml`，独立 3 s 复验通过：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 最终线速度 | 0.00094 m/s | <= 0.03 m/s |
| 最终角速度 | 0.00035 rad/s | <= 0.10 rad/s |
| Torso 最终倾角 | 1.51 deg | <= 5 deg |
| 四足触地 | 4 | >= 4 |
| 内部/Torso 接触 | 0 s | 0 s |
| 水平漂移 | 1.66 mm | <= 20 mm |
| 峰值力矩 | 0.35 N m | <= 2.1 N m |

复验命令：

```powershell
python scripts/validate_3d_park_pose.py --keyframe park
```

当前结论只证明 `park` 是可行的静态终态，不证明机器人能从高速滚动安全到达该姿态。
HOLD 不再作为训练阶段。下一步先可视化低速主动滚动、采集快照并搜索确定性展开模板，
随后才训练 BRAKE residual。

## 12. 低速主动滚动可视化

2-D 真实几何模型使用三阶段 CEM 最终控制器。保存控制器的名义振荡频率为
`0.677 Hz`。查看器现在支持两种互斥设置：

- `--frequency-hz`：直接给出名义 reference 频率，推荐使用；
- `--phase-rate-scale`：相对保存频率缩放，例如 `0.6` 对应约 `0.406 Hz`。

交互查看当前已验证的低速主动滚动：

```powershell
python -m scripts.replay_active_controller `
  --geometry real `
  --controller results/staged_cem_real_geometry_180_d50_foot60/03_foot_gap_2mm/best_phase_controller.json `
  --frequency-hz 0.40 `
  --duration 10 `
  --viewer
```

输出 GIF：

```powershell
python -m scripts.replay_active_controller `
  --geometry real `
  --controller results/staged_cem_real_geometry_180_d50_foot60/03_foot_gap_2mm/best_phase_controller.json `
  --frequency-hz 0.40 `
  --duration 10 `
  --output results/low_speed_roll_0p40hz_10s.gif `
  --diagnostics
```

`frequency-hz` 是关节 reference 的命令频率，不是机体实际滚动频率。回放结束时会同时
打印 `commanded_frequency_hz`、`actual_average_roll_frequency_hz`、实际圈数和位移。
当前快速 10 s 扫描结果为：

| reference 命令 | 实际平均滚动 | 10 s 圈数 | 结论 |
|---:|---:|---:|---|
| 0.30 Hz | 0.043 Hz | 0.428 | 启动后失速 |
| 0.35 Hz | 0.045 Hz | 0.450 | 启动后失速 |
| 0.38 Hz | 0.444 Hz | 4.442 | 持续滚动 |
| 0.40 Hz | 0.475 Hz | 4.752 | 持续滚动，当前推荐低速入口 |
| 0.50 Hz | 0.544 Hz | 5.440 | 持续滚动 |

这说明现有 CEM 控制器存在约 `0.35--0.38 Hz` 的维持滚动阈值，不能只靠无限降低
reference 频率获得任意低速。BRAKE-LOW 快照应先从 `0.40 Hz` 附近采集；更低速度
需要专门的低速控制器、频率渐变控制或 braking residual，不能把失速状态误当作
可控低速滚动。这里的阈值仅用于选择第一批可重复快照；最终刹车轨迹可以在展开前
短暂低于该阈值，因为它不需要继续维持滚动。
