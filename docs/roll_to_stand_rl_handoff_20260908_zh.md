# Roll to Stand 参考残差强化学习交接

## 当前目标

将已经通过的手写 Roll to Stand 包进强化学习：滚动阶段使用 CEM reference，在机身 nose-up pitch 进入 +90° 时切换，随后在 150 ms 内从触发瞬间的电机目标线性插值到 Stand。RL 输出叠加在该时变参考上，只学习修正量。Stand 的四个 ABD 目标必须为 0。

当前指定资产：

- Reference：`results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`
- 模型：`assets/rollingquad_description_2/mjcf/rollingquad_abd10_no_self_collision.xml`
- 触发：nose-up pitch `+90 deg`
- 展开时间：`0.15 s`
- Stand ABD：四腿均为 `0 rad`

## 已完成

1. `scripts/run_handcrafted_roll_to_stand.py`
   - 默认 reference 已切到上述 high-speed zero-contact refine 结果。
   - 仍使用 abd10 no-self-collision 模型。
   - Stand 控制目标的 actuator 0/3/6/9 已强制置零。
   - 支持 `--deploy-at-pitch-deg 90`。

2. `curl_robot_2d_mjx/environment_transition_3d.py`
   - 增加 `handcrafted_reference_residual` 模式。
   - reset 使用真实 +90° 触发状态的完整 `qpos/qvel/ctrl`，不改写速度。
   - 零策略动作复现 `handoff_ctrl -> ABD=0 Stand` 的 150 ms 线性参考。
   - 参考按每个 1 ms MuJoCo 物理步更新；RL 残差在一个 20 ms 策略周期内保持。
   - 残差映射为 `reference + action * [0.17, 0.50, 0.50]x4 * residual_scale`，默认 `residual_scale=0.35`。

3. Actor observation 已与 `scripts/train_ppo_deploy.py` 对齐
   - 单帧 36 维，20 帧历史，总计 720 维，最新帧在前。
   - `[0:3]` body gyro；`[3:6]` projected gravity；`[6:9]` 零速度命令；`[9:12]` desired world z；`[12:24]` 关节角减 ABD=0 Stand；`[24:36]` 上一帧原始 RL 残差动作。
   - 噪声对齐为 gyro `0.20`、gravity `0.05`、joint position `0.01`，其余为 0。

4. 新增 `scripts/collect_reference_roll_to_stand.py`
   - 只收集真实滚动轨迹进入 +90° 触发窗口时的状态，不再把整段滚动都当作 transition reset。
   - 每个样本保存完整 `qpos/qvel/ctrl/time` 和 provenance。
   - train/eval 可使用不同滚动圈数形成独立集合。

5. 新增 `scripts/train_roll_to_stand_reference_residual.py`
   - 固定使用指定模型、`brake_full`、动态 Roll to Stand、ABD=0、150 ms reference residual。
   - 使用 XML 一致的 accurate 求解配置：Newton 20 iterations、10 line-search iterations、1 ms timestep。
   - 新策略初始标准差为 `0.05`，actor 均值初始接近 0。

6. reference bank 校验
   - `reference_bank_contract_3d.py` 校验 reference SHA256、+90° 触发、ABD=0、模型及物理参数，并要求训练集与评估集来源不同。

## 当前验证结果

- `tests.test_handcrafted_roll_to_stand` 与 `tests.test_dynamic_roll_to_stand`：共 9 项，通过。
- 修改文件已通过 Python compileall。
- 扩展测试中存在一个既有不一致：`tests.test_transition_3d.Transition3DModelAndCliTests.test_named_walking_start_matches_current_deploy_env` 仍假设普通 `rollingquad_2` 的 Stand ABD 为 0，但当前模型 keyframe 是前腿 -15°、后腿 +15°。这不属于本次指定的 ABD=0 residual 路径。

## 尚未完成，接手后按顺序执行

### 1. 用新 reference 和 ABD=0 重跑手写基线

此前用户确认 +90° 手写方案通过，但切换到最新 reference 并强制 ABD=0 后还没有完成动力学复验。

```powershell
python -m scripts.run_handcrafted_roll_to_stand `
  --headless --video --deploy-at-pitch-deg 90 --min-roll-turns 1 `
  --out results/handcrafted_roll_to_stand_abd0_high_speed/plus90
```

验收：`summary.json` 中 `stable_stand_success=true`，`stand_abduction_deg` 四项为 0，并检查视频落地姿态和碰撞。

### 2. 生成独立 train/eval +90° handoff banks

```powershell
python -m scripts.collect_reference_roll_to_stand `
  --out results/roll_to_stand_reference_residual/handoffs_train.npz `
  --samples 8 --first-turn 1 --seed 0

python -m scripts.collect_reference_roll_to_stand `
  --out results/roll_to_stand_reference_residual/handoffs_eval.npz `
  --samples 8 --first-turn 9 --seed 1000
```

重点检查每个 handoff 的 `pitch_deg` 接近 +90°，速度有限，两个 summary 的 reference hash 和 task physics 一致。

### 3. 做 MJX 零残差等价验证

在正式 PPO 前，应从每个 handoff 以 action=0 rollout，比较 CPU 手写轨迹与 MJX 环境：150 ms 内的 ctrl、最终姿态、角速度、触地顺序。当前实现已按 1 ms 更新参考，但尚未跑数值对比。

### 4. 跑 PPO smoke

```powershell
python -m scripts.train_roll_to_stand_reference_residual `
  --roll-snapshots results/roll_to_stand_reference_residual/handoffs_train.npz `
  --eval-roll-snapshots results/roll_to_stand_reference_residual/handoffs_eval.npz `
  --preset cpu_smoke `
  --out results/roll_to_stand_reference_residual_smoke
```

Smoke 主要检查环境编译、reset、PPO 更新和独立评估链路。正式训练再换 `smoke`、`4090` 或 `h200` preset，并使用新的输出目录。

### 5. 部署路径必须补齐时变 reference

现有 C++ `neural_controller` 只执行静态 `default_joint_pos + action * action_scale`。当前训练动作的语义是“叠加在 150 ms 时变参考上的残差”，因此仅导出网络权重和静态 JSON 还不能在实机复现训练动作。

部署前必须选择并完成一种方案：

- 在 C++ Roll-to-Stand 状态机里保存触发瞬间的 12 维 ctrl，在 150 ms 内生成相同线性 reference，再叠加网络 residual；这是与当前训练最直接一致的方案。
- 将 residual policy 蒸馏或再训练成直接输出绝对 Stand 控制量；这种方案需要重新验证动作 ABI。

目前 metadata 已写入 `action_semantics=residual_over_150ms_linear_reference`，用于防止误把 residual 网络当普通 walking policy 部署。

## 风险与约束

- 该模型禁用了自碰撞，但腿部 CAD mesh 仍参与和地面的碰撞，所以 MJX 训练计算量可能较大。当前实现遵循最近一次“模型不变”的要求。如果恢复此前“训练不用 mesh、最后 mesh 验证”的要求，需要新增与该模型质量、关节、接触代理一致的 primitive abd10 no-self-collision 训练模型，并重新生成 handoff bank，不能直接混用本次 mesh 状态。
- 只有少量确定性的整圈 handoff 样本可能覆盖不足。基础 smoke 通过后，应增加初始状态、摩擦、质量、触发角和传感器噪声扰动，但触发角扰动应围绕 +90°，且不能修改保存的真实 qvel 来伪造低动态状态。
- 当前 reward 的接触冲击项尚未获得可靠 MJX 接触力，因此配置中没有声称该项有效。评估应额外统计峰值电机力矩、非足端接触、落地角速度和稳定站立持续时间。

## 本次相关文件

- `scripts/run_handcrafted_roll_to_stand.py`
- `scripts/collect_reference_roll_to_stand.py`
- `scripts/train_roll_to_stand_reference_residual.py`
- `scripts/train_mjx_3d_transition_ppo.py`
- `curl_robot_2d_mjx/config_transition_3d.py`
- `curl_robot_2d_mjx/environment_transition_3d.py`
- `curl_robot_2d_mjx/transition_initialization_3d.py`
- `curl_robot_2d_mjx/reference_bank_contract_3d.py`
- `curl_robot_2d_mjx/deployment_transition_3d.py`
- `curl_robot_2d_mjx/config_3d.py`
- `curl_robot_2d_mjx/environment_3d.py`
- `curl_robot_2d_mjx/environment_walking_3d.py`
- `tests/test_dynamic_roll_to_stand.py`
- `tests/test_handcrafted_roll_to_stand.py`
