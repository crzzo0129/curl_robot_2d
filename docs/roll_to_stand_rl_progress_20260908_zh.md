# Roll to Stand 参考残差 RL 交接 · 进展与 GPU runbook

本文件是对 [`roll_to_stand_rl_handoff_20260908_zh.md`](roll_to_stand_rl_handoff_20260908_zh.md)
的续接。记录已在本机完成的步骤、需要在 Linux GPU 实例上执行的任务，以及部署方案决策。

## 已在本机完成（CPU MuJoCo 3.10.0，无需 JAX）

### 任务 1：新 reference + ABD=0 手写基线（通过）

```powershell
python -m scripts.run_handcrafted_roll_to_stand `
  --headless --deploy-at-pitch-deg 90 --min-roll-turns 1 `
  --out results/handcrafted_roll_to_stand_abd0_high_speed/plus90
```

`results/handcrafted_roll_to_stand_abd0_high_speed/plus90/summary.json`：

| 验收项 | 结果 |
|---|---|
| `stable_stand_success` | **true**（最长稳定站立 1.797 s） |
| `stand_abduction_deg` | **`[0, 0, 0, 0]`** |
| 触发 pitch / 角速度 | 89.90° / 7.44 rad/s（1.751 圈） |
| 最终高度 / 倾斜 | 0.1555 m / 0.0045° |
| 自接触 / 峰值力矩 | 0 次 / 1.938 N·m |

同目录另存 `rollout.csv/npz` 和关键帧 PNG（`trigger_plus90_replay.png`、
`mid_deploy_075ms_replay.png`、`deploy_end_150ms_replay.png`、`final_stand_replay.png`）。
本机默认 Python 缺 `imageio-ffmpeg`，未生成 MP4；关键帧 PNG 已替代视频用于落地姿态与碰撞目检。

### 任务 2：独立 train/eval +90° handoff banks（通过）

```powershell
python -m scripts.collect_reference_roll_to_stand `
  --out results/roll_to_stand_reference_residual/handoffs_train.npz `
  --samples 8 --first-turn 1 --seed 0
python -m scripts.collect_reference_roll_to_stand `
  --out results/roll_to_stand_reference_residual/handoffs_eval.npz `
  --samples 8 --first-turn 9 --seed 1000
```

`validate_reference_split` 通过：

- `reference_sha256` 一致：`3cafc56df34dffddc7cb3ec00be0626de99605fd7f0f6a07d6b71d25c1cf2c60`
- `nominal_dynamics_match=true`（Newton 20 / 10 ls / 1 ms，与 XML 一致）
- train seed 0（1.75–8.75 圈）、eval seed 1000（9.75–16.75 圈），来源独立
- 每个 handoff：pitch ≈ 89.1–90.0°，线速度 ≈ 0.94 m/s，角速度 ≈ 7.8 rad/s，qvel 全部有限

### 任务 3：新增等价验证脚本（CPU 侧已自证）

新增 [`scripts/verify_mjx_zero_residual_equivalence.py`](../scripts/verify_mjx_zero_residual_equivalence.py)：
每个 handoff 以 action=0 重放 150 ms 线性 reference（`handoff_ctrl -> ABD=0 Stand`），
比较 CPU 与 MJX 的 ctrl、最终姿态、角速度与触地顺序。

本机已跑 `--cpu-only`，并交叉验证：

- 手写脚本触发态与 `handoffs_train[0]` 的 qpos/qvel/ctrl **逐元素一致**；
- 160 子步后 ctrl 精确到达 ABD=0 Stand；
- CPU 重放在 150 ms 处与手写 rollout 的 qpos 最大偏差 0.022（剩 3 ms 采样对齐误差）；
- 触地顺序：前 100 ms 无足接触，约 100 ms 起后腿（rear_L/rear_R）先着地，前腿在 160 ms 窗口内不着地。

**注意**：150 ms 窗口结束时机器人仍在运动中（最终倾斜约 44.6°、y 角速度约 2.7 rad/s），
“站稳”发生在随后的 stand-hold 阶段。等价比较只覆盖 150 ms 窗口，不是最终稳定站立。

## 需要在 Linux GPU 实例上执行（JAX 0.6.2 / MJX，本机无此栈）

环境按 `requirements-mjx.txt` 准备（`jax[cuda12]>=0.6,<0.7`、`flax>=0.10,<0.11`、
`brax>=0.14,<0.15`、`mujoco-mjx>=3.9,<4`）。本机 `.j/` 是本地 CPU 版（JAX 0.11.1 /
MuJoCo 3.12.0），与 GPU 训练栈版本不同，仅作兼容性检查，不能替代正式结果。

### 任务 3：MJX 零残差等价验证

```bash
python -m scripts.verify_mjx_zero_residual_equivalence \
  --train results/roll_to_stand_reference_residual/handoffs_train.npz \
  --eval  results/roll_to_stand_reference_residual/handoffs_eval.npz \
  --out   results/roll_to_stand_reference_residual/zero_residual_equivalence.json
```

（去掉 `--cpu-only`，脚本会在 GPU 上构建 MJX env、逐 handoff 以 action=0 步进 8 个
策略步并与 CPU 重放比较。）验收：`comparison_{train,eval}.qpos_max_abs_dev` 与
`qvel_max_abs_dev` 应小（数量级 `1e-3` 或更小），`foot_contact_mismatched_handoffs`
应为 0 或仅由接触阈值边界引起。

### 任务 4：PPO smoke

```bash
python -m scripts.train_roll_to_stand_reference_residual \
  --roll-snapshots results/roll_to_stand_reference_residual/handoffs_train.npz \
  --eval-roll-snapshots results/roll_to_stand_reference_residual/handoffs_eval.npz \
  --preset cpu_smoke \
  --out results/roll_to_stand_reference_residual_smoke
```

Smoke 只验证环境编译、reset、PPO 更新与独立评估链路（`cpu_smoke` = 2 env / 1536 步）。
通过后正式训练换 `--preset smoke`、`4090` 或 `h200`，并使用新的输出目录。

## 任务 5：部署方案已选定「方案 B（绝对 Stand 输出）」

用户选择 **方案 B**：不修改 C++ 做 150 ms 时变 reference，而是把策略
**蒸馏/再训练为直接输出绝对 Stand 控制量**，复用现有 C++ 静态映射
`default_joint_pos + action * action_scale`。

据此部署 ABI 应回到（`transition_controller_metadata_3d` 在非 residual 分支已生成）：

- `action_semantics = "absolute_about_default"`（不再是 `residual_over_150ms_linear_reference`）
- `default_joint_pos` = ABD=0 的 Stand 目标
- `action_scale` = 全量程 `action_range_fraction * max(high - target, target - low)`，
  不再是 `(0.17, 0.50, 0.50)*4 * 0.35` 的残差缩放
- C++ `neural_controller` **无需改动**（静态映射已支持）；需要重新验证导出 ABI 与实机动作。

### 方案 B 的训练路径（B1 已落地，B2 存接触代理风险）

已实测确认：对 abd10 模型用非 residual 配置（`handcrafted_reference_residual=False`
+ `dynamic_roll_to_stand=True` + `stand_abduction_zero=True`）时，`transition_controller_metadata_3d`
输出即满足方案 B 的 ABI：

- `action_semantics = "absolute_about_default"`
- `default_joint_pos` 的 ABD 四项 = `[0, 0, 0, 0]`
- `action_scale = [3.6652, 2.6453, 0.9755] × 4`（全量程，非残差 0.35 缩放）
- `policy_frequency_hz = 50.0`、`control_timestep = 0.02`

**B1（绝对输出训在 mesh abd10，复用现有 bank）—— 已落地，非 JAX 校验链全部通过。**
为让绝对输出复用 +90° handoff bank，改了三处（把「bank 格式」与「动作语义」解耦）：

1. `train_mjx_3d_transition_ppo.build_task` 放宽守卫：非 residual 的 `dynamic_roll_to_stand`
   允许 `rollingquad_2_primitive` 或 `rollingquad_2_abd10_no_self_collision`（仍拒绝无关的
   `rollingquad_2` 行走 mesh）；
2. `reference_bank_contract_3d.validate_reference_bank` 让绝对路径也接受
   `handcrafted_roll_to_stand_90_reference`（此前只接受 `cem_reference_zero_residual`）；
3. `transition_initialization_3d.load_roll_snapshots_3d` 依据 bank 来源标记
   `trigger_pitch_deg=90`（而非仅 `handcrafted_reference_residual`）识别 +90° handoff，
   使绝对路径也走 `measured_pitch_90_handoffs` 选择分支。

已本地验证：`validate_reference_split` 通过、train/eval `inspect_bank` 均返回
`measured_pitch_90_handoffs`（8 样本、coverage=True、velocities_modified=False）、
`--dry-run` 配置正确（`handcrafted_reference_residual=False`、`geometry=rollingquad_2_abd10_no_self_collision`、
`stand_abduction_zero=True`、solver 20/10）。正式训练命令：

```bash
python -m scripts.train_mjx_3d_transition_ppo \
  --geometry rollingquad_2_abd10_no_self_collision --stage brake_full \
  --dynamic-roll-to-stand --stand-abduction-zero --physics-profile accurate \
  --initial-policy-std 0.05 \
  --roll-snapshots results/roll_to_stand_reference_residual/handoffs_train.npz \
  --eval-roll-snapshots results/roll_to_stand_reference_residual/handoffs_eval.npz \
  --preset cpu_smoke --out results/roll_to_stand_absolute_smoke
```

**B2（绝对输出训在 primitive_abd10，便宜但需验证接触代理）—— 存风险。**
`assets/rollingquad_description_2/mjcf/rollingquad_primitive_abd10.xml` 已存在，关节结构、
`nq/nv/nu`、总质量（3.1372）、求解参数与 mesh abd10 一致，且 `mesh_geoms=0`。但实测接触几何
**不一致**：

- mesh abd10 的 `*_foot_proxy` 是 box（`type=7`，size `[0.020, 0.031, 0.066]`，`contype=0/conaffinity=1`）；
- primitive_abd10 的 `*_foot_proxy` 是 sphere（`type=2`，radius 0.0195，`contype=8/conaffinity=15`），
  且位置不同（mesh `(0.006, -0.048, ±0.027)` vs primitive `(0, -0.088, ±0.027)`）。

因此 B2 不能直接当作 mesh 的等价接触代理；要走 B2 还需：(1) 把 `rollingquad_2_primitive_abd10`
接入 `TRANSITION_GEOMETRY_NAMES_3D` 与 `build_task`；(2) 修正/验证 primitive 足端与壳体接触代理
与 mesh 一致；(3) 从该 primitive 模型重新生成 handoff bank（不得混用本次 mesh 状态）。

**建议**：smoke 与首轮正式训练走 B1（复用 mesh + 现有 bank，正确性有保证；mesh 仅与地面单边接触、
自碰撞已禁用，训练成本可控）。只有确认 B1 的 mesh-vs-floor 吞吐不可接受时，再投入 B2 的
接触代理验证与 bank 重生成。

## 本次新增/修改文件

- `scripts/verify_mjx_zero_residual_equivalence.py`（新增）
- `scripts/train_mjx_3d_transition_ppo.py`（修改：`build_task` 放宽非 residual 允许 mesh abd10，即 B1）
- `curl_robot_2d_mjx/reference_bank_contract_3d.py`（修改：绝对路径也接受 +90° handoff bank）
- `curl_robot_2d_mjx/transition_initialization_3d.py`（修改：按 bank 来源标记识别 +90° handoff，解耦动作语义）
- `tests/test_dynamic_roll_to_stand.py`（修改：`test_mesh_training_rejected_before_runtime` 错误信息断言同步）
- `results/handcrafted_roll_to_stand_abd0_high_speed/plus90/*`（任务 1 产物）
- `results/roll_to_stand_reference_residual/handoffs_{train,eval}.npz{,.summary.json}`（任务 2 产物）
- 本文件

## 遗留提示

- `tests.test_dynamic_roll_to_stand.DynamicContracts.test_reference_dynamics_and_split`
  在本沙箱下因 `tempfile.TemporaryDirectory` 被拒写而 ERROR（`PermissionError`），
  非代码回归；其余 8 项通过。该测试逻辑（seed 重叠 / 动力学不一致应抛错）已由
  上面的 `validate_reference_split` 实跑确认。
- 本机默认 Python 无 `jax/brax/flax/mjx`，任务 3/4 必须到 Linux GPU 实例执行。
- 回归检查：本次改动涉及的 transition 相关 5 个测试模块共 74 项，除沙箱 tempfile 报错
  （37 项 `PermissionError`）外，只有两项 FAIL，且**均为既有、非本次改动引入**——
  `tests.test_transition_3d.Transition3DModelAndCliTests.test_named_walking_start_matches_current_deploy_env`
  与 `tests.test_transition_deployment_3d.TransitionDeploymentTests.test_metadata_matches_hardware_names_pose_and_rate`
  都是因普通 `rollingquad_2` 的 Stand keyframe 外展从 0 改为 ±15°（max diff 0.2618 rad =
  15°，4/12 项）所致，与本次 abd10 / 方案 B 改动无关。本次 4 个文件改动未引入新回归。
