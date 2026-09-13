# DR 策略：从 stand-to-roll 接管状态续训速度跟踪

本次继续使用 DR 检查点 **696320**，不重新蒸馏。目标是让策略接住启动后的较高速度，并逐步跟踪用户指定的滚动速度。最终命令保留 **0.45–0.75 m/s**，无需因为启动瞬间超速就把整个目标区间上移。

代码已准备；本地只做源码语法与差异检查，没有运行本轮采集、仿真、单元测试或训练。效果以云端同一评估面板的 step=0 和后续检查点比较为准。本次没有修改或部署机器人控制器。

## 改动与定义

| 项目 | 新配方 |
|---|---|
| 初始策略 | `results/rolling_dr025_lateral050_20260912_213612/checkpoints/000000696320` 的完整 PPO actor、critic、normalizer |
| 起点分布 | 约 50% 实际启动策略回放后的接管快照，约 50% CEM 稳定滚动快照 |
| 接管快照时刻 | stand-to-roll 滚一圈，经过现有接管窗口、15 帧混合、15 帧健康直行稳定，再随机延迟 0–0.30 s；不是第一帧混合动作 |
| 保存内容 | qpos、qvel、ctrl、仿真时间、滚动相位、过去观测和上一实际动作；reset 加入当前帧，形成 36×20 历史 |
| 速度口径 | 每个 20 ms 时间段内，平面位移投影到该段中点水平航向，除以 0.02 s；航向来自 torso 滚动轴，处理 ±π 跨越 |
| 进度口径 | 累积上述航向前向位移，不用世界 x 净位移抵消转弯进度 |
| 最终命令 | vx=0.45–0.75 m/s；yaw=0 或 ±0.02–0.07 rad/s，直行占 40%；每个 episode 固定一个最终命令 |
| 接管段命令 | 接管样本的初始 vx 用最近约 1 s 的航向速度均值，限在 0.45–1.05 m/s；每秒最多变化 0.15 m/s，直到最终目标；yaw 从 0 以 0.07 rad/s² 变化到目标 |
| 稳定滚动样本 | 直接使用最终目标，不套接管初速斜坡 |
| 物理 | 名义所有关节 P=5、D=0.1；DR=0.25；直行 lateral 失败阈值 0.50 m，转弯继续使用既有转弯判据 |
| 奖励 | 前向速度跟踪权重 6、yaw 跟踪权重 3，保留原滚动和稳定性项，anchor 权重 0.01 |
| 优化 | critic 约 204800 steps；actor 约 1966080 steps、初始/最高 LR=3e-6、PPO clip=0.05；实际步数按 rollout 批次取整 |
| 评估 | 从各自 reset 状态开始 500 步、10 s，失败提前终止；采集时的启动和混合阶段不计入这 10 s |

例如初始 0.90 m/s、最终 0.60 m/s 时，指令在 2 s 内从 0.90 下降到 0.60。这个 2 s 是**命令斜坡时长**，不是声称机器人已经在 2 s 内减速成功。真实响应由速度误差和存活率反映。

训练/评估按独立启动轨迹 ID 划分，约 75%/25%；同一次启动的帧不会跨组。256 次采集约提供 192 个训练初态、64 个留出初态。与不同命令组合可以形成更多 episode，但不代表更多独立启动轨迹。采集用名义 CPU MuJoCo；DR 在 PPO reset 后施加并重算派生物理量和 critic 观测，尚不等价于随机物理下的完整启动分布。

## 云端命令

在云端 `curl_robot_2d` 仓库根目录操作。传输包为 `results/rolling_handoff_tracking_v1.zip`，包含当前源码、检查文件，以及已经核对来源的三个 JSON 模型。它不包含 PPO 检查点；继续使用云端已有的 696320 检查点。

先解压覆盖同名源码，运行轻量检查，再采集接管快照。采集只使用云端 CPU，不运行 JAX，也不需要显示器：

```bash
unzip -o results/rolling_handoff_tracking_v1.zip -d .

python -m unittest discover -s tests -p test_rolling_handoff_tracking.py -v

bank="results/rolling_handoff_bank_$(date +%Y%m%d_%H%M%S).npz"
python -u -m scripts.collect_rolling_handoff_bank \
  --startup handoff_assets/stand_to_roll.json \
  --rolling handoff_assets/rolling_dr696320.json \
  --stop handoff_assets/roll_to_stand.json \
  --count 256 \
  --out "$bank"
```

每次尝试会显示 accepted/attempts，详细启动日志在同名 `.log`。成功后生成 `.npz` 和 `.json`；初速分布在 JSON 的 `speed_summary_m_s`。若健康接管不足会报错，不会悄悄换回 CEM 初态。必须使用新的 bank 文件名；失败后也换新文件名保留诊断。

接着自动完成 critic → actor 两阶段：

```bash
run="results/rolling_heading_handoff_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.run_rolling_handoff_tracking \
  --source results/rolling_dr025_lateral050_20260912_213612 \
  --step 696320 \
  --handoff-bank "$bank" \
  --out "$run"
```

脚本会检查源检查点和原 CEM/steering 配置路径，运行轻量检查，然后建立跨 GPU 快照池。训练使用 4 张 GPU、2048 个环境。已有输出会被拒绝，不覆盖旧训练结果。

如果希望先看 critic 结果，把上面的命令加 `--stage critic`。确认 critic 正常后，使用**同一个** `$run`、`$bank`，把选项改成 `--stage actor`。不要在已完成 critic 的目录重新运行默认 `all`。

重开终端后需手动将 `bank` 和 `run` 设置成之前生成的实际路径。可加 `--dry-run` 只打印训练命令，不启动采集、JAX 或训练。

## 怎么判断训练有效

固定评估为名义物理、无部署 DR；常规 `[Student DR PPO eval]` 为训练 DR 设置。两者分别比较，不能把固定面板的成功率当成 DR 鲁棒性。

查看 `$run/actor/fixed_eval_history.json`：

- `tracking_by_reset_source.handoff` / `mature`：分别看成功率、完整 10 s 比例和速度误差，避免一个组改善掩盖另一个组退化。
- `transition_forward_mae_m_s`：命令仍在斜坡期间，相对**当前斜坡指令**的误差。
- `steady_forward_mae_m_s`：命令已到最终目标后，相对最终目标的误差；名称表示命令阶段，不保证实际运动已经稳定。
- `final_target_forward_mae_m_s`：全程相对最终目标的误差，保留启动速度差，不隐藏接管减速成本。
- `steady_phase_reached_episodes`、`steady_steps`：实际贡献稳态指标的样本量；提前失败的 episode 可能没有稳态样本，必须同时看失败率。
- `command_evaluation`：按最终速度和左右转命令分组，不按较高的接管初速分组。

奖励始终与选动作时实际看到的指令一致，下一帧指令不会用于给上一动作打分。世界 x 速度仍保留为 `world_forward_velocity_m_s`，用作诊断，不能再用它判断转弯时的真实前向减速。

新面板改变了起点和进度口径，旧的 92% 与新结果不直接可比。先比较新面板的 step=0 与新检查点。critic 阶段的 actor 参数应不变，固定面板表现也应一致；如果这一步发生明显改变，先看诊断再训练 actor。actor 若比自身 step=0 的固定成功率下降超过 10 个百分点，现有监控会保存检查点并停止。

脚本自动输出 `critic_diagnostics.zip` 和 `actor_diagnostics.zip`。选模仍先看成功率，再以稳态 vx MAE、yaw MAE 打破平局；这只是在固定验证面板选候选，最终还应换一组采集种子做独立验证。

## 后续真机接入

本轮先训练，不替换当前真机 696320 策略。新导出带 `heading_handoff_v3` 契约，当前机器人加载器不接受它，避免训练使用高初速斜坡而部署仍固定从 0.60 起步的错配。

新模型部署时需让接管结束后的命令变化与训练一致。配置已保存训练组的初速中位数供没有可靠线速度估计时使用；它是近似值，不是真机实时测速。真机运行逻辑、Python 回放逻辑和观测历史需要一起对齐后再替换新策略。
