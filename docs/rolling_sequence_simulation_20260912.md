# Stand-to-roll → PPO 转弯 7 秒 → roll-to-stand 仿真

> **已撤回的旧仿真结论。** 本文下面的轨迹使用了错误的角速度坐标：MuJoCo `mjOBJ_BODY` 的 local 输出位于惯性主轴系，不是躯干系。它影响了三套网络的输入和接管判断，因此旧数值不能用来判断策略或接管能力。修正后旧逻辑也能完成接管、正向转弯和站起。有效的新结果见 [平滑接管验证](rolling_smooth_handoff_20260912.md)。下文仅保留诊断历史。

本次按用户要求执行 CPU MuJoCo 单轨迹仿真，没有使用机器人电机，也没有运行训练。三段都使用神经网络 JSON；启动不是 CEM 快照，过程中不重置物理状态。结果是：当前组合在该初态下不能顺利完成动作链。

## 条件

- XML：`rollingquad_abd10_no_self_collision.xml`，cg20、implicitfast、1 ms 物理步长、50 Hz 策略、关闭自碰撞、原模型 3 Nm 电机力矩限幅。
- 从站立关节姿态 `[0, 0.9, 1.15] × 4`、零初速度开始，按几何支撑最低点调整初始高度，先保持 0.5 秒。
- 启动：`stand_to_roll_student_clipped_bc_norm_v1.json`；SHA256 `4f16bf9607b2a40ff40ae3b52bc26eab57091c538213596cf0c459920c0b6c7b`。
- PPO：第 81,920 步；SHA256 `902d8c08f530cfe752e394332e45d5ccbe7e641fba7f5bcf316b4c4938f74e97`。
- 停止：本地 Downloads 的 `policy_rtneural.json`；SHA256 `98cae471b8659c1d967ee4aac5b763e21282c4f40fcf58016007e3af4f72d7d5`。其接口配置与此前从机器人读取的停止模型一致；机器人此次离线，无法再次核对文件哈希。
- PPO 命令 vx=0.60 m/s、vy=0、yaw=+0.07 rad/s。7 秒从 PPO 实际接管时刻起算；到时预约停止，沿用 +85°～+95° pitch 门限及 6 rad/s 目标限速，再观察停止策略 5 秒。
- 观察、历史排列、动作中心转换、锁定外展通道、停止阶段冷历史和上一实际目标限速按照当前控制器逻辑复现。Python/NumPy float32 的算术顺序与 C++/Eigen 不保证逐位一致；没有加入真实传感器噪声、通信延迟或 SPI 错包。

## 按真机连续性门限

`results/rolling_sequence_turn_7s_20260912/` 保存报告、轨迹和 `sequence.mp4`。

0.50 秒启用启动策略，2.26 秒达到一整圈后请求 PPO 接管。随后等到 12.28 秒仍未满足条件，请求被拒绝。有效候选中“所有关节目标差的最大值”的最小值是 **0.3171 rad**，大于真机 **0.12 rad** 门限；PPO、转弯和停止阶段均未执行。

这里的 0.3171 rad 是所有关节中最大的目标差，不表示已定位到某个具体关节；本次没有增加自动外展过渡或放宽真机门限。

## 用户指定时序的直接切换诊断

为检查后两段，在第二条仿真里仅绕过 0.12 rad 连续性门限，保留历史转换、滚动条件和停止相位门限。此模式只用于仿真诊断，不等同于当前真机执行结果。

`results/rolling_sequence_prescribed_turn_7s_20260912/` 保存报告、轨迹和 `sequence.mp4`，视频顶部明确标注门限被绕过。

| 时刻 | 事件 |
|---|---|
| 0.50 s | 启用 stand-to-roll |
| 2.26 s | 累计滚完一圈，发出接管请求 |
| 2.28 s | PPO 接管并接收左转命令；最大目标跳变 **0.5448 rad** |
| 9.28 s | PPO 接管满 7 秒，请求 roll-to-stand |
| 10.064 s | pitch=94.81°，停止策略接管 |
| 15.065 s | 停止策略运行满 5 秒，仍未站稳，姿态已翻倒 |

PPO 前 7 秒的采样统计：世界 X 平均前进速度 **0.3966 m/s**，滚动轴水平航向的平均变化率 **−0.1322 rad/s**，累计航向变化 **−52.87°**，世界 Y 位移变化 **−1.513 m**。正 yaw 命令下实际发生负向转弯。数据区间为 6.98 秒的 50 Hz 采样端点跨度，预约停止仍严格发生在接管 7 秒后。

停止后按现有返回 Walking 的条件检查站稳：躯干倾角 <0.35 rad、各关节速度绝对值 ≤0.25 rad/s、各轴身体角速度绝对值 ≤0.3 rad/s，持续 0.5 秒。本次最长连续满足时间 **0 秒**；最终躯干倾角约 **171.15°**。

两条结果均不能当成成功率估计。它们说明至少这个启动初态下，当前门限会拒绝接管，而直接切换又没有实现预期转向和站立恢复。下一步应使用真实 stand-to-roll 输出状态验证/训练接管，并从 PPO 转弯末态验证停止策略，不能用放宽门限代替。

## 复现

入口 `python -m scripts.simulate_rolling_policy_sequence`，依赖 NumPy、MuJoCo；视频额外需要 Pillow、imageio、imageio-ffmpeg。

```bash
python -m scripts.simulate_rolling_policy_sequence \
  --startup /path/to/stand_to_roll_student_clipped_bc_norm_v1.json \
  --rolling /path/to/rolling_command_ppo_000000081920.json \
  --stop /path/to/policy_rtneural.json \
  --out results/sequence_controller --video
```

追加 `--handoff-mode prescribed` 并更换输出目录，可复现直接切换诊断。源模型和结果哈希在各自 `report.json` 中记录。
