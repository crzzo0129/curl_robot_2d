# 持续滚动 PPO 接管与手柄控制

这份实现基于 2026-09-12 从 `pi@pupper.local` 读取的真实控制器版本，保留 PS5 经 8BitDo 的按键映射、Walking 返回检查和现有 pitch-gated roll-to-stand。

## 操作

| 操作 | 结果 |
|---|---|
| × | 原 Walking 流程 |
| △ | 站稳后启用原 stand-to-roll 策略 |
| ○ | 请求切到新 PPO 持续滚动策略；控制器确认后才显示接管成功 |
| □ | 从启动滚动、等待接管或 PPO 滚动中请求原 roll-to-stand |
| 原急停键 | 优先急停；保持原解除流程 |

PPO 接管后，线速度杆的纵向轴（axis 1）居中为 0.60 m/s，后拉到底为 0.45，前推到底为 0.75；横向输入忽略，vy 恒为 0。转向轴沿用 axis 3，正负方向与原 teleop 配置一致。两根轴使用 10% 死区并重新映射剩余行程。yaw 在死区内为 0，死区外从 ±0.02 到 ±0.07 rad/s，避免使用训练中未覆盖的非零小幅 yaw 命令。回中是恢复直行巡航，不是停车。

新命令通过 `/<rolling_controller>/rolling_cmd_vel` 发送，不使用 Walking 的全局 `/cmd_vel`。启动和停止策略继续使用原来的零命令。PPO 运行中指令不合法或超过 1 秒未收到更新会触发急停，不会把失联当成摇杆回中。

## 接管

`rolling_model_path` 是可选的新参数，默认空。保持原 `model_path` 为 stand-to-roll，保持 `roll_to_stand_model_path` 为原停止策略。没有配置 PPO 权重时，原流程照常运行，○ 不会激活一个替代模型。

○ 使用上升沿触发，连按不重置历史。请求最多等待 10 秒，同时要求：

- 观察更新间隔在 15–25 ms，连续至少 50 次；用于检查实际 50 Hz 执行和有效历史。
- IMU 累计完整滚动至少一圈，且当前身体 y 轴角速度绝对值至少 0.5 rad/s。
- 新命令有效且最近 0.25 秒内收到。
- 新网络输出有限、在 tanh 动作范围内，锁定外展输出为零。
- 所有新关节目标与最后实际下发目标的差不超过 `rolling_handoff_max_target_delta_rad`，默认 0.12 rad。

这些是初版运行时门限，不代表真实接管分布已经验证。尤其旧策略外展动作可变、新策略固定前 −10°/后 +10°，可能始终无法满足目标连续性门限；此时会拒绝接管，应分析记录，不能靠强制切换或随意放宽门限解决。

接管时转换全部 20 帧关节偏移和上一动作坐标，保留 IMU 与时间顺序。当前上一动作从最后实际下发目标转换；更早的上一动作按旧策略映射和关节限位重建。锁定通道写 0，不除以零。历史中的命令改为本次请求的滚动任务命令，它们不表示启动阶段曾收到这些命令。首帧重新读取当前传感器。通过连续性检查后整体切换网络、归一化和动作映射，没有冷启动或额外 fade。

□ 在接管请求之前或计算过程中到达均优先处理。控制器状态 topic 为 `/<rolling_controller>/rolling_policy_state`，可取 `inactive/startup/pending/continuous/stop/rejected`。

## 权重与导出

当前候选为 `rolling_low_speed_20260912_065438/actor/checkpoints/000000081920/student_params`。其固定面板成功率 84.0%，vx MAE 0.1171 m/s，yaw MAE 0.05394 rad/s；它是复用面板选择出来的候选，未独立验证，且 DR strength 为 0。不要改用 critic 或最后一步的 `student_rtneural.json`。

在云端仓库更新代码后执行：

```bash
JAX_PLATFORMS=cpu python -m scripts.prepare_rolling_hardware_export \
  --run results/rolling_low_speed_20260912_065438/actor \
  --step 81920 \
  --out results/rolling_hardware_actor_81920
```

输出 `results/rolling_hardware_actor_81920.zip`，包含模型、校验向量、固定评估和哈希清单。脚本只加载/导出网络，不训练或仿真。使用原生 batchnorm 保留先减均值再缩放的数值顺序，避免低方差输入折叠归一化时的精度损失。合成向量校验仅证明导出数值一致，不证明真实闭环性能或树莓派 RTNeural 推理已经通过验证。

权重返回本地后，先完成树莓派原生推理一致性检查，再填写 `neural_controller_roll.ros__parameters.rolling_model_path`。原启动模型路径保持不变。

## 安装与验证

`install.py --stage /home/pi/<new_build_dir>` 创建独立构建目录；`--install` 会先检查机器人源文件与读取时的哈希一致，再备份并安装修改。二者都不替换模型、修改手柄 YAML、启动 ROS 控制器或使能电机。

无 ROS/电机检查在机器人上执行：

```bash
python3 test_gamepad.py
g++ -std=c++17 -I neural_controller/include test_rolling_history.cpp -o /tmp/test_rolling_history
/tmp/test_rolling_history
```

部署记录、机器人备份路径和构建结果见同目录 `deployment_status.json`。本地仅做源码与语法检查，没有进行训练、仿真或本地测试。
