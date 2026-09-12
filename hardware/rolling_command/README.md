# 持续滚动 PPO 接管与手柄控制

这份实现基于 2026-09-12 从 `pi@pupper.local` 读取的真实控制器版本，保留 PS5 经 8BitDo 的按键映射、Walking 返回检查和现有 pitch-gated roll-to-stand。

当前已安装 v5 平滑接管逻辑并绑定第 81,920 步 PPO 模型，机器人原生 RTNeural 校验通过。64 组输入的最大动作误差为 5.4389e-7，网络推理中位耗时 0.1495 ms。该检查使用合成观察向量，不是实际滚动接管测试；没有启动控制器或使能电机。模型校验记录见 `model_binding_report.json`；本次更新的仿真、构建和备份记录见 `smooth_handoff_validation.json`。

模型路径：`/home/pi/pupperv3-monorepo/ros2_ws/src/neural_controller/models/rolling_command_ppo_000000081920.json`。
已绑定的配置：`/home/pi/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/config_rollingquad_gamepad.yaml`，安装目录的配置是指向它的符号链接。
绑定前配置备份：`/home/pi/pupperv3-monorepo/ros2_ws/rolling_actor_binding_backup_20260912_202425/`。

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

○ 使用上升沿触发，请求最多等待 10 秒。启动历史须连续至少 50 次以 15–25 ms 更新，IMU 净累计滚动至少一圈，命令在最近 0.25 秒内有效。之后连续两个策略 tick 都须满足：完整 pitch 在 −60°～+60°、滚动轴离水平不超过 15°、身体 y 轴角速度绝对值 0.5～12 rad/s、PPO 直行候选目标与上一实际目标最大差 ≤0.12 rad。网络输出须有限且符合 tanh/锁定通道约定。

进入接管窗口后执行：

1. 15 个 20 ms tick（0.30 秒）的 smoothstep 混合。启动网络和 PPO 都继续从真实状态计算动作，不冻结旧动作；关节目标和增益一起混合。
2. 以 vx=0.60、vy=0、yaw=0 直行，滚动轴与角速度条件连续满足 15 tick 后允许加入手柄命令。混合结束后 2 秒仍未满足，自动请求原 pitch-gated roll-to-stand。
3. vx 每秒最多变化 0.15 m/s，yaw 每秒最多变化 0.07 rad/s；最大转向约 1 秒达到。手柄回中及左右反向也经过限速。过渡期间会短暂经过训练非零 yaw 区间以下的数值。
4. 混合及后续 PPO 的每步实际目标变化都受 `rolling_handoff_max_target_delta_rad` 限制，默认 0.12 rad / 20 ms。高速度时该限幅可能影响跟踪，需要按实际数据评估。

接管前转换全部 20 帧关节偏移和上一动作坐标，保留 IMU 与时间顺序，候选历史命令设为直行 0.60 m/s。混合期间两套策略维护各自真实历史，上一动作由实际下发的混合/限速目标反算。PPO 锁定外展通道仍写 0，启动策略保留自身外展动作历史。后续命令只写当前帧，不追溯改写历史。原权重未重训。

□ 和急停继续优先。请求停止后 yaw 平滑回零，停止策略仍按原 pitch 门限接管。状态 topic `/<rolling_controller>/rolling_policy_state` 新增 `blending/settling/command_ramp`；原 `inactive/startup/pending/continuous/stop/rejected` 保留。`continuous` 表示已经接稳且当前命令斜坡到达目标。

实现与仿真对照见 `../../docs/rolling_smooth_handoff_20260912.md`。此前仿真把 MuJoCo 惯性主轴角速度作为躯干角速度使用，旧“接管被拒绝/反向转弯”的结论已撤回；不能据此判断真机策略缺少接管能力。

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

本次权重已返回并完成树莓派原生推理一致性检查，已填写 `neural_controller_roll.ros__parameters.rolling_model_path`。原启动模型路径保持不变。更换权重时应重新核对哈希与原生推理；`verify_rtneural` 目录提供只进行网络推理、没有 ROS 或电机接口的校验程序。

沿用已存在的手柄启动入口：

```bash
cd /home/pi/pupperv3-monorepo/ros2_ws
source install/setup.bash
ros2 launch neural_controller rollingquad_gamepad.launch.py
```

按原流程进入 Walking 后，△ 启动 stand-to-roll；○ 请求 PPO 持续滚动；□ 请求停止策略。控制器报告 `PPO handoff blend started` 表示开始过渡；手柄节点报告 `PPO continuous rolling takeover confirmed` 表示已接稳并完成当前命令斜坡。`pending` 表示仍在等条件，`rejected` 表示本次请求已拒绝，不能视为 PPO 已运行。

## 安装与验证

`install.py --stage /home/pi/<new_build_dir>` 创建独立构建目录；`--install` 会先检查机器人源文件与读取时的哈希一致，再备份并安装修改。二者都不替换模型、修改手柄 YAML、启动 ROS 控制器或使能电机。

无 ROS/电机检查在机器人上执行：

```bash
python3 test_gamepad.py
g++ -std=c++17 -I neural_controller/include test_rolling_history.cpp -o /tmp/test_rolling_history
/tmp/test_rolling_history
```

部署记录、机器人备份路径和构建结果见同目录 `deployment_status.json`。本地仅做源码与语法检查，没有进行训练、仿真或本地测试。

## 真机转向对照录包

2026-09-12 已把机器人实际使用的录包配置从 7 个话题补到 17 个，下次正常 launch 生效。新增原始手柄、实际转向命令、PPO 接管状态、IMU、关节反馈及日志。三组操作和录包方法见 [真机转向检查](../../docs/rolling_hardware_steering_check.md)，配置备份与哈希见 `steering_recording_update.json`。只改变录包话题列表，原运动控制与录包启动/停止方式保留。
