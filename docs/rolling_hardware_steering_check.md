# 真机转向：下一轮怎么检查

先完成三组短对照，确认“自然偏航、转向响应不足、左右不对称”中的哪种情况更符合数据。暂时保持当前网络、动作映射和控制增益，不根据一次弱响应反转 yaw 或放大差动。

上一包已经证明：左转指令进入了正确 PPO，实际目标没有受到新增限幅，关节也产生了差动；撞椅前的航向响应仍接近零。撞椅后的运动不计为策略自主失败。

## 三组手柄操作

同一块平整地面、相同起始方向，预留足够的无障碍滚动和停靠空间。每组都从站立重新开始。速度杆始终居中，即 vx=0.60 m/s、vy=0。

| 录包标签 | 接管后转向杆 | 预期 yaw 指令 |
|---|---|---|
| straight | 居中 | 0 |
| left | 左推到底并保持 | +0.07 rad/s |
| right | 右推到底并保持 | −0.07 rad/s |

每次沿用现有流程：△ 启动 stand-to-roll，○ 请求持续滚动接管，然后固定该组转向输入约 5 秒。转向指令约 1 秒渐变到端点，分析时使用渐变结束后的完整滚动周期。随后用原 □ 请求停止流程，必要时使用原急停。碰撞、扶住、抬起等情况要说明，对应片段剔除，不能与自由滚动混在一起统计。若场地不足，提前结束，不追求固定秒数。

第一轮共三次；若表现不一致，再针对有疑问的组补做重复，不一开始就增加大量试验。

手机固定拍摄即可，尽量同时看到机器人朝向和地面参照线；这是对 IMU 航向结果的独立核对。不要手持追随机器人来判断转了多少角度。

## 录包

已于 2026-09-12 23:52 更新机器人实际使用的 `config_rollingquad_gamepad.yaml`：原 7 个话题扩展到 17 个，核对只有 record_topics 改变，安装目录的符号链接也指向新配置。原启动和录包按键逻辑保留，下次正常启动生效，无需重新编译。备份位于 `/home/pi/pupperv3-monorepo/ros2_ws/steering_recording_backup_20260912_235227`。

可以沿用原录包方式完成三组测试，每组单独保存。另已准备按试验标签命名的独立录包工具，源文件为 `hardware/rolling_command/record_steering_check.py`，机器人路径为 `/home/pi/record_steering_check.py`。它只订阅数据，不发送运动指令；与原录包器任选一种即可，避免重复录制同次数据。

如需使用带标签的独立录包工具，在单独终端运行：

```bash
source /home/pi/pupperv3-monorepo/ros2_ws/install/setup.bash
python3 /home/pi/record_steering_check.py straight
```

开始录制后再做对应动作，结束时在录包终端按 Ctrl+C，等待完成收尾。另两次分别将标签改成 left、right。文件默认保存在 `~/bags/rolling_steering/`，旁边的 `.trial.json` 只是预期试验标签，实际输入以 bag 内容为准。

没有脚本也可以直接运行以下命令，将输出目录的 left 改成对应组名；重做时用新的目录名：

```bash
source /home/pi/pupperv3-monorepo/ros2_ws/install/setup.bash
ros2 bag record -s mcap -o ~/bags/steering_left \
  /joy /joint_states /imu_sensor_broadcaster/imu /emergency_stop \
  /neural_controller_roll/rolling_cmd_vel \
  /neural_controller_roll/rolling_policy_state \
  /neural_controller_roll/request_rolling_policy \
  /neural_controller_roll/enable_policy \
  /neural_controller_roll/request_roll_to_stand \
  /neural_controller_roll/observation \
  /neural_controller_roll/policy_output \
  /neural_controller_roll/position_command \
  /neural_controller_roll/imu_latency_seconds \
  /neural_controller_roll/policy_inference_latency_seconds \
  /joy_util_node/sequence_state /joy_util_node/sequence_detail /rosout
```

结束后用 `ros2 bag info <输出目录>` 确认 /joy、rolling_cmd_vel、rolling_policy_state、IMU 和 joint_states 都有消息。某个话题为 0 条时说明没有采到，不能因为它出现在命令中就认定数据完整。零条请求消息也可能是该按钮没有触发，应结合测试动作解释。

## 数据回来后逐项检查

1. 对齐原始手柄轴、私有 rolling_cmd_vel 与 observation 中的命令。区分手柄符号/输入异常、命令渐变和策略接收的问题。
2. 从 IMU 四元数直接计算滚动轴水平航向 `atan2(-R[0,1], R[1,1])`，与陀螺仪重建值互相核对，再与固定视频核对。四元数和陀螺仪来自同一 IMU，视频才是独立参照；不用会随滚动翻转的普通身体 gyro_z 直接判断转向。
3. 将三组自由滚动数据按滚动相位和近似滚动速度对齐，比较左/右指令相对直行基线的航向变化。分离自然偏航与输入引起的转向，不能仅凭一组的绝对转角决定符号。
4. 对比网络目标、实际下发目标、左右关节实测差动；看幅度、相位滞后和限幅。joint_states 的 effort 需先确认驱动含义和有效性，不能默认它是准确的电机扭矩。
5. 若指令和动作均正确而转向响应不足，检查实际关节零位/外展姿态、有效 kp/kd、扭矩限制、供电和接触条件。只改变一个有证据支持的量；必要时再做受限幅度的差动标定与仿真/DR 调整。

这三组主要用于定位问题，不足以估计真机长期成功率。此前 SPI/NaN 故障是否消失仍需驱动日志确认；本方案不绕过原来的故障停机保护。
