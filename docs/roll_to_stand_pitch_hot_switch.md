# 滚动策略 → roll-to-stand：话题预约、pitch 门控热切换

当前实现适用于 `controller_backend: rtneural`，两个网络均为 36×20 输入、12 个位置动作。原有 CEM 自动展开功能独立，不能同时启用。

## 1. 云端导出

在更新后的训练项目目录执行：

```bash
python -m scripts.export_transition_rtneural \
  results/roll_to_stand_absolute_guided_hold_robust_v1/brake_full/params_final \
  results/roll_to_stand_absolute_guided_hold_robust_v1/brake_full/policy_rtneural.json \
  --config results/roll_to_stand_absolute_guided_hold_robust_v1/brake_full/deployment_config.json \
  --runtime pitch_hot_switch
```

将 JSON 复制到机器人。必须搭配本次更新后的 C++ 控制器；旧控制器不会实现 JSON 所声明的限速机制。

## 2. 机器人端配置与编译

在现有 neural_controller 的 `ros__parameters` 下增加这些参数。`model_path` 仍指向当前滚动策略，`roll_to_stand_model_path` 替换成机器人上的真实绝对路径：

```yaml
controller_backend: rtneural
roll_to_stand_model_path: /absolute/path/policy_rtneural.json
neural_stop_pitch_deg: 90.0
neural_stop_pitch_tolerance_deg: 5.0
roll_to_stand_enabled: false  # 原 CEM 手工展开开关
enable_body_angle_estop: false  # 整圈滚动不能使用站立姿态角急停
```

保留原滚动策略的启动参数和 repeat_action；接管单独按 JSON 的 policy_frequency_hz 调度。机器人端在 ROS 工作空间编译并重新启动控制器：

```bash
colcon build --packages-select neural_controller
source install/setup.bash
```

## 3. 滚动中发送请求

控制器名称为 `neural_controller` 时：

```bash
ros2 topic pub --once /neural_controller/request_roll_to_stand std_msgs/msg/Empty '{}'
```

如控制器有 namespace 或其他名称，相应修改话题前缀。

- 收到请求后记录 `Roll-to-stand armed`，滚动策略继续执行。
- 每个硬件更新周期检查 pitch，进入 +85° 到 +95° 时立即执行新网络，日志记录 `HOT SWITCH` 和实际 pitch。
- pitch 定义为 `atan2(R20, R22)`，直立为 0、抬头为正，与训练快照筛选一致；不是范围仅 ±90° 的 Euler asin。这里没有额外的旋转方向门控。
- 启动/fade-in 尚未完成、急停中或已切换后的请求会被忽略，不预存到下次启动。
- 切换没有 fade-in，不插值回站姿。新策略持续执行，不自动切回滚动，也不另行判断站稳后停网络。

## 接管契约

动作先转换为绝对关节目标并裁剪关节范围，再以**上一条实际下发目标**为中心限速。当前 JSON 为 6 rad/s × 0.02 s，即每次策略推理最多变化 0.12 rad。该限速是训练的一部分，与 fade-in 无关；不使用实测关节位置替代初值。

接管后平均 50 Hz 推理（受硬件周期量化，例如 520 Hz 控制循环下有一拍内的调度抖动），延迟时不连续补发多个动作。限速仅在推理时执行一次。kp/kd、动作中心/比例、关节范围和观测裁剪取自新 JSON；现有 gain_multiplier 仍生效。

观测历史按训练 `reset_from_roll_state(actor_history=None)` 冷初始化，当前帧读取真实传感器，第一帧 last_action 由最后滚动目标转换到新策略动作坐标。之后 last_action 保存新网络的原始输出。命令速度固定为零、desired world z 固定为 [0,0,1]。

本次只修改代码，未在本地编译、导出或仿真。机器人/云端验收应确认：预约时仍滚动；日志切换角在窗口内；接管无初始化动作；`position_command` 的连续策略目标差不超过 0.12 rad；最终站立表现由实际接管测试判定，不能仅凭 pitch 命中保证成功。
