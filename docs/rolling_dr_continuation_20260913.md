# 持续滚动 25% DR 续训

这一级从当前部署的第 81,920 步 PPO 完整 checkpoint 继续，保留 Actor、特权 Critic
和归一化统计。训练使用 25% deploy DR、真实控制器 720 维观测、滚动快照启动，
并把 lateral 终止阈值由 0.20 m 放宽为 0.50 m。

```bash
cd /inspire/qb-ilm2/project/leverage-robot/ky26210/curl_robot_2d
# 先把本次源码同步到云端仓库
bash scripts/run_rolling_dr025_continuation.sh
```

默认输入：

- `results/rolling_low_speed_20260912_065438/actor/checkpoints/000000081920/student_params`
- `results/rolling_low_speed_20260912_065438/actor/checkpoints/000000081920/params`

若云端路径不同，用 `ROLLING_DR_STUDENT`、`ROLLING_DR_RESTORE` 和
`ROLLING_DR_STEERING_CALIBRATION` 覆盖；用 `ROLLING_DR_OUT` 指定新的输出目录。
脚本拒绝覆盖已有目录、日志或诊断包。

## 本级设置

- DR strength 0.25：摩擦、质量、惯量、COM、逐电机 KP/KD/力矩、延迟、deadline
  miss、电机零偏和编码器偏置都从名义值小幅展开。
- 保留当前策略使用的 `rollingquad_2_abd10_no_self_collision` 几何、高速零接触
  CEM reference 和 `rollingquad_abd10_high_speed_steering_v1.json` 转向标定。
- lateral failure 为 `abs(y-y0) > 0.50 m`；其他跌倒、非法接触和数值失败标准不变。
- 保持 `student_anchor_weight=0.02`，学习率 `3e-6`，KL 自适应上限也为 `3e-6`。
- 训练和评估都从缓存滚动快照开始。快照保留位置、速度、相位、指令和 20 帧历史；
  每次 reset 会用该 lane 的随机模型重新计算接触、运动学和加速度。
- 使用均匀快照采样；不采用此前没有改善低速误差的 tracking-focus 配方。

DR 下常规 eval 与训练使用不同的一批随机模型。`fixed-eval-envs` 设为 0，因为现有
固定评价器只支持名义物理；不要把 DR eval 成功率和旧名义 fixed panel 的 84% 直接
比较。训练结束先发回 `${ROLLING_DR_OUT}_diagnostics.zip`，根据分组速度/转向跟踪、
非 lateral 失败率和参数变化选择 checkpoint，不默认采用最后一步，更不要直接覆盖
当前真机模型。

若 25% DR 保持稳定，再另开输出目录升到 50%；不要在本级目录上覆盖续跑。
