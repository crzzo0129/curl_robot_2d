# 静止启动监督：云端最小流程

目标仍然是同一个 actor 从静止 stand 起滚。当前先补齐静止 compact
附近的启动能力，之后再扩展站姿。本轮没有运行本地测试、仿真或训练。

## 1. 原始 controller 采集启动全过程

在 curl_robot_2d 目录运行：

```bash
python -m scripts.collect_cem_startup_data \
  --out results/cem_startup_data
```

默认 32 个回合，每回合从 alpha=0～0.1 的 compact 附近姿态、严格零速度
开始，关节扰动 ±0.01 rad，最长 10 秒。使用原始 CEM 振荡器、相位反馈、
目标生成与足间距投影，不使用轨迹查表教师。默认相位为零，不增加未经验证
的启动 ramp。与旧采集器的区别：不先把关节摆成 phase=0 的动作；复位姿态
与学生 compact 评估一致。目标每 20 ms 更新并保持，匹配学生接口；旧采集器
每物理子步更新。因此原始 controller 的既有成功不能代替本次预检。

检查 summary.json：至少 80% 回合无失败并完成一圈有效滚动才导出
startup_bc.npz。失败时保留报告，不导出数据；不要跳过门槛，应检查原始
controller 的实际启动初始化、控制频率和物理配置。CPU MuJoCo 成功仍需
后续 MJX 学生评估确认，不能视为 full-stand 成功。

每个成功回合从第一帧保存 PRE-action 720 维观测及实际执行的 12 维动作。
历史按部署方式初始化、不丢弃前 19 帧；各回合历史独立。前 2 秒标为启动
窗口，可用 --startup-seconds 调整。此文件仅用于 BC，不能替换稳态
cem_cycles.npz 匹配器数据。

## 2. 在已有 DAgger 学生上补学启动

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage bc \
  --bc-params results/stand_to_roll_dagger/bc_params \
  --startup-data results/cem_startup_data/startup_bc.npz \
  --bc-learning-rate 3e-5 \
  --out results/stand_to_roll_startup
```

继续使用旧学生的 v2 网络和冻结归一化。每批约 50% 样本来自启动窗口，
另外 50% 来自原有稳态数据及新轨迹的后半段。新轨迹按完整 episode 划分
训练与验证，历史不会跨回合。bc_summary.json 单独报告 startup_validation_rmse。
误差下降只说明动作拟合，不是起滚验收；尤其旧稳态验证数据可能已在先前
DAgger 中使用过，不是新独立验证。若不传 --bc-params，则重新初始化网络
并用新训练集计算归一化。

## 3. 纯静止起点闭环验收

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage compact --eval-only --static-eval \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --out results/stand_to_roll_startup
```

这里没有动态快照，复位速度严格为零；姿态仍为 compact 附近 alpha=0～0.1。
查看 eval_bc_compact/bc_closed_loop_eval.json 的 sustained_success、failed、
capture_time_s 与 avg_episode_length。以持续成功率 >=80%、失败率 <=20%
作为初步门槛，换 seed 与新 out 目录复测后再进入更难姿态。

同时用 --stage rolling_orbit --eval-only（不带 --static-eval）检查持续滚动
是否遗忘。失败就保留旧 DAgger 参数，不直接启动大规模 PPO。
## 4. 验收后直接开始静止姿态课程

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage slightly_open --static-curriculum --preset smoke \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --learning-rate 5e-6 --max-kl 0.2 \
  --out results/stand_to_roll_static_ppo
```

--static-curriculum 允许 slightly_open 直接从 BC 初始化，不需要 compact
PPO checkpoint。训练和评估均为零初速度、无快照、无观测噪声；保留原有
关节位置小扰动及姿态 alpha 范围。后续 crouch、semi_stand、full_stand
仍恢复前一姿态阶段的具体 checkpoint，继续传同一个 BC 文件和
--static-curriculum。不要恢复补学启动之前的 PPO checkpoint。
