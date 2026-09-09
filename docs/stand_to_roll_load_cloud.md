# 先评估负载，再做力矩微调

本地未运行测试、仿真或训练。以下全部在云端执行。

对同一旧 checkpoint、同一 seed 分别执行基线和限幅评估：

```bash
python -m scripts.render_stand_to_roll_checkpoint \
  --checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --load-eval --episodes 32 --seed 20000 \
  --out results/full_stand_load_baseline

python -m scripts.render_stand_to_roll_checkpoint \
  --checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --load-eval --episodes 32 --seed 20000 --limit-torque \
  --out results/full_stand_load_3nm
```

不更新参数、不渲染视频。各目录生成 load_report.json 和 episodes.npz。
报告的关节峰值是所有有效物理子步的最大绝对值；RMS 和时间占比按有效回合时长加权。
超过 2Nm 的比例按每个关节统计，触顶指 >=2.99Nm。
32 回合没有失败不等于所有状态都可靠。检查 finite_loads、成功率和最大力矩。

--limit-torque 对一关节一执行器的直接关节传动设置执行器 force range，
按 gear 换算至关节输出端 3Nm，保留模型原先更严格的限制。
统计使用 qfrc_actuator；不含接触/关节约束等外载造成的结构应力。
微调新增每秒惩罚：sum_j(max(abs(tau_j)/2-1,0)^2)，系数 0.1，物理子步平均后乘控制周期。
原有较小整体力矩惩罚保留。2Nm 是软目标，不能保证从不超出。

接触统计按实际 friction cone 解码约束法向力，每个物理子步取地面接触：
单接触点峰值、所有地面接触法向力之和的峰值，以及法向力时间积分。
总积分包含支撑力，不是仅碰撞冲量。当前不设接触硬上限或新惩罚，
待这次量级出来再选宽松范围，避免编造硬件安全阈值。

对比评估后再决定是否续训；命令如下：

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage full_stand --static-curriculum --limit-torque \
  --preset smoke --steps 1000000 --insurance-turns 1 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --restore-checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --learning-rate 1e-6 --max-kl 0.2 --num-evals 11 --max-success-drop 0.10 \
  --out results/stand_to_roll_torque_guarded_1m
```

力矩微调时 best_checkpoint 优先 capture、保险成功率，再比较超 2Nm 占比、
平均平方力矩和 capture 耗时。后续续训也要带 --limit-torque；
视频及负载评估则默认读 checkpoint 保存的任务限制。
训练日志 eval/episode_*peak* 是回合峰值的均值，独立负载报告给出跨回合最大值。
