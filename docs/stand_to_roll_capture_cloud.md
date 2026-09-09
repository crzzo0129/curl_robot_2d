# Capture 优先：云端训练

训练入口现在默认采用 capture + 1 圈保险；`--insurance-turns 2` 改为两圈。
BC 数据、网络和归一化不变。教师及数据采集工具的默认任务保留旧行为。

Capture 条件仍为 CEM 距离 <1.2、滚动角速度 >0.5 rad/s，连续 0.08 秒；
失败帧不能触发 capture。随后从 capture 当时的位置和累计转角重新计数：
净转角与沿世界 x 方向位移/滚动半径取较小者，达到目标圈数且未失败、仍正向旋转即结束。
这是短程保险，不保证过程中完全没有暂停，也不是直接交接另一策略。

Capture 奖励 10，保险完成奖励 2；滚动进度权重降至 0.1。
等待 capture 每秒扣 0.1；CEM 接近奖励改为 exp(-D) 的有界增量，
取消持续 orbit/sustain 奖励，避免鼓励拖长回合。
侧移终止放宽为 ±2m，侧移每秒惩罚为 0.5*(y/2)^2。
高度、轴偏转、异常接触、非有限状态的失败条件仍保留，回合上限仍为 500 步。
这些权重是待云端检验的初值，不保证成功率。

验收：capture >=80%，完成保险 >=80%，failed <=20%。
终端 Insurance 为完成保险的回合比例；sustained_success 在新任务中是同一指标的兼容别名。
post_capture_turns 是终止时的净保险圈数，经评估聚合后为回合平均。
旧任务和新任务的 sustained_success、reward、timeout 不能直接横向比较。

允许同阶段或前一阶段的旧 checkpoint 初始化，仍要求同一个 BC 文件。
新输出目录不可覆盖已有训练。推荐从 crouch 最佳策略重新训练 semi_stand：

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage semi_stand --static-curriculum \
  --preset smoke --steps 1000000 --insurance-turns 1 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --restore-checkpoint results/stand_to_roll_stability_1m/crouch/ppo_checkpoint/000000675840 \
  --learning-rate 5e-6 --max-kl 0.2 \
  --out results/stand_to_roll_capture_1m
```

确认云端数字 checkpoint 目录实际存在。训练结束生成 best_checkpoint.json：
仅在实际保存、且有对应评估步数的 checkpoint 中，依次按 capture、保险成功率、
成功 capture 回合的平均耗时排序。也记录包含 step=0 的最佳评估；
step=0 可能没有保存目录。params_final 和 summary 的 stage_passed 仍描述最终策略，
不要把最佳 checkpoint 的指标与最终参数混用。该排名只有评估样本内的含义。

未做本地运行、测试或仿真；请在云端确认新奖励与终止逻辑的实际效果。
