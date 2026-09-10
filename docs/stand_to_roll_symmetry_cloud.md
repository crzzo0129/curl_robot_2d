# 强制左右对称动作对照

仅修改评估时施加的动作，不训练、不修改 checkpoint。
--symmetric-actions 将前左/前右、后左/后右各三个对应动作取平均。
当前 rollingquad_2_abd10 的镜像符号已包含在关节轴中，对应关节目标同号；
不对 abduction 额外取反。前后腿不绑定。控制器动作中心/尺度左右相同。
投影后的动作进入 env.step，历史观测记录实际施加的动作。
开关作用于整个回合，含 capture 后保险圈；不修改初始状态/噪声或模型几何。
动作对称不保证动力学轨迹对称，强制平均也可能削弱纠偏或起滚能力。

云端对同一个 checkpoint、seed 执行以下两组（输出目录必须为空）：

```bash
python -m scripts.render_stand_to_roll_checkpoint \
  --checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --load-eval --episodes 32 --seed 20000 \
  --out results/symmetry_original

python -m scripts.render_stand_to_roll_checkpoint \
  --checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --load-eval --episodes 32 --seed 20000 --symmetric-actions \
  --out results/symmetry_projected
```

比较 capture、保险成功率，以及交接 y、vy、axis 均值和 P95，
axis<5° 的比例仅以成功 capture 回合为分母。负载统计照常输出。
默认恢复 checkpoint 的任务配置，不传 --limit-torque，避免同时改变奖励配置。
若要视频：去掉 --load-eval，使用新 --out，保留 --symmetric-actions。
本地未运行验证，以上实验由用户在云端执行。
