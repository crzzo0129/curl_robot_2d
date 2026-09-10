# PPO 自主学习左右对称动作

--symmetry-reward 0.05 启用小幅正奖励，不平均、不绑定左右输出。
比较控制器最终目标角：前左/前右、后左/后右的外展、髋、膝共六对；
当前模型对应关节坐标同号，前后不比较。它奖励目标动作对称，不要求实际力矩相等。

每步奖励 = w * dt * exp(-(max(abs(q_target_left-q_target_right))/sigma)^2)。
w 由 --symmetry-reward 控制，sigma 由 --symmetry-scale-rad 控制，二者独立。
当前 sigma 默认 0.01rad：最大左右角差 0.003rad 时获得约 91.4% 奖励，
0.005rad 时 77.9%，0.01rad 时 36.8%，0.02rad 时 1.8%。
只有六对目标角全部相等才满分。较窄范围也会使远离对称的动作得到很弱信号，
不能保证仅凭此软奖励实现严格相等。
以下示例 w=0.05：
dt=0.02s，最大每步 0.001 分、每秒 0.05 分；10 秒回合最多 0.5 分。
仅在 capture 前生效，包含触发 capture 的一步，失败步不发放。
capture 后保险圈不再发放。等待 capture 的原有每秒 0.1 扣分仍保留，
因此仅靠原地对称等待不能抵消等待扣分。

从原始成功策略开始，不使用蒸馏学生或 --symmetric-actions：

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage full_stand --static-curriculum --limit-torque \
  --symmetry-reward 0.05 --symmetry-scale-rad 0.01 \
  --preset smoke --steps 1000000 --insurance-turns 1 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --restore-checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --learning-rate 1e-6 --max-kl 0.2 \
  --num-evals 11 --max-success-drop 1.0 \
  --out results/stand_to_roll_symmetry_narrow_1m
```

成功率下降不触发停训，KL/非有限数值保护保留。使用新版有界交接奖励、
3Nm 硬限幅，新增力矩超限惩罚仍为零；其他奖励保持不变。
如需严格对照，另一组用相同命令、相同 seed，将 --symmetry-reward 改为 0，换新输出目录。

日志 Symmetry 给出回合奖励及 capture 前六对目标角差的时间加权 RMS（rad）。
同时查看成功率和交接 axis、vy、y；动作更对称不自动等同于交接更好。
最终用不带 --symmetric-actions 的独立评估检验所选 checkpoint。
该奖励不保证严格对称，允许策略保留必要的左右差异。
本地未运行验证，训练与评估在云端执行。
