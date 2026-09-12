# 标定 CEM 轨迹上的独立速度 estimator v3

依据 [cem_steering_calibration_20260912.md](cem_steering_calibration_20260912.md) 完成新数据采集、
100 epoch 监督训练及独立测试。新模型位于
`results/velocity_estimator_cem_calibrated_v3/model/estimator.npz`；未接入或训练 policy。
保留前滚线速度和轨迹转向率两个输出。本版实际产生了明显的持续左右转，而非仅设置转向命令。

## 标定接口与输入/标签契约

- CEM：`results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`。
- 标定表：`assets/controllers/rollingquad_abd10_high_speed_steering_v1.json`。
- 模型：`assets/rollingquad_description_2/mjcf/rollingquad_abd10_no_self_collision.xml`。
  严格沿用标定的关闭自碰撞、关闭 root damping、MuJoCo 3.9.0、cg20、1 ms 物理步和
  20 ms 控制步。无质量、摩擦或电机增益随机化；有随机初始世界朝向、±0.005 的初始关节
  位置/速度扰动和 observation 噪声/偏置。
- 通过生产接口验证模型/CEM 哈希、参考参数和关键物理设置。二维表输出已经是有效归一化差分，
  **不再乘 residual_gain 或 differential_scale**。速度、转向命令范围为 0.4111–0.811 m/s、
  ±0.02–±0.08 rad/s，另含直行；直行保留表中的小幅校正。
- 使用与标定一致的 `reference_action`、`reference_startup_scale_3d`、平面动作复制及两次
  [-1,1] 动作限幅，最后按关节/执行器范围限幅。这不同于旧采集器直接缩放物理目标的路径。
- 每帧原始 36 维 observation，选择 `0:6,12:36` 共 30 维；20 帧历史组成 600 输入。
  gyro、gravity、关节偏移和上次实际电机目标编码进入网络，command 不进入网络。
- **新版调用频率为 50 Hz**，20 帧覆盖 0.38 s 的首末时间间隔。旧 v2 为 52 Hz，不能直接按旧频率使用。
- 标签定义保持原约定：根原点水平速度在由 body y 导出的水平前向上投影得到有符号线速度；
  根水平 vx/vy 先经 tau=0.06 s 因果 EMA，再计算相邻方向夹角 / dt 得到轨迹转向率。
  低速 <0.08 m/s 等无效转向标签被 mask；没有用标定命令或轴 heading rate 替代监督真值。

## 数据与训练

| 数据 | 轨迹数 × 每条时长 | 指令保持时间 | 种子 | 用途 |
|---|---|---|---|---|
| smoke | 3 × 32 s | 4 s，固定顺序 | 20260914 | 采集链路验证，不参与训练/选模 |
| rollouts | 64 × 32 s | 4 s，随机变速/转向 | 20260915 | 按整条轨迹分为 52 训练、6 验证、6 测试 |
| scripted_test | 12 × 40 s | 5 s，固定顺序 | 20261915 | 独立测试，不参与训练或选模 |

随机指令直行/左转/右转概率为 40%/30%/30%，首段直行暖机。指令切换不重置状态、CEM 相位、
observation 历史或标签滤波器。独立固定顺序的 8 段如下：

| 段 | 前进速度命令 (m/s) | 转向命令 (rad/s) |
|---|---:|---:|
| 0 | 0.61105 | 0 |
| 1 | 0.4111 | 0 |
| 2 | 0.811 | 0 |
| 3 | 0.61105 | +0.08 |
| 4 | 0.61105 | -0.08 |
| 5 | 0.811 | +0.02 |
| 6 | 0.4111 | -0.02 |
| 7 | 0.61105 | 0 |

正式 76 条轨迹都通过本次有限状态、根高度 0.025–0.8 m、轴仰角 <0.5 rad、2 s 后平均前滚速度
>0.15 m/s 的检查。训练/独立测试最大轴仰角分别约 2.52° / 1.67°。自碰撞关闭，因此不把零接触
计数解释为打开自碰撞后的安全验证。

训练数据 102,400 帧、101,184 个窗口；MLP 600→256→128→2、ReLU、masked Huber、AdamW；
训练集归一化，训练/划分种子 7。100 epoch 中最佳验证 checkpoint 为 **epoch 69**。
本机训练含预处理、导出与内部评估约 249.5 s；正式采集约 172.6 s，独立采集约 77.4 s，
两路采集曾同时运行。这些是本次本机计时。

## 已完成的评估

| 评估 | 线速度逐帧 RMSE (m/s) | 转向逐帧 RMSE (rad/s) | 分段平均转向率 RMSE (rad/s) | 完整段累计转角 RMSE (度) |
|---|---:|---:|---:|---:|
| 随机留出 6 条 | 0.00675 | 0.03986 | 0.00407 | 0.831（4 s 段） |
| 独立固定顺序 12 条 | 0.00875 | 0.04340 | 0.00520 | 1.374（5 s 段） |

分段指标排除首个暖机段；平均转向率排除每段前 2 s，再对各 episode/段等权计分。
累计角误差对完整段的逐帧预测误差积分，仅计该段所有转向标签都有效的段。
随机测试有 42 段，独立测试有 84 段，以上全部完整有效。
这些统计不改变模型输出，也没有额外平滑训练标签或逐帧预览。

独立测试的分段平均转向率 MAE 为 0.00396 rad/s；恒零基线的分段 RMSE 为 0.04653 rad/s，
observation 的世界竖直 gyro 投影基线为 0.03054 rad/s。按实际段均值绝对值 >=0.02 rad/s
筛选出的 36 个明确转向段，预测左右符号全部正确；这不是所有瞬时帧都正确的声明。
指令切换后前 0.5 s 的逐帧速度/转向 RMSE 为 0.01412 m/s、0.06039 rad/s。

独立测试中实际持续转向与估计均值（排除各段前 2 s，对 12 条轨迹平均）：

| 转向命令 | 实际轨迹转向率 (rad/s) | 估计转向率 (rad/s) | 完整 5 s 实际净转角 (度) |
|---:|---:|---:|---:|
| +0.08 | +0.08308 | +0.08366 | +23.31 |
| -0.08 | -0.08676 | -0.07750 | -23.14 |
| +0.02 | +0.01503 | +0.01700 | +1.20 |
| -0.02 | -0.02182 | -0.01591 | -6.06 |

右转存在幅度低估：负命令段分段均值 RMSE 为 0.00843 rad/s，正命令段为 0.00280 rad/s。
直行段 estimator 均值 RMSE 为 0.00323 rad/s，恒零基线为 0.00196 rad/s，仍有小幅残余偏置。
原始轨迹转向标签仍有明显接触/滚动短时波动，不能把分段 0.00520 的误差解释成逐帧误差。

本次与 v2 的模型碰撞设置、参考动作执行路径、采样频率和动力学随机化范围不同，未进行
直接公平的旧模型对照，不能将两版测试误差差值全部归因于标定表。
未做实机、跨摩擦、强打滑或接入 policy 的验证；运行时没有低速有效性/置信度输出头。

## 复现与使用

在项目根目录执行，重跑时使用新的结果路径（采集器拒绝覆盖已有 NPZ）：

```bash
python -m scripts.collect_calibrated_rolling_velocity_data --out results/velocity_estimator_cem_calibrated_v3/rollouts.npz --episodes 64 --duration 32 --workers 4 --command-mode random --command-interval 4 --seed 20260915
python -m scripts.collect_calibrated_rolling_velocity_data --out results/velocity_estimator_cem_calibrated_v3/scripted_test.npz --episodes 12 --duration 40 --workers 2 --command-mode scripted --command-interval 5 --seed 20261915
python -m scripts.train_rolling_velocity_estimator --data results/velocity_estimator_cem_calibrated_v3/rollouts.npz --out results/velocity_estimator_cem_calibrated_v3/model --epochs 100 --seed 7
python -m scripts.evaluate_rolling_velocity_estimator --model results/velocity_estimator_cem_calibrated_v3/model/estimator.npz --data results/velocity_estimator_cem_calibrated_v3/rollouts.npz --out results/velocity_estimator_cem_calibrated_v3/evaluation
python -m scripts.evaluate_rolling_velocity_estimator --model results/velocity_estimator_cem_calibrated_v3/model/estimator.npz --data results/velocity_estimator_cem_calibrated_v3/scripted_test.npz --out results/velocity_estimator_cem_calibrated_v3/scripted_evaluation --all-episodes
python -m scripts.analyze_rolling_estimator_turns --data results/velocity_estimator_cem_calibrated_v3/scripted_test.npz --predictions results/velocity_estimator_cem_calibrated_v3/scripted_evaluation/predictions.csv --out results/velocity_estimator_cem_calibrated_v3/scripted_evaluation --plot
```

推理仍只需要 NumPy，每 20 ms 输入一个真实 36 维 observation，`update` 在积满 20 帧前返回 None，
之后返回 `[rolling_speed_m_s, trajectory_turn_rate_rad_s]`。新回合调用 `reset()`。

```python
from curl_robot_2d_mjx.rolling_velocity_estimator import RollingVelocityEstimator
estimator = RollingVelocityEstimator("results/velocity_estimator_cem_calibrated_v3/model/estimator.npz")
# 每 20 ms：prediction = estimator.update(raw_observation_36)
```

`python -m unittest tests.test_rolling_velocity_estimator tests.test_steering_calibration -v`
共 20 项通过，包括新增标定采集实际动作历史/时间契约、错误模型和指令域拒绝加载检查。
训练导出通过 JAX/NumPy 推理一致性检查；分段报告另以已知恒定误差验证均值与累计角误差计算。

## 结果入口

- [模型](../results/velocity_estimator_cem_calibrated_v3/model/estimator.npz)
- [独立测试预览：轨迹、逐帧与持续转向](../results/velocity_estimator_cem_calibrated_v3/scripted_evaluation/calibrated_estimator_preview.png)
- [独立测试逐帧指标](../results/velocity_estimator_cem_calibrated_v3/scripted_evaluation/metrics.json)
- [独立测试分段指标](../results/velocity_estimator_cem_calibrated_v3/scripted_evaluation/turn_summary.json)
- [独立测试各段明细](../results/velocity_estimator_cem_calibrated_v3/scripted_evaluation/turn_segments.csv)
- [训练指标与模型来源](../results/velocity_estimator_cem_calibrated_v3/model/metrics.json)

所有模型、轨迹及图像均为本地 results 产物，需要单独同步；旧 CEM 模型和作废 neural-policy
数据均未被覆盖，也未用于这次训练。
