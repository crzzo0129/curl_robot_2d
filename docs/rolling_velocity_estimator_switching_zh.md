# CEM 回合内变速、转向与独立 estimator v2

本版按用户要求，让 CEM 在同一条轨迹内改变速度和转向，然后训练独立状态估计器。
不训练控制 policy。旧的固定振幅/差分 CEM v1 保留为对照；被拒绝的 neural policy
数据仍为作废状态，不参与本次训练。

## 沿用的项目接口

来源：[3d_forward_velocity_command_tracking_zh.md](3d_forward_velocity_command_tracking_zh.md)，
实现位于 `curl_robot_2d_mjx/environment_3d.py`：

- `forward_command_to_target_scale_3d(np, v_cmd)`：按已验证查表把速度命令转成 CEM 振幅。
  查表范围约 0.4111–0.8110 m/s，对应振幅 0.36–1.0；本次使用 0.45–0.80 m/s。
- `steering_prior_3d`：`raw_a = clip(5 * turn_cmd, -0.5, 0.5)`，
  再乘文档中的 `residual_gain=0.15` 和 `differential_scale=0.25`，
  形成 `[a,a,-a,-a,a,-a,-a,a]` 的 8 维差分。转向命令幅值 0.02–0.08 rad/s
  对应 a=0.00375–0.015；没有额外 policy residual。
- CEM 原有相位锁定、脚部间隙处理、0.25 s 启动斜坡和物理步更新继续使用。

控制器与模型保持 CEM v1 的来源：
`results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`
和 `assets/rollingquad_description_2/mjcf/rollingquad_abd10.xml`，选择性自碰撞开启。
Kp=5、Kd=0.1、力矩限制 3 Nm、observation 52 Hz、每帧 20 个物理步。

文档中的转向命令针对滚动轴 heading 的控制先验，不是保证跟踪的轨迹转向率。
**本 estimator 的标签仍是根节点水平速度方向的变化率**，不使用命令当真值，
也不改成滚动轴 heading rate。速度查表源于固定指令的长轨迹测量，切换后的瞬态
速度可能明显偏离命令；这些瞬态也保留为训练数据。

## 指令序列

采集脚本新增 `--command-mode`，旧默认 `fixed` 保持 v1 可复现：

| 模式 | 作用 |
|---|---|
| `fixed` | 每个 episode 固定振幅和差分，复现 CEM v1 |
| `random` | 每 1.5 s 换速度和转向；40% 直行、30% 左、30% 右；首段 0.60 m/s 直行 |
| `scripted` | 固定顺序的减速、加速、左右反转及联合切换，独立评估用 |

指令边界采用整数 observation 步，不重置 CEM 振荡器、身体相位、标签滤波器或
observation 历史。速度目标和差分在边界切换，身体按实际动力学响应。
训练指令每 78 个 observation 步切换；独立测试每 104 步切换。

默认范围下，scripted 每轮 8 段为：

| 段 | 速度命令 (m/s) | 转向命令 (rad/s) |
|---|---:|---:|
| 0 | 0.625 | 0 |
| 1 | 0.450 | 0 |
| 2 | 0.800 | 0 |
| 3 | 0.625 | +0.08 |
| 4 | 0.625 | -0.08 |
| 5 | 0.800 | +0.02 |
| 6 | 0.450 | -0.02 |
| 7 | 0.625 | 0 |

`--speed-range`、`--turn-range`、`--command-interval` 可调整；范围检查阻止越过
上述已验证查表/转向命令范围。改变查表对应的 CEM 文件后需要重新验证速度映射。

## 运行命令

在项目根目录运行；依赖与 v1 相同，`--plot` 额外需要 matplotlib。
如尚未安装，先执行 `python -m pip install matplotlib`。

```bash
# 64 x 12 s 随机切换轨迹：每条 8 段；52/6/6 条训练、验证、测试。
python -m scripts.collect_rolling_velocity_data --command-mode random --episodes 64 --duration 12 --workers 4 --seed 20260913 --out results/velocity_estimator_cem_switching_v2/rollouts.npz
python -m scripts.train_rolling_velocity_estimator --data results/velocity_estimator_cem_switching_v2/rollouts.npz --out results/velocity_estimator_cem_switching_v2/model --epochs 100
python -m scripts.evaluate_rolling_velocity_estimator --model results/velocity_estimator_cem_switching_v2/model/estimator.npz --data results/velocity_estimator_cem_switching_v2/rollouts.npz --out results/velocity_estimator_cem_switching_v2/evaluation --plot

# 完全独立的 8 x 16 s 固定顺序测试：新种子、2 s 指令间隔，不用于选模。
python -m scripts.collect_rolling_velocity_data --command-mode scripted --command-interval 2 --episodes 8 --duration 16 --workers 2 --seed 20261913 --out results/velocity_estimator_cem_switching_v2/scripted_test.npz
python -m scripts.evaluate_rolling_velocity_estimator --model results/velocity_estimator_cem_switching_v2/model/estimator.npz --data results/velocity_estimator_cem_switching_v2/scripted_test.npz --out results/velocity_estimator_cem_switching_v2/scripted_evaluation --all-episodes --plot
```

训练仍为 20 帧历史、600→256→128→2 MLP，训练数据归一化，验证 loss 选模。
每帧 command 写入原 deployment frame 的 `6:9`，但 estimator 排除整个 `6:12`；
新增诊断字段也不进入输入或监督标签。

## 评估和追溯

数据增加 `command`、`command_segment`、`command_age_s`、速度/转向 command 差分、
当前 target scale 和 steering amplitude；每条轨迹 metadata 保存完整指令序列。
原有根节点状态、电机目标和相位记录保留，模型 provenance 包含采集参数及源文件 SHA256。

除总误差外，`metrics.json` 新增：

- `command_transition_0p5s`：非首段的指令切换后前 0.5 s。
- `speed_increase_transition` / `speed_decrease_transition`：按指令升降分组的切换窗口。
- `turn_reversal_transition`：相邻非零转向指令换符号后的切换窗口。
- `after_command_transition`：非首段且指令已保持至少 0.5 s；不宣称此时已经物理稳态。
- `commanded_straight` / `commanded_left` / `commanded_right`：按驱动指令分类；
  原来的 `left_turn` / `right_turn` 则仍按实际轨迹转向标签分类。

`predictions.csv` 含预测、真值、mask、时间和命令。`command_response.csv` 报告每条轨迹
每段指令保持 0.5 s 后的实际速度/转向均值，帮助检查是否真的发生变速与变向。
`prediction_preview.png` 显示第一条评估 episode；灰色虚线仅为 CEM 驱动命令。
转向标签有因果 EMA 的 0.06 s 时间常数；没有在出图时另外平滑真值或预测。

检查命令：`python -m unittest tests.test_rolling_velocity_estimator -v`。
14 项测试覆盖已有标签/动作契约、文档查表和转向 gain、序列可复现性、切换边界、
相位/历史连续性以及 command 不影响实际监督标签。

这一评估依然限于仿真内的 CEM 前滚运动。固定顺序测试有独立轨迹和指令间隔，但使用相同
CEM 参考和物理随机化范围；它不是跨控制器、强打滑或实机验证。

## 本次结果（2026-09-11）

按上述命令完成采集和 100 epoch CPU 训练，最佳验证 checkpoint 为 epoch 24。
训练/划分 seed=7；随机轨迹 768 s、采集用时约 118 s；独立固定顺序轨迹 128 s，
采集约 38 s（两路采集同时运行）；训练及评估约 81 s。这些仅是本工作站本次计时。

| 评估集 / 模型 | 线速度 RMSE (m/s) | 轨迹转向 RMSE (rad/s) |
|---|---:|---:|
| 随机切换留出 6 条 / v2 | 0.01395 | 0.04740 |
| 独立固定顺序 8 条 / v2 | 0.01737 | 0.04904 |
| 同一独立固定顺序 8 条 / 旧 CEM v1 | 0.02336 | 0.05582 |
| 独立固定顺序切换后前 0.5 s / v2 | 0.01911 | 0.04970 |
| 同一切换窗口 / 旧 CEM v1 | 0.02809 | 0.05671 |

随机留出集有 3,630 个线速度 / 3,592 个有效转向标签；独立固定顺序测试分别为
6,504 / 6,454 个。预先登记的随机留出门槛（速度 RMSE <0.05 m/s、转向 <0.10 rad/s）通过。
旧 CEM v1 仅用于同测试集对照，没有使用作废 neural-policy 模型。

独立轨迹按段统计、排除每段前 0.5 s 后，+0.08 转向命令段的实际轨迹转向均值约
+0.024 rad/s，-0.08 段约 -0.030 rad/s。实际运动发生了方向切换，但没有精准跟踪
指令幅值；训练目标是实际运动。转向曲线仍存在接触引起的瞬时尖峰，0.049 rad/s
总 RMSE 不能解释成对每个微小慢转都相对准确。

随机轨迹全部原始采样的平均前滚速度约 0.647 m/s；瞬时峰值 1.288 m/s，不能用来
代表持续速度。随机训练采集的自碰撞物理步比例为 0；这是该批次的诊断结果。

- [v2 模型](../results/velocity_estimator_cem_switching_v2/model/estimator.npz)
- [独立测试对照曲线](../results/velocity_estimator_cem_switching_v2/scripted_evaluation/prediction_preview.png)
- [独立测试指标](../results/velocity_estimator_cem_switching_v2/scripted_evaluation/metrics.json)
- [逐帧预测和命令](../results/velocity_estimator_cem_switching_v2/scripted_evaluation/predictions.csv)
- [逐段真实运动响应](../results/velocity_estimator_cem_switching_v2/scripted_evaluation/command_response.csv)

训练数据、测试数据和模型均位于本地 results 目录，按项目惯例被 Git 忽略，需单独同步。

## 转向信号复核（2026-09-12）

用户指出转向曲线主要像噪声。复核同一批 8 条独立 scripted 轨迹后，确认持续转弯
确实很弱，不能把上面的总 RMSE 门槛通过解释为已经验证了明显转弯的估计能力。

| 驱动转向命令 (rad/s) | 实际段均值 (rad/s) | 估计段均值 (rad/s) | 完整 2 s 段的净转角 (度) | 段内真值波动 RMS (rad/s) |
|---|---:|---:|---:|---:|
| +0.08 | +0.02420 | +0.02490 | +2.00 | 0.12386 |
| -0.08 | -0.03039 | -0.02429 | -2.71 | 0.10354 |
| +0.02 | +0.00332 | +0.00085 | +0.305 | 0.11164 |
| -0.02 | -0.00738 | -0.00501 | -0.822 | 0.07002 |

表中段均值及波动均排除每段前 0.5 s，再对 8 条轨迹等权平均；波动 RMS 是各段
真值减去本段均值后的 RMS。净转角则对完整 2 s 的原有因果滤波转向标签积分，
这四类完整转向段标签全部有效。没有更换训练标签或对展示曲线额外低通。

这说明有微弱的左右转信号，但 2 s 只转约 2–3 度，短时摆动幅度又是持续转向的
数倍。尖峰已经出现在仿真真值中，并非全由 estimator 产生；具体接触事件与尖峰
的因果对应尚未逐事件验证。段均值估计仍有价值：+0.08 / -0.08 两段跨 episode 的
均值误差 RMSE 分别为 0.00553 / 0.00801 rad/s。因此逐帧总 RMSE 大于持续转向均值
也不等于模型完全没有学到转向；它说明需要分别评估瞬时波动和持续转向。

源头限制是 `steering_prior_3d` 原本为策略残差提供小幅辅助，这次没有策略残差，
且同向命令仅保持 1.5–2 s。后续应先在当前 CEM 和模型上标定更强差分、延长同向保持，
用实际地面轨迹净转角、曲率和自碰撞情况确认产生了稳定明显的转弯，再补充训练与
独立测试；不能仅放大命令数值或平滑曲线来宣称改进。

仓库另有 `results/steering_authority_8d_rollingquad_abd10_pupper_cem` 的差分实验，
但其 CEM 文件与本 estimator 不同，且报告的 heading rate 是滚动轴方向变化率，
不能直接当作当前控制器的轨迹转向标定结果。

- [地面轨迹、累计方向变化和段均值诊断图](../results/velocity_estimator_cem_switching_v2/turn_diagnostic/turn_diagnostic.png)
- [逐轨迹逐段诊断](../results/velocity_estimator_cem_switching_v2/turn_diagnostic/segment_diagnostic.csv)
- [跨轨迹汇总](../results/velocity_estimator_cem_switching_v2/turn_diagnostic/summary.json)
