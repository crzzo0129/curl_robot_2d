# 独立滚动速度 estimator v1

**状态：轨迹来源待指定。** 用户已明确拒绝此前使用的 rolling student policy。
此前 smoke/v1 数据、模型和误差结果已作废，不作为本任务可用产物；通用代码保留。

此模块只估计状态，不接入、不训练或修改 policy。已有 RTNeural rolling policy
仅用于驱动数据采集仿真。支持 CPU 多进程采集、JAX/Optax 监督训练和纯 NumPy 在线推理。

## 输入与输出

每帧使用原始 36 维 deployment observation，抽取索引 `0:6, 12:36`：
gyro 3、projected gravity 3、关节位置偏移 12、上一帧实际 action 12。
排除 command 和 desired world z；不把仿真真值放入输入。20 帧按最新帧优先排列，
形成 600 维特征。模型为 `600 → 256 → 128 → 2`，隐藏层 ReLU，输出层线性。

输出顺序：

1. `rolling_speed_m_s`：根节点原点沿水平滚动朝向的有符号瞬时线速度。
2. `trajectory_turn_rate_rad_s`：水平速度方向的变化率，逆时针为正。

设 `b = R_world_from_body[:, 1]` 为机体滚动 y 轴，则水平前向为
`h = normalize([b_y, -b_x])`，线速度为 `dot(root_velocity_world[:2], h)`。
这一定义在身体绕 y 轴翻滚整圈时保持稳定；水平 y 轴投影长度低于 0.2 时屏蔽
线速度标签，避免滚动轴接近竖直时的奇异。后滚速度为负，侧滑不计入前进速度。

根节点速度从 **free joint 的平移 qvel** 获取；MuJoCo `cvel[..., 3:]`
是不同参考点的空间线速度分量，不能直接当作根节点原点速度。
每帧记录前调用 `mj_forward`，使姿态、观测和速度来自同一仿真时刻。

对根节点水平速度做因果 EMA，默认时间常数 0.06 s，然后计算：

```text
alpha = 1 - exp(-dt / tau)
u[t] = u[t-1] + alpha * (v_xy[t] - u[t-1])
turn[t] = atan2(cross2d(u[t-1], u[t]), dot(u[t-1], u[t])) / dt
```

不使用未来帧、不对机体 yaw 求导。相邻两帧的原始及平滑水平速率均需 ≥ 0.08 m/s，
否则屏蔽转向 loss；首帧也无有效转向标签。屏蔽值虽然存为 0，**不作为零转向真值**。
`--velocity-filter-tau 0` 可关闭平滑，但冲击时的方向导数会更尖锐。
转向输出估计的是该因果平滑目标，有相应延迟；线速度输出不经过该 EMA。
原地转动没有有效轨迹转向标签。停下后反向起步也不应被解释为可靠的瞬时转向。

## 本地与服务器运行

在 `curl_robot_2d` 项目根目录运行。已有 MJX 训练环境通常已经包含训练依赖；
独立 CPU 环境可执行 `python -m pip install -r requirements-estimator.txt`。
服务器若已配置 CUDA JAX，会自动使用 GPU 训练；采集仍为 CPU MuJoCo，
通过 `--workers` 并行，与 MJX/GPU 仿真吞吐不同。

以下为模板命令；先指定正确的轨迹来源，不能直接沿用作废实验。
`/path/to/selected_rolling_policy.json` 是占位符。如果选择 CEM 或已有日志，需适配采集入口。

本地最小验证模板：

```bash
python -m scripts.collect_rolling_velocity_data --policy /path/to/selected_rolling_policy.json --out results/velocity_estimator_selected_smoke/rollouts.npz --episodes 12 --duration 4 --workers 2
python -m scripts.train_rolling_velocity_estimator --data results/velocity_estimator_selected_smoke/rollouts.npz --out results/velocity_estimator_selected_smoke/model --epochs 40
```

训练与评估模板：

```bash
python -m scripts.collect_rolling_velocity_data --policy /path/to/selected_rolling_policy.json --out results/velocity_estimator_selected/rollouts.npz --episodes 48 --duration 8 --workers 4 --seed 20260911
python -m scripts.train_rolling_velocity_estimator --data results/velocity_estimator_selected/rollouts.npz --out results/velocity_estimator_selected/model --epochs 100
python -m scripts.evaluate_rolling_velocity_estimator --model results/velocity_estimator_selected/model/estimator.npz --data results/velocity_estimator_selected/rollouts.npz --out results/velocity_estimator_selected/evaluation
```

服务器上把 `--policy` 替换为已同步的 JSON 路径；例如增加到 `--episodes 512 --workers 16`。
也需同步模型 XML 引用的 meshes。可以重复传入 `--policy`，按 episode 轮换控制器。
若是 command-conditioned policy，使用 `--command VX VY TURN` 指定其输入；
estimator 始终忽略这些 command 通道。

默认控制频率 **52 Hz**，每次控制 20 个 physics substep，physics dt 为 `1/(52*20)`。
这与旧的 `analyze_rolling_student_5s` 中 50 Hz/1 kHz 设置不同：只复用其 JSON 推理与
frame 拼装函数，显式覆盖时间步。需要 50 Hz 时采集传 `--control-hz 50`；训练自动读取
数据的 dt，在线推理必须使用相同采样间隔，不能混用 50 Hz 与 52 Hz 的历史。

每个 episode 随机化初始世界 yaw、地面摩擦、各电机 Kp，并加入观测噪声与固定偏置。
三组 action 扰动分别为零、正、负左右差分，另有小幅随机 action 偏置，启动后渐入。
这些用于产生转弯及偏离标称的轨迹，**不保证指定转弯半径或滚动成功**。
实际施加的上一帧 action 会写入下一帧 observation。当前未模拟质量/惯量变化、
传感器延迟、真实姿态滤波误差、地形或受控外推力，需要后续扩充。

## 数据划分与产物

数据 NPZ 存储 `frames[N,36]`、`velocity_world[N,3]`、`body_y_world[N,3]`、
`episode[N]`、`time_s[N]` 和字符串 `metadata_json`。metadata 包含 dt、采集参数、
policy/model SHA256、MuJoCo 版本和各 episode 随机化信息。允许直接从其他采集流程
生成此格式，再使用同一训练程序。各 episode 时间戳必须按 dt 递增。

完整 episode 按约 80/10/10 分成训练、验证、测试；小数据至少各留一个验证/测试 episode。
历史及标签滤波器在每个 episode 重置，前 19 帧只用于填充历史，不作为训练窗口。
归一化只使用训练集，按验证 loss 选择 checkpoint，测试集不参与选模。
少于 5 条完整 episode，或任何划分没有有效输出标签时明确报错。

训练目录包含：

- `estimator.npz`：权重、输入输出归一化、配置、特征顺序与来源信息。
- `metrics.json`：训练/验证/测试 RMSE、MAE、bias、标签覆盖和分组结果。
- `test_predictions.npz`：保留测试窗口的预测、真值、mask、episode、原始行号。
- `learning_curve.csv`：训练/验证 Huber loss。

评估另导出 `predictions.csv`，带时间戳和两种标签的有效标志。
基线包括训练集平均值与 `-dot(gyro, projected_gravity)`；后者只是比较项，
不把竖直角速度等同于轨迹转向速度。分组会分别报告前进、后退、低前向速度、左转和右转；
没有样本的组返回 null，不能视为已经验证。

默认评估严格使用原始数据的已保存 test episode，并校验数据文件 SHA256。
评估新采集的独立数据时增加 `--all-episodes`；请用新的 seed，并确认未复用训练轨迹。
同来源的随机 episode 留出不代表跨策略、跨地形或实机泛化测试。

## 独立在线推理

```python
from curl_robot_2d_mjx.rolling_velocity_estimator import RollingVelocityEstimator

estimator = RollingVelocityEstimator("results/velocity_estimator_selected/model/estimator.npz")
# 每 1/52 s 输入一帧未经 policy normalizer 处理的原始 observation。
prediction = estimator.update(raw_observation_36)
if prediction is not None:  # 第 20 帧起有输出
    rolling_speed, trajectory_turn_rate = map(float, prediction)

# 传感器流重启、episode 重置或明显丢帧后重新填充历史：
estimator.reset()
```

部署推理仅依赖 NumPy，不需要 JAX 或 MuJoCo。当前没有速度有效性/置信度预测头；
历史填满只表示输入就绪，**不表示低速时的转向有物理意义**。训练时的真值 mask
不能在实机直接获取，也不应简单用有符号前进速度代替水平总速度作同等可靠的判断。
大量滑移情况下，本体感知对平移速度存在可观测性限制。目前没有基于用户指定轨迹训练的可用模型。

## 检查

2026-09-11：通用标签及在线/离线一致性等 8 项测试已通过。
此前使用未获用户认可的 policy 进行的仿真训练结果已撤回。
`results/velocity_estimator_v1` 和 `results/velocity_estimator_v1_smoke` 均有 `DEPRECATED.md` 标记，
仅留存用于追溯，不继续用于训练、评估结论或部署。

```bash
python -m unittest tests.test_rolling_velocity_estimator -v
```

覆盖直行、反向、侧滑、正反圆周运动与 ±π 跨界、完整身体翻滚、低速 mask、因果 EMA、
command 排除、episode 隔离、在线/离线窗口一致性，以及根节点和偏置质心速度参考点。
训练结束还检查导出 NumPy 模型与 JAX 网络的数值一致性。
