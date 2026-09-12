# 独立滚动速度 estimator v1

**最新工作：** 回合内变速、左右转向切换的采集、训练和评估见
[rolling_velocity_estimator_switching_zh.md](rolling_velocity_estimator_switching_zh.md)。
本页保留通用输入/标签定义和固定 CEM v1 对照结果。

**当前来源：CEM 滚动控制器。** 用户已明确拒绝此前使用的 neural rolling policy。
此前 `velocity_estimator_v1` / `velocity_estimator_v1_smoke` 数据、模型和误差结果作废；
新数据使用独立目录 `results/velocity_estimator_cem_v1`。

模块独立估计状态，不接入 policy。采集器只接受 `phase_locked_oscillator` CEM JSON，
不加载或执行 neural policy。支持 CPU 多进程采集、JAX/Optax 监督训练和 NumPy 在线推理。

## CEM 轨迹的具体来源

沿用项目现有 `scripts/collect_cem_cycle_data.py` 的配套来源：

- 控制器：`results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`。
- 模型：`assets/rollingquad_description_2/mjcf/rollingquad_abd10.xml`，保留该模型的选择性自碰撞。
- 相位锁定、平面关节目标、足间距投影和左右映射使用已有 CEM 回放函数。
- Kp=5、Kd=0.1、力矩限制 3 Nm；关节 action 中心来自模型 compact 关键帧，
  前腿 abduction 为 -10°、后腿为 +10°，不会读取作废 policy 的 default_joint_pos。
- 从 compact 静止启动，CEM 振幅在 0.25 s 内平滑增加到该 episode 的目标振幅。
- 默认每条轨迹振幅在 0.45–1.0 之间采样；这是 **CEM 振幅比例，不是速度命令**。
  左右转轨迹使用项目已有的 `[a,a,-a,-a,a,-a,-a,a]` 差分模式，默认 a 为 0、+0.03、-0.03，
  1–2 s 之间渐入。它是 CEM 加固定差分的轨迹，不是另一个学习 policy，也不保证精确转弯半径。

控制器和模型文件 SHA256、物理参数、随机种子、振幅、左右差分、逐帧根节点位置/速度、
电机目标、CEM 相位、身体累计滚动相位及自碰撞接触比例均记录在数据或 metadata 中。
日志分别报告整段平均速度、2 s 后的平均速度和瞬时最小/最大速度，不能把单帧峰值当持续速度。

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

小规模检查（默认即为上述 CEM 文件和配套模型）：

```bash
python -m scripts.collect_rolling_velocity_data --out results/velocity_estimator_cem_probe/rollouts.npz --episodes 6 --duration 4 --workers 3 --seed 20260912
```

第一版采集、训练和评估：

```bash
python -m scripts.collect_rolling_velocity_data --out results/velocity_estimator_cem_v1/rollouts.npz --episodes 48 --duration 8 --workers 4 --seed 20260912
python -m scripts.train_rolling_velocity_estimator --data results/velocity_estimator_cem_v1/rollouts.npz --out results/velocity_estimator_cem_v1/model --epochs 100
python -m scripts.evaluate_rolling_velocity_estimator --model results/velocity_estimator_cem_v1/model/estimator.npz --data results/velocity_estimator_cem_v1/rollouts.npz --out results/velocity_estimator_cem_v1/evaluation
```

服务器上同步 CEM JSON、模型 XML 和它引用的 meshes；例如增加到 `--episodes 512 --workers 16`。
`--controller` 可指定另一份配套 CEM JSON，多次传入则按 episode 轮换。
`--model` 可显式指定模型，采集会检查其 compact 髋/膝姿态是否符合当前 Pupper CEM 几何。
没有 `--policy` 参数。`--target-scale-range 1 1 --steering-bias 0 --dr-strength 0`
可复现标称振幅、无差分、无物理随机化的 CEM；初始世界 yaw 仍随机，观测噪声另由
`--observation-noise` 控制。初始位置保持不变，没有人为指定前进速度。

默认 observation 采样频率 **52 Hz**，每帧之间运行 20 个物理步。
CEM 相位反馈和电机目标在每个物理步更新，dt 为 `1/(52*20)`，沿用原 CEM 的更新方式；
并非把 CEM 命令冻结 20 个物理步。`last_action` 是采样时刻前最后一次实际电机目标，
按 `[0,0.8,1.2] × 4` 编码，abduction action 为 0。
在线输入必须使用相同的动作中心、缩放和采样频率。元数据明确区分 observation dt 和 controller dt。

旧 CEM 周期采集为 50 Hz / 1 kHz；若需完全保持此时间步，传 `--control-hz 50`。
训练自动读取数据 dt，不能把 50 Hz 数据训练的历史模型当作 52 Hz 使用。

默认 `--dr-strength 0.5`：摩擦缩放 0.875–1.125、各电机 Kp 缩放 0.925–1.075；
另加入 gyro/encoder 噪声与固定偏置，以及小幅 gravity 投影噪声。
未模拟质量/惯量变化、传感器延迟、真实姿态滤波器、地形或受控外推力。
观测噪声只影响记录的 estimator 输入，CEM 相位反馈仍使用仿真真实角速度。

## 数据划分与产物

数据 NPZ 存储 `frames[N,36]`、`velocity_world[N,3]`、`body_y_world[N,3]`、
`episode[N]`、`time_s[N]` 和字符串 `metadata_json`。metadata 包含 dt、采集参数、
CEM/model SHA256、MuJoCo 版本和各 episode 随机化信息。允许直接从其他采集流程
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
同来源的随机 episode 留出不代表跨控制器、跨地形或实机泛化测试。

## 独立在线推理

```python
from curl_robot_2d_mjx.rolling_velocity_estimator import RollingVelocityEstimator

estimator = RollingVelocityEstimator("results/velocity_estimator_cem_v1/model/estimator.npz")
# 可查看 estimator.provenance["observation_contract"] 获取动作中心和缩放。
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
大量滑移情况下，本体感知对平移速度存在可观测性限制。本版本仅针对 CEM 仿真轨迹，尚未验证实机。

## 检查

当前 CEM 版本本地结果（2026-09-11）：48 条 × 8 s，38 条训练、5 条验证、5 条测试。
采集 seed=20260912，训练/划分 seed=7；CPU 4 进程采集 24.3 s，CPU 100 epoch
训练和评估 53.0 s，按验证 loss 选取 epoch 28。

| 留出测试方法 | 线速度 RMSE (m/s) | 轨迹转向 RMSE (rad/s) |
|---|---:|---:|
| CEM 数据训练的历史 MLP | 0.02144 | 0.06112 |
| 训练均值基线 | 0.21776 | 0.18116 |
| 竖直 gyro 分量（仅转向） | — | 0.20553 |

测试包含 1,985 个线速度标签、1,972 个有效转向标签。数据来自同一 CEM 参考与相近
随机物理参数，未证明跨控制器或实机泛化；测试集中只有 2 个低前向速度窗口，没有有效
后退窗口，不能据此声称低速/后退效果。测试左/右转窗口分别为 428/304 个（阈值 ±0.1 rad/s）。

各条 8 s 轨迹的整段平均速度为 **0.531–0.783 m/s**，前 2 s 后均值为 **0.609–0.893 m/s**。
全部采样帧中的瞬时最大值为 1.347 m/s，它不是持续速度。完整轨迹包含启动瞬态；
少量自碰撞发生，各 episode 的物理步自碰撞比例最高约 1.80%，已记录，未把这些轨迹
当作“无碰撞成功”数据筛选。测试指标是状态估计误差，不是控制器安全或成功率认证。

产物：[模型](../results/velocity_estimator_cem_v1/model/estimator.npz)、
[评估指标](../results/velocity_estimator_cem_v1/evaluation/metrics.json)、
[逐帧预测](../results/velocity_estimator_cem_v1/evaluation/predictions.csv)。
结果目录按项目惯例被 Git 忽略，需单独同步模型/数据或使用上面的命令重建。

2026-09-11：标签、在线/离线一致性及 CEM 采集等 11 项测试已通过。
此前使用未获用户认可的 policy 进行的仿真训练结果已撤回。
`results/velocity_estimator_v1` 和 `results/velocity_estimator_v1_smoke` 均有 `DEPRECATED.md` 标记，
仅留存用于追溯，不继续用于训练、评估结论或部署。

```bash
python -m unittest tests.test_rolling_velocity_estimator -v
```

覆盖直行、反向、侧滑、正反圆周运动与 ±π 跨界、完整身体翻滚、低速 mask、因果 EMA、
command 排除、episode 隔离、在线/离线窗口一致性，以及根节点和偏置质心速度参考点。
CEM 检查覆盖标称目标与已有回放一致、clipping 后 action 对应真实电机目标、采样时间正确。
训练结束还检查导出 NumPy 模型与 JAX 网络的数值一致性。
