# 从 stand 起步的 CEM / 正弦行走控制器

默认机器人是 `assets/rollingquad_description_2/mjcf/rollingquad.xml`，使用模型原有碰撞、质量、伺服增益和 1 ms 物理步长，不依赖 JAX 或 PPO checkpoint。

在 `curl_robot_2d` 项目目录执行：

```powershell
# 默认加载已保存的 CEM 参数，stand 起步，打开 MuJoCo 窗口
python -m scripts.walk_cem --view

# 无窗口评估，保存 evaluation.json 和 rollout.npz
python -m scripts.walk_cem --duration 30

# 不加载 CEM 参数，使用内置正弦基线
python -m scripts.walk_cem --controller sine --view

# 导出预览动画（依赖 requirements.txt 中的 Pillow）
python -m scripts.walk_cem --duration 10 --gif results/walk_cem/walking.gif

# 重新执行离线 CEM 搜索；每次候选都从 stand 重置
python -m scripts.walk_cem optimize --population 48 --iterations 20 --workers 4 --duration 10 --out results/walk_cem_new

# 加载新搜索的参数
python -m scripts.walk_cem --policy results/walk_cem_new/best_controller.json --duration 30 --view

# 从已有参数继续搜索
python -m scripts.walk_cem optimize --policy assets/controllers/walk_cem.json --duration 10 --out results/walk_cem_refine
```

如果本机 `python` 没有加入 PATH，请使用已有 Python 3.12 解释器的完整路径。

默认初始化直接采用 `stand` 的 qpos 和 ctrl，先保持 0.5 秒，再用 1 秒 smoothstep 渐变进入周期步态。`--hold` 和 `--ramp` 可以调整这两个时间，整个评估时长必须大于两者之和。控制周期是 20 ms；输出是模型执行器顺序的关节位置目标，按关节和执行器范围的交集限幅。

步态采用前后方向余弦轨迹和正半周期抬脚轨迹，依据实际 CAD 模型 stand 姿势的足端 Jacobian 转成关节偏移。左右腿相差半周期，后腿相位可由 CEM 微调；加入 roll/pitch 反馈以及沿世界 +X 方向的航向与横向偏移反馈。模型关节 qpos 顺序与执行器顺序不同，代码全部按名称映射。

CEM 是**离线搜索周期控制器参数**，不是每个控制周期在线规划的 MPC。12 个参数包含频率、步长、抬脚高度、前后腿偏置、后腿相位及 roll/pitch 反馈增益。每轮采样、仿真、保留精英并更新均值和标准差；固定随机种子、保留最优候选，并在每轮保存 JSON。评分结合前进位移、速度误差、倾斜、横向漂移、非足端触地、执行器负载和跌倒惩罚。

`--speed` 是搜索/评分中的目标速度；运行已保存参数时修改它只改变评估评分，不会自动重新生成步态。当前控制器用于沿 +X 方向前进，没有提供任意速度、转弯或实机通信接口。

默认参数存放在 `assets/controllers/walk_cem.json`。初次搜索使用种子 42、32 个候选、12 轮、6 秒评估；随后添加固定航向反馈并独立进行 30 秒验证。MuJoCo 3.10.0 的结果：

| 指标 | 默认控制器，30 秒 |
| --- | ---: |
| 前进距离 | 12.064 m |
| 全程平均速度（包括起步） | 0.402 m/s |
| 起步后平均速度 | 0.419 m/s |
| 最终侧向偏移 | 0.036 m |
| 最大机身倾角 | 9.476° |
| 最低基座高度 | 0.137 m |
| 非足端地面接触采样占比 | 0 |
| 跌倒 | 否 |

2026-09-11 按要求将默认 CEM 参数的 `lift_m` 从 0.028 提高至 **0.050 m**，上表为修改后的 30 秒验证。实际足底离地高度按碰撞 mesh 最低点测量，完整周期峰值的中位数为前脚 33.0–34.3 mm、后脚 55.8–56.3 mm。参数幅度不等于实际离地高度；机身运动、姿势偏置和伺服跟踪都会影响结果。可用 `python -m scripts.evaluate_walk_lift --lifts 0.028 0.05 --duration 30` 重现高度对比。JSON 中的 `training` 保留原始 CEM 搜索记录，`validation_30s` 对应当前参数。

验证采用固定平地、原始物理参数及准确 stand 初始化；尚未验证随机扰动或复杂地形。接触统计以 50 Hz 采样，`foot_liftoffs` 是原始接触切换次数，可能包含接触抖动，不能解释为完整步数。GIF 和 NPZ 可用于检查具体步态。JSON 内保存模型 XML SHA-256，加载时检查一致性；模型的外部 mesh 文件仍须与项目版本一致。

可复用 API 在 `curl_robot_2d/walking_cem.py`：

```python
import mujoco
from curl_robot_2d.walking_cem import MODEL_PATH, SineWalkingController
from scripts.walk_cem import DEFAULT_POLICY, load_policy

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
parameters, metadata = load_policy(DEFAULT_POLICY)
controller = SineWalkingController(model, parameters,
                                   hold_s=metadata["hold_s"], ramp_s=metadata["ramp_s"],
                                   heading_feedback=metadata["heading_feedback"])
data = mujoco.MjData(model)
controller.reset(data)
for _ in range(500):
    data.ctrl[:] = controller.targets(data.time, data)
    mujoco.mj_step(model, data, nstep=20)
    mujoco.mj_forward(model, data)
```

测试命令：`python -m unittest discover -s tests -p test_walking_cem.py -v`。
