# CEM 转向与独立速度 estimator 交接（2026-09-12）

## 当前交接范围

用户已把 CEM 转向部分交给另一个对话，本对话暂停该部分，只整理交接。用户最后明确指出：
两个 CEM 应该差不多，应在当前高速 CEM 上使用历史实验同样的差分方法验证。
**已完成验证，用户判断成立：当前高速 CEM 也能明显转弯。** 尚未根据这次验证修改采集器
默认转向映射，也未使用强转向轨迹重新训练 estimator。

项目根目录：`C:/Users/12481/Desktop/OH-WorkSpace/robot_description/curl_robot_2d`。
本文代码、结果路径均相对于该目录。工作区存在多个对话的未提交修改；请只接续本任务相关文件，
不要回退其他修改。results 按项目惯例被 Git 忽略；本地文件存在，但 Git 提交不会自动携带。

## 用户确定的目标与约束

- 独立 estimator，暂不接 policy。输入机器人 observation 历史。
- 滚动轴为 body y；输出有符号前滚线速度和水平轨迹转向率。
- 线速度：根节点原点世界水平速度，投影到由 body y 导出的地面前向
  `h = normalize([body_y_world.y, -body_y_world.x])`。
- 转向率：根节点世界水平速度方向的变化率。现有标签先对 vx/vy 做因果 EMA（tau=0.06 s），
  再算相邻方向夹角 / dt；原始与滤波水平速度在相邻端点均须 >=0.08 m/s，否则 mask。
- 不能把转向命令、gyro_z、Euler yaw 或滚动轴 heading rate 当成该标签。
- 用户明确拒绝先前 neural-policy 数据来源，只允许 CEM 生成轨迹。作废数据/模型位于
  `results/velocity_estimator_v1` 和 `results/velocity_estimator_v1_smoke`（含 DEPRECATED.md），勿使用。
- 不应为了让曲线好看而悄悄改标签或增加平滑。需要真实、明显的持续左右转运动。

## 已澄清的根因与此前说法纠正

历史实验 `results/steering_authority_8d_rollingquad_abd10_pupper_cem` 使用旧三阶段 CEM，
raw 差分 `[a,a,a,-a]`，gain=0.30、differential_scale=0.25；a=+0.50 时记录
滚动轴平均 heading rate +0.07443 rad/s，a=-0.50 时 -0.07384，a=+0.75 时 +0.10449。
这是持续 10 s 的实验，转向效果有原始数据支持。

estimator v2 采集使用当前高速 CEM，但错误地把较弱的辅助 prior 当作产生足够转弯的数据源：
`raw=clip(5*turn_command, ±0.5)`，再乘 gain=0.15、differential_scale=0.25。
所以命令 +0.08 只对应归一化差分 +0.015；历史 raw +0.50 对应 +0.0375。
**本次差分仅为历史实验的 40%，且同向保持时间只有 1.5–2 s。**

不能因为 reference 文件不同就认定当前 CEM 转向不足；最新实测已经推翻这种解释。
也不能说必须有 learned policy residual 才能明显转弯；CEM 加固定差分已经能做到。
历史报告称作 yaw rate 的指标准确名称是滚动轴水平 heading 的全程平均变化率。
它与用户要的轨迹转向率应分别测量；本次在相同时间窗口内，两者平均值接近。

## 最新完成的同方法验证：这是接续工作的直接依据

控制器：
`results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`

模型：`assets/rollingquad_description_2/mjcf/rollingquad_abd10.xml`，**自碰撞开启**。

方法：现有 CPU reference evaluator；MuJoCo 3.9.0；cg20；compact 初始姿态；每条 10 s；
target_scale=1；前外展 -10°、后 +10°；Kp=5、Kd=0.1、力矩限制 3 Nm；
gain=0.30、differential_scale=0.25；50 Hz 记录；0.25 s 启动振幅斜坡。
差分从启动时就恒定施加，沿用历史实验行为。每个幅度一条确定性仿真，无随机化。

| raw a | 全程轴 heading rate (rad/s) | 2 s 后轨迹转向率 (rad/s) | 2 s 后前滚速度 (m/s) | 自碰撞物理步比例 |
|---:|---:|---:|---:|---:|
| 0 | +0.00005 | +0.00041 | 0.898 | 0% |
| +0.25 | +0.04290 | +0.04572 | 0.902 | 0.53% |
| -0.25 | -0.02211 | -0.02037 | 0.900 | 2.47% |
| +0.50 | +0.09137 | +0.09339 | 0.902 | 1.72% |
| -0.50 | -0.07538 | -0.07481 | 0.896 | 3.29% |
| +0.75 | +0.13510 | +0.14168 | 0.897 | 2.55% |
| -0.75 | -0.12884 | -0.13649 | 0.886 | 3.61% |

raw ±0.50 的全程轴转角为 +52.35° / -43.19°；2 s 后的轨迹累计转角约
+42.81° / -34.29°。所有轨迹保持数值有限，2 s 后的轨迹转向标签均有效。
“数值有限”不是通过完整稳定性验收；左右响应也不是严格对称。

自碰撞不是零：+0.50 的 `front_right_foot_proxy__rear_right_thigh_geom` 接触累计
0.172 s，最大穿透约 0.371 mm；-0.50 的对应左侧接触累计 0.329 s，最大穿透约 0.638 mm。
后续需要判断接触是否可接受，或调整参数以恢复无自碰撞，不能把本次结果称为 zero-contact。
target_scale=1 是单一振幅，不能据此声称整个变速范围都能精准跟踪转向命令。

结果目录：`results/steering_authority_high_speed_cem_20260912/`

- `summary.json` / `summary.csv`：上述指标及相同 2 s 后窗口的轴 heading rate。
- `provenance.json`：控制器、模型 SHA256 与协议。
- `raw_*.json`：每个幅度的完整 evaluator 结果，含接触对、穿透、关节范围等。
- `raw_*_motion.npz`：time_s、position_world、velocity_world、body_y_world。
- `steering_paths.png`：实际地面轨迹和轴方向变化对照图。文件已生成，本对话暂停前未目视验图。

**这些 motion NPZ 没有 observation/action 历史，不能直接拿来训练 estimator。**
强转向训练集需要通过 estimator 的采集器重新生成。

## 本轮新增代码与验证状态

1. `scripts/evaluate_3d_symmetric_cem_reference.py`：增加可选 `--motion-series-out`。
   保存每个记录时刻的根节点 qpos[:3]、qvel[:3] 和 body-y；body-y 用当前根四元数转换，
   避免 mj_step 后 xmat 可能滞后一物理步。仅增加记录/导出，不改控制律。
2. `scripts/probe_high_speed_cem_steering.py`：按上述 7 个幅度调用原 evaluator，
   导出 MotionLabeler 的轨迹统计和可选绘图；输出目录必须不存在，避免覆盖旧实验。

七次仿真及汇总、绘图已成功退出；脚本检查了记录时间间隔与 control_dt 一致。
这两项改动之后尚未跑额外回归测试或“开关导出不影响动力学”的精确对照。
历史 estimator 14 项单元测试在上一轮 v2 工作中通过，不代表已检查本轮新增导出。
未提交、未推送，未修改模型参数或旧结果。

复现（项目根目录，选择一个不存在的输出目录）：

```powershell
python -m scripts.probe_high_speed_cem_steering --out results/steering_authority_high_speed_cem_repeat --plot
```

本机可用 Python：
`C:/Users/12481/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`。
包含 NumPy、MuJoCo、JAX/Optax CPU；无 Torch、pytest。matplotlib 在
`C:/Users/12481/Desktop/OH-WorkSpace/robot_description/.experiment_deps`，需加入 Python 搜索路径；
MPLCONFIGDIR 设在工作区。无需绘图时省略 `--plot`。

## 已有 estimator 工作，保留作基线

主要说明：`docs/rolling_velocity_estimator_zh.md` 和
`docs/rolling_velocity_estimator_switching_zh.md`。后者末尾有弱转向数据诊断。

主要代码：`curl_robot_2d_mjx/rolling_velocity_estimator.py`、
`scripts/collect_rolling_velocity_data.py`、`scripts/train_rolling_velocity_estimator.py`、
`scripts/evaluate_rolling_velocity_estimator.py`、`tests/test_rolling_velocity_estimator.py`。

- 36 维原始帧选 `0:6,12:36`（gyro、gravity、关节偏移、上次动作），排除 command 等 `6:12`。
- 20 帧历史，600→256→128→2 MLP。采集 52 Hz，20 个物理子步，按完整 episode 划分。
- 数据和导出模型：`results/velocity_estimator_cem_switching_v2/`。
  `rollouts.npz` 为 64×12 s 随机切换；`scripted_test.npz` 为独立 8×16 s、2 s 切换。
  `model/estimator.npz` 是当前 v2；`results/velocity_estimator_cem_v1` 为有效 CEM v1 基线。
- 独立 scripted 测试 v2 线速度 RMSE 0.01737 m/s，转向 RMSE 0.04904 rad/s。
  但实际 ±0.08 命令段的持续转向仅约 +0.024 / -0.030 rad/s，2 s 净转角 +2.0° / -2.7°。
  逐帧真值波动约 0.10–0.12 rad/s，因此用户认为曲线像噪声是有依据的。
- 段均值仍有信号：上述左右段的估计均值约 +0.0249 / -0.0243 rad/s，不能反过来说模型
  完全没有学到转向；总逐帧 RMSE 也不能作为慢转相对准确性的充分证据。
- `results/velocity_estimator_cem_switching_v2/turn_diagnostic/` 有逐段汇总和诊断图。

## 建议接续步骤（尚未执行）

1. 保持当前高速 CEM 和用户指定模型，先检查上述地面轨迹、接触对与穿透，确认可用差分范围。
2. 在不同速度/振幅下分别标定左右差分响应；同时报告轴 heading rate 与用户定义的轨迹转向率。
   不要直接把 raw 幅度叫作 rad/s，也不要仅提高命令数字而仍经原来的弱映射。
3. 若接续训练，改 estimator 采集配置，让同向保持足够久；覆盖变速、明显左右转、反转及过渡。
   记录原始 observation/action 和真实根运动；训练标签保持原定义，command 仍不进 estimator。
4. 新训练/验证/测试按完整 episode 划分，保留旧模型；检查持续转向段均值、净转角误差、
   逐帧误差及切换瞬态，不能只看总 RMSE。当前 probe 是 50 Hz，estimator 是 52 Hz，注意契约。

工作区另有 `docs/cem_steering_comparison_20260912.md` 和
`scripts/probe_cem_steering_comparison.py`，来自其他对话；其中 current teacher 的实验使用
no-self-collision 模型，不能与本文自碰撞开启的结果混为一谈，也不要覆盖那些文件。
