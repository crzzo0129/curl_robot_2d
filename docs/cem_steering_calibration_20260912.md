# 当前高速 CEM 的转向标定（2026-09-12）

已完成本地 CPU MuJoCo 标定。控制器保持为
`results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`，没有修改 CEM 系数或任何学生权重。

可用标定文件：`assets/controllers/rollingquad_abd10_high_speed_steering_v1.json`。
这是“前进速度指令、转向速度指令 → 有效归一化差动幅值”的二维表，不再使用统一的 `5 * yaw_command * .15 * .25`。
正负转向分别测量，零指令包含小幅静态校正，速度和转向幅值之间用双线性插值。

## 适用条件

- `rollingquad_2_abd10_no_self_collision`，前外展 -10°、后外展 +10°。
- MuJoCo 3.9.0，cg20，物理步 0.001 s、控制步 0.02 s，平地，无动力学随机化。
- 前进速度**指令**范围 0.4111–0.8110 m/s；转向指令 -0.08–+0.08 rad/s。
- 使用原始前进速度查表。本次没有重新标定前进速度；转弯时的实际前进速度不保证等于指令。
- 输出直接是归一化动作偏移 a，按 `[a,a,-a,-a,a,-a,-a,a]` 叠加，随后使用现有动作和关节限幅。髋/膝物理偏移分别为 `0.8*a`、`1.2*a` rad。**不要再乘 residual_gain 或 differential_scale。**
- 加载时验证参考文件和 XML 哈希、参考参数、关键物理设置和指令范围。不匹配时拒绝加载，不偷偷套用。校验哈希统一 CRLF/LF 换行，兼容 Windows 提交后在 Linux 云端拉取；原始字节哈希也保留在 provenance 中。

## 测量协议与结果

共 207 次 CPU rollout：91 次幅值扫描、56 次额外标定验证、56 次旧映射配对对照、4 次回合内切换测试。标定扫描取 7 个速度点、13 个有符号幅值，选择包含零点的连续、稳定、单调分支反插值到 9 个转向指令节点。

扫描先直行暖机 3 s，再施加恒定差动 10 s；拟合的是施加差动后第 2–10 s 的滚动轴 heading 平均变化率。验证改用 2.4 s 暖机、不同速度点/转向幅值，以及 ±0.005 rad 初始关节位置、±0.005 rad/s 关节速度扰动。56 次旧映射测试使用完全相同的速度、目标、种子、暖机时长和初始扰动。

| 指标 | 旧映射 | 新标定 |
|---|---:|---:|
| 56 次测试的平均绝对稳态转向误差 | 0.029479 rad/s | **0.001846 rad/s** |
| 最大绝对稳态转向误差 | 0.066815 rad/s | **0.007052 rad/s** |

平均误差下降约 93.7%。这是每回合稳态平均角速度的误差，不是每个 20 ms 控制步的瞬时 MAE，不能直接与 PPO 日志的 `yaw_mae` 数值横向比较。

部分验证结果：

| 前进速度指令 | -0.08 指令实测 | +0.08 指令实测 |
|---|---:|---:|
| 0.4111 m/s | -0.0782 | +0.0838 |
| 0.46 m/s | -0.0864 | +0.0871 |
| 0.58 m/s | -0.0803 | +0.0816 |
| 0.64 m/s | -0.0816 | +0.0812 |
| 0.76 m/s | -0.0792 | +0.0798 |
| 0.811 m/s | -0.0814 | +0.0809 |

表内转向单位均为 rad/s。0.70 m/s 的 ±0.08 指令实测约 ±0.074，说明插值和动力学波动仍带来误差，不能将此表视为精确闭环控制。

4 条切换轨迹使用另一种初始扰动种子，直行暖机 4 s 后，依次执行 `+.08, 0, -.08, +.04, -.04`，每段 4 s，共 20 s。段内不重置物理状态或相位。排除各段前 1 s 后，20 个分段的平均绝对角速度误差 **0.003742 rad/s**，最大 **0.011836 rad/s**。

56 次恒定验证和 4 次切换均满足本次机械稳定检查：状态有限、根高度在 0.025–0.8 m 内、滚动轴仰角小于 0.5 rad、平均朝向前进速度大于 0.15 m/s。最大轴仰角约 **0.0359 rad（2.06°）**。自碰撞关闭，这些测试不能证明打开自碰撞后的安全性，也没有复用完整 PPO 接触失败判据。

原始记录位于 `results/cem_steering_calibration_20260912/` 的 `scan.json`、`validation.json`、`baseline.json`、`switch.json` 和 `validation_summary.json`。标定表保留 provenance 和验证摘要。CPU 回放使用生产环境的归一化 reference、动作映射、限幅和每物理步相位反馈函数；没有运行完整 MJX 环境、云端策略训练或硬件验证。

## 同时处理的转弯判据问题

旧环境的轴倾斜定义为 `acos(abs(body_y_world.y))`，它把水平朝向变化也计为倾斜。以 .08 rad/s 正常转弯约 6.25 s 后，朝向偏移就能达到 .5 rad，继而触发持续 .1 s 的轴倾斜终止；`-8*axis_tilt²` 奖励也会惩罚这个正常转弯。

**仅启用标定表时**，有转向指令的轴倾斜改为 `asin(abs(body_y_world.z))`，即滚动轴离开水平面的真实仰角。直行仍使用原判据，无标定表的旧实验也保持原定义。这是新模式下奖励/失败判据的明确变更，因此它的成功率不能直接当作旧评估协议下的成功率提升。单元测试覆盖水平转弯不误判、真正倾斜仍超阈值和旧模式保持原值。

## 云端使用

代码和数据包：`results/cem_steering_calibration_v1.zip`。通过 Git 同步后，在项目根目录解包：

```bash
git pull
unzip -o results/cem_steering_calibration_v1.zip
```

在现有 `scripts.train_mjx_3d_roll_distillation` 或 `scripts.train_mjx_3d_roll_student_dr_ppo` 命令中，保留 `--command-conditioned`、上述 geometry 和速度范围，增加：

```bash
--steering-calibration assets/controllers/rollingquad_abd10_high_speed_steering_v1.json
```

没有该参数时使用旧映射，已有 checkpoint 不会自动获得新转向能力。蒸馏训练启用后，教师标签由新映射生成；PPO 启用后，CEM 快照使用新映射，学生仍输出自己的完整动作。下一轮策略训练建议先用新教师重新蒸馏或补充训练并做独立对照；本次未启动这些训练。

## 本地复现

```bash
python -m scripts.calibrate_cem_steering --stage scan --workers 2
python -m scripts.calibrate_cem_steering --stage fit
python -m scripts.calibrate_cem_steering --stage validate --workers 2
python -m scripts.calibrate_cem_steering --stage baseline --workers 2
python -m scripts.calibrate_cem_steering --stage switch --workers 2
python -m scripts.calibrate_cem_steering --stage finalize
python -m unittest tests.test_steering_calibration tests.test_mjx_3d_reward
```

标定脚本缓存已完成的 case。修改仿真代码或物理配置后必须指定新的 `--out` 目录，避免沿用旧测量。18 项单元测试已通过，包括 JAX JIT 批量查表、幅值不重复缩放、模型/配置不匹配拒绝加载、跨平台换行哈希及奖励回归测试。
