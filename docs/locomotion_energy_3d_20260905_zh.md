# 三维机器人滚动与行走能耗基线（2026-09-05）

在相近实际速度下，当前滚动参考控制器的每米正机械功低于当前行走策略约 **16.5%–17.6%**；如果采用绝对机械功，滚动反而高约 **2.0%–2.6%**。这是一组仿真机械功对比，不能据此宣称电池节能。

## 对照设置

- 模型：`assets/rollingquad_description_2/mjcf/rollingquad.xml`，完整 CAD 碰撞，质量 3.137152 kg。
- 滚动：`assets/rollingquad_2_3d_self_collision_cem_reference.json`，0.25 s 启动幅值渐变；从 compact 开始。
- 行走：上级目录 `rollingquad_2_deploy_robust_dr_policy_stable.json`；从 stand 加 0.5 mm 初始离地余量开始，与部署训练一致。观测排列、历史初始化及 50 Hz 更新与部署训练代码对齐。
- 两种模式使用相同模型、质量、摩擦、接触、增益和力矩上限；未为单一模式关闭碰撞。使用 pyramidal cone、Newton 20 次迭代、line search 10、impratio 10、禁用 Euler damping。
- MuJoCo 3.12.0，确定性 CPU 仿真；每次 10 s，同时输出全程与 2–10 s 窗口。
- 先扫描行走指令 0.3、0.5、0.7 m/s，再以 0.63 m/s 获得接近滚动的实际速度。额外将物理步长从 2 ms 减到 1 ms 复核，行走控制仍为 50 Hz。

## 结果：2–10 s 窗口

| 物理步长 | 模式 | 实际前向速度 m/s | 正机械功率 W | 正机械功 J/m | 正功 CoT | 绝对功 CoT |
|---|---|---:|---:|---:|---:|---:|
| 2 ms | 滚动 | 0.5873 | 6.5636 | 11.1751 | 0.3631 | 0.6392 |
| 2 ms | 行走（指令 0.63） | 0.5910 | 8.0104 | 13.5543 | 0.4404 | 0.6264 |
| 1 ms | 滚动 | 0.5837 | 6.5877 | 11.2860 | 0.3667 | 0.6316 |
| 1 ms | 行走（指令 0.63） | 0.5906 | 7.9826 | 13.5163 | 0.4392 | 0.6156 |

2 ms 时实际速度差约 0.62%，1 ms 时约 1.18%。减小步长后，两种模式的正功 J/m 分别变化约 +0.99% 和 −0.28%；正功优势方向保持一致。这是有限的步长敏感性检查，不是完整的收敛性证明。

## 指标口径与验证

逐执行器功率为 `actuator_force * actuator_velocity`，取同一仿真状态的数据。每一步交叉检查其求和与 `qfrc_actuator @ qvel` 相等，最大误差小于 6e-14 W。

正机械功 `E+ = ∫Σmax(Pi,0)dt`；负机械功幅值 `E− = ∫Σmax(−Pi,0)dt`；绝对机械功 `Eabs = E+ + E−`。CoT 为相应机械功除以 `m*g*|Δx|`，距离使用前向净位移而非累计摆动路径，横向漂移单独保存。

行走在匹配工况的 2–10 s 内高度最低约 0.148 m、最大机身倾角约 1.2°，未记录到足/小腿以外的地面接触。这里的接触检测将 shank 几何算作足腿支撑，不能证明接触只发生在足底。滚动全程约 6.25–6.28 圈。没有数值非有限状态；匹配窗口行走无力矩饱和，滚动饱和执行器时间比例约 0.26%。

## 结论边界

滚动的正功较低，但制动负功较大，所以换成绝对功指标后并没有显示优势。实际电池消耗还与电机铜损、驱动效率、制动方式及是否回馈有关，不能由这里的正功或绝对功直接替代。

本轮仅为一个名义初态的确定性基线，没有跨种子或扰动的置信区间，也没有计入行走—滚动切换成本。2–10 s 是启动后窗口，并未证明所有状态已达到稳态。两种控制器具有不同控制频率和控制方法，结论只适用于当前控制器组合，不代表两种运动模态各自的理论最优能效。

下一轮可在同速条件下加入摩擦、载荷和初态扰动，再单独测量蜷缩、启动、制动和展开的机械功，以估计不同任务距离下切换是否值得。

## 复现与产物

在 `curl_robot_2d` 下运行（Python 环境需安装 requirements.txt）：

```powershell
python -m scripts.compare_locomotion_energy_3d --walk-speeds 0.63 --out results/energy_new_matched
python -m scripts.compare_locomotion_energy_3d --dt 0.001 --walk-speeds 0.63 --out results/energy_new_dt1ms
```

输出目录必须为空。每组包含 manifest（输入路径、SHA256、版本、假设和通过条件）、summary、report 及逐物理步 CSV。

- `results/energy_comparison_3d_20260905/`：三速度初步扫描。
- `results/energy_comparison_3d_20260905_matched/`：2 ms 匹配工况。
- `results/energy_comparison_3d_20260905_dt1ms/`：1 ms 复核。
- `results/energy_comparison_3d_20260905_smoke/`：1 s 接口验证，不用于最终节能结论。

旧的上级目录 rolling_energy_test.py 与 walking_energy_test.py 保留原样。本报告由新增的统一入口生成数据，后续能耗对比应使用该入口，避免旧脚本的执行器/关节顺序及历史观测更新问题。
