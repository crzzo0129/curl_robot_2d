# 指定 Pupper reference + RollingQuad mesh：前 ABD −10°、后 ABD +10°

最终采用用户更正后的配置：前左/前右外展关节目标均 −10°，后左/后右均 +10°。该组合能够持续滚动，且本轮没有检测到自接触或力矩饱和。之前四腿均 +10° 的测试不代表本配置。

## 模型、控制和能耗口径

- Reference：`results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json`，未经重新 CEM 优化。
- 模型：`assets/rollingquad_description_2/mjcf/rollingquad.xml`，完整 CAD mesh 碰撞，保持原接触掩码；未使用简化碰撞体或关闭自碰撞。
- 质量 3.137152 kg；kp=5、kd=0.1、力矩上限 ±3 Nm。ABD 初始化到指定角度，并由相同伺服器保持目标，未刚性锁定关节。
- 保留 0.25 s 周期幅值渐变。MuJoCo 3.12.0，implicitfast、pyramidal cone、Newton 20、line-search 10、impratio 10、禁用 Euler damping，沿用此前统一能耗对照设置。
- 正机械功为各执行器正功率积分；绝对机械功包含负功的幅值，均包含四个 ABD 电机。功率采用同一状态的 actuator_force × actuator_velocity，和广义关节功率核对到约 1e-14 W。
- 每米指标按前向净位移归一化。所有数值是机械功，不是电池耗电。

## 滚动结果

| 运行条件/统计窗口 | 前向距离 m | 平均速度 m/s | 正功 J | 绝对功 J | 正功 J/m | 绝对功 J/m | 正功 CoT | 绝对功 CoT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 ms，0–10 s | 6.195 | 0.620 | 21.709 | 33.048 | 3.504 | 5.334 | 0.1139 | 0.1733 |
| 2 ms，2–10 s | 5.558 | 0.695 | 18.484 | 27.467 | 3.326 | 4.942 | 0.1081 | 0.1606 |
| 1 ms，2–10 s | 5.562 | 0.695 | 18.425 | 27.234 | 3.312 | 4.896 | 0.1076 | 0.1591 |
| 2 ms，2–20 s | 13.587 | 0.755 | 45.176 | 67.847 | 3.325 | 4.994 | 0.1080 | 0.1623 |
| 2 ms，10–20 s | 8.029 | 0.803 | 26.692 | 40.380 | 3.324 | 5.029 | 0.1080 | 0.1634 |

2 ms 全程 10 s 净滚转 7.562 圈，20 s 净滚转 17.830 圈；20 s 总前向位移 14.225 m、横向漂移 −0.0758 m。所有上述正确角度工况的自接触比例和力矩饱和比例均为零。地面接触仍存在，包括滚动所需的躯干/壳体地面接触，不能混同为自接触。

2–10 s 实际前 ABD 均值 −10.086°、后 ABD +9.901°；四电机绝对机械功合计 0.0776 J，占总绝对功约 0.28%。保持力矩造成的铜损没有包含在机械功中，因此不能由这个小比例断言 ABD 保持几乎不耗电。

减小物理步长后每米正功变化约 −0.41%，绝对功变化约 −0.93%；说明这组能耗结果对这次步长变化较稳定。

## 与行走的探索性对照

沿用同模型的 `rollingquad_2_deploy_robust_dr_policy_stable.json`，行走仍使用其自身 ABD 动作，不强行保持前后 ±10°。速度指令 0.76 m/s。

| 2 ms，2–10 s | 滚动 | 行走 |
|---|---:|---:|
| 实际平均速度 m/s | 0.6947 | 0.6862 |
| 正功率 W | 2.3105 | 9.6280 |
| 绝对功率 W | 3.4334 | 14.0605 |
| 正功 J/m | 3.3258 | 14.0303 |
| 绝对功 J/m | 4.9421 | 20.4896 |
| 正功 CoT | 0.1081 | 0.4559 |
| 绝对功 CoT | 0.1606 | 0.6658 |

这组窗口平均速度差约 1.22%，滚动正功 J/m 低 76.30%，绝对功 J/m 低 75.88%。1 ms 对照对应约 76.38% 和 75.66%。行走保持直立且无检测到的自接触。

这仅是当前策略组合的探索性结果，有两个重要边界：

1. 滚动仍在加速，2–4 s 平均约 0.477 m/s，10–20 s 平均约 0.803 m/s，因此 2–10 s 只是平均速度相近，不是严格稳态同速试验。20 s 的滚动与同一行走指令速度已不匹配，不能把该组标为稳态同速节能结论。
2. 当前本地行走训练源码的 CMD_VX 范围为 [−0.6,0.6] m/s。0.76 指令超出这个范围；导出的 policy JSON 未提供训练速度范围证明。不能把这次外推测试当作行走策略在该速度下的最优能效。

与此前 ABD=0、另一个 rollingquad 专用 reference 的结果相比，本次能耗明显更低，但 reference 与姿态同时改变，不应全部归因于其中某一项。本轮指定 reference 的 ABD=0 以及四腿 +10° 对照都没有持续滚动，不能用于正常前进能效排名。

## 错误角度对照与复现

`pupper_reference_mesh_abd10_energy_20260907_*` 为更正前的四腿 +10° 工况，10 s 前进约 0.30 m，启动后回退；原有 reference 回放设置也未实现持续滚动。这些文件保留为诊断，不用于最终配置结论。

正式结果：

- `results/pupper_reference_mesh_frontm10_rearp10_energy_20260907_matched10s/`
- `results/pupper_reference_mesh_frontm10_rearp10_energy_20260907_matched1ms/`
- `results/pupper_reference_mesh_frontm10_rearp10_energy_20260907_matched20s/`（目录名沿用 matched 前缀，但20 s结果并非严格同速；另有 `late_10_20s.json`）

每组含输入哈希、summary、逐物理步 CSV。未覆盖官方模型、reference 或之前的实验产物。代码支持前后 ABD 独立设置，原有 CEM 目标函数测试全部通过。

在 curl_robot_2d 下运行：

```powershell
python -m scripts.compare_locomotion_energy_3d --controller results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json --xml assets/rollingquad_description_2/mjcf/rollingquad.xml --front-abduction-deg -10 --rear-abduction-deg 10 --walk-speeds 0.76 --out results/pupper_mesh_abd_energy_new
```

只测滚动可加 `--roll-only`；长时验证加 `--duration 20`，步长验证加 `--dt 0.001`。输出目录必须为空。
