# 指定 Pupper reference 在 primitive 模型上的回放测试

## 配置

- 控制器：`results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json`
- 模型：`assets/rollingquad_description_2/mjcf/rollingquad_primitive_abd10.xml`
- ABD 目标：前腿 −10°，后腿 +10°
- 回放时长：10 s
- 判据：前向位移和滚动圈数持续增加，且无非有限状态、自碰撞或力矩饱和。

`rollingquad_primitive_abd10.xml` 使用 primitive 碰撞几何，并把当前 ABD 姿态写入初始模型；回放时仍明确施加前 −10°、后 +10°的 ABD 目标。

## 结果

正式 reference 评估得到：

| 指标 | 结果 |
|---|---:|
| 10 s 前向位移 | 4.661 m |
| 横向位移 | −0.025 m |
| 实际滚动 | 5.194 圈 |
| 由姿态累计的绝对转动 | 5.259 圈 |
| 滚动轴倾斜 RMS / 最大 | 0.357° / 0.755° |
| reference 跟踪 RMSE | 5.14° |
| 自接触占比 | 0% |
| 力矩饱和占比 | 0% |

统一能耗评估器独立复核得到 4.727 m 和 5.278 圈，结论一致。去掉前 2 s 启动段后，平均速度为 0.508 m/s，正机械功为 3.675 J/m，绝对机械功为 5.791 J/m；功率恒等式最大误差为 `3.6e-15 W`。

因此，这个由 shell150/45 二维 CEM 得到的 reference 可以直接驱动当前 primitive 三维模型，并在 10 s 内形成持续、方向稳定的滚动。

## 与 mesh 模型对照

同一 reference 和 ABD 配置在 mesh 模型上的 2–10 s 平均速度约为 0.695 m/s，绝对机械功约为 4.942 J/m。primitive 模型仍能滚动，但速度低约 27%，单位距离绝对机械功高约 17%。这说明控制器具有跨几何表示的可迁移性，但 primitive 与 mesh 的接触形状差异会明显改变速度和机械功，二者不应混用为同一能耗基线。

## 产物

- `results/pupper_reference_primitive_abd10_20260907/canonical_reference_10s.json`：正式 reference 回放结果。
- `results/pupper_reference_primitive_abd10_20260907/energy_10s/summary.json`：统一能耗汇总。
- `results/pupper_reference_primitive_abd10_20260907/energy_10s/roll.csv`：逐时刻轨迹和功率数据。
- `results/pupper_reference_primitive_abd10_20260907/energy_10s/report.md`：自动生成的能耗报告。
