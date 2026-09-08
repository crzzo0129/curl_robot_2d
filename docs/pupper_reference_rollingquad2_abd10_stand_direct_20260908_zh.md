# RollingQuad 2 ABD10 从 Stand 直接切入滚动 reference

## 配置

- 模型：`assets/rollingquad_description_2/mjcf/rollingquad_abd10.xml`（RollingQuad 2 mesh）
- reference：`results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json`
- 初态：`stand` keyframe，静止
- ABD：前腿 −10°、后腿 +10°
- 控制：第一个仿真周期开始执行 reference，不使用 stand→compact 过渡控制器；保留 reference 原有的 0.25 s 周期幅值启动
- 时长：10 s

## 结果

该模型能够从 Stand 直接进入持续滚动。正式 1 ms/Newton reference 评估得到：前进 5.246 m，滚动 6.537 圈，最后 2.5 s 平均速度 0.777 m/s；横向漂移 0.024 m，滚动轴最大倾斜 0.739°，无自碰撞，力矩饱和占比 0.055%。

统一 2 ms 能耗评估器复核得到前进 6.569 m、滚动 8.099 圈；去掉前 2 s 后平均速度为 0.738 m/s，正机械功 3.444 J/m，绝对机械功约 5.120 J/m。两套求解设置的绝对速度不同，但都明确进入持续滚动。

Stand 与 reference 基准姿态的最大初始关节差约 45.2°，因此启动仍有明显冲击：绝对机械功率峰值约 99.2 W，前 0.25 s 绝对机械功约 7.02 J。完整周期幅值在第 0 秒直接打开也能滚动，但正式评估位移下降到 4.989 m，因此保留 reference 原有幅值启动更合适。

这与 primitive 模型的失败结果不同，说明 mesh 接触形状提供了足够的启动推进与相位捕获能力。两个模型的 Stand 直启结论不能互换。

## 产物

- `results/pupper_reference_rollingquad2_abd10_stand_direct_20260908/stand_direct_standard_ramp_10s.mp4`
- `results/pupper_reference_rollingquad2_abd10_stand_direct_20260908/stand_direct_standard_ramp_10s.gif`
- `results/pupper_reference_rollingquad2_abd10_stand_direct_20260908/canonical_standard_ramp_10s.json`
- `results/pupper_reference_rollingquad2_abd10_stand_direct_20260908/canonical_full_reference_10s.json`
- `results/pupper_reference_rollingquad2_abd10_stand_direct_20260908/standard_ramp_10s/summary.json`
- `results/pupper_reference_rollingquad2_abd10_stand_direct_20260908/standard_ramp_10s/roll.csv`
