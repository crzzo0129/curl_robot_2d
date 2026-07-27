# Curl Robot 2D

用于探索四足机器人全身蜷缩滚动的二维侧视仿真子项目。最终目标是实现
**行走—滚动双模态机器人及其双向切换**；二维模型是从滚动动力学和纵向
切换机制出发、逐步迁移到完整三维四足机器人的第一阶段。

本方向不同于已有的 `disk_robot`，也不同于原先给 Torso 增加自由度的
`robot_curl`：

- 不把 Torso 改成圆盘；
- 暂不增加或改变现有机器人关节自由度；
- Torso 保持为单个刚体；
- 机器人利用现有腿部关节蜷缩；
- 在侧视图中，由 Torso、前腿大腿/小腿、后腿大腿/小腿共 5 个刚性部分
  形成近似五边形；
- 当前几何基线令五条中心线边全部等长，公共边长为 0.15 m；
- `compact` 关键帧保持五边等长，并让两个有限半径足端球表面接触、中心不再
  重合，因此它不再是严格正五边形；
- 在这 5 个部分外侧增加分段弧形外壳，使蜷缩姿态的外围轮廓近似圆形。

## 当前研究范围

当前子项目只研究滚动及蜷缩/展开过程的侧视二维问题。二维模型中的前腿和
后腿分别代表真实机器人左右对称的一对腿；后续设置质量与转动惯量时，需要
计入投影中重合的两侧结构。

真实行走模态仍由完整三维四足机器人承担，不在第一阶段二维仿真的直接评价
范围内。二维模型首先回答以下问题：

1. 现有关节是否能够形成适合滚动的蜷缩构型？
2. 弧形外壳应该采用怎样的曲率、弧长、厚度与间隙？
3. 质量和质心位置如何影响滚动启动、平顺性和能耗？
4. 机器人应该采用被动滚动、主动滚动，还是二者结合？
5. 强化学习能否发现有效且可迁移的滚动步态？
6. 如何制动并停在允许展开和恢复站立的滚动相位？
7. 二维蜷缩、滚动和展开机制如何迁移到三维双模态系统？

## 文档

- [`docs/design_discussion_zh.md`](docs/design_discussion_zh.md)：截至
  2026-07-26 的方案讨论整理、当前共识和待研究问题。
- [`docs/fixed_parameters_zh.md`](docs/fixed_parameters_zh.md)：第一版二维
  等效模型的固定参数、来源、合并规则和暂定仿真参数。
- [`docs/rigid_phase_analysis_zh.md`](docs/rigid_phase_analysis_zh.md)：
  compact 构型的刚性滚动相位、质心、势能和重力矩分析。
- [`docs/rigid_release_baseline_zh.md`](docs/rigid_release_baseline_zh.md)：
  刚性 compact 从静止释放后的完整动态基准和接触损失结果。
- [`docs/servo_release_baseline_zh.md`](docs/servo_release_baseline_zh.md)：
  有限刚度和力矩下的可变形释放基准，以及与刚性结果的对照。
- [`docs/active_roll_controller_baseline_zh.md`](docs/active_roll_controller_baseline_zh.md)：
  从纯相位控制的停滞，到相位锁定周期控制实现持续主动滚动的完整基准。
- [`docs/leg_crossing_analysis_zh.md`](docs/leg_crossing_analysis_zh.md)：
  对无自碰撞主动 baseline 的前后腿拓扑交叉、杆身交叉和结构间隙诊断。
- [`docs/collision_model_revision_zh.md`](docs/collision_model_revision_zh.md)：
  外壳设计缝隙、有限足端、模型自碰撞和旧控制器回放的当前修订记录。
- [`docs/collision_constrained_cem_zh.md`](docs/collision_constrained_cem_zh.md)：
  碰撞分级、CEM 代价设计、正式搜索设置和当前主动滚动基准。
- [`docs/nominal_com_mjx_rl_zh.md`](docs/nominal_com_mjx_rl_zh.md)：
  固定当前名义 COM、从零开始的 MJX/PPO 环境、奖励和云端运行流程。

## 当前状态

目前处于参数化模型和评价体系设计阶段。主仿真器已经确定为 MuJoCo：

- 前期使用 CPU MuJoCo 建模、调试接触并完成结构评价；
- 使用平面复合根关节建立严格的侧视模型；
- 已实现五条等长中心线边和参数化离散圆弧外壳；
- 每条边的名义 \(72^\circ\) 圆弧两端各裁去约 \(7.29^\circ\)，实际覆盖
  \(57.42^\circ\)，暂由 6 个短 capsule 近似；
- 相邻刚体的弧壳在大小腿共线的极端姿态下仍保留 2 mm 设计缝隙；
- 腿、Torso、足端和弧壳的相关自碰撞已经开启；足端允许接触但不能自由穿越；
- 当前外壳只改变视觉和碰撞轮廓，尚未计入质量与惯量；
- 模型与评价稳定后，再接入 MJX 进行批量强化学习；
- 历史无自碰撞模型曾由相位锁定控制器在 10 s 滚动约 10.99 圈，但该结果
  包含腿部穿越，现仅作为历史控制基准；
- 同一旧控制器直接回放到当前碰撞模型时不再发生腿杆交叉，10 s 只滚动
  1.54 圈；
- 已在当前碰撞模型上重新运行 CEM：新控制器 10 s 滚动 9.91 圈，腿杆
  交叉为 0，非允许自接触占 1.36%，最大非允许穿透约 0.459 mm。
- 已建立读取同一 XML 的名义 COM MJX/PPO 后端；云端基准环境固定为
  Python 3.12、Linux、CUDA 12.8，并在 RTX 4090 或 H200 上先做编译
  冒烟，再验证纯 RL 能否从零学会滚动。
- RL 奖励权重已集中到 `curl_robot_2d_mjx/reward_config.py`，奖励计算、
  普通训练指标和失败原因分别记录，便于后续频繁调参且不覆盖历史实验。

第一版带注释和离散弧壳的等边五连杆模型位于 `assets/curl_robot_2d.xml`，由
`curl_robot_2d/model.py` 根据集中参数生成。可使用：

```powershell
python -m scripts.generate_model
python -m scripts.inspect_model --keyframe compact
python -m scripts.analyze_roll_phase
python -m scripts.run_release_baseline --joint-mode both
python -m scripts.optimize_phase_controller
python -m scripts.replay_active_controller
python -m scripts.replay_active_controller --viewer
python -m scripts.analyze_leg_crossing
python -m scripts.view_model --keyframe compact --simulate
python -m scripts.view_model --keyframe compact --simulate --camera-distance 1.0
python -m scripts.render_model --keyframe all
python -m unittest discover -s tests -v
```

`--simulate` 默认使用跟随 Torso 的侧视 tracking camera，机器人向前滚动时
不会离开画面。使用 `--camera-distance` 调整视野大小，数值越大，看得越远；
如需恢复 XML 中的固定相机，可添加 `--camera-mode fixed`。

当前主线是先在名义 COM 下完成二维纯 RL 滚动，再加入启动、调速、制动和
停止相位控制；随后验证二维蜷缩/展开链路并粗略寻找双模态可行的 COM 区域。
选定具有鲁棒裕度且可制造的结构后，将二维滚动机制迁移到完整三维四足模型，
与行走策略和高层状态机组合，最终验证完整的 `walk → roll → walk`。
