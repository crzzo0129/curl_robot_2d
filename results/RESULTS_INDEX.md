# 结果目录索引

## 2026-09-05 三维能耗对比

- `energy_cem_3d_20260905/`：能耗目标 CEM，两个种子各 6 代、24 候选；最优控制器及全部候选记录。相对原滚动绝对功 J/m 降低约 6.6%–8.4%，未达到预设 10% 目标。详见 [`能耗 CEM 实验报告`](../docs/energy_cem_3d_20260905_zh.md)。
- `energy_cem_3d_20260905_validation10s/`、`energy_cem_3d_20260905_validation1ms/`、`energy_cem_3d_20260905_validation20s/`：最优候选与同速行走的回放验证。

- `energy_comparison_3d_20260905/`：同模型滚动与三档行走速度扫描。
- `energy_comparison_3d_20260905_matched/`：相近实际速度（约 0.59 m/s）的 2 ms 对照。
- `energy_comparison_3d_20260905_dt1ms/`：1 ms 步长复核。
- 方法、完整结论和复现入口见 [`三维能耗报告`](../docs/locomotion_energy_3d_20260905_zh.md)。滚动正功 J/m 较低，但绝对功 CoT 略高；不能解释为电池节能。

`results/` 保持原路径不大规模搬迁，以免破坏脚本、JSON 和汇报材料中的引用。
下面按用途区分当前正式结果、历史基线和大体积中间搜索。

## A. 当前 Pupper 几何设计主线

| 结果目录 | 用途 | 关键产物 |
|---|---|---|
| `pupper_original_geometry_shell_r127p5/` | R=127.5 mm 解析几何 | `geometry.json`, `compact.png` |
| `pupper_r127p5_three_stage_cem/` | 完整圆壳失败基线 | `summary.json`, 三阶段 controller/rollout |
| `pupper_r127p5_shank_shell_trim_three_stage_cem/` | 60° 开口、旧 90/75 分配 | `summary.json`, 阶段 3 controller, 碰撞截图 |
| `pupper_r127p5_shank_shell_trim90_reference_eval/` | 60° reference 直接用于 90° 模型 | `evaluation_summary.json` |
| `pupper_r127p5_shank_shell_trim90_three_stage_cem/` | 90° 开口重新 CEM | `summary.json`, 阶段 3 controller |
| `pupper_r127p5_open60_3d_width120/` | 旧 90/75 分配的 3D 迁移与修复 | `evaluation_migration_corrected.json` |
| `pupper_r127p5_open60_shell150_45_three_stage_cem/` | 当前 150/45 分配的匹配 2D CEM | `summary.json`, 阶段 3 controller |
| `pupper_r127p5_open60_3d_width120_shell150_45/` | 当前方案的 3D paired evaluation | `evaluation_new_2d_reference.json`, `reference_new_2d_cem_tracking.gif` |
| `rollingquad_2_3d_reference/` | corrected RollingQuad 2 完整 CAD 的 3D reference | `cpu_newton20_reference_10s.json`, 关节曲线与 10 s GIF |
| `design_logic_exploration_ppt/` | 设计探索汇报 | `CURL_robot_design_exploration_CN.pptx` |

### 当前推荐 reference

```text
pupper_r127p5_open60_shell150_45_three_stage_cem/
  03_strict_forbidden_collision/best_phase_controller.json
```

说明：阶段 3 的指标与阶段 2 相同，pipeline 保留的最终文件位于阶段 3 目录；不要与旧
90/75 分配或 90° 开口 reference 混用。

对应的正式 XML 已复制/再生成到：

```text
assets/curl_robot_2d_pupper_r127p5_open60.xml
assets/curl_robot_3d_pupper_r127p5_open60_width120.xml
```

### 当前推荐 3D 评估

```text
pupper_r127p5_open60_3d_width120_shell150_45/
  evaluation_new_2d_reference.json
  reference_new_2d_cem_tracking.gif
```

关键结果：10 s 位移 7.034 m，位移等效 8.781 圈，实际滚动 8.568 圈，跟踪
RMSE 6.34°，力矩饱和 0%，横向漂移近似 0。

当前训练默认使用 corrected RollingQuad 2：

```text
rollingquad_2_3d_reference/
  cpu_newton20_reference_10s.json
  joint_angles_10s.png
  joint_angles_10s.csv
  reference_10s.gif
```

该模型沿用同一 CEM reference，使用完整 CAD 碰撞和 reference 求解配置
（1 ms、Newton 20、line-search 10）。10 s 位移 7.046 m，位移等效 8.795 圈，
实际滚动 8.799 圈，滚动轴倾斜 RMS 1.58° / 最大 3.19°，无非有限值、
无自碰撞、3 Nm 力矩饱和 0%。

## B. Real-geometry 与旧模型基线

| 结果目录 | 用途 |
|---|---|
| `staged_cem_real_geometry_180_d50_foot60/` | 60 mm 足端 real-geometry 的关键旧 reference；训练默认仍可能引用 |
| `staged_cem_real_geometry_180_d50_foot39/` | 39 mm 足端对照 |
| `old_reference_shell_radius_160mm/` | 旧模型仅放大外壳半径的 reference 回放 |
| `shell_radius_160mm_warm_start_cem/` | 160 mm 外壳 warm-start CEM |
| `shell_radius_160mm_motors_54x33/` | 160 mm 外壳加入电机碰撞体后的统计 |
| `contact_free_cem_real_geometry/` | real-geometry 无 Torso 碰撞 CEM |
| `contact_free_curriculum_real_geometry/` | 对应 curriculum 中间结果 |

其中 `staged_cem_real_geometry_180_d50_foot60/03_foot_gap_2mm/best_phase_controller.json`
被现有训练/冒烟流程保留，清理时不可删除。

## C. 早期滚动、行走与停止基线

- `collision_constrained_cem/`：早期碰撞约束 CEM。
- `collision_constrained_cem_foot_gap_2mm_short_contact/`：早期 2 mm 足端间隙版本。
- `phase_controller/`、`rigid_release/`、`servo_release/`：早期动力学基线。
- `walking_exploration_*`：行走控制器探索。
- `rolling_stop/`：滚动停止任务；`low_speed_snapshots_0p40hz.*` 是训练依赖。
- `park_pose_*`：3D park pose 搜索与验证。

## D. 大体积搜索结果

以下目录主要由逐候选/逐代 CSV 构成，占用空间最大，但保留了完整可追溯数据：

- `cold_start_cem_torso_com_upper_half/`：约 189 MB。
- `per_point_cem_torso_com_upper_half/`：约 189 MB。
- `torso_com_cem_sweep_5x5_cpu_short/`：约 65 MB。
- `staged_cem_com_grid/`：约 64 MB。
- `torso_com_cem_sweep_formal/`：约 48 MB。
- `staged_cem_selected_com/`：约 38 MB。

这些不是当前 Pupper 设计主线，但属于历史实验数据。本次整理保留它们；若以后需要释放
空间，优先将整目录压缩归档，而不是只留下无法解释的零散文件。

## E. 清理规则

可以安全删除：

- `__pycache__/`、`.pytest_cache/`。
- PPT 的解压检查目录、重复 ZIP、逐页渲染 PNG。
- 明确标记为 smoke、dry-run 且不被测试引用的临时结果。

需要确认后再删除：

- 任意 `best_phase_controller.json`。
- 任意 `evaluation*.json`、`summary.json` 和接触统计。
- 论文/PPT 使用过的图片和 GIF。
- `rolling_stop/low_speed_snapshots_0p40hz.*`。
