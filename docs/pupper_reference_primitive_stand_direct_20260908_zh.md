# Stand 姿势直接切入滚动 reference 测试

## 测试定义

- 模型：`assets/rollingquad_description_2/mjcf/rollingquad_primitive_abd10.xml`
- reference：`results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json`
- 初态：模型的 `stand` keyframe，静止释放
- ABD：前腿 −10°、后腿 +10°
- 切换：不使用 stand→compact 过渡控制器，不改写仿真状态
- 时长：10 s；reference 物理配置为 1 ms、Newton 20、50 Hz 控制

主测试保留 reference 原有的 0.25 s 周期幅值启动。另做“第 0 秒完整周期幅值”和正常 compact 初态对照。

## 结论

Stand 直接切入不能进入原滚动 reference 的稳定滚动轨道。主测试先向前翻滚约 0.65 圈，随后反向回摆，约 6 s 后基本停住；10 s 净前进仅 0.152 m，净滚动 0.103 圈，最后 2.5 s 平均前向速度为 −0.0036 m/s。没有自碰撞和数值发散，失败形式是“翻倒后回摆并停住”，不是仿真崩溃。

第 0 秒施加完整周期幅值也没有解决问题：10 s 净前进 0.459 m、净滚动 0.520 圈，机器人持续在约半圈附近前后摆动，未形成连续转动。作为对照，compact 初态在相同 10 s 内前进 4.727 m、滚动 5.278 圈。

| 初态/切换 | 10 s 位移 | 10 s 滚动 | 结果 |
|---|---:|---:|---|
| compact，标准 reference 启动 | 4.727 m | 5.278 圈 | 持续滚动 |
| stand 直接切入，标准幅值启动 | 0.152–0.177 m | 0.090–0.103 圈 | 回摆后停止 |
| stand 直接切入，第 0 秒完整幅值 | 0.459–0.461 m | 0.520–0.524 圈 | 半圈附近往复摆动 |

数值范围来自 reference 评估器和统一能耗评估器，两套求解设置给出一致的失败判断。

## 启动冲击

Stand 的髋/膝关节约为 51.6°/65.9°，而 reference 的基准姿态接近 compact（髋约 6.4°、膝约 52.1°）。标准直接切换在第 0 秒产生 45.2°的最大关节目标误差；绝对机械功率在 14 ms 达到 99.0 W。前 0.25 s 的绝对机械功为 7.10 J，是正常 compact 启动 1.93 J 的约 3.7 倍。

该 reference 的相位反馈用于维持已经建立的滚动，并没有把 Stand 收敛到滚动极限环的能力。Stand 初态位于其吸引域之外；一次大幅折叠虽然能使机器人翻倒，却不能建立正确的姿态、角速度和 reference 相位关系。需要单独的 stand→compact/低速滚动启动控制器，或把 Stand 初态和启动成功指标加入 CEM/PPO 训练。

## 产物

- `results/pupper_reference_primitive_stand_direct_20260908/stand_direct_standard_ramp_10s.mp4`：主测试 MP4。
- `results/pupper_reference_primitive_stand_direct_20260908/stand_direct_standard_ramp_10s.gif`：主测试 GIF。
- `results/pupper_reference_primitive_stand_direct_20260908/stand_direct_comparison.png`：三种初态/切换的状态对照。
- `results/pupper_reference_primitive_stand_direct_20260908/canonical_standard_ramp_10s.json`：正式主测试结果。
- `results/pupper_reference_primitive_stand_direct_20260908/canonical_full_reference_10s.json`：完整幅值敏感性测试。
- 三个子目录分别保存逐时刻轨迹、功率和汇总 JSON。
