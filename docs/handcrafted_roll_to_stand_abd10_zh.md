# 手写 Rolling → Stand

使用指定 reference：`results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json`。

模型固定默认为 `assets/rollingquad_description_2/mjcf/rollingquad_abd10_no_self_collision.xml`；保持 XML 的接触设置、物理步长和电机参数，ABD 沿用前 −10°、后 +10°。

上仰为正 pitch。默认至少滚动 2 s 且累计超过 1 圈，然后等待下一次从正 pitch 进入 `[0°,15°]`、朝 0° 回落的窗口。必须是进入窗口的瞬间，不能因启动等待刚结束而在窗口末端迟触发。pitch 从实际机身旋转矩阵读取，不使用 reference 振荡器相位作为姿态。

切换时记录当前 12 个电机目标，用 150 ms 线性插值到 XML 的 `stand` 关键帧目标，然后保持 3 s。不制动、不重置速度、不覆盖 qpos；快速变化的是伺服目标，实际关节仍由物理仿真响应。

在项目目录运行：

```powershell
python -m scripts.run_handcrafted_roll_to_stand
```

无窗口并保存视频：

```powershell
python -m scripts.run_handcrafted_roll_to_stand --headless --video
```

可用 `--lead-deg 20 --deploy-duration 0.10` 调整提前量和展开速度；角度越大越早展开。默认输出 `results/handcrafted_roll_to_stand_abd10/` 中的 MP4、触发和最终截图、CSV、NPZ 及 summary.json。

本次数值回放：4.325 s、pitch +14.974°、pitch 角速度约 −3.701 rad/s 时触发；150 ms 完成目标插值。最终高度 0.15295 m、倾斜 2.194°，物理子步峰值电机力矩 2.290 N·m，自接触采样数为零。末段连续满足站立判据约 1.345 s。站立判据为至少三足接触、无非足端地面支撑、倾斜小于 0.22 rad、线速度小于 0.12 m/s、角速度小于 0.45 rad/s；连续满足 1 s 记为本基线成功。

这是单个确定性初态上的手写基线，未做扰动鲁棒性或自碰撞模型验证。

## −15° 对照

按同样的 150 ms 插值，在同一圈附近延后到 −15°（低头）再展开：

```powershell
python -m scripts.run_handcrafted_roll_to_stand --headless --video --deploy-at-pitch-deg -15 --min-roll-turns 1.5 --out results/handcrafted_roll_to_stand_abd10/minus15_015
```

`--deploy-at-pitch-deg` 直接指定进入触发窗口时的角度，优先于旧的目标角加提前量参数。`--min-roll-turns 1.5` 保证对比同一圈附近，避免 −15° 在第一圈后就触发而混入启动速度差异。

实测 4.438 s、pitch −15.087° 触发，比 +15° 基线晚 113 ms。触发时 pitch 角速度约 −5.167 rad/s；随后翻倒，最终倾斜约 158.98°，连续稳定站立时间为零。按本脚本的定义，+15° 是上仰、尚未转过水平，−15° 是低头、已经转过水平。本次对照支持在正 pitch 时提前展开，不能据此断言所有 −15° 参数组合都会失败。
