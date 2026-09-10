# 连续 Stand → CEM Roll → Stand 视频

在云端项目根目录、原训练 Python/JAX 环境中运行：

```bash
python -m scripts.render_stand_roll_stand --preflight
python -m scripts.render_stand_roll_stand --out results/stand_roll_stand_video
```

默认使用用户指定的两个参数文件：

- `results/stand_to_roll_symmetry_student/student_params`
- `results/roll_to_stand_absolute_guided_hold_robust_v1/brake_full/params_final`

还需要 student 同目录的 `student_source.json`、其原始 `training_config.json`、
`results/stand_to_roll_startup/bc/bc_params`（冻结归一化）、
`results/cem_cycle_data/cem_cycles.npz`（capture 与接管相位匹配），
以及 stand 参数同目录的 `training_config.json`。
如果 student_source 里的原始绝对路径已失效，用 `--student-config` 指定原配置；
脚本校验配置及 BC 的 SHA256，不要求加载原始 PPO checkpoint 权重。

默认 CEM reference 为项目高速零自碰撞版本：
`results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`。
可用 `--reference` 显式指定其他 CEM；没有对策略重新训练。

流程：

1. 从 full stand reset 开始运行对称 student，不额外做左右动作平均。
2. 沿用 student 环境的持续 capture 门槛；capture 时以 matcher 返回的振荡器相位启动 CEM。
3. CEM 连续运行至少 5 秒，之后保持控制并每个物理步检测 pitch 是否进入 +90° ±1°。
   `pitch=atan2(R[2,0],R[2,2])`，与 +90° handoff bank 相同。不会把 −90° 当成 +90°。
   默认总共 30 秒仍未到达则报失败，不强制切换、不冻结画面。
4. 将完整 MJX state 交给 `reset_from_roll_state`，检查 qpos/qvel/ctrl/time 完全不变。
   观测按训练的冷启动历史方式初始化，last_action 由最后 CEM ctrl 映射到站起策略的动作约定。
5. 站起策略持续控制，达到训练配置中的 ready hold + stand verification 才判成功。

全过程统一使用 roll-to-stand 训练的 nominal 模型和物理配置，关闭传感器噪声、DR 和外推力；
student 的初始关节/速度 reset 噪声保留，seed 固定为 0。
这是在同一模型上的组合验收，不等于两份 checkpoint 各自原训练环境的独立复现。
脚本检查两模型的形态、名称及关节排序，结构不兼容时拒绝运行；不会交接时更换模型或重置速度。

成功后输出 `stand_roll_stand.mp4`（1280×720、50 fps）、`final_frame.png`、
含实际时间戳/阶段/qpos/qvel/ctrl/pitch 的 `rollout.npz` 和 `report.json`。
报告包含参数哈希、两次切换时间、CEM 实际时长和最终站稳指标。
仿真失败仍保存已产生的轨迹与错误报告，不导出“成功”视频。
MP4 需要 `imageio`、`imageio-ffmpeg` 和 Pillow；云端默认 EGL。

本地仅验证 CLI、角度/时间门槛和缺失依赖报告。端到端仿真与视频目检需在取得云端参数后执行。
