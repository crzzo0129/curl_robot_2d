# 碰撞模型修订结果说明

本目录只存放有限足端、外壳缝隙和自碰撞修订后的结果，不覆盖
`results/phase_controller/` 或 `results/leg_crossing_analysis/` 中的历史
无自碰撞产物。

- `compact.png`、`open.png`：当前最终模型静态图；
- `old_controller_replay_after_collision.gif`：当前 1 ms 时间步和最终接触
  参数下的 10 s 旧控制器回放，应作为本目录的当前回放；
- `old_controller_crossing/`：与上述最终模型一致的 10 s 交叉和接触诊断；
- `old_controller_replay.gif`：接触调节过程中的较早中间回放，仅为保留已有
  结果而未删除，不应用于最终数值判断。

当前定量结论以
`old_controller_crossing/leg_crossing_summary.json` 为准。
