# 碰撞约束 CEM 正式结果

正式搜索配置为 14 代、每代 48 个候选、8 个精英、3 代启动越障课程、
6 s 持续评价、10 s 最终验证，随机种子 23。搜索从历史控制器附近热启动，
但所有候选均在当前自碰撞模型上重新评价。

- `best_phase_controller.json`：最终参数、碰撞权重和 10 s 汇总；
- `cem_history.csv`：逐代搜索记录；
- `best_rollout.csv`：10 s 完整时序；
- `best_rollout.png`：运动、力矩、能量、形变、自接触和穿透汇总；
- `active_roll.gif`：带接触诊断的 10 s 回放；
- `crossing_analysis/`：独立腿部交叉和具体接触对分析。

定量解释见 `docs/collision_constrained_cem_zh.md`。
