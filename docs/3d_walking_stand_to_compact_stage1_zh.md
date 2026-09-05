# 第一阶段：行走名义初始站姿 → 低速 compact

本阶段只验证固定站姿到 compact 的可达性。入口已实现，尚未完成 PPO 训练或收敛验收；不代表已能起滚或从行走中切换。

## 起点、模型与动作

- 使用 `rollingquad_2` 的 stand keyframe，根部额外抬高 5 mm，关节速度和根部速度为零。这对应行走代码的名义初始站姿及基础离地余量，不包含行走 reset 的随机偏航、关节噪声与额外高度噪声。
- 完整 mesh 与外壳碰撞保持启用，采用现有滚动环境 CG20、1 ms 物理步长、50 Hz 控制及原模型伺服/力矩限幅。行走入口关闭外壳接触，因此本阶段不声称逐项复用了行走物理配置。
- 先复用 8 维髋膝动作，4 个外摆保持零位伺服；观测为现有 53 维特权观测。它是固定站姿基线，并非 12 维行走策略的直接续训或可部署学生。
- 不加载行走或滚动权重。模型不会在 episode 中切换；reset 之后不改写姿态或清零速度。不播放固定收腿轨迹。

## 目标与终止

最多 3 s 内，连续 5 个控制帧（0.10 s）进入 compact 状态窗口：

| 指标 | 上限 |
| --- | ---: |
| 全部 12 关节最大角度误差 | 0.02 rad |
| 全部 12 关节最大速度绝对值 | 0.05 rad/s |
| 根部高度误差 | 0.01 m |
| 根部各轴线速度绝对值 | 0.02 m/s |
| 根部各轴角速度绝对值 | 0.10 rad/s |
| 姿态角距离及滚动相位误差 | 各 0.05 rad |
| 横向偏移 | 0.05 m |
| 滚动轴倾斜 | 0.10 rad |

同时要求没有当前禁止接触或物理失败。越过任一门槛会清除连续确认计数。达标即结束，不启动教师，不要求长期静止。原有物理失败检查、足端接触滑动惩罚、收拢势函数、余速惩罚和防弹跳项保留。

本模式不检查滚动教师首帧命令差，因为没有加载教师；这一项留给第二阶段。`--continuation-s` 和 `--minimum-turns` 在此模式不参与验收。

## 运行

以下命令在 `curl_robot_2d` 目录及已有 MJX/Brax 训练环境执行。配置检查不训练：

```bash
python -m scripts.train_mjx_3d_startup_ppo --compact-only \
  --preset smoke --dry-run --out results/walking_stand_compact_stage1_check
```

短接口测试（只保持初始站姿，不是成功演示）：

```bash
python -m scripts.train_mjx_3d_startup_ppo --compact-only \
  --preset smoke --smoke-steps 2 --out results/walking_stand_compact_stage1_smoke
```

通过接口测试后训练：

```bash
python -m scripts.train_mjx_3d_startup_ppo --compact-only \
  --preset h200 --max-devices 1 --out results/walking_stand_compact_stage1_seed0
```

使用新输出目录。此模式拒绝 `--teacher` / `--teacher-config`，并使用独立 contract 阻止误续训旧启动任务。默认 5 帧确认，可显式调整 `--confirmation-steps`，但验收报告必须注明实际时长。

## 结果解释与后续

独立评估输出 `compact_reach_rate`、拖地指标、到达时速度以及原始轨迹。共享日志中的 `handoff` 仅表示 compact 窗口已确认，`success` 仅表示本阶段成功；报告明确标注 `rolling_continuation_evaluated=false`。当前固定且无噪声的初态配合确定性评估会产生重复轨迹，批量 100% 不能解释为鲁棒成功率。

先检查动作轨迹和真实状态，确认不是弹跳、碰撞或短暂穿透造成的表面达标。然后增加站姿小扰动与多训练种子测试；若 8 维基线难以保持平衡，再扩展 12 维动作。

第二阶段使用实际到达状态连续接入冻结滚动策略，保留物理状态、上一命令和观测历史，检查首帧命令连续性并续滚验收。primitive 滚动策略需先在本阶段使用的同一模型上验证，不能通过切换碰撞几何宣称接管成功。最后才扩展为行走各步态相位起步。

## 本次本地检查（2026-09-05）

Python 语法检查及本次改动的 diff 空白检查通过。相关三个测试模块共 32 项：23 通过、5 项可选 MJX 集成测试跳过、4 项出错。其中 3 项因当前解释器没有 MuJoCo，另 1 项为未修改的历史候选库与当前模型指纹不一致。未修改历史指纹或放宽校验。依赖下载未完成，已停止安装；模型配置 dry-run、MJX stepping 与 PPO 均尚未验证。上述命令是待在完整训练环境执行的入口，不是已跑通的训练记录。
