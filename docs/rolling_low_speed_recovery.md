这套入口从校准教师蒸馏后的 `student_params` 继续训练，先补低速 DAgger，再分别运行 critic 预热和 actor 小步 PPO。默认源目录是 `results/rolling_distill_calibrated_20260912_034719`。源码修改与命令只做本地静态检查，未在本地运行测试、JAX、训练或仿真；实际收益需要云端确认。

**第一步：云端低速补训**

在云端 `curl_robot_2d` 项目根目录，先通过 Git 同步本次修改。如只同步了 `results/rolling_low_speed_recovery_v2.zip`，在项目根目录解压：

```bash
python -m zipfile -e results/rolling_low_speed_recovery_v2.zip .
```

2026-09-12 更新：v1 在各段改变训练 seed 时也改变了评价环境 reset 的 seed salt，导致命令面板不一致。已有权重无需重训，先按文末的 v2 复评命令修复比较。

该增量包依赖云端现有的校准蒸馏/PPO实现；它包含更新后的蒸馏入口、环境 reset 的可选 seed 参数、采样/筛选模块、恢复入口、轻量检查和说明，不包含训练权重。原 CEM controller、标定表与模型文件仍使用原运行记录中的路径。

云端先运行轻量检查（只需 NumPy，无仿真），再启动训练：

```bash
python -m unittest discover -s tests -p test_distillation_curriculum.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN="results/rolling_low_speed_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"
python -u -m scripts.run_rolling_low_speed_recovery distill \
  --source results/rolling_distill_calibrated_20260912_034719 \
  --out "$RUN"
```

记录打印的 RUN 路径，之后的阶段使用同一路径。每次新训练生成独立目录；脚本检测到已有目录时会报出实际路径，不覆盖历史结果。若源训练目录不同，只改 `--source`，脚本从其中的 `distillation.json` 继承 controller、标定表、模型、命令范围、网络、设备数、评价种子等设置。

可在执行前追加 `--dry-run` 查看完整命令；该模式只读配置并打印命令，不要求本地具备云端权重，不导入 JAX、不生成输出目录。

实际运行顺序：

1. 原 DAgger 策略重新评估，作为 `baseline`。
2. 最多 3 段续训，每段 500 次更新。恢复观察归一化、actor 与辅助速度头；跳过统计和 BC，每段重新初始化 Adam。学习率 `2e-5`，教师干预概率每段从 10% 降至 0%。
3. 仅训练初态/重置池按低、中、高速度目标概率 60%/20%/20% 重采样，保留原直行/左右转比例。完整物理状态、观察历史和上一动作使用同一组索引。重采样有放回，实际比例及不同快照数写入 `dagger_sampling_history`，不是保证每一步活跃样本比例都精确相同。
4. 聚焦模式把训练接管时的超时计数清零，允许后续完整 500 步；物理时间、相位、运动状态、历史和原始侧漂参考均保留。失败仍会提前重置。此前预热占用了训练的超时预算，新配置能够覆盖诊断中 5–10 秒的误差增长区间。旧的默认 `uniform` 模式不受此修改影响。
5. 每段评估仍使用完整原命令范围、原均匀速度采样、至少 256 环境、接管后 10 秒，并保存轨迹/同状态教师标签诊断。

**如何自动挑选候选**

候选必须同时提高低速成功率、降低低速 vx MAE，才替换此前选中的策略。另对原基线和当前最佳候选分别检查：

| 退化检查 | 允许变化上限 |
|---|---:|
| 总体成功率下降 | 2 个百分点 |
| 中速、高速各自成功率下降 | 3 个百分点 |
| 跑满时长比例下降 | 2 个百分点 |
| 总体 vx MAE 增加 | 0.005 m/s |
| 总体 yaw MAE 增加 | 0.003 rad/s |

出现以上退化、空分组、非有限指标或命令/评价口径变化时，停止后续 DAgger 段。所有已经完成的候选都保留，不覆盖原权重。未退化但尚未改善的候选可以继续下一段；最终所选仍是此前最佳。没有新候选通过时，`selected_student` 保留原 DAgger 策略。

这些阈值是本轮保守筛选规则，不是统计显著性证明。v2 将评估环境 seed 与训练 seed 隔离，并复用首次生成的 `evaluation_snapshots.npz`，其中保存完整状态、观察历史和上一动作。各候选的 `command_evaluation.json` 必须具有相同 `initial_state_sha256`，否则拒绝筛选。缓存兼容性检查包括命令/物理配置、环境代码、模型 XML、controller、标定表和状态结构；缓存不含可执行的 pickle 对象。后续候选只做初始模板 reset，不重复 CEM 预热。在独立面板验证前，不把小幅提升当作已确定收益。当前成功判据仍不等于速度跟踪合格率。

训练结束查看：

- `$RUN/recovery.json`：实际命令、每段筛选原因、选中的 checkpoint、阶段完成状态。
- `$RUN/baseline/` 和 `$RUN/dagger_01/` 等：分组评价及轨迹。
- `${RUN}_diagnostics.zip`：自动生成的诊断包，不含权重。异常退出也尽量保存已有日志和诊断。

优先发回这个 ZIP，核查低速改善是否伴随中高速退化，再使用以下 PPO 阶段。通过 Git 回传可执行：

```bash
git add -f "${RUN}_diagnostics.zip"
```

再按现有 Git 工作流提交并推送。脚本不会替你提交或推送。

**第二步：critic 预热**

沿用第一步的 RUN（新终端需重新设置成实际路径）：

```bash
python -u -m scripts.run_rolling_low_speed_recovery critic --out "$RUN"
```

读取 `recovery.json` 中所选学生，使用相同标定表和命令范围；冻结 actor，训练 critic 409600 环境步，学习率 `1e-4`、每批更新 2 次。4 卡/2048 环境从原记录继承；训练快照池 2048，评价快照池 1024。使用均匀训练采样，先保持 DR=0，训练观察噪声为 1，固定评价观察噪声为 0。沿用当前默认跟踪奖励权重，不启用此前退化的 tracking_focus 配方。

查看 `$RUN/critic/fixed_eval_history.json`：actor 参数差必须为零（训练器已有检查），固定评价应基本稳定；结合回报尺度检查 value loss。每个 PPO 阶段内部复用缓存评价初态。它与 DAgger 的评价采样入口不同，成功率应在各自基线内比较。critic 非零退出时不会标记阶段完成，actor 入口会拒绝继续。

**第三步：小步 actor PPO**

```bash
python -u -m scripts.run_rolling_low_speed_recovery actor --out "$RUN"
```

恢复本轮 critic 的 `params_final`，运行 819200 环境步，每批更新 1 次；actor 初始学习率及上限均 `3e-6`，KL 自适应、目标 KL=0.01，下限 `1e-7`。继续 DR=0；固定成功率相对该阶段起点下降超过 5 个百分点时，训练器保存现场并停止。这个 PPO 止损针对总体成功率，不能替代分组跟踪审查。

各 PPO 阶段会额外生成 `$RUN/critic_diagnostics.zip`、`$RUN/actor_diagnostics.zip`，同时更新外层诊断包。最终候选按 `best_fixed_checkpoint.json` 查看，不能默认最后一步最好。确定低速、直行漂移与转向均改善后，再制定逐步开启 DR 的下一阶段；当前入口不自动扩大 DR。

**已有第一段结果：只复评，不重训**

这条命令针对 `rolling_low_speed_20260912_065438`，读取旧 `recovery.json` 中的原策略和已完成候选路径。先同步 v2 源码；若只传 ZIP：

```bash
python -m zipfile -e results/rolling_low_speed_recovery_v2.zip .
python -m unittest discover -s tests -p test_distillation_curriculum.py
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m scripts.run_rolling_low_speed_recovery recheck \
  --out results/rolling_low_speed_20260912_065438
```

原策略与 `dagger_01/student_params` 使用同一实际状态池，原训练 seed 作为显式评价环境 salt，以保留旧基线命令面板。只运行评价，不修改任何网络权重。新结果放在运行目录的 `recheck_<时间>/`，筛选结论在其中的 `comparison.json`。尚未启动 critic/actor 时，按相同筛选规则更新 `recovery.json` 中的 `selected_student`；若 PPO 已经启动，记录建议但不更改对应学生。

复评结束更新外层 `results/rolling_low_speed_20260912_065438_diagnostics.zip`，发回这个 ZIP 即可。状态缓存留在云端，不放入诊断包；包内报告保留实际初态校验值。后续 PPO 先等这次可比评价的结果，不要直接把 79.7% 视为已确认提升。

直接使用蒸馏入口时，`--eval-seed` 现在同时决定默认的评价环境 salt；复现旧评价面板可显式指定 `--eval-environment-seed 原训练seed`。固定 seed 仍可能有 GPU 数值差异，严格配对还需让两个命令共享 `--eval-snapshot-cache 同一路径.npz`。不要删除正在用于候选比较的缓存。
