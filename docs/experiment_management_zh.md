# 实验记录与决策规范

更新日期：2026-07-28

## 1. 目标

实验记录的目的不是保存所有终端输出，而是让三个月后仍能回答：

- 当时验证的假设是什么；
- 与父实验相比只改变了什么；
- 使用了哪个模型、代码和随机种子；
- 结果是否通过预先定义的门槛；
- 这个结果导致了什么下一步决策。

机器可读索引位于 `experiments/registry.csv`。大型 checkpoint、rollout 和视频
仍保存在各自的 `results/<run_id>/` 目录中。

## 2. 实验 ID 与目录

推荐 ID：

```text
<METHOD>-<YYYYMMDD>-<SHORT_NAME>-S<SEED>
```

例如：

```text
PPO-20260728-ROOTFIX-S0
CEM-20260727-COLLISION-S23
PPO-20260729-CLOCK-RAMP-S1
```

目录名使用小写下划线形式，并保持与 ID 可互相识别：

```text
results/mjx_ppo_rootfix_seed0
results/collision_constrained_cem
results/mjx_ppo_clock_ramp_seed1
```

聚合多个种子的实验可以省略 ID 末尾的 seed，并在 `seeds` 字段写完整列表。

## 3. 开始实验前

必须先填写：

- `hypothesis`：一个可证伪判断；
- `parent_id`：直接对照实验；
- `change`：与父实验相比的唯一改动或改动组；
- `stage`：对应路线图 M0-M9；
- `seeds`、训练预算和评价级别；
- `pass_gate`：开始前确定的通过条件；
- `artifact_path`：输出目录，禁止复用非空目录。

不建议使用“试试更久”“调一下参数”作为假设。合格示例：

> 在 root_low 已禁用且其他终止条件不变时，降低学习率并增加 entropy，
> 能使 PPO 通过 S1 门槛。

若一次同时改变学习率、entropy、discounting 和 terminal penalty，记录中必须
写明它是一个“稳定性参数组”实验，结果不能归因于其中单个参数。

## 4. 运行完成后

必须补齐：

- `status`：`PLANNED`、`RUNNING`、`COMPLETED`、`FAILED` 或 `ABORTED`；
- `key_result`：独立物理指标，不写“reward 变高”作为唯一结论；
- `gate_result`：`PASS`、`FAIL` 或 `NOT_EVALUATED`；
- `decision`：继续、停止、复现、升级评价或修改假设；
- `notes`：异常、云实例、手工操作、日志路径等。

失败实验不能删除。`ABORTED` 用于人工停止或基础设施中断，`FAILED` 用于实验
正常结束但未达到门槛或出现数值失败。

## 5. 最低产物

### CEM/轨迹优化

- 完整配置和 seed；
- 搜索 history；
- 最优控制器；
- 独立 rollout 时间序列；
- 汇总 JSON；
- 至少一个可视化回放。

### PPO/RL

- `training_config.json` 和 `reward_config.json`；
- metrics/reward history；
- best/final checkpoint；
- deterministic evaluation rollout 和 summary；
- 完整控制台日志；
- 进入 V1 后的多 seed 和 CPU 回放汇总。

### 结构或物理扫描

- 被扫描参数及单位；
- 固定不变的 controller/policy；
- 参数网格或采样 seed；
- 每个候选的硬失败和任务指标；
- 明确区分 fixed-policy sensitivity 与 retrained potential。

## 6. 单实验记录模板

需要更详细说明时，在结果目录放置 `experiment.md`：

```markdown
# <EXPERIMENT_ID>

Status:
Stage:
Date:
Owner:
Parent:
Artifact:

## Hypothesis

## Change From Parent

## Fixed Conditions

## Budget And Seeds

## Predefined Gate

## Results

## Failure Analysis

## Decision

## Next Experiment
```

## 7. 当前记录说明

`registry.csv` 已加入三个关键节点：

- 当前碰撞模型下的 CEM 可行性基准；
- 2026-07-27 未产生合格候选的 PPO 参数扫描；
- 正在云端运行的 rootfix 稳定性参数组实验。

云端任务结束后，先把完整结果目录同步回来，再更新 `key_result`、`gate_result`
和 `decision`。不能根据训练过程中的单个 eval 点提前标记 `PASS`。

