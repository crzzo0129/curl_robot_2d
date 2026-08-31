# 自主 stand→rolling 启动 PPO

> 历史 v1 设计，以下命令不再适用于当前训练入口。当前默认已改为
> [stand→低速 compact v2](3d_compact_startup_training_zh.md)，请使用新文档；
> 不再传 candidate bank，也不能恢复旧 v1 启动参数。

入口：`python -m scripts.train_mjx_3d_startup_ppo`。

这是**新的启动教师**，不是给旧 rolling PPO 加 stand reset，也不是旧插值启动原型。
从 stand 自主控制，满足状态门控后交给冻结滚动教师，再把实际续滚结果计入同一个
episode。脚本可训练不意味着新策略已经通过课程或可部署。

## 训练内容与时序

- reset 只放置 stand 一次，保留原配置的关节 q/qd 噪声。
- 启动 actor 输出 8 个 hip/knee 的完整归一化位置目标，以原 compact 为动作零点；
  范围为 ±1，不叠加 reference，不乘 0.15 residual gain，不压缩左右差动。
  按关节名称映射 actuator，四个 abduction 保持原零位目标，仍有原有限力矩伺服。
- 初始网络的确定性均值为 stand；这是网络初始化，不是随时间执行的轨迹。
- 最多 3 s 自主收腿、倾转、加速；没有“到 1 s 就切换”的规则。
- 找到一个完整候选，连续 3 个控制帧满足条件后，进入冻结教师接管段。
- 接管后继续模拟 3 s，无配置内失败且至少新增 1.5 圈、净转角为正，才记启动成功。
  超时未接管、接管后失败、续滚进度不足分开记录。

启动和续滚共最多 6 s / 300 个控制步。若 1 s 接管，则约 4 s 就结束该 episode。
同一失败标准贯穿两段，尤其 `|y| > 0.20 m` 不清零、不放宽。
模型仍为 `_2` 完整几何、CG20、1 ms 物理步长、每动作 20 子步、±3 Nm 力矩限制。

## 如何接管而不伪造成功

接管时保持实际 qpos/qvel/ctrl、仿真时间、接触/求解器状态、全局原点、真实滚动相位、
上一实际动作及失败计数不变。**不把实际状态替换成候选快照**。

只初始化 CEM 上下文：

- oscillator phase = 候选 oscillator phase + 实际与候选 rolling phase 的有符号相位差；
- reference 控制时钟用 offset 对齐到候选的教师年龄；物理时钟继续前进，不倒拨；
- 教师观测由真实当前状态、实际上一动作及新 CEM 上下文重新计算。

切换前预测教师第一个物理子步的目标，限制它与当前 ctrl 的差。这个检测不是偷偷
混合控制器输出：通过后才切换，不通过就继续由启动 actor 控制。

PPO 环境在接管后仍逐控制步推进，不在 gate 内嵌一个巨大 rollout。此时 actor 的
动作被忽略，冻结教师输出真正生效；同一 episode 的回报和 value bootstrap 将
后续结果传回启动阶段。actor 观测包含阶段标志，教师参数不参与 PPO 优化。
这会增加教师段的采样开销，尚未实现只对启动段做 policy-loss mask 的专用 PPO。

## 候选库与暂定门槛

随代码提供 `assets/startup_handoff_gain_teacher_t1.json`，来自前次实际试验的 3 条
合格教师轨迹 t≈1 s 状态。它绑定用户提供的 `params_best (3)` 的 SHA-256、MJCF
及 task/reference/reward 配置，不允许换一个教师 checkpoint 后继续盲用这个库。
文件是小型 JSON，不必同步全部 NPZ 试验目录，也不包含教师网络权重。

| 每个候选分别检查的项目 | 默认上限 |
| --- | ---: |
| 所有 12 关节位置的最大误差 | 0.10 rad |
| 所有 12 关节速度的最大误差 | 1.0 rad/s |
| 根部高度误差 | 0.035 m |
| 根部各轴线速度最大误差 | 0.15 m/s |
| 根部各轴角速度最大误差 | 1.0 rad/s |
| 四元数姿态角距离 | 0.15 rad |
| 实际滚动相位误差（wrap） | 0.15 rad |
| 相对于 episode 起点的横向偏移 | 0.05 m |
| 实际滚动轴倾斜 | 0.10 rad |
| 教师首个物理子步的最大命令差 | 0.18 rad |

还要求没有 forbidden contact、物理失败或非有限教师动作。不限制 x 必须恰好等于
候选原轨迹的 x，避免把“先移动到某个世界坐标”当作目标。

**这些是用于训练探索的暂定容差，不是前次小扰动试验已认证的接管区域**。
`--gate-scale` 同时缩放前七项匹配容差，但不放宽 y、轴倾斜、命令连续性和任务失败标准。
最近候选由最坏归一化误差最小选取，不能拼接不同候选的关节/速度；候选切换会清掉
连续确认次数。最终是否真的接得住由后续真实 3 s 续滚验证。

更换教师、模型或候选时刻，应重新运行 probe，再导出：

```bash
python -m scripts.export_startup_handoff_bank \
  --probe results/handoff_teacher_new \
  --time-s 1.0 --out assets/startup_handoff_new.json
```

只导出完整来源合格、exact 回放正确、四类扰动均达到原 probe 成功标准的候选。
MJCF 校验保留原始 `model_sha256` 作为溯源，新增 `model_lf_sha256`：仅将 CRLF 换成
LF 后计算 SHA-256，兼容 Windows/Linux 及混合换行；不忽略其它空白、参数、元素顺序等
任何内容变化。新的 probe、候选导出和启动训练元数据都会保存这两个值。
它们只校验 MJCF 文件本身，不覆盖外部 mesh/include 文件；仍需同步完整模型资产。

随附候选库已补充 portable 哈希。补充前确认本地原始 XML 与原试验的原始哈希完全一致：
`9b3177d67c170cc5126d0cc46952eaf109a807b9c7d4708f4119a4b3961a58b8`；原文件 252 个换行中
44 个是 CRLF，其余是 LF。统一为 LF 后为
`f292c7e76d27721d40e93159012d7a1b905987d9dd8338c9bd73ef20a30c776a`。
没有修改候选状态、教师权重或 XML 物理参数。

遇到 `candidate bank MJCF does not match`，先同步新版代码**以及候选库 JSON**，原命令加
`--dry-run` 检查；通过后去掉该选项训练。若仍失败，错误会显示实际模型路径、预期/实际
哈希；此时不能归因于普通 CRLF/LF 差异，应核对模型版本，不要手工改哈希绕过验证。
旧 probe/旧训练记录没有 portable 字段时仍严格比较原始字节，不会自动信任当前模型。

## 奖励和观测

启动奖励包含到候选的连续 potential shaping、时间代价、动作变化和力矩代价。
有一次接管奖励，但完整成功奖励更大；教师段的新滚动进度、失败和最终成功也计入回报。
potential 在退出启动段时归零，避免重复接管刷奖励。没有预设收腿轨迹跟踪项。

目前是 **53 维特权观测 / 8 维动作** 的启动教师：根部高度/偏移/航向、机体轴、全部
速度、关节位置、实际上一动作、滚动相位，以及启动/接管计数。不是实机 720/12 维学生。
状态门控依赖仿真信息；最终要把两段专家一起蒸馏并评估独立学生，不能直接把这个
checkpoint 放进现有 rolling student 部署器或旧 65 维 rolling 教师入口。

训练物理频率与已有滚动教师保持 50 Hz。实机控制仍为 52 Hz，260 Hz 是 IMU 发布频率；
本脚本没有悄悄修改这些频率，也未完成 52 Hz 部署适配。

## 云端命令

下例教师目录沿用此前课程结果；其中的参数必须确实是用户本次提供的那份教师，且
`training_config.json` 与之对应。若重命名/复制了 `params_best (3)`，只改参数路径即可。

先检查配置、哈希和维度，不启动 JAX：

```bash
python -m scripts.train_mjx_3d_startup_ppo \
  --teacher results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --teacher-config results/rollingquad2_floor_mass_gain_v3_h200_seed0/training_config.json \
  --preset h200 --out results/rollingquad2_autonomous_startup_v1_seed0 \
  --dry-run
```

先跑小型完整 PPO 流程，验证本机依赖、采样、更新、保存和评估：

```bash
python -m scripts.train_mjx_3d_startup_ppo \
  --teacher results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --preset smoke --max-devices 1 \
  --out results/rollingquad2_autonomous_startup_smoke
```

正式名义训练：

```bash
python -m scripts.train_mjx_3d_startup_ppo \
  --teacher results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --teacher-config results/rollingquad2_floor_mass_gain_v3_h200_seed0/training_config.json \
  --candidate-bank assets/startup_handoff_gain_teacher_t1.json \
  --preset h200 --max-devices 1 \
  --startup-budget-s 3 --continuation-s 3 --minimum-turns 1.5 \
  --confirmation-steps 3 --seed 0 \
  --out results/rollingquad2_autonomous_startup_v1_seed0
```

默认 PPO 使用单卡；`--max-devices 1` 不阻止 JAX 发现其它卡。要连初始化都限制到单卡，
在命令前加 `CUDA_VISIBLE_DEVICES=0`。需要四卡时保持四卡可见并改 `--max-devices 4`。
脚本不设置 JAX 跨主机分布式初始化。显存不足先降 `--envs 256 --batch-size 128
--num-minibatches 4`。总环境步数包含教师续滚段，不全部都是启动 actor 动作。

## 在哪里看成功率

日志：

```text
[startup PPO] step=... handoff=...% success=...% failed=...% timeout=...%
```

- `handoff`：从 stand 达到状态门控并切换的比例。
- `success`：从 stand 出发、接管后完整续滚且新增圈数达标的比例。
- `timeout`：3 s 内未接管的比例。
- `evaluation_best.json`：训练结束后以新 seed 做独立整段评估。
  `success_rate` 才是完整启动成功率；`conditional_success_after_handoff` 是接管后的条件成功率。
- `metrics_history.json`：每次 PPO eval 的原始指标。
- `params_best` / `params_final`：按完整成功、接管、失败、回报依次排序的最佳参数及最终参数。
  即使尚未成功也保存最佳中间策略；保存不等于通过。
- `evaluation_best_arrays.npz`：逐 episode 统计，以及前 4 条的 qpos/qvel/ctrl、相位切换标志、
  时间、候选 ID、gate 误差和有效动作轨迹。

成功、失败和接管指标是一次性事件，不按剩余存活步数重复累计。
完整 autoreset 会重置控制上下文和 gate 状态、重新采样初态噪声，避免上一局信息泄漏。

## 续训和复评

`--restore-startup` 只接受本入口产生的启动参数及同目录元数据；不接受旧滚动教师参数。
它恢复网络和归一化器，优化器重新开始。每次使用新输出目录，不覆盖已完成的实验。

```bash
python -m scripts.train_mjx_3d_startup_ppo \
  --teacher results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --restore-startup results/rollingquad2_autonomous_startup_v1_seed0/params_best \
  --preset h200 --learning-rate 5e-5 \
  --out results/rollingquad2_autonomous_startup_v2_seed0
```

```bash
python -m scripts.train_mjx_3d_startup_ppo \
  --teacher results/rollingquad2_floor_mass_gain_v3_h200_seed0/params_best \
  --restore-startup results/rollingquad2_autonomous_startup_v1_seed0/params_best \
  --eval-only --eval-envs 256 --seed 100 \
  --out results/rollingquad2_autonomous_startup_v1_eval_seed100
```

复评需使用训练时相同的 hidden layers、候选库和 gate 参数。改变容差/续滚窗口属于
新的评价协议，记录在新输出目录，不能将两种成功率混为一谈。
当前只做名义物理参数训练，保留基础 reset 噪声；没有重新执行 floor/mass/gain curriculum。
在名义启动通过后再扩展初态、扰动与随机化，不建议仅扩大 gate 来提高 handoff 数字。

## 本地验证记录（2026-08-31）

- 9 项契约测试通过；5 项实际 MJX 集成测试通过，覆盖 stand reset、完整独立目标、
  接管不改物理状态、真实 gate 转换、教师段忽略启动动作、终止事件与完整 autoreset。
  部分集成测试使用构造的候选状态和零 residual 测试替身，只验证机制，不计入启动成功率。
- 相关回归测试共 178 项：基础 Python 下通过 172 项、跳过 6 项；其中 5 项新 MJX
  集成测试已在 MJX 环境另行运行通过，剩余 1 项为旧插值启动的可选 MJX 测试。
- 使用用户真实滚动教师，完成 1 env、4 个训练步的小型 PPO 更新、最佳/最终参数保存及
  新 seed 独立评估。成功率 0，失败原因为启动超时，没有非有限物理值；不能据此判断收敛。
- 已另行通过 `--restore-startup --eval-only` 重载评估；归一化器维度为 53，原先零初始化
  的 location kernel 已有非零更新，验证不是仅保存初始化参数。
- 本地 JAX 0.11.1 比项目目标 JAX 0.6.x 新，Brax 使用的复制入口在本地公开 API 中已移除。
  这次 PPO smoke 仅在验证进程内恢复 JAX 保留的旧入口实现，没有修改正式训练脚本、
  教师、物理或优化算法。详见 [本地验证上下文](../results/autonomous_startup_ppo_local_compat_smoke/validation_context.json)
  与 [smoke 汇总](../results/autonomous_startup_ppo_local_compat_smoke/summary.json)。

正式训练使用已有 `mjx312` 环境；没有在本地证明原生 GPU、多卡性能、课程鲁棒性或实机启动。
