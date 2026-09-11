# 滚动策略 PPO 退化：云端检查与恢复

以下命令只在 Linux 云端、`curl_robot_2d` 项目目录运行。此次修改只经过静态审阅，没有在本地运行测试、训练或仿真。参数是诊断起点，尚未证明能解决此次退化。

## 1. 先保留旧现场

成功率从约 80% 降至 0%，同时非横漂失败升至约 98%，说明原有滚动能力严重退化。快照影响初始难度，但目前不能据此认定快照是主因，也不能仅凭这三行日志认定优化器或奖励存在某个具体错误。

**先只上传 `scripts/collect_rolling_ppo_diagnostics.py`，暂时不要覆盖云端训练代码。** 在原来的训练 Python 环境执行：

```bash
python -m scripts.collect_rolling_ppo_diagnostics \
  results/rolling_command_ppo_stage1 \
  --out results/rolling_ppo_old_diagnostics.zip
```

将目录替换为实际退化训练目录。有终端日志时追加 `--log 实际日志路径`。脚本仅使用标准库，读取配置、指标、检查点文件清单、项目源码和已安装 Brax 源码/版本，不导入 JAX、不启动仿真，也不打包模型权重。源码反映采集时的文件；如果此前改过代码，请说明。发回 ZIP 和实际启动命令即可，不需要云端密码或 SSH 密钥。

## 2. 同步诊断功能

采集后同步这次修改的训练入口及依赖，至少包括：

- `scripts/train_mjx_3d_roll_student_dr_ppo.py`
- `curl_robot_2d_mjx/rolling_ppo_diagnostics.py`
- `curl_robot_2d_mjx/rolling_student_snapshot_pool.py`
- `curl_robot_2d_mjx/environment_rolling_student_dr_3d.py`
- 保持配套的 distillation 入口、`distillation_execution.py`、`distillation_evaluation.py`、PPO 网络转换与 wrapper 文件一致。

后续命令共用下面的 Bash 数组，在同一个终端定义。`STUDENT` 始终指向原来能滚动的蒸馏模型，不能换成已退化的 PPO 导出模型。四卡训练的 env 数量是全局数量。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
STUDENT=results/rolling_command_distill_medium_fast/student_params
COMMON=(
  "$STUDENT"
  --geometry rollingquad_2_abd10_no_self_collision
  --command-conditioned --rolling-snapshots
  --snapshot-pool-size 512 --eval-snapshot-pool-size 256
  --snapshot-warmup-min-steps 100 --snapshot-warmup-max-steps 300
  --forward-command-min-m-s 0.4111 --forward-command-max-m-s 0.8110
  --turn-command-min-rad-s 0.02 --turn-command-max-rad-s 0.08
  --turn-command-straight-fraction 0.4 --command-interval-s 10
  --episode-length 500 --minimum-success-turns 5
  --preset h200 --max-devices 4 --envs 2048 --eval-envs 64
  --batch-size 256 --num-minibatches 8 --unroll-length 20
  --dr-strength 0 --observation-noise-scale 1
  --student-anchor-weight 0 --initial-policy-std 0.02
  --entropy-cost 0 --discounting 0.995 --reward-scaling 1
  --fixed-eval-envs 64 --fixed-eval-seed 123456 --seed 0
  --training-metrics-steps 40960
)
```

如果原运行使用了其他 CEM controller，需要在 COMMON 中指定原来的 `--controller`。训练和普通 Brax 评估保留观测噪声；新增固定评估设观测噪声为 0、确定性动作、DR=0。这两个评估系列分别比较，不要直接混合数值。

## 3. 原策略与退化策略做相同初态对照

```bash
python -u -m scripts.train_mjx_3d_roll_student_dr_ppo "${COMMON[@]}" \
  --eval-only --out results/rolling_ppo_fixed_original

python -u -m scripts.train_mjx_3d_roll_student_dr_ppo "${COMMON[@]}" \
  --eval-only --restore-ppo results/rolling_command_ppo_stage1/params_final \
  --out results/rolling_ppo_fixed_degraded
```

如果旧训练尚未保存 `params_final`，使用旧运行 `ppo_checkpoint` 下**实际存在的某个数字步骤目录**作为 `--restore-ppo`，不要传父目录。新版也支持 `checkpoints/<步骤>/params` 文件。没有任何退化检查点时，先发第 1 步诊断包，再进行下面的短程检查。

核对两个 `fixed_eval_manifest.json` 的 `initial_state_sha256` 相同。相同 seed 但哈希不同，说明源码、模型、快照或运行环境未完全一致，不能声称是严格相同初态对照。

每次接管后评估完整 500 步，即 10 秒，教师 warmup 不计入。物理失败提前终止，不自动 reset。成功仍是无失败且至少 5 个有效滚动圈；这还不是按每条速度命令定义的达标率。速度误差从接管第一步统计。快照是 CEM 滚动态代理，并非真实 stand-to-roll 输出；接入真实 handoff 分布是后续工作。

## 4. 阶段 A：冻结 actor，只预热 critic

```bash
python -u -m scripts.train_mjx_3d_roll_student_dr_ppo "${COMMON[@]}" \
  --critic-only --steps 409600 --num-evals 11 \
  --learning-rate 0.0001 --updates-per-batch 2 \
  --clipping-epsilon 0.05 --max-grad-norm 0.5 \
  --stop-success-drop 0.15 \
  --out results/rolling_ppo_critic_warmup

python -m scripts.collect_rolling_ppo_diagnostics \
  results/rolling_ppo_critic_warmup \
  --out results/rolling_ppo_critic_diagnostics.zip
```

**先把此包发回审阅，再决定是否执行阶段 B。** 阶段 A 冻结 actor 的全部参数，包括探索 std，并保持 actor 观测归一化不变。检查：

- `actor_max_parameter_delta_from_start` 必须严格为 0；代码检查异常变化。
- 固定评估的成功率、失败数、动作差和跟踪误差应保持一致（设备浮点误差除外）。普通 Brax 评估重新抽样，可能波动。
- 原始模型的 `same_state_student_action_rmse` 应接近 0，用来检查 actor 转换、归一化和动作通道；不能只看 `anchor_rmse/step`，anchor 权重为 0 时旧日志的该值是占位零。
- 查看 value loss 是否相对回报尺度趋于合理，不能要求随机采样的每个点单调下降。解释 PPO loss/KL 必须结合实际 Brax 版本；未输出 KL 不等于 KL 为零。

`discounting=0.995` 的名义衰减时间尺度约 4 秒，用于比原来 0.99 更重视持续滚动；这是待验证的选择，A/B 保持一致。critic-only 仍有随机动作探索，与确定性部署并不相同。

## 5. 阶段 B：小步长 actor 微调（阶段 A 通过后）

```bash
python -u -m scripts.train_mjx_3d_roll_student_dr_ppo "${COMMON[@]}" \
  --restore-ppo results/rolling_ppo_critic_warmup/params_final \
  --steps 819200 --num-evals 21 \
  --learning-rate 0.00001 --updates-per-batch 1 \
  --clipping-epsilon 0.05 --max-grad-norm 0.5 \
  --stop-success-drop 0.15 \
  --out results/rolling_ppo_actor_probe
```

每约 40,960 个环境 transition 检查一次，而不是训练一百多万步后才看到退化。Brax 会按完整 rollout 取整。初次快照生成和 JIT 编译仍需等待，固定评估本身额外使用一张卡；训练使用最多四张卡。

这里先保持现有奖励配方和 anchor=0，减少同时改变的因素。若仍发生较大动作偏离，结合 KL/std/奖励分项再决定是否加入 anchor、限制 std 或调整损失；环境 anchor 是奖励惩罚，**不等同于直接监督损失或硬 KL 约束**。PPO clipping 和梯度裁剪也不保证动作改变很小。

每次回调先保存 PPO 参数和确定性 student，再运行固定评估。成功率低于本轮 step=0 基线超过 15 个百分点时，写入 `stopped.json` 并退出；不会自动回滚或继续训练。15 个百分点是防止大幅退化的临时阈值，不是可接受性能损失。非有限参数或诊断也会触发保存后停止。

检查 `fixed_eval_history.json` 的成功率、具体失败数、全时长比例、vx/yaw MAE 及有符号偏差、动作偏差、std、饱和比例和奖励分项。MAE 按有效 transition 加权；提前失败会缩短样本，不能把“早死但 MAE 更低”当成提升。保留滚动稳定性的前提下，再以低/中/高速和直行/左右转分组评估跟踪改进。不要仅用总体成功率挑最终策略。

每个 `checkpoints/<步骤>/student_params` 均可交给现有 distillation 分组评估器复测。`params_final` 和 `checkpoints/<步骤>/params` 保存 actor、critic 和统计量，**不保存优化器状态**，续训会重新初始化优化器。恢复时须保持原 student、网络结构、命令设置和观测定义一致。
