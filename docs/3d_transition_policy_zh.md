# 3D ROLL → BRAKE → DEPLOY → STAND 第一版

## 结论

第一版采用三个独立策略和一个确定性监督器：

```text
ROLL policy --stop request--> Transition policy --READY hold--> WALK policy
                              BRAKE -> DEPLOY -> STABILIZE
```

ROLL 和 WALK 保持已有的两个训练策略。新增的 Transition policy 是一个 12
自由度策略，统一学习 BRAKE、DEPLOY 和 STABILIZE；高层不直接生成关节轨迹，
只负责不可逆地切换策略，并对 READY TO WALK 做连续时间消抖。

## 为什么 DEPLOY 不是直接到 park

最终 Pupper 模型同时包含 `compact`、`park` 和 `stand` 关键帧。CPU MuJoCo
检查表明 `park` 可作为接触捕获中间姿态，但 WALK policy 的训练初态是
`stand`。因此参考中心采用：

1. BRAKE：`compact`；
2. DEPLOY 前段：`compact → park`；
3. DEPLOY 后段：`park → stand`；
4. STABILIZE：`stand`。

策略输出不是这条轨迹本身，而是围绕参考中心的 12 维关节目标残差。这样既
保留了容易训练的姿态先验，也允许策略根据碰撞、速度和姿态误差主动调整。

## 训练课程

训练按反向课程逐阶段热启动：

1. `deploy_near_stand`：在接近站立的小扰动状态学习稳定和 READY；
2. `deploy_capture`：扩大到 compact/park/stand 之间并加入中等速度和倾角；
3. `brake_low`：从低速蜷缩滚动态学习 BRAKE 后接入 DEPLOY；
4. `brake_full`：覆盖完整滚动相位和目标速度范围。

前三阶段可以直接使用代码中的合成 reset 分布。正式 `brake_full` 训练前，应
冻结已经训练好的 ROLL policy，在相同 12 自由度 Pupper 模型中采集终止快照，
然后用这些快照替换合成分布。Actor 的 66 维观测和 12 维动作接口无需改变。

## READY TO WALK 门限

只有以下条件连续满足 0.40 s，监督器才允许切换到 WALK：

- 线速度不高于 0.12 m/s；
- 角速度不高于 0.45 rad/s；
- 机身倾角不高于 0.22 rad；
- 相对 `stand` 的关节 RMS 误差不高于 0.20 rad；
- 根节点高度位于 0.145–0.235 m；
- 至少 3 足接触；
- 状态均为有限值。

任意一项失效都会清零 READY 累计时间。收到 stop 后不会自动退回 ROLL，
切到 WALK 后也不会因单帧噪声退回 Transition。

## 本地检查

无需 JAX 的检查：

```powershell
python -m unittest tests.test_transition_3d -v
python -m scripts.train_mjx_3d_transition_ppo --stage deploy_near_stand --dry-run
```

## 云端顺序

每一阶段先 smoke，再正式训练，并把上一阶段 checkpoint 作为下一阶段的恢复点：

```bash
python -m scripts.mjx_3d_transition_smoke --stage deploy_near_stand
python -m scripts.train_mjx_3d_transition_ppo --stage deploy_near_stand --preset h200

python -m scripts.mjx_3d_transition_smoke --stage deploy_capture
python -m scripts.train_mjx_3d_transition_ppo --stage deploy_capture --preset h200 \
  --restore-checkpoint results/mjx_3d_transition_ppo/deploy_near_stand/ppo_checkpoint
```

随后以相同方式训练 `brake_low` 和 `brake_full`。第一轮验收重点是成功率、失败
率、READY 连续保持时间、切换时速度/倾角/关节误差，以及 WALK policy 接管后
前 1 秒是否仍保持站立。

