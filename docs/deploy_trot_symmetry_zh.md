# 左右镜像与实际周期步态约束

训练入口 scripts/train_ppo_deploy.py 新增三项可选设置；均默认 0，保持旧实验
和旧续训的目标不变。推荐通过四卡 symmetry 实验比较，不能仅凭总 reward 排名。

| 参数 | 对比实验启用值 | 作用 |
| --- | ---: | --- |
| --lr-symmetry-weight | 0.01 | actor 左右镜像一致性，包含直行、侧移、转向 |
| --trot-phase-weight | 0.05 | 稳定直行时两组对角腿相差半周期的奖励 |
| --cycle-balance-weight | 0.02 | 左右对应腿实际周期统计量不一致的惩罚 |

前后镜像保持 0.01，action rate 保持 0.10；其余奖励、DR、指令分布、网络结构、
720 维观测和 12 维部署输出不变。没有给策略或真机添加外部相位时钟。
左右转向仍从原混合指令分支采样，没有新增纯转向采样桶。

## 空间镜像

左右反射为机身坐标 y→−y，交换 FL↔FR、RL↔RR。当前模型对应关节不用反号，
启动时检查默认值、限位、关节轴、足端映射；CAD 和足端球体两种模型均支持。
局部静态采样最大足端误差约 1.07 mm，不表示质量分布和每一次 DR 实现都严格对称。

每帧 gyro 符号为 [-1,+1,-1]，重力和期望竖直向量为 [+1,-1,+1]，
指令 [vx,vy,wz] 变成 [vx,-vy,-wz]，关节偏移和 last action 随腿交换。
全部 20 帧保持时间顺序，在归一化之前反射原始观测。左右损失只在最新指令
norm(command)>0.05 时有效，前后损失仍只在直行样本上有效。

使用 tanh 后的确定性 action 比较，原始和镜像 actor 分支都求导。
原始 actor 前向在两个镜像项间共享；关闭的分支不做额外前向。
这是 actor loss，不直接计入环境 reward；不会镜像 PPO 回报或 critic 目标。

## 周期关系

期望 FL/RR 同相、FR/RL 同相，两组相差 0.5 个周期。周期约束只在
abs(vx_cmd)>0.05、abs(vy_cmd)<0.05、abs(wz_cmd)<0.15 时启用。
指令变化、站立、转向、终止清除统计；新指令前 0.5 s 不开始记录完整周期。

用实际足端接触（沿用环境的足底高度与上一帧接触过滤）记录两次触地之间的周期。
有效周期为 0.2–1.2 s，其中有其他脚支撑的摆动至少 0.06 s、峰值离地至少 8 mm。
这只是“有效迈步”的识别门槛，原来的 4 cm 抬脚奖励目标仍然保留。
周期统计过期或任一腿没有有效周期时，不给相位奖励和周期平衡罚分。

在每次有效触地时，比较其他腿最近触地时间与自己的实测周期。循环相位误差
使用高斯质量 exp(-mean(error²)/0.15²)，允许有限时间误差和双支撑区间，
不用硬编码四脚接触的二值模板。最近一次质量保留至下一个周期或统计过期。
需要四条腿各自完成有效周期才能拿到完整的相位反馈，刚启动时日志为 0 正常。

正确方向的实际机身 vx 必须超过 0.05 m/s；奖励从该门槛线性增加，在
min(abs(vx_cmd), 0.15) m/s 达到全额，避免为了领奖而超出低速指令。
站着、反向运动、四脚同时腾空不能靠旧记录获利。
周期奖励最多 0.05/步，不替代速度追踪和原有 hip ROM 不足惩罚。

左右对应腿（FL/FR、RL/RR）比较：完整周期 hip 时间均值、hip 峰峰值、
足端在机身前后方向的行程。三项误差尺度分别为 0.20 rad、0.20 rad、0.03 m，
归一化平方误差各截断至 1，取平均再乘权重。罚分最多 0.02/步，同样使用有效周期
和实际运动门槛。这里约束的是触地相位与周期统计量，不保证完整关节波形逐点相同。
不要求同一时刻左右 hip 角相等，也不强迫转弯时内外侧步幅相等。

## 四卡对照实验

| GPU | 配置 | 前后镜像 | action rate | 左右镜像 | 相位奖励 | 周期平衡 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | baseline | 0.01 | 0.10 | 0 | 0 | 0 |
| 1 | lr_only | 0.01 | 0.10 | 0.01 | 0 | 0 |
| 2 | cycle_only | 0.01 | 0.10 | 0 | 0.05 | 0.02 |
| 3 | lr_cycle | 0.01 | 0.10 | 0.01 | 0.05 | 0.02 |

同步 train_ppo_deploy.py、deploy_symmetry.py、新增 deploy_cycle_gait.py、
launch_deploy_reward_sweep.py。服务器 curl_robot_2d 目录下，在四张 GPU 可用时：

```bash
python -m scripts.launch_deploy_reward_sweep \
  --sweep symmetry \
  --resume rollingquad_2_deploy_robust_dr_checkpoints/000167772160.bin \
  --prefix trot_symmetry_v1 --gpus 0,1,2,3 \
  --collision-model foot-spheres --launch
```

可将 --resume 换为服务器上另一个仍正常行走的 checkpoint，四组必须共用
同一个初始文件。不要用旧四组各自的 checkpoint 开始这轮对照，否则起点不同。
每组默认 1024 环境、batch64、300M 新采样步数，使用同一默认随机种子。
去掉 --launch 可预览。旧 reward 实验仍可通过原 --continue-from 命令续训；
未指定 --sweep 的新实验仍使用旧 reward 组合。

本轮中断后的续训：

```bash
python -m scripts.launch_deploy_reward_sweep \
  --continue-from results/trot_symmetry_v1 \
  --prefix trot_symmetry_v2 --launch
```

自动继承四组各自的新权重与碰撞模式，从各自最新 checkpoint 恢复参数。
日志和 manifest 在 results/<prefix>；模型和视频位于项目目录下对应实验名。

## 观察指标与验证范围

lr_symmetry_action_rmse 衡量左右镜像误差，trot_phase_quality 在有效周期齐全时
接近 1 表示触地相位符合 trot，cycle_hip_mean_error / cycle_hip_rom_error /
cycle_foot_span_error 分别报告 rad、rad、m。cycle_valid_fraction 衡量完成有效
周期的腿的比例。训练打印的是整段 episode 均值，包含站立/转向等无效区间，
不能把较低的均值直接解释为直行时的相位都很差。

仍需结合前后速度、左右转速、是否持续迈步和视频；更低的镜像误差不等于
更好的行走。新增正奖励会改变总 reward，四组应按相同新采样步数比较。
四组验证只用于初筛，最终候选需要换随机种子，并回到 CAD 碰撞模型评估。

本地通过 NumPy 合成触地/关节序列、损失接线、四组续训配置及原生 MuJoCo
镜像几何静态检查；未运行 JAX 测试、自动微分、训练或 GPU 性能测试。
