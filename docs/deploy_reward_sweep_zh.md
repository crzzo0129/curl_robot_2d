# 四卡独立奖励对比

新增 `--sweep symmetry` 可运行“基线 / 左右镜像 / 周期步态 / 两者都有”的
四组对照，详见 [对称步态实验](deploy_trot_symmetry_zh.md)。本页原有组合仍是
默认 `--sweep reward`，旧实验续训保持原设置。

四张 GPU 分别训练四个独立策略；同一初始 checkpoint、默认随机种子、
DR 配置、采样分布和训练步数，仅改变下表两项。镜像项是 actor loss，
action rate 项是环境奖励中的相邻 action 差值平方和惩罚。

| GPU | 镜像权重 | action rate 权重 |
| --- | ---: | ---: |
| 0 | 0 | 0.08 |
| 1 | 0 | 0.10 |
| 2 | 0.01 | 0.08 |
| 3 | 0.01 | 0.10 |

每组默认 1024 个环境、batch size 64、300M 新采样步数；不启用 terrain。
默认使用 CAD 碰撞；可加 `--collision-model foot-spheres`，让四组都使用
原生足端球体模型来减少碰撞计算，详见 [足端球体模式](deploy_foot_spheres_zh.md)。
每个进程通过 CUDA_VISIBLE_DEVICES 只看到指定的单张卡。请在四张卡可用时
启动；已有训练需要先自行安排停止或迁移。四组并行可以同时比较，单组运行
速度不保证与四卡协同训练相同。恢复模型参数，不恢复优化器和累计训练步数。

在服务器 curl_robot_2d 目录中，先预览：

```bash
python -m scripts.launch_deploy_reward_sweep \
  --resume rollingquad_2_deploy_robust_dr_checkpoints/000167772160.bin \
  --prefix reward_ab_v1
```

正式启动四个后台进程：

```bash
python -m scripts.launch_deploy_reward_sweep \
  --resume rollingquad_2_deploy_robust_dr_checkpoints/000167772160.bin \
  --prefix reward_ab_v1 --gpus 0,1,2,3 --launch
```

可以将 --resume 换成另一个仍能正常行走的 checkpoint，四组共用该文件。
--num-envs 和 --batch-size 按每组设置；降低规模时必须保持
batch_size * 32 能被 num_envs 整除，四组使用相同规模。

启动器在 results/reward_ab_v1/ 保存四份日志、PID 文件、manifest.json。
模型和视频仍保存在项目目录，前缀是
rollingquad_2_deploy_reward_ab_v1_<组合名>，生成 XML 也按实验名隔离。
如果目标实验已存在，启动器拒绝覆盖；新实验请换 --prefix。
打印 PID 仅表示进程已创建，模型加载、编译或训练错误请看各组日志。

```bash
tail -f results/reward_ab_v1/sym001_rate010.log
```

需要停止某一组时，用 manifest 或对应 .pid 文件中的 PID 执行
`kill -INT PID`，让训练程序处理停止。四组停止后可整组续训：

```bash
python -m scripts.launch_deploy_reward_sweep \
  --continue-from results/reward_spheres_v1 \
  --prefix reward_spheres_v2 --launch
```

启动器读取旧 manifest，为四组分别选择各自目录中步数最大的数字命名
checkpoint，继承各组奖励、GPU、碰撞模式、环境数和 batch size。
在新目录保存结果，保留旧 checkpoint 和视频；四组来源文件会逐一打印并
记录到新 manifest。缺少任一组 checkpoint 会在启动前报错，不会退回
共同的初始策略。如果旧 manifest 记录的进程仍在运行，也会拒绝重复启动。
去掉 --launch 可预览命令。再次续训 v2 应指向 results/reward_spheres_v2，
同时选择新的 --prefix。恢复的是模型参数，优化器和训练计数会重新开始，
默认再训练 300M 步，eval 从 1/30 重新显示。

比较时按相同采样步数检查前后速度追踪、是否持续迈步、动作震荡、
打滑/擦地和失败情况。不同 action rate 权重会改变总 reward，不能仅靠
总 reward 排名；镜像误差较低也不等于走得好。一轮相同种子的实验用于
筛选设置，最终候选还应换种子复核。

本地仅进行语法检查和启动命令预览；未启动 JAX、GPU 训练或性能测试。
