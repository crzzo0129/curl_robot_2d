# 仅足端球体碰撞的走路模型

训练入口新增 `--collision-model foot-spheres`，默认仍为 `cad`。
该模式只启用 4 个 MuJoCo 原生 sphere 与 floor 的碰撞，不使用 mesh
模拟球体。四个球心与现有 foot site 重合，半径均为 0.0195 m；与奖励
里的足端净高度计算使用同一组位置、半径。站姿下球底与原 CAD 脚底
高度差约 0.16–0.22 mm，默认站姿和重置高度可以沿用。

所有原 CAD geom 保留外观，逐个关闭 contype/conaffinity；清除显式
contact pair，避免绕过碰撞掩码。没有躯干碰撞盒，没有腿部、躯干或
自碰撞。球体 mass=0，各刚体已有的显式质量、惯量、关节和电机参数
保持原值。原来关闭自碰撞的 CAD 模型有 13 个地面候选对，新模式只有
4 个球体地面候选对。训练前会检查编译后模型的碰撞掩码和球心/半径。

这能减少 mesh 碰撞计算，实际显存和训练提速幅度尚未在 GPU 上测量。
物理步长、求解器和奖励没有随碰撞模式改变。模型忽略躯干/腿部触地，
因此适合先快速筛选走路策略；选出策略后要回到 CAD 碰撞模型评估。
低矮姿态和摔倒时的接触行为与原模型不同，现有高度、姿态终止条件仍在。

四卡对比使用同一种碰撞模式，保证四组间仅奖励设置不同：

```bash
python -m scripts.launch_deploy_reward_sweep \
  --resume rollingquad_2_deploy_robust_dr_checkpoints/000167772160.bin \
  --prefix reward_spheres_v1 --gpus 0,1,2,3 \
  --collision-model foot-spheres --launch
```

四张卡应可用；从同一个旧 CAD 策略加载参数开始适应新碰撞模型。
单卡训练示例（将设备号换为可用 GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_deploy dr \
  --resume rollingquad_2_deploy_robust_dr_checkpoints/000167772160.bin \
  --run-name walk_spheres_v1 --collision-model foot-spheres \
  --num-envs 1024 --batch-size 64 \
  --fb-symmetry-weight 0.01 --action-rate-weight 0.10
```

新生成的运行 XML 带 `_foot_spheres` 后缀，检查点目录记录
`collision_model_config.json`。中断后恢复时也必须带相同 collision-model。
不指定 run-name 时该模式自动使用独立的 foot_spheres 输出名称。
支持与 terrain 组合；高度场只与四个球体碰撞，配对平地视频使用相同球体。

可单独生成用于 MuJoCo 查看器的模型，不加载策略、不导入 JAX：

```bash
python -m scripts.deploy_collision
```

生成 `assets/rollingquad_description_2/mjcf/rollingquad_walk_foot_spheres.xml`。
其原始物理选项来自源 XML，训练时仍使用 walk3d 的求解器设置。
球体属于 geom group 3；在查看器启用该组能看到半透明橙色球体，CAD
外观仍保留。训练运行时从当前源 CAD XML 动态生成，不依赖预生成文件。

静态验证覆盖完整 XML 解析、原生 MuJoCo 加载、质量/惯量/电机/站姿
一致性、四个实际球地接触、显式 pair 清除和高度场兼容性。
未运行 JAX、策略 rollout 或 GPU 性能测试。
