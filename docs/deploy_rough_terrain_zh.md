# Deploy：从平地策略继续训练轻度崎岖地形

通过地形的任务难度促使策略改善落脚和支撑，保留 hip 摆幅奖励。
当前版本另有持续的指数姿态惩罚：站立严格、行走较弱，偏离默认关节
位置较远时罚分明显增大，详见 `deploy_hip_rom_reward_zh.md`。
站立 abd 仍为四腿全 0，自碰撞仍关闭。
观测保持 36 × 20 = 720 维，不添加地形高度输入，导出接口不变。

## 第一阶段命令

在 `curl_robot_2d` 目录执行，把 resume 路径换成刚训练好的实际检查点：

```bash
python -m scripts.train_ppo_deploy terrain dr --resume rollingquad_2_deploy_robust_dr_policy.bin
```

如果原训练没有启用 DR，去掉 `dr` 并指定对应的平地检查点。
需要同步 `scripts/train_ppo_deploy.py`、新增的 `scripts/deploy_terrain.py`，
以及当前版本的 `scripts/train_ppo_walk3d.py`、`scripts/deploy_gait.py` 和
abd 全 0 的 `assets/rollingquad_description_2/mjcf/rollingquad.xml`。
不要直接换成已有的 `abd10_terrain.xml`，它的站立参考与这次训练不同。

不带 `terrain` 的旧命令仍使用平地。带 `terrain dr` 时输出为：

- `rollingquad_2_deploy_terrain_dr_policy.bin`
- `rollingquad_2_deploy_terrain_dr_checkpoints/`
- `rollingquad_2_deploy_terrain_dr_videos/`
- `rollingquad_2_deploy_terrain_dr_policy.json`（执行 export 时生成）

不带 `dr` 的 terrain 输出去掉名称中的 `_dr`。首次使用 terrain 应显式
指定原平地检查点；不指定时只会尝试自动加载对应的 terrain 输出文件，
文件不存在则从零训练。已有平地策略的输出文件不会被 terrain 模式覆盖。

## 地形分布与逐级加难

每个并行环境以 40% 概率使用平地、60% 概率使用随机崎岖地形。
崎岖地形是二维平滑噪声，目标谷峰高度从 0.005～0.015 m 采样；
它不是 ±1.5 cm 的起伏。出生区及边缘的平滑过渡可能让实际最大高度略低。
不同并行环境的地形独立，单个环境的地图在本次训练中保持固定；
不是每个 episode 重新生成。正反行走和侧向/转弯指令都会接触地形。

地形范围 16 × 16 m，161 × 161 网格，间距 10 cm。
中心半径 45 cm 为平坦出生区，随后用 35 cm 平滑过渡到完整起伏。
机器人保持与原训练相同的初始姿态和朝向。接近地图边缘 50 cm 时终止，
避免走出高度场后利用不存在的地面。

第一阶段稳定后，可以手动提高到最高 2.5 cm：

```bash
python -m scripts.train_ppo_deploy terrain dr --terrain-max-height 0.025 --resume rollingquad_2_deploy_terrain_dr_policy.bin
```

该参数单位为米，不会自动随训练步数增长。地形参数会保存到检查点目录的
`terrain_15mm_config.json` 或对应高度的配置文件中。配置类的其他参数位于
`scripts/deploy_terrain.py`。

## 首次 reset 显存不足

若报错发生在 Brax `env_state = reset_fn_(key_envs)`，并且单个 GPU
申请约 284 GiB，失败发生在 PPO 更新前，不能归因于 PPO 数据批量。
日志中的碎片化提示是通用提示，换 allocator 不能消除巨大的计算数组。
当前 CAD 脚部原始凸包约 1800 个顶点、3600 个三角面；MJX 将局部
heightfield 拆成三角棱柱，与 CAD 凸包逐一检测，可能产生很大的批量
临时数组。关闭机器人自碰撞并不会关闭这些地形碰撞。

terrain 默认使用 `NUM_ENVS=1024`、`BATCH_SIZE=64`，以及 256 顶点
凸包上限。先用更小配置在服务器重跑，可沿用平地策略：

```bash
python -m scripts.train_ppo_deploy terrain dr --num-envs 256 --batch-size 32 --resume rollingquad_2_deploy_robust_dr_policy.bin
```

`--num-envs` 是总并行环境数，多 GPU 时还会分摊；
`batch_size * num_minibatches` 必须能被 `num_envs` 整除，当前
`num_minibatches=32`。仅减少 batch_size 不会解决首次 reset 的碰撞开销。
运行后核对日志中的凸包上限、`robot-robot pairs=0` 和并行环境数。
这次只做语法和原生 MuJoCo 静态检查，没有运行 JAX reset、训练或
测量 GPU 峰值；是否完全解决显存不足仍需服务器重跑确认。

## 物理地面和 reward 一致

地面平面被单个 MuJoCo heightfield 替换。terrain 模式保留 CAD 显示网格，
但将每个 mesh 的碰撞凸包限制到最多 256 个顶点（`maxhullvert`），以控制
CAD 与地形三角棱柱碰撞检测的临时数组规模。该设置也用于同一次运行的
平地对照视频；不带 terrain 的平地训练保持原配置。
碰撞轮廓是近似的，不等同于完整 CAD 凸包；质量、惯量、关节、站立
keyframe 和策略接口保持一致。本地 MuJoCo 3.9 静态检查中，站立时
四脚凸包最低点变化约 0.06～0.27 mm；512 个方向的支撑面抽样差异中，
脚部最大约 1.92 mm，全部部件最大约 4.39 mm。这不是任意姿态的误差上界。
原来的无自碰撞掩码仍然保证机器人各部件只和地面碰撞。
高度样本在加载模型后写入 `MjModel.hfield_data`，再转成 MJX；
域随机化回调把它替换为每个环境自己的样本。生成的 XML 单独打开时没有
这些运行时样本，不能用它的初始平面判断训练地形是否生效。

reward 查的是当前环境 `self.sys.hfield_data`，与该环境的物理碰撞一致。
查询按 MuJoCo 网格的三角面进行插值；高度场底座深度不计入地表高度。

- 触地：使用脚底到脚下地表的距离，继续使用原来的 1 mm 阈值和一帧滤波。
  这是几何接触近似，不是接触力传感器。
- 4 cm 抬脚奖励、离地不足、擦地惩罚及 hip 有效摆动统计：均使用脚下
  地表作为零点。
- 机身高度惩罚及最低高度终止：均使用机身中心下方地表作为零点。
- 世界重力方向、直立奖励及策略观测的含义保持一致。

## 怎样判断是否改善

### 先确认地形是否显示

在 `curl_robot_2d` 目录执行，不需要策略文件，也不会运行 JAX 或训练：

```bash
python -m scripts.deploy_terrain --max-height 0.015 --out terrain_preview
```

生成 `terrain_preview_surface.png`（真实比例的地表渲染）和
`terrain_preview_height.png`（出生区附近的高度彩图，单位 mm）。
高度彩图白圈标记半径 45 cm 的平坦出生区；45～80 cm 为过渡区。
最大起伏仅 1.5 cm，跟随机器人时可能仍像平地，高度彩图更容易辨认。
独立预览使用与视频相同的固定参考地图；它不代表某个随机训练环境。

录制策略在地形上的表现时，必须带 `terrain`，仅使用 terrain 策略文件名
不会自动启用地形。把文件名替换成实际策略路径：

```bash
python -m scripts.train_ppo_deploy terrain dr video rollingquad_2_deploy_terrain_dr_policy.bin
```

默认输出 `rollingquad_2_deploy_terrain_dr_videos/showcase.mp4`，以及从
实际渲染模型样本生成的 `showcase_terrain_height.png`。终端会打印
`video terrain=True; native heightfields=1` 和实际高度范围，默认应为
`0.00 .. 15.00 mm`。若训练使用 2.5 cm 地形，录制命令也加
`--terrain-max-height 0.025`，独立预览则使用 `--max-height 0.025`。

地形材质改为无棋盘的哑光表面，增加侧光帮助观察浅起伏。视频提前结束时
会打印 `TERMINATED`、实际时长、离原点距离、机身离地高度和边界标志。
提前结束不一定是摔倒；若机器人仍在出生区附近，视频也不足以展示
策略在完整崎岖地形上的行走效果。速度统计使用实际完成时间。

### 训练与策略表现

每个检查点分别生成固定崎岖地形和纯平地视频，后缀为 `_terrain.mp4`
和 `_flat.mp4`，使用相同的初始随机种子和指令脚本。崎岖视频使用固定种子、
当前最大难度的一张参考地图；它不代表全部随机地形的通过率。
常规 Brax eval 使用与训练同样的平地/崎岖随机分布及启用的 DR。

新增日志包含：

- `hip_mean_rad FL/FR/RL/RR`：各腿实际 hip 的 episode 时间平均角度，
  用于发现某条腿持续偏置；不是原始 action。
- `terrain_rough_time_fraction`：崎岖地形在有效评估步中的比例，受不同
  地形存活时间影响，不等同于精确的 60% 环境数量。
- `terrain_span_mean_m`：各环境实际高度范围的时间平均。
- `base_clearance_mean_m`：机身相对当地地面的平均高度。
- `terrain_boundary`：触发边界终止的累计次数。

同时观察速度跟踪、episode 长度、擦地、滑动、hip 有效周期数量，以及
平地视频是否退化。目标是可靠落脚和通行，不要求各腿每一帧角度一致。

地形预览更新经过语法、静态几何检查及原生 MuJoCo 静态渲染验证；
没有执行 JAX 测试、训练或策略 rollout。
