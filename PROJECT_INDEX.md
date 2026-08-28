# CURL 项目索引

本文件是项目的主入口。当前有效开发项目为 `curl_robot_2d`；上级目录中的
`robot_curl`、`disk_robot`、`pupperv3_mjx` 等是其他阶段或独立项目，不应与本项目混用。

## 当前方案

- 运动学：保留 Pupper 原髋距与腿长，不再强制足端、电机中心构成正五边形。
- 髋中心距：150.40 mm。
- 大腿长度：84.45 mm。
- 小腿投影长度：88.01 mm。
- 电机碰撞包络：直径 64 mm、厚度 33 mm。
- 足端：直径 39 mm。
- 紧凑态足端中心距：43 mm，表面间隙 4 mm。
- 外壳半径：127.5 mm。
- 底部开口：60°。
- 外壳角度归属：躯干 150°、每条大腿 45°、每条小腿 30°。
- 3D 厚度/左右中心面间距：120 mm。

## 目录职责

| 目录 | 内容 |
|---|---|
| `curl_robot_2d/` | CPU MuJoCo 的 2D/3D 几何、模型和参数源代码 |
| `curl_robot_2d_mjx/` | MJX 环境、奖励、随机化和 RL curriculum |
| `scripts/` | 模型生成、CEM、评估、碰撞分析、渲染和训练入口 |
| `assets/` | 生成或固定使用的 MuJoCo XML 模型 |
| `tests/` | 单元测试与迁移一致性检查 |
| `docs/` | 设计、评估协议、训练和历史 handoff 文档 |
| `experiments/` | 实验登记表 |
| `results/` | 本地实验结果；大部分被 Git 忽略 |
| `renders/` | 可再生成的演示渲染 |

## 当前正式结果

详细文件级入口见 [`results/RESULTS_INDEX.md`](results/RESULTS_INDEX.md)。最重要的结论是：

1. 完整圆壳在严格碰撞惩罚下会使 CEM 退化到几乎不动。
2. 足端附近裁出 60° 开口后，2D 可达 9.772 圈/10 s，禁碰撞时间 0.188 s。
3. 90° 开口进一步减少壳体碰撞，但最终速度降至 9.140 圈/10 s。
4. 第一次 3D 失败来自缺少 0.25 s 启动 ramp 和允许足端接触对，不是力矩不足。
5. 修复迁移后，旧 90/75 壳体分配在 3D 达到 9.754 圈。
6. 壳体改为躯干 150°/大腿 45° 后，必须重新优化匹配 reference。
7. 当前匹配方案在 3D 达到位移等效 8.781 圈、实际滚动 8.568 圈，力矩饱和 0%。
8. corrected RollingQuad 2 完整 CAD 模型复用该 reference 后，10 s 达到位移等效
   8.795 圈、实际滚动 8.799 圈；滚动轴倾斜 RMS 1.58°，力矩饱和 0%。

当前正式模型已提升到 `assets/`：

- `assets/curl_robot_2d_pupper_r127p5_open60.xml`
- `assets/curl_robot_3d_pupper_r127p5_open60_width120.xml`
- `assets/rollingquad_description_2/mjcf/rollingquad.xml`（当前行走、部署和 3D 滚动默认模型）

二者都是 60° 开口、R=127.5 mm、躯干 150°/大腿 45° 的匹配模型；正式
`assets` 中不再保留 90° 开口 Pupper 模型。

## 推荐工作流

```powershell
# 当前 RollingQuad 2 行走训练主入口
python -m scripts.train_ppo_walk3d probe
python -m scripts.train_ppo_walk3d
python -m scripts.train_ppo_deploy probe
python -m scripts.train_ppo_deploy

# 生成模型
python -m scripts.generate_model
python -m scripts.generate_3d_model

# 测试
python -m unittest discover -s tests -v

# 2D CEM 与 3D reference 验证
python -m scripts.optimize_phase_controller --help
python -m scripts.evaluate_3d_symmetric_cem_reference --duration 10 --physics-profile reference --out results/rollingquad_2_3d_reference/cpu_newton20_reference_10s.json

# 进入 RL 前先做 reference-only 鲁棒性验证
python -m scripts.compare_mjx_3d_reference --help
```

旧 `train_mjx_3d_walking_ppo.py` 行走流程仅用于历史实验复现；后续行走与实机部署 policy 使用上述两个 `train_ppo_*` 入口。

## 文件管理规则

- 源代码、固定参数、关键 controller 和实验摘要应保留。
- `results/` 中的逐代 CSV、重复 GIF 和 sweep 中间帧默认不提交 Git。
- 删除实验前先检查 `results/RESULTS_INDEX.md`，避免删除当前 reference 的来源。
- 新实验应在 `experiments/registry.csv` 登记假设、模型、reference 和结果目录。
- `__pycache__`、PPT 展开目录和临时渲染均为可再生成文件，可以安全清理。
