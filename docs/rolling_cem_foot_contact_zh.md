# 滚动 CEM 足端接触时间优化

更新日期：2026-07-29

## 1. 问题

原始 compact 姿态让两个有限尺寸足端球表面相切。现有 CEM 的周期项又围绕该姿态
振荡，因此滚动中会反复触发足端自接触。当前模型上直接回放旧最佳控制器时，10 秒
内足端累计接触 1.047 秒，最长连续接触 36 ms。长时间接触会约束两条开链的相对
运动，也会给 residual RL 制造容易失败的接触状态。

## 2. 修改

`scripts/optimize_phase_controller.py` 新增可选参数：

- `--minimum-foot-gap-mm`：名义足端表面间隙；
- `--foot-gap-tracking-margin-mm`：覆盖位置伺服跟踪误差的目标余量；
- 对累计接触时间和最长连续接触显式计分；
- 记录最小表面间隙、间隙亏欠积分和足端接触时长；
- 对膝关节目标做最小几何投影，避免周期项直接命令两足闭合；
- 正间隙控制器从分离姿态复位，启动斜坡不再从足端相切开始。

默认两个参数不启用时，旧 CEM 的目标和分数保持不变。

`curl_robot_2d_mjx/cem_reference.py` 和 MJX environment 同步支持保存的膝角偏置、
几何投影和分离 reset，使 CPU CEM 与 residual RL reference 使用相同目标。

## 3. 结果

同一当前模型、10 秒 CPU MuJoCo 回放：

| 指标 | 原始最佳 | 2 mm 短接触候选 |
|---|---:|---:|
| 净滚动 | 9.049 圈 | 8.763 圈 |
| 水平位移 | 8.831 m | 8.456 m |
| 足端累计接触 | 1.047 s | 0.075 s |
| 最长连续足端接触 | 36 ms | 16 ms |
| 最大足端重叠 | 1.662 mm | 0.377 mm |
| 腾空比例 | 11.21% | 12.74% |
| 非允许接触时间 | 0.097 s | 0.073 s |
| 最大非允许穿透 | 0.608 mm | 0.740 mm |
| 正执行器功 | 36.59 J | 32.44 J |

足端累计接触减少 92.8%，最长连续接触减少 55.6%；代价是净圈数下降 3.2%，位移
下降 4.2%。最大非允许穿透略高于原始最佳，因此新结果保留为候选，没有替换默认
控制器。

## 4. 文件保留

原始最佳保持不变：

```text
results/collision_constrained_cem/
```

推荐的短接触候选：

```text
results/collision_constrained_cem_foot_gap_2mm_short_contact/
```

中间实验也分别保存在独立目录中，没有覆盖上述两套结果。

## 5. 复现

```powershell
python -m scripts.optimize_phase_controller `
  --generations 10 `
  --population 48 `
  --elite-count 8 `
  --duration 10 `
  --final-duration 10 `
  --barrier-generations 0 `
  --workers 8 `
  --minimum-foot-gap-mm 2 `
  --foot-gap-tracking-margin-mm 4 `
  --initial-controller `
    results/collision_constrained_cem/best_phase_controller.json `
  --output-dir `
    results/collision_constrained_cem_foot_gap_2mm_short_contact
```

Linux 云端渲染：

```bash
python -m scripts.replay_active_controller \
  --controller results/collision_constrained_cem_foot_gap_2mm_short_contact/best_phase_controller.json \
  --duration 10 \
  --output results/collision_constrained_cem_foot_gap_2mm_short_contact/active_roll.gif
```

使用该候选进行 residual RL 时，通过 `--controller` 指向候选 JSON；在正式替换旧
最佳前，还应完成 MJX/CPU 一致性和扰动鲁棒性评估。
