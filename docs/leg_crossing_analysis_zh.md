# 主动滚动 baseline 的腿部交叉检测

## 1. 检测目的

此前主动滚动 baseline 没有启用腿部之间的自碰撞。可视化中观察到前后腿
可能互相穿过，因此本实验固定原模型和原最优控制器，只增加只读几何诊断，
不改变动力学，也不覆盖 `results/phase_controller/` 中的旧结果。

全部新输出位于：

`results/leg_crossing_analysis/`

## 2. 杆件定义

侧视平面内使用四条结构中心线：

- front thigh：前髋到前膝；
- front shank：前膝到前足端；
- rear thigh：后髋到后膝；
- rear shank：后膝到后足端。

检查四组前后腿组合：

- front thigh / rear thigh；
- front thigh / rear shank；
- front shank / rear thigh；
- front shank / rear shank。

同一条腿内部相邻的大腿和小腿共享关节，不作为交叉对象。

## 3. 两级交叉定义

### 3.1 完整中心线拓扑交叉

如果两条有限线段在各自内部发生严格相交，则记为一次拓扑交叉。只在共同
端点相接不计为交叉，因此初始 compact 姿态中两个小腿在五边形底部的共同
端点不会被误报。

但是，当两个足端从共同点继续运动并交换前后顺序时，即使交点仍靠近足端，
也会被记为拓扑交叉。这一指标用于检测链条拓扑是否发生交换。

### 3.2 杆身交叉

对于 front shank / rear shank，额外从两条小腿远端各排除一个足端半径对应
的中心线长度，再检查剩余杆身是否相交。该保留区长度占小腿中心线约
$0.01995/0.15=13.3\%$。

其他三组没有预期的共同端点，因此杆身交叉与完整中心线交叉定义相同。

杆身交叉可以排除“只在 compact 底部接触”的解释，是后续选择性自碰撞最
重要的诊断量。

## 4. 10 s 检测结果

| 指标 | 结果 |
|---|---:|
| 首次完整中心线交叉 | 0.154 s |
| 首次交叉时 Torso 相位 | 0.668° |
| 完整中心线交叉事件 | 23 次 |
| 完整中心线交叉累计时间 | 1.098 s |
| 完整中心线交叉时间比例 | 10.98% |
| 最长完整中心线交叉事件 | 0.090 s |
| 首次杆身交叉 | 1.010 s |
| 首次杆身交叉时 Torso 相位 | 225.46° |
| 杆身交叉事件 | 20 次 |
| 杆身交叉累计时间 | 0.574 s |
| 杆身交叉时间比例 | 5.74% |
| 最长杆身交叉事件 | 0.034 s |
| 同时交叉的最大杆件对数 | 1 |

分杆件结果：

| 杆件组合 | 杆身交叉 | 首次时间 | 事件数 | 累计时间 |
|---|---:|---:|---:|---:|
| front thigh / rear thigh | 否 | — | 0 | 0 |
| front thigh / rear shank | 是 | 3.070 s | 9 | 0.278 s |
| front shank / rear thigh | 否 | — | 0 | 0 |
| front shank / rear shank | 是 | 1.010 s | 11 | 0.296 s |

结构碰撞代理的半径分别为 12 mm 和 10 mm。按中心线距离减去两杆半径估计，
最小结构间隙达到 -22 mm。这不是精确接触穿透深度，因为当前并未启用物理
自碰撞，但足以说明相关实体碰撞代理会发生显著重叠。

## 5. 结论

1. 当前控制器确实反复交换两个小腿的前后拓扑关系。
2. 即使排除 compact 底部的预期足端接触，仍存在明确的杆身交叉。
3. 除了前后小腿互穿，3.07 s 后还周期性出现前大腿与后小腿交叉。
4. 因此当前 10.99 圈结果应保留为“无自碰撞探索性 baseline”，不能直接
   视为物理可实现的最终 baseline。
5. 下一步应在同一模型中设计选择性腿部自碰撞，然后固定旧控制器直接复测；
   在得到无交叉控制器前，不进行正式鲁棒性扫描。

## 6. 可复现文件

- 检测脚本：`scripts/analyze_leg_crossing.py`
- 几何函数：`curl_robot_2d/planar_geometry.py`
- 摘要：`results/leg_crossing_analysis/leg_crossing_summary.json`
- 时间序列：`results/leg_crossing_analysis/leg_crossing_timeseries.csv`
- 诊断图：`results/leg_crossing_analysis/leg_crossing_diagnostic.png`
- 首次完整中心线交叉：`results/leg_crossing_analysis/first_crossing.png`
- 首次杆身交叉：`results/leg_crossing_analysis/first_core_crossing.png`

复现实验：

```powershell
python -m scripts.analyze_leg_crossing
```
