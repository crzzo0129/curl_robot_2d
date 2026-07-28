# 二维展开行走控制探索

更新日期：2026-07-28

## 1. 目标与模型限制

本轮目标不是继续调整滚动 PPO，而是先回答一个更基础的问题：

> 当前前膝、后肘的二维机构，能否在现有质量、伺服、力矩、外壳和碰撞参数下产生连续足端行走？

二维模型把左右腿同步合并成前、后两个虚拟腿组。因此它不能复现四条腿依次落地的
静态 crawl，只能形成类似二维双足的交替支撑、trot 或 bound。这个限制决定了单支撑
阶段必须依靠动力学和状态反馈稳定机身俯仰。

## 2. 参考方法

实现采用以下组合：

- CHAMP 风格的 stance/swing gait scheduler 和足端轨迹；
- 前膝、后肘构型的解析逆运动学；
- 机身高度、俯仰 PD、速度和落脚点反馈；
- CEM 搜索机身高度、步长、抬脚高度、频率、占空比、质心偏置及反馈增益；
- 严格 walking 约束：非足端连续触地超过 50 ms 或连续腾空超过 80 ms 即失败。

相关参考：

- CHAMP: <https://github.com/chvmp/champ>
- Walk These Ways: <https://github.com/Improbable-AI/walk-these-ways>
- MuJoCo Playground: <https://github.com/google-deepmind/mujoco_playground>
- Changeable Configuration Quadruped:
  <https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202500713>

实现入口为 `scripts/explore_walking_controller.py`。

## 3. 外壳动态工作区

静态扫描条件为：

- 机身水平；
- 足端球与地面相切；
- 物理足端相对髋关节在 `[-70, 70] mm` 内扫描；
- 显式计算腿部 shell capsule 的最低表面高度。

结果：

| 髋到足端深度 | 扫描内最低外壳间隙 |
|---:|---:|
| 0.205 m | -1.0 mm |
| 0.225 m | +2.9 mm |
| 0.245 m | +7.2 mm |
| 0.265 m | +12.5 mm |
| 0.285 m | +20.1 mm |

结论是：缩短后的外壳没有封死行走工作区。当有效腿长不低于约 0.225 m 时，
`+-70 mm` 的支撑摆幅仍有正间隙。动态外壳触地主要是控制失败导致机身掉高，
不是正常展开姿态必然碰撞。

## 4. CEM 结果

| 实验 | 独立评估 | 位移 | 主要问题 |
|---|---:|---:|---|
| 手工 IK gait | 1.61 s | +0.232 m | 俯仰失稳，非足端触地 24% |
| 非严格 CEM | 9.61 s | +1.518 m | 27.6% 腾空、27.5% 非足端触地，实际是 bound |
| 严格 CEM | 4.00 s 搜索窗口通过 | +0.016 m | 8 s 外推在 4.34 s 失稳并后退 |
| 严格长时 CEM | 未通过 8 s | - | 固定周期控制器没有形成稳定吸引域 |

非严格 CEM 证明机构可以通过交替蹬地快速前进，但画面显示后半段逐渐进入腾空和
滚转，不能作为 walking 成功。严格 CEM 证明足端支撑且不碰壳的短时轨迹存在，
但继续增加总体参数搜索不能消除长时俯仰漂移。

## 5. 诊断

1. 外壳不是当前首要障碍，维持 `root_z` 才是。
2. 名义摆动高度必须补偿负载下的机身下沉，否则摆动脚仍会拖地。
3. 支撑腿需要 root 高度闭环，否则位置伺服的负载静差会让外壳接地。
4. 质心偏置和落脚点反馈能减少短时俯仰，但固定周期参数无法覆盖接触时刻误差。
5. 二维两虚拟腿模型缺少三维四足步态中的左右对角支撑多边形，因此比最终三维
   trot 更难通过开环轨迹稳定，不能把二维开环失败直接解释为三维机构不可行。

## 6. 推荐训练路线

下一阶段应建立独立 walking task，不应复用当前 rolling task 的复位和奖励：

1. 从展开站立姿态复位，而不是 `compact`。
2. 动作使用足端 IK 参考附近的四关节 residual，避免 PPO 从全关节空间盲搜。
3. 观测加入 root 高度、俯仰/角速度、水平速度、关节状态、两足接触、步态相位、
   参考动作和上一动作。
4. 主奖励使用命令速度跟踪、机身水平、目标高度和存活。
5. 辅助奖励约束摆动脚净空、计划接触一致性、动作变化、力矩、非足端触地和自碰撞。
6. 课程从双支撑站立开始，再加入原地换脚、低速前进，最后提高速度和随机化。
7. CEM 轨迹只作为参考和早期课程教师；策略必须能够根据状态改变落脚点和接触时刻。

这条路线对应 Walk These Ways 的命令/相位条件化训练，也保留 MuJoCo Playground
中 residual locomotion 和 domain randomization 的扩展空间。

## 7. 复现

本地 CPU CEM：

```powershell
python -m scripts.explore_walking_controller `
  --search `
  --generations 30 `
  --population 96 `
  --search-duration 4 `
  --evaluation-duration 8 `
  --workers 8 `
  --output-dir results/walking_exploration
```

Linux 云端可以使用相同命令。该脚本是 CPU MuJoCo 搜索，不需要 EGL；只有渲染时
使用 EGL：

```bash
python -m scripts.render_mjx_policy \
  results/walking_exploration/walking_rollout.npz \
  --output results/walking_exploration/walking_rollout.gif \
  --control-dt 0.001 \
  --mujoco-gl egl
```
