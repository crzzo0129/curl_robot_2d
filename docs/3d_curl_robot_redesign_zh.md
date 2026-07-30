# 3D Curl Robot 重新设计记录

更新日期：2026-07-30

## 1. 决策

`disk_robot/assets/pupper_v3_disk_structure_candidate.xml` 不作为 curl 项目的 3D
基线。它属于另一条 disk-body quadruped / walking-first 设计路线，虽然已有完整
MJX/Teacher/Student 工具链，但结构思想与当前二维 curl 机器人不一致。

当前 3D 主线改为从二维等边 curl 模型直接提升：

```text
2D equal-edge curl geometry
-> left/right side-rail 3D curl model
-> mirrored 2D CEM rolling reference
-> small 3D residual for roll/yaw/lateral/asymmetry
```

## 2. 第一版 3D 几何合同

新模型文件：

```text
assets/curl_robot_3d.xml
```

生成器：

```text
curl_robot_2d/model_3d.py
scripts/generate_3d_model.py
```

设计原则：

- 保留二维等边五边形中心线思想；
- 不使用 Pupper/Disk 的圆盘躯干；
- 将二维前/后两条 sagittal 链复制到左、右两条 side rail；
- 每条 side rail 具有 torso / thigh / shank 弧壳接触代理；
- 左右足端分别允许前后足接触；
- 使用 freejoint 根节点，允许真正 3D roll/yaw/lateral drift 被测量；
- 每个 3D actuator 代表单侧电机，力矩/增益为二维成对关节的一半。

当前 3D 关节顺序：

```text
front_left_hip
front_left_knee
front_right_hip
front_right_knee
rear_left_hip
rear_left_knee
rear_right_hip
rear_right_knee
```

模型合同：

```text
nq = 15      freejoint qpos 7 + internal joints 8
nv = 14      freejoint velocity 6 + internal joints 8
nu = 8
mass = 3.170 kg
keyframes = open / walk / compact
```

## 3. 2D CEM 到 3D 的映射

二维 reference 输出：

```text
front_hip, front_knee, rear_hip, rear_knee
```

第一版 3D 对称映射：

```text
front_left_hip    <- front_hip
front_left_knee   <- front_knee
front_right_hip   <- front_hip
front_right_knee  <- front_knee
rear_left_hip     <- rear_hip
rear_left_knee    <- rear_knee
rear_right_hip    <- rear_hip
rear_right_knee   <- rear_knee
```

这一步故意不引入左右差异。左右差异留给后续 3D residual：

```text
symmetric residual      rolling/forward correction
antisymmetric residual  roll/yaw/lateral/asymmetric contact correction
```

## 4. 当前 smoke 结果

命令：

```powershell
& 'C:\Users\12481\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m scripts.evaluate_3d_symmetric_cem_reference --duration 2.0 --out results\3d_symmetric_cem_smoke.json
```

结果摘要：

```text
status                         ok
nonfinite                      false
distance_x                     +0.346 m
distance_as_shell_turns        +0.373
lateral drift                  ~0
rolling_axis_tilt_rms          ~2.45e-8 rad
torque_saturation_fraction     0
```

注意：完整滚动中 Euler roll/yaw 会在翻转附近跳到 `pi`，所以最小 3D smoke
优先看 `rolling_axis_tilt_*`，而不是直接用 Euler yaw/roll 做失败判断。

## 5. 下一步

1. 做初始相位和相位速度方向 sweep，选择 3D nominal reference；
2. 增加短时渲染，视觉检查 side rail 接触和足端/弧壳碰撞；
3. 加轻微侧向初始误差，评估纯对称 CEM 的自然 3D 稳定性边界；
4. 设计 3D residual observation/action，把 symmetric 与 antisymmetric 分量拆开；
5. 只在 3D 新问题上训练 residual，不回头证明二维 residual 有鲁棒性提升。
