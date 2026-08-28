# curl_2d：滚动到行走的可运行基线

## 方法

采用同一 MuJoCo 模型、同一组四个关节伺服器和一个不可逆状态机：

```text
ROLL -> BRAKE -> DEPLOY -> WALK -> COMPLETE
```

- `ROLL`：复用 `results/phase_controller/best_phase_controller.json` 中的相位锁定滚动控制器；如果文件不存在，使用代码内置的同一数值基线。
- `BRAKE`：记录切换瞬间的展开相位，使用相位展开量计算目标窗口，并沿用已有 braking CEM 的相位进度、振荡频率缩放、振荡幅值缩放和关节偏置。
- `DEPLOY`：用五次时间轨迹把当前制动末端关节姿态连续过渡到二维行走控制器的初始支撑姿态。
- `WALK`：复用已有的足端轨迹和解析二连杆 IK，前后虚拟腿相差半个步态周期，并用机身俯仰、俯仰速度和根部速度做小幅反馈。
- `COMPLETE`：完成指定时长，输出全程诊断；它不等价于“行走一定稳定”。

这条链路的关键是：滚动控制器只负责把机器人送到可制动窗口，展开控制器只负责改变形态，行走控制器只在形态切换结束后接管。任何阶段都不会瞬间覆盖关节目标。

## 运行

在 `curl_robot_2d` 目录执行：

```powershell
python -m scripts.run_roll_to_walk
```

也可以缩短流程做烟测：

```powershell
python -m scripts.run_roll_to_walk --roll-duration 0.5 --brake-duration 0.5 --deploy-duration 0.8 --walk-duration 1.0
```

默认输出到 `results/roll_to_walk/`：

- `summary.json`：模式链路、位移、姿态、接触和执行器诊断；
- `rollout.csv`：逐 MuJoCo 步的时序数据。

## 当前验收含义

`completed_all_modes=true` 且 `status=ok` 表示数值仿真完整走完滚动、制动、展开和行走接口；`walking_baseline_safe=true` 才表示现有二维行走 baseline 同时满足足端支撑、无非足端地面接触、俯仰和高度阈值。当前工程中的旧行走 baseline 仍可能触发非足端接触，因此该字段会明确报告为 `false`，不会把流程完成误报为行走性能达标。
