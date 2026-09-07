# stand → compact 物理诊断

目的：区分“策略没有发出收腿命令”和“实际收腿受接触、关节限位或力矩约束影响”。此工具不训练策略，也不把执行完插值当作转换成功。

在 `curl_robot_2d` 目录的已有 MuJoCo 环境执行（只需 CPU，不加载 JAX/Brax 或策略）：

```bash
python -m scripts.diagnose_compact_physics \
  --out results/compact_physics_probe_01
```

输出目录必须是新的。默认按顺序完成两个独立实验：

1. `fixed`：把 torso 固定在 0.8 m 高处，删除根部自由关节，保留关节、质量、重力、碰撞和原有伺服/力矩限制。脚不应接触地面；若触地立即停止。这里存在外部支撑，只检查收腿的几何与关节跟踪，不能证明地面可达性。生成的 `fixture.xml` 仅属于该诊断输出，不覆盖源模型。
2. `ground`：原始 primitive 模型，机身自由，stand reset 加 5 mm 离地余量，与 stage1 名义起点一致。后续只设置电机目标，不修改 qpos/qvel，不固定机身，不在收拢末尾清零速度。

两者均使用 CG20、1 ms 物理步长、50 Hz 控制、原始 primitive 自碰撞与外壳接触。控制全部 12 个原始伺服，外摆的 stand 与 compact 目标均为零，与 stage1 一致。

## 默认探测轨迹与停止

先保持 stand 1 s，然后按最大关节目标变化不超过 0.05 rad 分段。每段用五次曲线移动 0.25 s，再停留 0.25 s。当前姿态共 16 段、约 8 s，最后保持 1 s。这是用于定位冲突的探测轨迹，没有声称解决支撑转移。

在每个 1 ms 子步检测下列事件，默认持续 10 ms 则停止仿真并保存此前全部数据：

- 根部向上速度超过 0.15 m/s（仅 ground）。
- 机身倾角超过 0.35 rad（仅 ground）。
- 地面接触法向力总和超过机器人重量的 3 倍。
- 最大接触穿透超过 4 mm。

任一关节力矩持续处于其限幅的 99% 以上达 0.2 s，也停止；非有限状态、MuJoCo 警告或 fixture 触地立即停止。这里的阈值是仿真诊断阈值，不是实机安全认证。站立稳定阶段也检查，不把初始弹跳隐去。停止后不再追加阻尼或改写状态，便于保留触发时刻。

若需要与此前连续插值对照，单独运行一个新目录：

```bash
python -m scripts.diagnose_compact_physics --mode ground \
  --trajectory smooth --fold-seconds 5 \
  --out results/compact_physics_smooth_01
```

它使用平滑五次插值，不等同于历史线性插值。可以调整时长，但不应为了“跑完”而先关闭碰撞或放宽力矩。

## 保存内容

每个实验子目录包含：

- `summary.json`：是否因事件停止、何时停止、收拢进度、最大力矩、跟踪误差、各关节持续饱和时间。
- `metrics.csv`：每 1 ms 的机身高度/向上速度/倾角、跟踪误差、四足法向力、接触点滑速、足球净离地间隙、外壳及其他结构的地面法向力。
- `trajectory.npz`：qpos/qvel/ctrl/actuator_force 全轨迹、关节索引和名称，以及逐接触点的几何名称索引、法向/切向力、位置和穿透距离。fixed 和 ground 根部自由度不同，分析时使用文件内的关节索引，不按固定偏移猜测。
- `events.json`：接触对首次出现/重新出现的时刻，以及停止原因。

接触力按法向力大小汇总，并非世界竖直方向投影；固定机身的外部支撑反力没有测量。脚滑速使用足球与地面接触点的切向速度，不用脚球质心速度替代。

## 观看已保存的运动

在有桌面的电脑上，从 `curl_robot_2d` 目录打开 MuJoCo 回放窗口：

```bash
python -m scripts.replay_compact_physics \
  results/compact_physics_probe_01/ground --speed 0.5 --loop
```

把 `ground` 改为 `fixed` 可看固定机身测试。默认半速；`--speed 1` 为原速。窗口中可用鼠标调整视角。若环境中之前设置了 `MUJOCO_GL=disable`，先取消它或改为 `glfw`。

没有桌面的 Linux GPU 服务器可导出 GIF，然后下载观看：

```bash
MUJOCO_GL=egl python -m scripts.replay_compact_physics \
  results/compact_physics_probe_01/ground --speed 0.5 \
  --gif results/compact_physics_probe_01/ground/replay.gif
```

需要 Pillow 和可用的 OpenGL/EGL 渲染环境。程序拒绝覆盖已有 GIF。跨机器复制时保留 `trajectory.npz` 与同目录的 `summary.json`，并同步相同 primitive 模型；固定夹具在新机器按保存的配置重建，不依赖旧机器的绝对 mesh 路径。

回放只显示保存的物理状态，不调用 `mj_step`，不构成新的动力学实验。模型指纹先核对。GIF 顶部显示时间、收拢进度、倾角和向上速度；`stop` 标注本次记录的最终停止原因。本地地面轨迹已成功导出 640×480、442 帧半速 GIF，并检查最后一帧。

先查看顶层和各实验的 `summary.json`，再回看停止前约 0.2 s 的轨迹：是足仍承重却在移动、外壳开始接触、自碰撞增多，还是力矩先饱和。接触同时出现并不足以单独证明因果。

fixed 能跟随而 ground 弹跳，才重点研究支撑转移/分足收拢；fixed 也跟不上，则先查关节范围、目标映射、自碰撞和伺服。即使 ground 完成命令序列，也仍需检查低余速 compact 门槛和后续真实滚动接管。

## 本地实际运行结果

已在 CPU MuJoCo 3.12.0 跑完以下单条无随机扰动实验。源模型 LF 指纹为 `567e2583c4ac2ec5a3129133236a84efad262a542ba0bd3c469a5afac4af4d49`，与本次提供的云端 stage1 配置一致；不保证 CPU 与 MJX 逐步轨迹完全相同。

| 实验 | 结果 | 最大关节终点误差 | 峰值力矩 | 峰值向上速度 |
| --- | --- | ---: | ---: | ---: |
| 固定机身，默认分段 | 运行完 10 s，无地面接触/穿透 | 0.00364 rad | 0.200 Nm | 0 |
| 自由机身，默认分段 | 8.805 s 因倾角停止 | 0.1251 rad | 1.072 Nm | 0.0327 m/s |
| 自由机身，1 s 平滑收腿 | 2.864 s 因倾角停止 | 0.1234 rad | 1.403 Nm | 0.0704 m/s |

默认地面轨迹在约 8.518 s 已倾斜超过 0.1 rad；外壳总法向力首次超过 1 N 出现在约 8.717 s。停止时倾角约 0.362 rad，后足球约离地 6.5 mm，外壳承重明显增加。该时间顺序和接触记录提示应继续检查末段支撑和平衡，不能直接认定外壳碰撞是起因。

这两种地面探测没有复现之前描述的明显弹跳，而是收拢末段倾倒；它们是不同于历史线性插值的路径。固定夹具结果说明本路径下关节能接近 compact；地面峰值力矩未触及 3 Nm，但存在承重跟踪误差。结果不证明 3 s 自主转换可达，也不证明 compact 已被稳定捕获。

原始输出分别位于 `results/compact_physics_probe_local_01` 和 `results/compact_physics_probe_local_fast_01`。默认两项实验的数值数组有限；4 项夹具/轨迹/连续事件测试通过，入口与测试语法检查通过。

## 一次一只脚的探测

```bash
python -m scripts.diagnose_compact_physics --mode ground --trajectory single-foot --out results/single_foot_01
python -m scripts.replay_compact_physics results/single_foot_01/ground --loop
```

当前验证顺序为左前、右前、右后、左后，各移动一次。每次目标向 compact 足位置沿世界 X 方向移动最多 8 mm，抬脚目标 15 mm。过程为按质心位置和待抬脚载荷闭环转移重心、抬脚、平移、下降、确认落地、四足重新承重。摆动期间还对三只支撑脚使用不超过 4 mm 的法向力微调。主体仍是 IK 和限速位置伺服，不是动力学 WBC，也没有训练策略。真实仿真状态仅在初始化时设置，之后由电机与接触动力学演化。

root 高度不固定在站姿值。第 `k` 个已规划小步把高度目标设为 `stand_z + k / planned_steps * (compact_z - stand_z)`，并在该步重心转移阶段与水平目标一起使用 smootherstep 变化；落脚和四足重置阶段保持本步高度。当前 MJCF 的 compact 映射高度高于落地稳定后的站姿高度，因此该模型实际执行的是逐步升高，而不是下蹲。事件和总结分别保存每阶段高度目标、初始高度与 compact 高度。

抬脚前检查其余三足承重和质心投影；转移不足时缓慢修正机身目标。落地要同时满足法向力、位置和低速条件。支撑足失去接触、过度滑移或 yaw 突变等会停止试验。`single_foot_events.json` 记录各阶段，`summary.json` 记录停止原因。回放中的 steps 仅表示已确认的小步数，不表示 compact 达成率。跨机器回放也应复制事件文件。

当前源模型站姿发生更新后，在 LF 指纹 `cbd3a5493cb71a52afebc7b7bdfe8f01609b6a81d1835f88241f683b037a8f9a` 上运行 `results/compact_single_foot_local_05`：左前脚在 3.80 s 确认落地，相对卸载起点向内移动 10.86 mm，峰值离地间隙 13.54 mm；右后脚抬起期间于 6.97 s 触发 `support_contact_lost`，完成 1/4 步。实际位移包含伺服和机身运动造成的偏差，因此不同于 8 mm 命令。这次未完成一整轮，更没有完成 stand-to-compact；下一问题是换脚期间的三足支撑控制。旧模型上的前三脚结果不能作为当前模型的验证结果。

加入载荷闭环、四足重置和支撑力微调后，`results/compact_single_foot_local_20` 已连续完成左前和右前两步：实际 X 位移分别为 -10.09 mm 和 -10.63 mm，峰值离地间隙均约 13.4 mm。第三步右后脚在重心转移后仍无法取得 8 mm 离地间隙，于 16.54 s 以 `foot_not_clear` 停止；峰值 yaw 0.0342 rad、峰值 yaw 角速度 0.282 rad/s，未触发 yaw、滑移或支撑失联保护。后脚优先对照 `results/compact_single_foot_local_21` 从原始站姿同样无法抬起，排除了仅由先收前脚导致工作空间不足的解释。现有位置 IK 的可靠边界是连续两只前脚；后脚需要带三足接触约束和机身姿态任务的全身控制求解器。

一次把现有运动学速度约束解直接映射为位置伺服目标的隔离实验也已执行：前脚使用该输出时约 0.6 s 即支撑失联；只对后脚启用时，前两步仍成功，但第三步约 0.34 s 后支撑失联。该接法未保留在默认控制器。后续求解必须包含位置伺服/阻抗的闭环响应，或直接在可控的力矩层执行接触约束，不能把浮动基速度解简单截取为关节位置目标。

按 compact 高度同步后的 `results/compact_single_foot_height_local_01` 从落地稳定高度 0.14886 m 分四步对齐 0.15993 m：前三步目标依次为 0.15163、0.15439、0.15716 m。左前和右前仍确认落地，实际 X 位移为 -8.35 mm 和 -8.87 mm；右后脚在第三步目标高度下仍以 `foot_not_clear` 停止。由此可见固定 stand 高度确实不合理，但在当前模型中只修正高度尚不足以让后脚形成相对机身的离地运动。
