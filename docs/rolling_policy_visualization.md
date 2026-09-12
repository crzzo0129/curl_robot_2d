在云端导出当前选中滚动策略的 MP4，下载后观看。入口读取 `recovery.json` 中的 `selected_student`，目前是原校准蒸馏策略 `rolling_distill_calibrated_20260912_034719/student_params`，不是被筛选器拒绝的低速补训候选。

先通过 Git 同步源码。若只同步增量包，在云端项目根目录解压（要求已安装上次成功复评使用的 v2 代码）：

```bash
python -m zipfile -e results/rolling_policy_visualization_v1.zip .
python -m pip install imageio imageio-ffmpeg

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m scripts.visualize_rolling_student \
  --run results/rolling_low_speed_20260912_065438
```

默认输出独立的 `results/rolling_policy_video_<时间>/` 目录和同名 ZIP。下载 ZIP 并解压，打开 `index.html` 可按速度/方向浏览并同步播放；也可以直接用播放器打开 MP4。

包括低、中、高速 × 直行、左转、右转，共 9 条轨迹，每条斜视和俯视两个视角，共 18 个 MP4。按命令接近各速度区间中心及 yaw=0/±0.05 rad/s 的原则选取已有评价环境，实际命令印在视频与页面中，不保证正好等于这些目标中心。不按成功率挑选，失败和低速未达标也会如实展示。只需要一个视角可在命令后追加 `--views oblique` 或 `--views top`。

轨迹来自真实 MJX 学生闭环，使用原来的模型、cg20 物理设置、标定表及评价命令。优先复用最近一次同状态复评的快照缓存，教师预热提供完整观察历史和上一动作；学生接管后最多运行 10 秒，失败提前结束。视频从学生接管时刻开始，没有把教师预热片段混入学生表现。渲染只将记录的 qpos 放回模型并执行 `mj_forward` 显示，不再用 CPU 物理重新运行策略。

视频叠加显示目标速度、目标 yaw、逐步实际 vx/yaw、世界 X 位移、侧向偏移、轴倾斜和失败标记。侧向偏移保留原评价相对最初 reset 的口径；转向轨迹的世界 Y 位移不等同于直行侧漂失败。页面同时列出运行时长和跟踪 MAE。9 条示例不能替代完整 256 环境评价。

输出保留 `evaluation/rollouts/episode_*.npz` 供后续回放，每份包含 qpos、qvel、逐步指标、学生来源及初态校验值。`visualization.json` 保存选取依据、各场景对应的评价编号、指标和视频文件名。初态缓存不加入视频 ZIP；不会修改任何权重，也不执行 BC、DAgger 或 PPO 更新。

通过 Git 下载可使用自定义输出路径，便于定位和暂存：

```bash
python -u -m scripts.visualize_rolling_student \
  --run results/rolling_low_speed_20260912_065438 \
  --out results/rolling_selected_policy_video
git add -f results/rolling_selected_policy_video.zip
```

然后按现有 Git 工作流提交和推送。输出已存在时请换新 `--out`，脚本不会覆盖历史视频。若需要查看补训候选，可显式追加 `--student results/rolling_low_speed_20260912_065438/dagger_01/student_params`，并使用另一个输出目录；默认始终读取当前选中策略。

渲染使用 EGL 和 H.264。入口预先检查视频编码依赖，避免跑完评价才发现缺少编码器。执行失败时也会打包已生成的日志和轨迹，便于诊断。本地仅进行语法和参数静态核对，没有启动训练、仿真或视频渲染；实际画面需要云端生成后检查。
