# Student 导出 neural_controller JSON

```bash
python -m scripts.export_stand_to_roll_student \
  --student-params results/stand_to_roll_symmetry_student/student_params \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --out results/stand_to_roll_symmetry_student/student_rtneural.json
```

需要 student_params 同目录的 student_source.json、原始 BC 文件，以及云端现有
Brax/MuJoCo/NumPy 环境。只导出推理网络，不重训、不启动机器人。
Student 的 PPO normalizer 不是实际输入归一化来源，必须使用 hash 匹配的 BC mean/std/clip。

导出保留 720 输入、12 输出，20 帧最新在前、ELU 隐藏层、tanh(location) 输出；
不包含左右动作平均，丢弃 Gaussian scale 和 critic。
为精确保留 clip((obs-mean)/std, ±clip)，增加 1440 单元 ReLU 层，
通过 relu(z+c)-relu(z-c)-c 实现裁剪，并将重组并入原第一隐藏层。
它使用现有 RTNeural dense/ReLU/ELU/tanh 格式，不要求新增 C++ 运算符；
JSON 更大、推理开销增加，需云端/控制器端测延迟。
云端导出时会用含裁剪区间外输入的 64 个样本核对 NumPy 推理，
动作最大误差 >1e-4 时不输出 JSON。该检查不是 RTNeural C++ 或硬件验证。
本地未运行转换或验证，JSON 在用户云端生成。

JSON 的 default_joint_pos/action_scale/关节限位/kp/kd/observation_history 会被
当前 neural_controller 对应字段读取。joint_names、controller_requirements 是配置信息，
不是全部自动应用的参数。部署需匹配 FL/FR/RL/RR、每腿外展/髋/膝的顺序，
位置控制、20ms 策略周期、原始观测限幅100、速度指令零、方向指令通道[0,0,1]。
default_joint_pos 是动作/观测中心，不是启动 stand 姿态；控制器启动插值、
action fade-in 和翻滚角保护需与现有站立起滚流程协调。导出不会修改这些控制器行为，
也不提供 capture 后自动切换/停止或 JSON 内的物理力矩限幅。
