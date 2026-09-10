# 将对称化教师蒸馏进学生网络

冻结原 PPO 策略，教师输出为前左/前右、后左/后右目标动作平均。
学生从原 actor 初始化，仍为 720 输入、12 输出，保持 BC 归一化。
第一轮采教师闭环轨迹，后两轮在当前学生访问的状态上查询冻结教师标签，累计数据。
Adam 1e-5，梯度范数上限 0.5，batch 256，每轮默认 1000 次更新，共三轮。
损失为学生确定性动作对教师动作的 MSE；不更新 critic，不执行 PPO。
学生收集和验收不施加左右平均，因此可检验网络自身是否学会。
冻结教师在大幅偏离状态上未必可靠，所以仅做有限轮数并闭环验收。

```bash
python -m scripts.render_stand_to_roll_checkpoint \
  --checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --distill-symmetry --episodes 32 --distill-updates 1000 --seed 0 \
  --out results/stand_to_roll_symmetry_student
```

教师门槛：capture、保险 >=95%，capture 时 axis P95 <5°。
各轮使用独立于采集的固定初始状态筛选；最后再用一批新的初始状态，
配对比较原策略与学生。最终要求成功率不低于原策略且 >=95%、失败率不增加、
axis P95 <5° 且优于原策略。32 个回合是初筛，不是任意扰动保证。
不通过也会保存 distillation_report.json；仅通过时导出 student_params。

学生负载/交接复验（无平均开关）：

```bash
python -m scripts.render_stand_to_roll_checkpoint \
  --checkpoint results/stand_to_roll_capture_1m/full_stand/ppo_checkpoint/000000675840 \
  --bc-params results/stand_to_roll_startup/bc/bc_params \
  --student-params results/stand_to_roll_symmetry_student/student_params \
  --load-eval --episodes 64 --seed 40000 \
  --out results/symmetry_student_eval
```

视频用同一入口、同一 --student-params，去掉 --load-eval 并换输出目录。
student_params 是推理参数文件，不是 Brax 数字目录训练 checkpoint，不能直接传给
PPO 的 --restore-checkpoint。保留原配置和 student_source.json 用于参数来源核对。
蒸馏模仿的是强制对称教师，不等同于镜像等变网络；更大扰动下的纠偏仍需验证。
本地未运行测试或仿真，以上操作在云端执行。
