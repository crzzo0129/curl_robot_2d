# Curl Robot 3-D Walking：训练问题复盘与后续重构交接

更新日期：2026-08-16

本文用于把当前 Curl Robot 四足行走训练的目标、代码演变、失败现象、已经完成的修改、已验证事实和下一步重构方案交接给新的 Codex 对话。后续讨论应以本文和当前工作区代码为准，不要从最初的 `anymal_v1` 配置重新猜测。

## 1. 项目与目标

- 工作区：`C:/Users/12481/Desktop/OH-WorkSpace/robot_description`
- 项目目录：`curl_robot_2d`
- 训练模型：`assets/curl_robot_3d_pupper_r127p5_open60_width120.xml`
- 训练后端：MuJoCo MJX + Brax PPO + JAX
- 当前主要训练入口：`scripts/train_mjx_3d_walking_ppo.py`
- 当前主要环境：`curl_robot_2d_mjx/environment_walking_3d.py`
- 当前主要奖励：`curl_robot_2d_mjx/reward_walking_3d.py`
- 目标：先训练稳定的 `0.10 m/s` 直线行走，再逐步加入横向速度、转向、扰动和 domain randomization，最终形成可部署、可恢复的四足速度跟踪策略。

最初希望参考 ETH ANYmal 的任务定义、reward shaping、domain randomization 和 PPO 流程。后续对比发现，当前代码的基本 observation 和关节位置控制形式很像传统 `legged_gym`，但整个训练组织仍缺少现代 Unitree/Isaac Lab/MjLab locomotion 中的课程、步态脚手架、非对称 Actor–Critic 和恢复训练。

## 2. 早期问题与已经做过的修正

### 2.1 Brax domain randomization 找不到 `env.sys`

最早运行 `anymal_v1` 时出现：

```text
AttributeError: 'CurlRobot3DWalkingMJXEnv' object has no attribute 'sys'
```

原因是 Brax 的 `DomainRandomizationVmapWrapper` 要求环境把 MJX 模型暴露为 `env.sys`。环境现在已经使用 `self.sys = mjx.put_model(...)`，所有 MJX dynamics 调用也使用 `self.sys`。

### 2.2 横向漂移不应直接终止横向指令训练

旧逻辑把相对初始 `y` 坐标的绝对漂移作为 failure。对于包含横向速度指令的训练，这会惩罚本来正确的横向运动。

现在 `lateral_drift` 只作为诊断指标和固定直行 evaluation 的辅助指标，不再直接参与 termination。

### 2.3 `alive=0`

`alive` 当前为零，不是因为生存不重要，而是避免每一步固定正奖励再次制造“原地站立就是最优”的局部最优。存活价值已经通过持续获得任务奖励和避免终止体现。

### 2.4 Observation 坐标系修正

基础 recipe 的 observation 共48维；启用 Stage-A phase 脚手架后追加
`sin(phase), cos(phase)`，共50维：

```text
3  body linear velocity
3  body angular velocity
3  projected gravity
3  velocity command
12 joint position relative to stand
12 joint velocity
12 previous action
```

Observation 中自由关节世界线速度会旋转到 torso body frame；MuJoCo 已经按机体局部坐标存储的自由关节角速度不重复旋转。

Reward 的平面速度另外使用“只包含 yaw 的 heading frame”。因此 roll、pitch 和世界竖直速度不会泄漏到 forward velocity tracking 中，避免机器人通过向前倒产生假的速度收益。

### 2.5 策略分布和 Actor 初始化

策略已经改为：

- Gaussian policy；
- state-independent scalar/log standard deviation；
- Actor mean 输出层使用很小但非零的随机初始化；
- mean 输出限制在动作范围内；
- 参数仍然正常接收梯度，并不是人为固定 Actor 输出为零。

这更接近 ETH/RSL-RL 的状态无关探索分布，但当前 stage-1 的初始标准差后来被压低到 `0.10`，这已经成为新的探索瓶颈，详见后文。

## 3. 训练中持续出现的现象

### 3.1 Step 0 经常成为 best checkpoint

随机初始化策略在精确 stand reset 下有时可以维持很久，但不会正确前进。训练后策略开始尝试动作，反而更早失败。因此按存活优先选择 checkpoint 时，`step=0` 会成为 `params_best`。

这不表示初始策略完成了任务，只表示当前 checkpoint 排序把“完整存活”放在“真实行走”之前。当前脚本仍保存 `params_final`，但后续应拆分 `best_survival`、`best_tracking` 和 `best_return`，避免一个 best 指标承担相互冲突的目标。

### 3.2 速度增加但 reward 下降

曾经出现 distance 和 forward velocity 增加，但 reward 反而下降。主要原因包括：

- 向前倾斜或扑倒造成短时速度；
- upright、angular velocity、collision 和 termination 同时恶化；
- 非足端结构接触地面；
- episode 迅速缩短。

这类速度不是真正周期步态，渲染 `params_final` 后也确认机器人主要通过前倾获取短时位移。

### 3.3 失败集中在 `upright_tilt` 或 `nonfoot_contact`

多轮 evaluation 中，失败原因曾集中为：

- `upright_tilt`：策略无法建立稳定支撑和摆腿切换；
- `nonfoot_contact`：前倾后腿部或机身结构触地；
- 极短 episode：策略一旦离开站姿就缺乏恢复能力。

典型后期点曾出现 forward velocity 接近甚至超过目标，但 episode 只有几十步，且 `nonfoot_contact=100%`。这说明策略发现的是“短时前扑”，不是可持续步态。

### 3.4 KL 爆升与随后过度保守

当 learning rate 提高到 `3e-4` 或一批数据重复更新较多时，曾出现 KL、policy loss 和 value loss 突然放大，随后 reward 崩溃。为避免崩溃，配置逐步被压到：

```text
learning_rate       = 2e-5
updates_per_batch   = 1
desired_kl          = 0.003
init_noise_std      = 0.10
reward_scaling      = 0.05
```

这些设置虽然降低了单次更新风险，但与主流四足 PPO 相比已经过度保守：策略很难跨出原地站立附近的局部最优，探索标准差也会逐渐收缩。

## 4. 当前 `forward_stage1_v1` 实际配置

### 4.1 Task

```text
control dt                 = 0.020 s (50 Hz)
episode length             = 500 steps = 10 s
command vx                 = exactly 0.10 m/s
command vy                 = 0
command yaw rate           = 0
reset                      = exact stand
reset joint/root noise     = 0
observation noise          = disabled
domain randomization       = disabled
action scale abduction     = 0.06 rad
action scale hip           = 0.50 rad
action scale knee          = 0.65 rad
```

因此所有训练环境都从几乎同一个状态、同一个指令开始。策略采样噪声仍会带来差异，但状态覆盖远窄于带 reset noise、command curriculum 和 push recovery 的训练。

### 4.2 Reward 关键项

```text
velocity tracking weight        = 4.0
velocity sigma                  = 0.05 m/s
overspeed weight                = 1.0
yaw tracking                    = 0.25
forward progress                = 0
upright                         = 0.2
upright sigma                   = 0.20 rad
stagnation penalty              = up to -0.2/step
foot air time                   = 0.8
swing clearance                 = 0.15
swing target height             = 0.025 m
angular velocity penalty        = 0.15
action rate penalty             = 0.04
termination                     = 20
early termination scale         = 0.5
```

注意当前 velocity tracking 仍然乘以 `upright_score`：

```python
velocity_tracking = weight * velocity_score * upright_score
```

这虽然抑制前倾作弊，却会在机器人已经倾斜时同时切断速度恢复信号。由于 forward progress 已经取消、reward velocity 已修正为世界水平 heading frame，后续建议把 tracking 与 upright 解耦，用独立 orientation/height/contact/overspeed 项约束前倾。

### 4.3 一秒停滞机制

已经实现50个控制步的位置环形缓冲。令最近1秒沿命令方向的位移为 `d`：

```text
p = clip((0.05 - d) / 0.05, 0, 1)
upright_reward = original_upright_reward * (1 - p)
stagnation_reward = -0.2 * p
```

效果：

- episode前1秒不处罚；
- 1秒完全不动：upright变为0，额外 `-0.2/step`；
- 1秒前进0.025 m：upright减半，额外 `-0.1/step`；
- 1秒前进至少0.05 m：upright完整，不扣停滞分；
- 固定指令仅被周期性重新采样但数值没变时，不会错误重置1秒宽限。

日志中对应 `progress_1s` / `progress_window_m` 和 `stagnation` / `stagnation_fraction`。

### 4.4 PPO

```text
envs                   = 2048 (H200 preset)
unroll length          = 40
samples per rollout    = 2048 * 40 = 81,920
minibatches             = 8
updates per batch       = 1
learning rate           = 2e-5
adaptive LR range       = [2e-6, 2e-5]
desired KL              = 0.003
entropy cost            = 0.01
initial policy std      = 0.10
discount                = 0.99
network                 = [256, 256, 128], ELU
observation norm        = disabled; fixed physical scaling is used
```

一批 rollout 的样本量与 Unitree 常用设置相近，但数据只复用1轮，而且 LR、KL 和初始 std 同时很小。

## 5. 已完成的机械可行性验证

已经为同一个 XML 编写了不依赖 RL 或 reward 的解析 IK 对角小跑：

```text
script       = scripts/demo_handcrafted_3d_walk.py
frequency    = 1.6 Hz
step length  = 0.045 m
foot lift    = 0.025 m
duty factor  = 0.68
gait         = FL+RR / FR+RL diagonal trot
```

本地 `cg12` 仿真结果：

```text
duration                   = 6.0 s
failed                     = false
x displacement             = 0.534 m
mean forward velocity      = 0.089 m/s
final lateral drift        = -0.018 m
maximum upright tilt       = 0.093 rad
mean foot contact count    = 2.68
```

结果文件：

- `results/handcrafted_3d_walk_default/handcrafted_summary.json`
- `results/handcrafted_3d_walk_default/evaluation_rollout.npz`
- `results/handcrafted_3d_walk_default/handcrafted_walk.gif`

结论：同一个机构、执行器、关节范围和物理配置可以接近 `0.10 m/s` 稳定行走。当前 RL 失败不能归因于模型根本无法行走。

## 6. 关节方向检查结论

有限差分检查显示：

- 前腿 hip 轴是 `-Y`；
- 后腿 hip 轴是 `+Y`；
- 正 hip 增量使前足向 `+X`、后足向 `-X`，即都向各自机身外侧移动；
- 世界坐标中让前后足同向摆动，本来就需要前后 hip 使用相反增量。

这属于镜像机构的正确设置，不是关节方向接反。

“前腿 hip 一弯机器人就向前倒”的原因是：单独改变前 hip 会让前足前移并略微抬起，前支撑卸载；如果 knee、对角腿和落脚时机没有同步协调，机身就会绕后足向前俯倒。手写IK通过同时协调 hip、knee 和对角相位避免了这个问题。

## 7. 与 Unitree/主流四足训练结构的关键差异

Unitree官方存在多套训练栈：旧版 `unitree_rl_gym/legged_gym`、Isaac Lab版 `unitree_rl_lab`、以及当前基于MuJoCo/MjLab的 `unitree_rl_mjlab`。当前Curl实现的48维 observation 和关节位置残差接近旧版 `legged_gym`，但缺少后两套训练栈的完整组织。

### 7.1 没有真正的自动 curriculum

`forward_stage1_v1` 只是静态 recipe。进入横向、转向、reset noise、push或domain randomization需要手工启动另一轮训练，且阶段之间容易发生分布突变。

Unitree当前实现把 command range 和 terrain difficulty作为 curriculum term，在训练中逐步扩展，而不是一次性从固定直行跳到完整随机指令。

### 7.2 没有步态相位和接触计划

当前策略必须从纯随机动作中同时发现：

1. 哪条腿抬起；
2. 哪条对角腿配合；
3. 支撑腿如何保持机身；
4. 摆腿方向；
5. 落脚时机；
6. 下一半周期如何交换。

Unitree官方MuJoCo velocity task把 phase 放进 Actor observation，并提供稠密 `foot_gait` 接触匹配奖励。对于标准四足，纯reference-free也可能自行发现步态；对于Curl这种非标准镜像机构，已有手写可行步态却完全不用它作为弱先验，会显著增加探索难度。

### 7.3 Actor和Critic没有分离

当前 Actor 和 Critic 使用同一48维 observation。Actor直接看到仿真中的精确base linear velocity，而Critic没有额外的足高、接触、接触力和执行器信息。

Unitree新实现通常让Actor只看可部署传感器状态，让Critic使用无噪声且更丰富的 privileged observations。当前Brax接口如果不方便直接支持，可以先保留现有Actor完成步态发现，之后再做 asymmetric critic 或 teacher-student/distillation。

### 7.4 Tracking和upright被耦合

主流实现通常把速度tracking和orientation penalty分开。当前把 tracking乘以upright，机器人进入倾斜状态后缺少恢复方向的稠密任务信号，这与“一旦探索失败就没有恢复能力”的观察一致。

### 7.5 PPO被压得过于保守

Unitree官方RSL-RL常用：

```text
rollout steps/env     = 24
learning epochs       = 5
minibatches           = 4
learning rate         = 1e-3
desired KL            = 0.01
initial std           = 1.0
entropy               = 0.01
```

不能把这些数字直接复制到Curl，但当前 `1 epoch + 2e-5 LR + 0.003 KL + 0.10 std` 是另一极端。以当前动作尺度计算，初始一倍标准差只产生：

```text
abduction target std  = 0.006 rad
hip target std        = 0.050 rad
knee target std       = 0.065 rad
```

这很难随机形成多腿协调的2.5 cm抬脚动作。

### 7.6 恢复训练和randomization不完整

当前domain randomization包括摩擦、刚体质量/惯量、执行器增益、关节阻尼和armature，但缺少：

- 训练中的周期性push；
- COM offset；
- encoder bias；
- restitution；
- action/sensor latency；
- 地形难度；
- 独立的recovery evaluation。

这些因素不应全部从第0步启用，但应作为 curriculum 的后半段，而不是完全缺失。

### 7.7 Evaluation和checkpoint职责混在一起

当前固定直行、精确stand evaluation适合比较nominal gait，但不足以判断恢复、指令泛化和随机化鲁棒性。checkpoint又把存活放在首位，因此经常选择step 0。

后续应至少保存：

- `best_survival`；
- `best_tracking`（先要求存活达到阈值）；
- `best_return`；
- `final`；
- 周期性原始PPO checkpoint。

## 8. 当前最可能的根因排序

按优先级判断：

1. **缺少步态发现脚手架**：没有phase和对角接触计划，摆腿/支撑协调只能靠低概率随机探索。
2. **初始探索过小且PPO更新过弱**：大batch只训练1轮，LR和KL上限过低，std从0.10开始。
3. **训练状态分布过窄**：精确stand和固定命令让策略没有学到跌倒边缘状态，也没有恢复经验。
4. **tracking乘upright形成恢复死区**：姿态一差，任务方向信号也消失。
5. **checkpoint选择掩盖后期进步**：不影响梯度，但会让可视化和人工判断长期围绕step 0。
6. **后续sim2real结构不足**：Actor/Critic未分离，缺少延迟、偏置、COM和push训练。这不是当前“完全不会走”的首要原因，但会成为下一阶段问题。

## 9. 建议的下一版训练结构（尚未实施）

以下内容是下一步方案，当前代码尚未全部实现。

### Stage A：phase-guided gait bootstrap

- Observation增加 `sin(phase), cos(phase)`；
- phase周期使用手写验证值：`1 / 1.6 = 0.625 s`；
- 对角偏移：FL/RR为0，FR/RL为0.5；
- 支撑比例先使用 `0.68`；
- 加稠密 contact schedule reward；
- 保留直接 `stand + action` 控制，不必立即使用完整关节轨迹模仿；
- tracking不再乘upright，orientation独立处理；
- clearance改为围绕 `0.025 m` 的目标误差，而不是只要超过目标就满分；
- reset加入很小的关节和速度噪声；
- 暂不启用完整domain randomization。

如果纯phase/contact仍无法学会，可采用更强的临时方案：

```text
q_target = handcrafted_q_ref(phase, command) + residual_scale * policy_action
```

先训练稳定器，再逐步衰减reference幅度或扩大residual自由度。这会从“纯reference-free”变为“残差步态bootstrap”，但对于非标准机构更务实。

### Stage B：稳定与恢复

- 逐步降低phase/contact reward权重，确认策略不是只在刷接触分；
- 增加轻微root velocity/reset noise；
- 每5–10秒加入小push；
- 要求连续10秒存活、速度接近0.10 m/s、无非足端触地；
- 观察push后1秒内速度和姿态是否恢复。

### Stage C：command curriculum

- 从前向小范围开始，而不是直接全范围：例如 `[0.05, 0.15] m/s`；
- 再加入少量停止指令；
- 然后逐步加入倒退、横向和yaw；
- 每个阶段按tracking和survival门槛自动扩展，而不是仅按固定训练步数切换。

### Stage D：robustness与部署

- 加摩擦、COM、质量、增益、阻尼、armature、encoder bias、latency；
- 增加地形变化；
- 使用privileged Critic；
- 进行MuJoCo CPU sim2sim、命令网格、扰动恢复和多seed测试；
- 最终导出部署所需策略格式和严格一致的observation/action contract。

## 10. PPO建议起点（尚未实施）

不要直接复制Unitree的 `LR=1e-3/std=1.0`。在加入phase脚手架后，可从以下较温和配置开始：

```text
updates_per_batch = 4
learning_rate     = 1e-4
desired_kl        = 0.01
init_noise_std    = 0.25 ~ 0.40
entropy_cost      = 0.01
clip              = 0.2
max_grad_norm     = 1.0
```

应记录每次更新的：

- approximate KL；
- clip fraction；
- policy std；
- explained variance或value prediction统计；
- advantage均值/标准差；
- gradient norm；
- episode state distribution。

如果KL再次突然爆升，应优先检查 advantage/value normalization、terminal transition、动作分布和batch数据多样性，而不是立刻把LR永久压回 `2e-5`。

## 11. 推荐evaluation矩阵

训练过程中应同时运行四种evaluation，而不是只看固定直行：

1. **Nominal gait**：精确stand、0.10 m/s、无噪声；
2. **Reset recovery**：小姿态/速度/关节扰动；
3. **Push recovery**：固定时刻施加不同方向速度扰动；
4. **Command grid**：多个vx、vy和yaw组合。

关键成功指标：

```text
episode survival        >= 95% of 10 s
mean vx error           <= 0.02~0.03 m/s
upright tilt            remains bounded
nonfoot contact failure near 0
clear alternating diagonal contacts
foot clearance          near 0.025 m, not excessive
push recovery           returns near command within ~1 s
```

不要只用total reward判断成功；必须同步渲染contact、root trajectory、joint trajectory和policy mean/std。

## 12. 当前可运行命令

### 手写步态验证

```bash
python -m scripts.demo_handcrafted_3d_walk \
  --duration 6 \
  --output results/handcrafted_3d_walk_default

python -m scripts.render_mjx_3d_policy \
  results/handcrafted_3d_walk_default/evaluation_rollout.npz \
  --model-xml assets/curl_robot_3d_pupper_r127p5_open60_width120.xml \
  --physics-profile cg12 \
  --output results/handcrafted_3d_walk_default/handcrafted_walk.gif
```

### 当前尚未加入phase脚手架的长训

下面命令仍使用现有 `forward_stage1_v1`，只适合复现当前基线，不代表下一版推荐结构：

```bash
python -m scripts.train_mjx_3d_walking_ppo \
  --preset h200 \
  --recipe forward_stage1_v1 \
  --steps 15000000 \
  --num-evals 32 \
  --save-ppo-checkpoints \
  --ppo-checkpoint-dir results/mjx_pupper_forward_stage1_v9/ppo_checkpoint \
  --out results/mjx_pupper_forward_stage1_v9
```

下一次训练应该使用新输出目录，不要恢复最近失败策略，也不要覆盖旧结果。

### 已实现的 Stage-A phase bootstrap（2026-08-16）

当前代码已新增 `forward_phase_bootstrap_v1`：

- observation 从48维增至50维，末尾追加 `sin(phase), cos(phase)`；
- 周期为0.625秒，足序 FL/FR/RL/RR 的相位偏移为 `0/0.5/0.5/0`，支撑比例为0.68；
- 接触奖励按四足实际接触与对角相位计划的匹配比例计算；
- velocity tracking 不再乘 upright score，姿态由独立 upright、角速度、碰撞和终止项约束；
- swing clearance 使用以0.025米为中心、sigma为0.0075米的目标误差奖励，过高抬脚不再满分；
- reset 加入小幅关节、速度、根部平面速度与 yaw-rate 扰动；
- PPO 使用 `updates_per_batch=4`、`LR=1e-4`、`desired_kl=0.01`、`init_noise_std=0.30`；
- 原 `forward_stage1_v1` 保持48维和旧奖励行为，用于基线复现。

旧48维 checkpoint 与新50维网络输入层不兼容，不应通过
`--restore-checkpoint` 直接载入。建议使用新的输出目录：

```bash
python -m scripts.train_mjx_3d_walking_ppo \
  --preset h200 \
  --recipe forward_phase_bootstrap_v1 \
  --steps 15000000 \
  --num-evals 32 \
  --save-ppo-checkpoints \
  --ppo-checkpoint-dir results/mjx_pupper_forward_phase_bootstrap_v1/ppo_checkpoint \
  --out results/mjx_pupper_forward_phase_bootstrap_v1
```

训练前可先编译并验证50维 phase 环境：

```bash
python -m scripts.mjx_3d_walking_smoke \
  --gait-phase \
  --physics-profile cg12 \
  --steps 8
```

## 13. 关键代码位置

- Stage recipe和PPO入口：`scripts/train_mjx_3d_walking_ppo.py`
- Environment reset/step/termination/observation：`curl_robot_2d_mjx/environment_walking_3d.py`
- Reward：`curl_robot_2d_mjx/reward_walking_3d.py`
- Task配置：`curl_robot_2d_mjx/config_walking_3d.py`
- Domain randomization：`curl_robot_2d_mjx/randomization_3d.py`
- 模型和执行器：`assets/curl_robot_3d_pupper_r127p5_open60_width120.xml`
- 手写步态：`scripts/demo_handcrafted_3d_walk.py`
- 现有说明：`docs/anymal_style_mjx_walking_zh.md`
- 本交接文档：`docs/walking_training_handoff_zh.md`

已经新增或更新了与手写步态、reward和训练recipe相关的单元测试。最近一次回归为39个测试通过，另有1个依赖环境的测试跳过；MJX smoke也曾在修改环形缓冲后通过。

## 14. 官方对比资料

- Unitree官方MuJoCo/MjLab velocity任务：<https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/velocity_env_cfg.py>
- Unitree Go2 MuJoCo配置：<https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/env_cfgs.py>
- Unitree Go2 PPO配置：<https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/rl_cfg.py>
- Unitree官方Isaac Lab Go2任务：<https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py>
- RSL-RL PPO配置说明：<https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst>
- ETH legged_gym：<https://github.com/leggedrobotics/legged_gym>

## 15. 给下一段对话的建议开场

可以在新对话中直接发送：

> 请先阅读 `curl_robot_2d/docs/walking_training_handoff_zh.md`，再检查当前git diff和walking相关代码。不要继续只调reward权重。优先设计并实现Stage A：加入0.625秒周期的sin/cos phase observation、FL+RR与FR+RL对角接触计划奖励，把velocity tracking与upright解耦，并把clearance改为0.025 m目标误差。实现前先列出observation维度变化、reward公式、课程开关和兼容旧checkpoint的影响，然后修改代码和测试。

## 16. 一句话结论

机构已经被手写步态证明可以稳定行走；当前失败的核心不是“reward或observation完全不通用”，而是**缺少步态发现脚手架、训练状态分布过窄、tracking与upright耦合，以及PPO在经历KL崩溃后被调得过度保守**。下一步应重构训练课程，而不是继续围绕单个reward系数反复试错。
