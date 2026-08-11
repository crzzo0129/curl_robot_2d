# 3D real-geometry 行走训练

此入口在 `curl_robot_3d_real_geometry.xml` 上从零训练 50 维观测、8 维动作的
3D PPO 行走策略。`000021299200.bin` 的接口是 20 维观测、4 维动作；在缺少原始
20 维观测定义时不能安全恢复到该任务，因此本入口不会假装使用不兼容的参数。

## H200 环境

要求 Linux、Python 3.12、CUDA 12，并在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mjx.txt
```

先验证 real-geometry MJX 编译和 stepping：

```bash
python -m scripts.mjx_3d_walking_smoke \
  --geometry real \
  --physics-profile cg12 \
  --batch-size 8 \
  --steps 4 \
  --mujoco-gl disable
```

smoke 成功后启动后台 H200 训练：

```bash
bash scripts/run_mjx_3d_real_geometry_walking_overnight.sh
```

默认配置为 20M steps、2048 个训练环境、256 个评估环境，并周期性保存 PPO
checkpoint。脚本会输出 PID、日志路径和结果目录。可用以下命令监控：

```bash
tail -f results/mjx_3d_real_geometry_walking_h200_seed0_*.log
nvidia-smi
```

训练结束后的主要产物为：

- `params_best`：按稳定行走指标选出的最佳参数；
- `params_final`：最后一次更新的参数；
- `ppo_checkpoint/`：用于中断恢复的 PPO checkpoint；
- `training_summary.json` 和指标历史：训练配置及评估记录。

如果 H200 显存不足，先把启动命令改为 `--preset 4090`，确认不是其他进程占用
显存后再提高并行环境数。
