# DAgger execution and multi-GPU training

Run from `curl_robot_2d` on the Linux training machine:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.train_mjx_3d_roll_distillation \
  --teacher-source cem \
  --geometry rollingquad_2_abd10_no_self_collision \
  --command-conditioned \
  --random-cem-snapshots \
  --preset h200 \
  --num-devices 4 \
  --envs 2048 \
  --snapshot-pool-refresh-steps 100 \
  --stats-steps 100 \
  --train-steps 2000 \
  --dagger-steps 1000 \
  --eval-envs 64 \
  --log-every 10 \
  --out results/rolling_command_distill_medium_fast
```

`--envs` and `--eval-envs` are global batch sizes and must divide evenly over
`--num-devices`. Four devices with 2048 environments means 512 environments
per device. Statistics, batched physics, snapshot generation, and learning use
the selected devices. Parameters and optimizer state are replicated; the loss
is a global mean, with cross-device gradient reductions. This trains one
student. Default device count remains one. Multi-device terrain/deploy-DR
is rejected explicitly; those modes retain their single-device path.

DAgger now compiles observation processing, teacher labeling, optimization,
student physics, and reset selection as one update. No Python boolean reads a
device termination flag every step. Unused teacher outputs can be removed by
the compiler; physics needed to compute the effective label is retained.
Logging synchronizes at the requested
interval and reports completed updates/s and environment transitions/s;
the first interval includes compilation. Refresh costs are included in the
following throughput interval. Physics substeps and teacher warmup transitions
are not counted as student transitions.

With random CEM snapshots, one complete reset snapshot per environment is
cached on device, including history and previous action. Failed environments
reuse their corresponding snapshot until the pool is refreshed (100 updates
by default), while active environments keep their current state. Refresh uses
fresh random keys and the original 20–300 step warmup range. It is independent
of the failure rate. For 1000 updates, the DAgger pool needs 10 generations
including initialization, rather than up to one generation per update.
BC/statistics snapshot behavior and the independently seeded evaluation reset
remain unchanged.

Reusing snapshots introduces correlation between repeated resets within each
refresh interval. Compare final closed-loop evaluation as well as throughput;
reduce `--snapshot-pool-refresh-steps` if more reset diversity is needed.
This optimization does not reproduce the original random-number trajectory.

Snapshot generation, first DAgger compilation, and DAgger training print a
heartbeat every 30 seconds during long waits. The checkpoint
`student_params_before_dagger` preserves the completed BC result. Resume with
the same model/command/normalization configuration and
`--restore-student <previous-output>/student_params_before_dagger`, using a new
output directory. It skips BC/statistics and starts DAgger with a fresh optimizer
and intervention schedule; it is not an exact interrupted-step resume.

CPU regression verification of four-device partitioning and global gradients:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  python -m unittest discover -s tests -p 'test_distillation_execution.py' -v
python -m unittest discover -s tests -p 'test_rolling_3d_distillation.py' -v
```

CPU correctness checks do not establish H200 throughput or scaling. Compare
one and four devices at the same global batch size on the training server;
communication and MJX compilation can limit scaling.
