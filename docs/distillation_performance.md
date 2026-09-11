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
BC/statistics snapshot behavior remains unchanged. Evaluation generates its
own snapshots and starts a fresh student timeout budget at takeover.

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

## Re-evaluate an existing student for a full 10 seconds

Use the final `student_params` checkpoint (not the RTNeural JSON). Replace its
directory below with the actual training output directory. No retraining is
needed. Run on the training server from `curl_robot_2d`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.train_mjx_3d_roll_distillation \
  --eval-only \
  --restore-student results/rolling_command_distill_medium_fast/student_params \
  --teacher-source cem \
  --geometry rollingquad_2_abd10_no_self_collision \
  --command-conditioned \
  --random-cem-snapshots \
  --num-devices 4 \
  --eval-envs 64 \
  --episode-length 500 \
  --minimum-closed-loop-turns 5 \
  --eval-seed 100000 \
  --log-every 50 \
  --out results/rolling_command_distill_medium_eval_10s
```

Snapshot warmup still lasts 20–300 teacher control steps. At takeover, only
`info["step_count"]` is reset to zero. Physics time, oscillator state, observation
history, previous action, failure persistence counters, and progress baselines
are preserved. Snapshot mode already requires a fixed command for the episode.
The student receives 500 control steps (10 seconds) regardless of warmup length;
failure still ends the trajectory early. Warmup progress is not counted toward
the student's five-turn threshold. Training/DAgger reset behavior is unaffected.

The five-turn threshold is unchanged. `evaluation.json` records `duration_basis`, teacher
warmup steps, student steps and actual durations per episode, and
`full_horizon_rate`. Full horizon means all 500 steps were executed, not
necessarily success (failure on the last step is still failure).

`--eval-only` skips BC, statistics and DAgger, and does not overwrite the student
checkpoint. It writes metrics to the new output directory. Detailed teacher
action comparisons and lateral traces now require `--record-diagnostics`;
ordinary evaluation avoids that extra teacher rollout. Keep geometry, command
ranges, warmup settings and hidden layers consistent with the original training
configuration. The fixed eval seed makes subsequent re-evaluations reproducible
under the same software/device configuration; it does not reproduce an earlier
training run's implicitly generated evaluation seed.

## Command-conditioned performance breakdown

Evaluation also writes `command_evaluation.json` and
`command_evaluation_episodes.csv`; the JSON is included in
`evaluation.json` under `closed_loop_evaluation.command_evaluation`.
Sync both `scripts/train_mjx_3d_roll_distillation.py` and
`curl_robot_2d_mjx/distillation_evaluation.py` to the server.

The report includes overall results, three equal-width target-speed bins,
straight/left/right groups, and all nine speed-by-turn combinations. For the
default speed range the boundaries are 0.4111, 0.5444, 0.6777 and 0.8110 m/s.
Positive yaw commands are labeled left, negative right; absolute yaw command
at most 0.001 rad/s is straight, consistent with the environment. Groups are
formed from commanded, not achieved, speed. Episodes whose command changes
are excluded from fixed-command groups and counted separately. MAE is weighted
by active transitions, including failure transitions. Empty groups report null
metrics rather than zero success. Failure counts may overlap.

Each populated group reports episode count, legacy success rate, strict success,
failure-free and full-horizon rates, mean effective turns and duration,
forward/yaw tracking MAE, and failure counts. Per-episode commands, errors,
durations, effective turns and failure flags are retained for later analysis.

A separate `minimum_speed_success_rate` implements a minimum-speed progress
criterion without changing legacy success:

```
required_effective_turns = configured_minimum_speed * student_horizon_seconds / (2*pi*rolling_radius)
success = full student horizon AND no strict failure AND effective_turns >= required_effective_turns
```

It uses the configured minimum (default 0.4111 m/s), not the smallest randomly
sampled command, and the environment's actual rolling radius. No arbitrary
tolerance is introduced. This is a progress criterion, not a speed-tracking
accuracy criterion: forward/yaw MAE must still be inspected. Effective progress
uses world-x displacement and rotation, so turns do not measure curved path
length. Keep this limitation in mind when comparing straight and turning runs.

Re-run the command above with `--eval-envs 256` and a fresh output directory
such as `results/rolling_command_distill_medium_eval_grouped_10s` for more
samples per subgroup. Keep `--eval-seed 100000`; repeated runs with the same
batch size/configuration use the same reset sampling. Changing batch size does
not preserve a prefix of old reset samples. Previous summary-only reports lack
the per-episode command/error association and cannot be retroactively grouped.
