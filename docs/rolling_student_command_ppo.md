# Fine-tune the command-conditioned rolling student with PPO

Run on the Linux training server, from `curl_robot_2d`. This path copies the
distilled actor and freezes its observation normalizer, initializes a fresh
privileged critic, and optimizes reward. The actor remains a 720-input MLP with
eight active motor outputs; exported checkpoints expand to twelve outputs with
locked abduction. Commands are present in actor history and appended to the
65-value privileged critic observation (68 total).

Sync `scripts/train_mjx_3d_roll_student_dr_ppo.py` and
`curl_robot_2d_mjx/environment_rolling_student_dr_3d.py`, plus the existing
distillation script and its helper modules, and
`curl_robot_2d_mjx/rolling_student_snapshot_pool.py`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.train_mjx_3d_roll_student_dr_ppo \
  results/rolling_command_distill_medium_fast/student_params \
  --geometry rollingquad_2_abd10_no_self_collision \
  --command-conditioned \
  --rolling-snapshots \
  --snapshot-pool-size 512 \
  --eval-snapshot-pool-size 256 \
  --forward-command-min-m-s 0.4111 \
  --forward-command-max-m-s 0.8110 \
  --turn-command-straight-fraction 0.4 \
  --command-interval-s 10 \
  --preset h200 \
  --max-devices 4 \
  --steps 10000000 \
  --envs 2048 \
  --eval-envs 256 \
  --num-evals 10 \
  --dr-strength 0 \
  --student-anchor-weight 0 \
  --initial-policy-std 0.02 \
  --learning-rate 0.00005 \
  --observation-noise-scale 1 \
  --out results/rolling_command_ppo_stage1
```

`--steps` counts environment transitions, unlike DAgger update counts. Brax PPO
uses synchronous data parallelism across up to `--max-devices` visible devices.
No teacher action loss is evaluated when anchor weight is zero. The CEM file
still supplies environment reference metadata; direct action mode applies the
student's motor actions rather than adding a CEM action. Extra deployment
domain randomization is disabled in this first stage, while observation noise
remains consistent with student training. No local runtime tests were run.

The initial command-tracking reward recipe emphasizes forward speed tracking
(weight 4, Gaussian sigma 0.15 m/s) on both straight and turning commands, and
turning heading-rate tracking (weight 2, sigma 0.05 rad/s). Rolling progress
weight is reduced from 6 to 0.5 to reduce incentive to exceed a low requested
speed. Straight lateral-drift reward is increased from 0.5 to 1.5. Straight-line
rewards are masked on commanded turns. Axis stability, action smoothness,
torque costs and failure penalties remain. The full reward configuration is
saved in `training_config.json`. These are initial tuning choices, not validated
optimal weights. Success thresholds do not replace the tracking reward.

Command-conditioned PPO now defaults to `--rolling-snapshots`. Both training
and internal evaluation begin from rolling states, not the compact startup.
Separate training/evaluation pools are generated once before PPO with different
seeds; subsequent resets sample cached snapshots with replacement, with no CEM
simulation at reset. The finite pool is reused for the run. Increasing pool
size increases coverage, initial generation time and device memory use; candidate
counts must divide by the selected device count.

The teacher alone starts from compact for 100–300 control steps by default.
Snapshots must have completed at least the minimum warmup, have no terminal
failure or nonfinite pose/velocity, and have forward speed >0.05 m/s and absolute
roll rate >0.5 rad/s. These gates reject stationary starts; they are not a proof
of future stability or exact target-speed tracking. If too few candidates pass,
generation raises an error rather than falling back to compact. Accepted pool
counts, commands, seeds and warmup steps are saved in `training_config.json`.

Each reset restores the full simulator state, pre-takeover observation history
and previous action. Physical time, phases, original lateral reference and
physical failure-persistence counters remain intact. The student episode
counter starts at zero; PPO reward/episode metrics start at takeover. Success
subtracts the initial rolling potential so teacher progress is excluded.
Tracking error is scored from the first student transition for up to a full
10 seconds (failure can end it earlier). There is no artificial zero-speed
startup and no grace period that would hide post-handoff tracking errors.

Snapshots currently require nominal physics (`--dr-strength 0`) and commands
fixed for the full episode. Explicit `--no-rolling-snapshots` retains compact
starts for separate startup experiments; it is not recommended for a policy
whose responsibility begins after stand-to-roll.

These are CEM-generated rolling proxies, not actual stand-to-roll handoff
states. They remove the inappropriate startup objective, but actual integration
requires a handoff dataset with velocity, phase, history and previous action
from the stand-to-roll controller. Random rolling phases are broader than a
specific handoff distribution and do not establish success of the complete
stand-to-roll-then-roll sequence.

After training, use the existing standalone grouped evaluator with
`--restore-student results/rolling_command_ppo_stage1/student_params`, the same
command ranges, `--random-cem-snapshots`, `--eval-seed 100000`,
`--eval-envs 256`, and a fresh output directory. Compare forward/yaw MAE by
command group and straight lateral failure counts, in addition to success.
PPO exports an action-only checkpoint. The updated evaluator marks auxiliary
velocity-estimator RMSE as unavailable (`null` in JSON) for such checkpoints,
instead of reporting the error of a freshly initialized head. Tracking MAE is
measured from physics and remains valid.

`params_final` is the PPO actor/critic checkpoint for PPO continuation;
`student_params` is the deterministic checkpoint for the common evaluator;
`student_rtneural.json` is the deployment export. A prior non-command PPO
checkpoint has a different critic input dimension and must not be supplied as
`--restore-ppo` for command-conditioned training. Start this stage from the
distilled `student_params` instead.
