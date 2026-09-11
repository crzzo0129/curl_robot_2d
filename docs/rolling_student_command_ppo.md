# Fine-tune the command-conditioned rolling student with PPO

Run on the Linux training server, from `curl_robot_2d`. This path copies the
distilled actor and freezes its observation normalizer, initializes a fresh
privileged critic, and optimizes reward. The actor remains a 720-input MLP with
eight active motor outputs; exported checkpoints expand to twelve outputs with
locked abduction. Commands are present in actor history and appended to the
65-value privileged critic observation (68 total).

Sync `scripts/train_mjx_3d_roll_student_dr_ppo.py` and
`curl_robot_2d_mjx/environment_rolling_student_dr_3d.py`, plus the existing
distillation script and its helper modules.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.train_mjx_3d_roll_student_dr_ppo \
  results/rolling_command_distill_medium_fast/student_params \
  --geometry rollingquad_2_abd10_no_self_collision \
  --command-conditioned \
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

This PPO entry resets from the compact pose; it does not implement random CEM
snapshot resets. Its internal evaluations therefore also include startup and
cannot be compared directly to the earlier snapshot-takeover success rates.
Inspect the initial PPO evaluation before interpreting improvement. If startup
dominates failure, a snapshot reset curriculum is a separate required change;
increasing reward weights alone will not fix that distribution mismatch.

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
