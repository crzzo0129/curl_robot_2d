# Stand-to-roll v2 cloud validation

## Optional minimal DAgger before PPO

```bash
python -m scripts.train_stand_to_roll_dagger \
  --bc-params results/stand_to_roll_v2/bc/bc_params \
  --out results/stand_to_roll_dagger
```

This is a separate, untested-locally experiment. It keeps the v2 normalization
and actor layout. Its teacher uses nearest-phase recorded NEXT CEM actions;
it is an approximation, not the original oscillator or a learned recovery
controller. Both nominal and small-perturbation teacher rollouts must achieve
80% sustained success with <=20% failures before training starts. If status is
teacher_gate_failed, the reference cannot supervise recovery reliably under
this setup; no training or replacement checkpoint is produced.

Three rounds collect at teacher intervention probabilities 0.5, 0.25 and 0.
Labels farther than matcher distance 3 are excluded. Each update mixes 50%
original BC data and 50% aggregated student-visited data. These are discrete
teacher interventions during sampling, not action blending or deployment
control switching. Pure-student evaluations select updates; independent final
seeds compare baseline/candidate with identical small-perturbation resets.

Only status=passed exports bc_params, requiring >=80% sustained success,
<=20% failures and improvement over baseline on the final seed batch. This is
finite-sample cloud evidence, not a guarantee of deployment performance.
For student_gate_failed, the original BC is left untouched.

Use the resulting results/stand_to_roll_dagger/bc_params for a NEW rolling_orbit
PPO run and consistently for every later stage. Do not restore checkpoints
trained with the old BC file. See dagger_summary.json for both teacher gates,
baseline, round decisions and final independent evaluations.

Run from `curl_robot_2d` in the Linux MJX environment. These changes have NOT
been tested locally. Keep the old runs for comparison. Do not reuse v1 BC or
PPO checkpoints: v2 changes label alignment and observation preprocessing.

## What changed

- Collector rows are post-action states: state/history at row i now predicts
  the target at row i+1, with target i in the previous-action field.
- Legacy collector angular_velocity is actually local free-joint angular
  velocity. BC now reads that local velocity from qvel when available.
- Recorded actuator targets and live controls are mapped by joint/servo names.
- BC and PPO share frozen, per-channel normalization floors and clipping at 5.
- BC reports a temporal holdout RMSE with a 20-frame separation. This is still
  the same recorded rollout, not evidence of closed-loop generalization.
- Reset curriculum: rolling_orbit (100% snapshots), mixed_75 (75%), mixed_25
  (25%), compact (0%), slightly_open, crouch, semi_stand, full_stand.
- Snapshots restore qpos, qvel and matching 20-frame history. XY translation is
  recentered on the flat floor. No teacher actions are executed after reset.
- Defaults: learning rate 2e-5, one update per batch, initial std 0.02, entropy 0.
- Metrics are written after every evaluation. A nonfinite metric or KL above
  --max-kl (default 1) aborts at the evaluation callback. This is NOT per-update
  KL early stopping and does not roll back a bad update. Inspect aborted runs.
- failure_nonfinite/height/lateral/axis_tilt distinguish terminal conditions.
- sustained_success requires timeout without failure, capture, and at least
  one turn of net conservative rolling progress. Each stage requires >=80%
  sustained_success, >=80% capture and <=20% failures. Mixed-stage evaluation
  includes snapshots; only compact and later stages establish static startup.

## Commands

```bash
python -m unittest tests.test_stand_to_roll_training

python -m scripts.train_mjx_3d_stand_to_roll \
  --stage bc --out results/stand_to_roll_v2

python -m scripts.train_mjx_3d_stand_to_roll \
  --stage rolling_orbit --eval-only \
  --bc-params results/stand_to_roll_v2/bc/bc_params \
  --out results/stand_to_roll_v2

python -m scripts.train_mjx_3d_stand_to_roll \
  --stage compact --eval-only \
  --bc-params results/stand_to_roll_v2/bc/bc_params \
  --out results/stand_to_roll_v2
```

Review `bc/bc_summary.json` and `eval_bc_*/bc_closed_loop_eval.json` before PPO.
The evaluator also checks BC/PPO deterministic actions agree within 1e-5.
If snapshot BC fails, inspect data/model physics consistency and trajectories
before increasing training duration. Low BC training error is not a pass.

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage rolling_orbit --preset smoke \
  --bc-params results/stand_to_roll_v2/bc/bc_params \
  --out results/stand_to_roll_v2
```

Smoke is an execution/stability check, not a sufficient learning budget. For a
longer fresh run, use --preset 4090 or h200 and a new --out path; keep using the
same v2 BC file. After reviewing a passing summary, run the next stage:

```bash
python -m scripts.train_mjx_3d_stand_to_roll \
  --stage mixed_75 --preset 4090 \
  --bc-params results/stand_to_roll_v2/bc/bc_params \
  --restore-checkpoint results/stand_to_roll_v2/rolling_orbit/ppo_checkpoint \
  --out results/stand_to_roll_v2
```

Continue mixed_25 -> compact -> slightly_open -> crouch -> semi_stand ->
full_stand, restoring the preceding stage's ppo_checkpoint each time. The CLI
does not automatically advance stages. Do not advance on capture alone.

`eval/episode_*` diagnostics such as height, distance and tilt are summed over
the episode. Event metrics and sustained_success count events; do not compare
summed height/tilt against single-step termination thresholds. Mean action
saturation can be estimated from its episode sum / avg_episode_length.

Deployment must reuse v2 mean/std AND clipping before the actor. Exporting
only the weights without the same preprocessing changes the policy.
