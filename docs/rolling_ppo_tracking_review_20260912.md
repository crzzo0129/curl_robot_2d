# Rolling PPO tracking v4 diagnostic review — 2026-09-12

Source: `rolling_ppo_tracking_20260911_145224_diagnostics.zip`. Parsed diagnostic JSON/logs only; no local training, simulation, tests, or dependency installation. Extracted inputs are under `results/ppo_tracking_review_20260912/`. This review does not change the trainer.

## Decision

Do not promote the stopped checkpoint or continue this configuration unchanged. Keep `results/rolling_ppo_actor_probe_20260911_123158/params_final` as the continuation baseline. The v4 panel-best checkpoint is not an established improvement: it gains seven episodes and loses six against its own step-zero baseline.

## Paired fixed evaluation

All 12 evaluations use the evaluator's cached initial states, 256 episodes, observation noise scale 1, and a 500-step / 10-second student horizon. Initial-state hash: `f32228e576473f352a398639336212338b29de7abbde6cddc563bfffbac62660`. The hash is recorded when the cache is created, not recomputed per evaluation. Source confirms cached initial states are reused. Numerical rollout variability remains possible, but there is no per-checkpoint snapshot resampling.

| Metric | Step 0 | Panel best: 245,760 | Stopped: 1,351,680 |
|---|---:|---:|---:|
| Success | 217/256 (84.77%) | 218/256 (85.16%) | 191/256 (74.61%) |
| Full horizon | 226/256 (88.28%) | 227/256 (88.67%) | 200/256 (78.13%) |
| Forward MAE, m/s | 0.104445 | 0.105296 | 0.105848 |
| Yaw MAE, rad/s | 0.044340 | 0.044238 | 0.045912 |
| Mean return | 2888.25 | 2882.21 | 2737.93 |
| Lateral failures | 28 | 27 | 55 |
| Axis-tilt failures | 2 | 2 | 1 |

The configured 10-percentage-point baseline regression guard stopped the run at 1,351,680 steps. It saved `results/rolling_ppo_tracking_20260911_145224/checkpoints/000001351680`. The panel-best pointer is `checkpoints/000000245760`. The stopped evaluation appears in fixed history but not the ordinary metrics history because the guard interrupts the run.

Against step zero, the stopped policy gains 13 successful episodes and loses 39. This is a sustained regression, not evidence of improvement hidden by changed evaluation snapshots. Different runs use different panels; do not compare this run's 84.8% baseline directly with a previous 92.2% result on 64 episodes.

## Which commands regress

| Command group | Episodes | Initial success | Best success | Stopped success |
|---|---:|---:|---:|---:|
| Low / straight | 16 | 15 | 15 | 14 |
| Medium / straight | 49 | 44 | 40 | 11 |
| High / straight | 31 | 8 | 13 | 15 |
| Low / left | 25 | 20 | 20 | 20 |
| Low / right | 25 | 22 | 22 | 22 |
| Medium / left | 23 | 23 | 23 | 23 |
| Medium / right | 29 | 29 | 29 | 29 |
| High / left | 21 | 21 | 21 | 21 |
| High / right | 37 | 35 | 35 | 36 |

Medium-straight failures change from two negative-Y and three positive-Y lateral failures to 38 positive-Y lateral failures. High-straight negative-Y failures decline from 23 to 13, with three new positive-Y failures. This pattern is consistent with a shift toward positive lateral motion that helps one command region while damaging another; episode summaries alone cannot establish the underlying action/phase mechanism.

At termination there are 56 failed episodes (55 lateral, one axis tilt) and nine surviving episodes below the five-turn success threshold. Collision and nonfinite failures remain zero. It is inaccurate to describe this as loss of rolling ability across all commands.

## Tracking and selection effects

On the same 187 episodes surviving the full horizon under both the initial and stopped policies:

- Per-step forward MAE: 0.103291 -> 0.104249 m/s.
- Mean absolute per-episode signed forward error: 0.025435 -> 0.025964 m/s.

The second metric averages signed error over each episode before taking its absolute value; it measures episode-average speed bias. It does not measure within-episode response speed and can hide compensating errors. Both metrics fail to show improvement here. Therefore changing the reporting metric alone would not establish a training gain.

On the 220 common survivors for step zero vs panel-best, per-step MAE changes 0.102238 -> 0.102974 m/s and episode-average absolute bias changes 0.024764 -> 0.026637 m/s.

Transition-weighted yaw target/actual rates:

- Left: initial +0.048951 / +0.008770; stopped +0.048951 / +0.015275 rad/s.
- Right: initial -0.048984 / -0.015431; stopped -0.049006 / -0.012141 rad/s.

Left response improves but remains far below its target; right response weakens. The slight change in weighted right target comes from episode duration changes. Turning success uses the survival/turn-count criterion and is not proof of correct turning-speed tracking.

## Optimizer and reward observations

Actual configuration: forward/yaw tracking weights 6/3, training snapshot probabilities low/medium/high 40/20/40%, straight/left/right 60/20/20%, DR 0, observation noise 1, anchor 0, discount 0.99. Medium-straight reset probability is 12%, versus 24% each for low-straight and high-straight. This can create competing generalization pressures, but this run simultaneously changed reward, sampling, and seeds, so it is not a causal ablation.

The first recorded interval KL is 0.007032; subsequent values are 0.002361–0.004308. Policy standard deviation stays near 0.0200. These diagnostics do not resemble the original catastrophic KL explosion. They also do not guarantee preservation of long-horizon behavior.

Adaptive learning rate rises from the requested initial 3e-6 to an interval mean of 6.46e-6, then reaches 1e-5 by step 245,760 and stays there. Thus this was not a constant-3e-6 run. Value loss ranges approximately 581.8–683.4 and ends at 670.0 in the available ordinary metrics; it does not establish critic convergence. Reward weights changed while restoring the previous critic, with no separate critic-only adaptation stage. That mismatch is a plausible contributor, not a proven cause.

Fixed-evaluation mean reward per active step drops 5.8873 -> 5.7268. Forward reward drops 3.7789 -> 3.7575 and lateral-drift reward drops 0.4528 -> 0.3815. Yaw-command reward increases slightly, 0.8341 -> 0.8468, despite aggregate yaw MAE worsening. The Gaussian reward and absolute-error metric weight errors differently. These results do not support a simple claim that fixed-evaluation return increased while success fell.

Current forward reward uses root world-X displacement divided by the control interval (20 ms), without a rolling-cycle average or projection onto horizontal heading. This is a task-definition concern, but not proven to explain this specific regression. The archive has episode aggregates, not the trajectories needed to quantify phase-dependent speed ripple or reconstruct filtered/heading-relative reward offline.

## Recommended next experiment

1. Retain the pre-v4 actor as baseline. Do not promote the stopped actor; the panel-best actor's net one-episode gain is insufficient evidence.
2. Define command speed explicitly as world-X or horizontal-heading-relative, and decide an averaging timescale before changing the reward. For stand-to-roll continuation, a heading-relative forward command and a short rolling-aware average are reasonable candidates; their latency must be checked in cloud trajectories.
3. Collect a small fixed trajectory panel covering all nine speed/turn groups, logging root XY position, rolling-axis heading, command, phase, and failure timing. Keep existing raw-speed metrics and add averaged-speed metrics so an apparent gain cannot come solely from metric replacement.
4. When reward definition changes, adapt the critic with the actor frozen before resuming actor updates. For an initial controlled run, cap actor learning rate at 3e-6 or use a constant smaller rate, keep the reset distribution fixed, and preserve per-group evaluation gates. Investigate middle-speed straight stability explicitly instead of selecting only by overall success.

These are proposed controlled experiments, not validated fixes or new training runs.
