# CEM steering authority: historical reproduction and current mapping

The user explicitly authorized local teacher simulation on 2026-09-12. Eleven CPU MuJoCo 3.9.0 rollouts were completed using `scripts/probe_cem_steering_comparison.py`. No policy training or controller changes were performed. Raw results: `results/cem_steering_comparison_20260912/summary.json` and individual case JSON files.

## Historical evidence is valid

The supplied `results/steering_authority_8d_rollingquad_abd10_pupper_cem/all_coherent_p0p5.json` records +0.0744338 rad/s with raw differential `[.5,.5,.5,-.5]`, gain .30, differential scale .25, the older Pupper three-stage CEM reference, rollingquad_abd10.xml, self collision enabled, and cg20 physics. The normalized actuator offset amplitude is .0375 (hip .030 rad, knee .045 rad). The current local reproduction gives +0.0743860 / -0.0737988 rad/s; zero differential gives +0.0001459 rad/s.

This demonstrates steering authority for the CEM reference plus differential joint offsets. It does not demonstrate that the unmodified symmetric reference accepts a yaw-rate command. The original command explicitly injected the differential offset. `steering_prior_3d` uses that same sign pattern; repository history places its introduction in commit 0855c4c on 2026-09-08, and the command-tracking document describes its original purpose as a guide for residual PPO to refine.

## Current teacher mapping differs

The current reference is `results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json`, with the no-self-collision model. Both distillation and PPO snapshot generation set the teacher reference gain to .15 and differential scale to .25. Note that the serialized direct-student task has differential scale null; the PPO script overrides it to .25 when constructing the snapshot teacher.

For command +.08 rad/s:

```
raw = clip(5 * .08, -.5, .5) = .4
normalized actuator offset = .4 * .15 * .25 = .015
hip target offset = .015 * .8 = .012 rad
knee target offset = .015 * 1.2 = .018 rad
```

This is 40% of the historical .0375 action offset, and the reference controller has also changed. It would be incorrect to infer proportional yaw-rate scaling without simulation.

## Completed CPU probes

| Reference/setup | Nominal speed command | Offset amplitude | Measured mean heading rate (rad/s) |
|---|---:|---:|---:|
| Historical / no differential | Historical scale 1 | 0 | +0.000146 |
| Historical / positive differential | Historical scale 1 | +.0375 | +0.074386 |
| Historical / negative differential | Historical scale 1 | -.0375 | -0.073799 |
| Current / zero | .60 m/s | 0 | -0.000244 |
| Current / +.08 command amplitude | .60 m/s | +.015 | +0.018385 |
| Current / -.08 command amplitude | .60 m/s | -.015 | -0.018666 |
| Current / historical offset amplitude | .60 m/s | +.0375 | +0.052761 |
| Current / zero | .80 m/s | 0 | -0.002417 |
| Current / +.08 command amplitude | .80 m/s | +.015 | +0.045497 |
| Current / -.08 command amplitude | .80 m/s | -.015 | -0.046818 |
| Current / historical offset amplitude | .80 m/s | +.0375 | +0.101054 |

All cases remain finite. Historical cases record zero self contact; current cases disable self collision, so their zero self-contact count does not establish collision-free motion if self collision were enabled. These probes do not apply the PPO failure/success thresholds.

Protocol: existing CPU reference evaluator, compact start, 10 s including startup, 50 Hz reporting, cg20 physics, front abduction -10 degrees / rear +10 degrees. Mean heading rate is unwrapped rolling-axis heading change divided by elapsed time. The current reference amplitude uses the existing forward-command lookup. Unlike MJX teacher execution, the CPU evaluator applies the constant differential during startup without the teacher's initial steering ramp. These are controlled authority probes, not exact replications of noisy MJX snapshot-takeover evaluation. Each setting has one deterministic rollout; robustness across snapshots and perturbations is not measured.

## Interpretation and next step

The earlier claim of approximately .074 rad/s steering authority is reproduced. The current command-to-offset mapping substantially under-delivers .08 rad/s in these probes, and its response depends on forward speed. At the same .0375 offset, the current setup produces .0528 rad/s at nominal .60 m/s and .1011 rad/s at nominal .80 m/s. Therefore restoring one global gain is not a validated solution across the speed range.

The teacher labels can encode a mismatch between requested yaw rate and actual motion even when the student copies actions accurately. Do not attribute weak student turning solely to lost imitation capacity. Calibrate steering against speed and both command signs using the current reference, then verify same-state rolling takeovers and command switches. A modest heading-rate feedback term is another candidate, but is not implemented or validated by this experiment. Existing policy checkpoints and teacher settings have not been changed.
