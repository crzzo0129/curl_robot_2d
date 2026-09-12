"""CPU calibration of the current CEM's signed, speed-dependent steering.

Uses the production normalized reference/target functions at every 1 ms physics
step. Warm up straight, then hold steering for ten seconds. No neural policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS as GEOMETRY
from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceGeometry, advance_oscillator, load_cem_reference, reference_action,
)
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.steering_calibration import text_sha256
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d, disable_rollingquad_self_collision_3d,
    duplicate_planar_action_3d, forward_command_to_target_scale_3d,
    model_path_3d, reference_startup_scale_3d, rolling_target_ctrl_3d,
)

CONTROLLER = Path("results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json")
PATTERN = np.array([1, 1, -1, -1, 1, -1, -1, 1])
SPEEDS = np.linspace(.4111, .811, 7)
TASK = physics_profile_3d("cg20", replace(
    Rolling3DConfig(), geometry="rollingquad_2_abd10_no_self_collision",
    self_collision_enabled=False, reset_joint_noise_rad=0., reset_velocity_noise=0.,
    disable_root_damping=True,
))


def rollout(job):
    speed, amplitude, warmup, seed, noise, target = job[:6]
    schedule = job[6] if len(job) > 6 else None
    duration = len(schedule)*4. if schedule else 10.
    control_steps = round(duration/.02)
    model = mujoco.MjModel.from_xml_path(str(model_path_3d(TASK.geometry)))
    apply_physics_options_3d(model, TASK)
    disable_rollingquad_self_collision_3d(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
    ids = np.array([model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D])
    joints = np.array([model.joint(name).id for name in JOINT_NAMES_3D])
    qids = model.jnt_qposadr[joints]
    rng = np.random.default_rng(seed)
    data.qpos[qids] += rng.uniform(-noise, noise, len(qids))
    data.qvel[6:] += rng.uniform(-noise, noise, len(data.qvel)-6)
    compact = model.key_ctrl[model.key("compact").id].copy()
    data.ctrl[:] = compact
    low = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[ids, 0])
    high = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[ids, 1])
    reference = load_cem_reference(CONTROLLER, minimum_residual_gain=.15)
    planar = np.array([GEOMETRY.compact_hip_angle, GEOMETRY.compact_knee_angle]*2)
    planar_low = np.array([GEOMETRY.hip.shell_compatible_range[0], GEOMETRY.knee.shell_compatible_range[0]]*2)
    planar_high = np.array([GEOMETRY.hip.shell_compatible_range[1], GEOMETRY.knee.shell_compatible_range[1]]*2)
    scales = np.asarray(TASK.action_scales)
    geometry = CEMReferenceGeometry(torso_length_m=GEOMETRY.torso_length,
        link_length_m=GEOMETRY.edge_length, foot_diameter_m=2*GEOMETRY.foot_radius,
        upper_link_length_m=GEOMETRY.upper_length, lower_link_length_m=GEOMETRY.lower_length)
    phase = spin = 0.
    scale = float(forward_command_to_target_scale_3d(np, speed))
    warmup_steps = round(warmup/.02)
    rows = []
    saturation = 0
    for step in range(warmup_steps + control_steps + 1):
        mujoco.mj_forward(model, data)
        axis = data.xmat[model.body("torso").id].reshape(3, 3)[:, 1]
        if step >= warmup_steps:
            rows.append([data.time-warmup, *data.qpos[:3],
                         np.arctan2(-axis[0], axis[1]),
                         np.arcsin(np.clip(axis[2], -1., 1.)),
                         np.arccos(np.clip(abs(axis[1]), 0., 1.))])
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all():
            break
        if step == warmup_steps + control_steps:
            break
        active_amplitude = amplitude
        if schedule and step >= warmup_steps:
            active_amplitude = schedule[min((step-warmup_steps)//200, len(schedule)-1)][1]
        for _ in range(20):
            phase = float(advance_oscillator(np, spin, phase, .001, reference))
            act = reference_action(np, phase, reference, compact_ctrl=planar,
                action_scales=np.array([.8, 1.2]*2), joint_low=planar_low,
                joint_high=planar_high, geometry=geometry)
            ramp_scale = reference_startup_scale_3d(np, data.time, TASK, target_scale=scale)
            base = np.clip(ramp_scale * duplicate_planar_action_3d(np, act), -1., 1.)
            raw = base + (active_amplitude * PATTERN if step >= warmup_steps else 0.)
            saturation += int(np.any(np.abs(raw) > 1.)) if step >= warmup_steps else 0
            command = np.clip(raw, -1., 1.)
            data.ctrl[:] = rolling_target_ctrl_3d(np, compact, ids, command, scales, low, high)
            mujoco.mj_step(model, data)
            spin += data.qvel[4] * .001
    values = np.asarray(rows)
    finite = bool(np.isfinite(values).all() and len(values) == control_steps+1)
    if not finite:
        return dict(speed=speed, amplitude=amplitude, warmup=warmup, seed=seed,
                    noise=noise, target=target, stable=False, nonfinite=True)
    heading = np.unwrap(values[:, 4])
    rate_full = (heading[-1]-heading[0])/duration
    rate_tail = (heading[-1]-heading[100])/(duration-2.)
    displacement = np.diff(values[:, 1:3], axis=0)/.02
    heading_mid = (heading[1:]+heading[:-1])/2
    local_speed = displacement[:, 0]*np.cos(heading_mid)+displacement[:, 1]*np.sin(heading_mid)
    lateral_speed = -displacement[:, 0]*np.sin(heading_mid)+displacement[:, 1]*np.cos(heading_mid)
    stable = bool(values[:, 3].min() > .025 and values[:, 3].max() < .8
                  and np.abs(values[:, 5]).max() < .5 and local_speed[100:].mean() > .15)
    result = dict(speed=float(speed), amplitude=float(amplitude), warmup=float(warmup),
        seed=int(seed), noise=float(noise), target=target, stable=stable, nonfinite=False,
        rate_full=float(rate_full), rate_tail=float(rate_tail),
        forward_tail=float(local_speed[100:].mean()), lateral_tail=float(lateral_speed[100:].mean()),
        world_vx_full=float(displacement[:, 0].mean()),
        axis_elevation_max=float(np.abs(values[:, 5]).max()),
        legacy_axis_tilt_max=float(values[:, 6].max()),
        legacy_axis_tilt_over_limit_fraction=float(np.mean(values[:, 6] > .5)),
        root_z_min=float(values[:, 3].min()), root_z_max=float(values[:, 3].max()),
        saturation_fraction=saturation/(control_steps*20))
    if schedule:
        result['segments'] = [dict(target=cmd, amplitude=a,
            rate_full=float((heading[(i+1)*200]-heading[i*200])/4.),
            rate_after_1s=float((heading[(i+1)*200]-heading[i*200+50])/3.))
            for i, (cmd, a) in enumerate(schedule)]
    return result


def run_jobs(jobs, path, workers):
    results = json.loads(path.read_text()) if path.exists() else []
    completed = {(r['speed'], r['amplitude'], r['warmup'], r['seed'], r['noise']) for r in results}
    jobs = [j for j in jobs if tuple(j[:5]) not in completed]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(rollout, job) for job in jobs]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{path.stem} {len(results)}: v={r['speed']:.4f} a={r['amplitude']:+.5f} "
                  f"yaw={r.get('rate_tail', float('nan')):+.4f} stable={r['stable']}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/cem_steering_calibration_20260912'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--stage', choices=['scan', 'fit', 'validate', 'baseline', 'switch', 'finalize'], default='scan')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'scan':
        amplitudes = np.array([-.08, -.06, -.045, -.03, -.02, -.01, 0., .01, .02, .03, .045, .06, .08])
        metadata = dict(controller=str(CONTROLLER), controller_sha256=hashlib.sha256(CONTROLLER.read_bytes()).hexdigest(),
            controller_text_sha256=text_sha256(CONTROLLER),
            model=str(model_path_3d(TASK.geometry)), model_sha256=hashlib.sha256(model_path_3d(TASK.geometry).read_bytes()).hexdigest(),
            model_text_sha256=text_sha256(model_path_3d(TASK.geometry)),
            task=asdict(TASK), mujoco_version=mujoco.__version__,
            protocol='3 s straight warmup then 10 s constant offset; fit heading change over seconds 2..10 after takeover; no PPO or MJX execution',
            stability='finite, root z in (.025,.8), axis elevation below .5 rad, mean heading-relative forward speed >.15 m/s; self collision disabled; legacy world-axis tilt recorded separately')
        (args.out/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        run_jobs([(float(v), float(a), 3., 0, 0., None) for v in SPEEDS for a in amplitudes], args.out/'scan.json', args.workers)
    elif args.stage == 'fit':
        from curl_robot_2d_mjx.steering_calibration import fit_calibration
        payload = fit_calibration(json.loads((args.out/'scan.json').read_text()), json.loads((args.out/'provenance.json').read_text()))
        (args.out/'calibration.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(json.dumps(payload, indent=2))
    elif args.stage in ('validate', 'baseline', 'switch'):
        from curl_robot_2d_mjx.steering_calibration import calibrated_steering_amplitude
        table = json.loads((args.out/'calibration.json').read_text())
        jobs = []
        # Unseen speeds/targets, different takeover phases, and small reset noise.
        if args.stage == 'switch':
            for v in (.46, .64, .76, .811):
                schedule = [(cmd, float(calibrated_steering_amplitude(np, table, v, cmd)))
                            for cmd in (.08, 0., -.08, .04, -.04)]
                jobs.append((v, 0., 4., 83, .005, None, schedule))
            run_jobs(jobs, args.out/'switch.json', args.workers)
            return
        for v in (.4111, .46, .52, .58, .64, .70, .76, .811):
            for target in (-.08, -.05, -.02, 0., .02, .05, .08):
                a = (float(np.clip(5*target, -.5, .5)*.15*.25) if args.stage == 'baseline'
                     else float(calibrated_steering_amplitude(np, table, v, target)))
                jobs.append((v, a, 2.4, 41, .005, target))
        run_jobs(jobs, args.out/('baseline.json' if args.stage == 'baseline' else 'validation.json'), args.workers)
    else:
        table = json.loads((args.out/'calibration.json').read_text())
        records = json.loads((args.out/'validation.json').read_text())
        baseline = json.loads((args.out/'baseline.json').read_text())
        switches = json.loads((args.out/'switch.json').read_text())
        errors = np.array([r['rate_tail']-r['target'] for r in records])
        old_errors = np.array([r['rate_tail']-r['target'] for r in baseline])
        switch_errors = np.array([r['rate_after_1s']-r['target'] for case in switches for r in case['segments']])
        passed = (len(records) == len(baseline) == 56 and len(switches) == 4
                  and all(r['stable'] for r in records+switches)
                  and np.abs(errors).mean() <= .005 and np.abs(errors).max() <= .015
                  and np.abs(switch_errors).max() <= .025)
        summary = dict(passed=bool(passed), episodes=len(records), switching_episodes=len(switches),
            rate_mae_rad_s=float(np.abs(errors).mean()), max_rate_error_rad_s=float(np.abs(errors).max()),
            baseline_rate_mae_rad_s=float(np.abs(old_errors).mean()),
            baseline_max_rate_error_rad_s=float(np.abs(old_errors).max()),
            switch_rate_mae_rad_s=float(np.abs(switch_errors).mean()),
            switch_max_rate_error_rad_s=float(np.abs(switch_errors).max()),
            max_axis_elevation_rad=max(r['axis_elevation_max'] for r in records+switches),
            stable_episodes=sum(r['stable'] for r in records+switches),
            gate='56 constant cases: mean abs settled yaw error <=.005, max <=.015 rad/s; 4 switching cases: max segment yaw error after first 1 s <=.025 rad/s; all mechanically stable',
            scope='CPU MuJoCo, flat/no self collision; bounded reset noise and phase variations only, not cloud MJX or hardware validation')
        table['validation'] = summary
        table['validation_status'] = 'passed_cpu_holdout' if passed else 'failed_cpu_holdout'
        (args.out/'calibration.json').write_text(json.dumps(table, indent=2), encoding='utf-8')
        (args.out/'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        if not passed:
            raise SystemExit('Calibration did not pass its holdout gate')


if __name__ == '__main__':
    main()
