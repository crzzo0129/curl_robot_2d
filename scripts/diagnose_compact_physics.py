"""CPU-only primitive stand->compact diagnostics; no policy or state playback.

Run both an elevated fixed-base fixture and a free-base ground probe. The
fixture tests joint tracking only, never ground-transition feasibility.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.environment_3d import (
    model_path_3d, apply_physics_options_3d, validate_rolling_morphology_3d,
)
from curl_robot_2d_mjx.autonomous_startup_3d import model_fingerprint

LEGS = ('front_left', 'front_right', 'rear_left', 'rear_right')


def smootherstep(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10. - 15.*u + 6.*u*u)


def fold_progress(elapsed, delta, args):
    """Bound each staircase increment by max joint displacement, with a dwell."""
    if args.trajectory == 'smooth':
        duration = args.fold_seconds
        return float(smootherstep(elapsed / duration)), duration
    segments = max(1, math.ceil(float(np.max(np.abs(delta))) / args.increment_rad))
    period = args.move_seconds + args.dwell_seconds
    duration = segments * period
    if elapsed <= 0:
        return 0., duration
    if elapsed >= duration:
        return 1., duration
    segment = int(elapsed // period)
    local = elapsed - segment * period
    return float((segment + smootherstep(local / args.move_seconds)) / segments), duration


def fixed_fixture_xml(source, height):
    """Remove only root DoFs, keeping links, inertia, servos and collisions.

    This explicit fixed-base fixture supplies external support. It is not a
    weld enforced by overwriting the free joint after each physics step.
    """
    root = ET.parse(source).getroot()
    body = root.find('./worldbody/body[@name="torso"]')
    free = body.find('freejoint')
    if free is None or free.get('name') != 'root':
        raise ValueError('expected torso root freejoint')
    body.remove(free)
    stand = np.fromstring(root.find('./keyframe/key[@name="stand"]').get('qpos'), sep=' ')
    body.set('pos', f'{stand[0]} {stand[1]} {height}')
    body.set('quat', ' '.join(map(str, stand[3:7])))
    for key in root.findall('./keyframe/key'):
        for field, offset in (('qpos', 7), ('qvel', 6)):
            if field in key.attrib:
                key.set(field, ' '.join(key.get(field).split()[offset:]))
    # Preserve asset references when loading the diagnostic XML from a string.
    for asset in root.findall('./asset/*'):
        if asset.get('file'):
            asset.set('file', str((source.parent / asset.get('file')).resolve()))
    return ET.tostring(root, encoding='unicode')


def contact_metrics(mj, model, data, floor, feet, geom_names):
    foot_force = np.zeros(4)
    foot_slip = np.zeros(4)
    shell_force = other_force = 0.
    contacts = []
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = map(int, con.geom)
        force = np.zeros(6)
        mj.mj_contactForce(model, data, i, force)
        if force[0] <= 0 and con.dist >= 0:
            continue
        normal_force = max(0., float(force[0]))
        contacts.append((g1, g2, float(con.dist), normal_force,
                         float(np.linalg.norm(force[1:3])), *map(float, con.pos)))
        if floor not in (g1, g2):
            continue
        other = g2 if g1 == floor else g1
        if other in feet:
            leg = feet[other]
            foot_force[leg] += normal_force
            jac = np.zeros((3, model.nv))
            mj.mj_jac(model, data, jac, None, con.pos, int(model.geom_bodyid[other]))
            velocity = jac @ data.qvel
            normal = con.frame.reshape(3, 3)[0]
            foot_slip[leg] = max(foot_slip[leg], float(np.linalg.norm(
                velocity - normal * np.dot(normal, velocity))))
        elif '_shell_' in geom_names[other]:
            shell_force += normal_force
        else:
            other_force += normal_force
    return foot_force, foot_slip, shell_force, other_force, contacts


def sustained_triggers(flags, timers, dt, duration):
    for name, flag in flags.items():
        timers[name] = timers.get(name, 0.) + dt if flag else 0.
    return [name for name in flags if timers[name] + 1e-12 >= duration]


def json_write(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')


def run(mode, args):
    import mujoco as mj
    folder = args.out / mode
    folder.mkdir()
    source = model_path_3d('rollingquad_2_primitive')
    task = physics_profile_3d('cg20', Rolling3DConfig(geometry='rollingquad_2_primitive'))
    # Validate the untouched rolling model before constructing the fixture.
    model = mj.MjModel.from_xml_path(str(source))
    validate_rolling_morphology_3d(model, task.geometry)
    if mode == 'fixed':
        xml = fixed_fixture_xml(source, args.fixture_height_m)
        (folder / 'fixture.xml').write_text(xml, encoding='utf-8')
        model = mj.MjModel.from_xml_string(xml)
        task = replace(task, disable_root_damping=False)
    apply_physics_options_3d(model, task)
    dt, control_steps = model.opt.timestep, task.action_repeat
    torso, floor = model.body('torso').id, model.geom('floor').id
    names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    vadr = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    stand, compact = (model.key(pose).ctrl.copy() for pose in ('stand', 'compact'))
    if not np.all(model.actuator_forcelimited):
        raise ValueError('all servos must retain torque limits')
    for ctrl in (stand, compact):
        if np.any(ctrl < model.actuator_ctrlrange[:, 0]) or np.any(ctrl > model.actuator_ctrlrange[:, 1]):
            raise ValueError('keyframe targets outside actuator limits')
    limits = model.actuator_forcerange.copy()
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, model.key('stand').id)
    if mode == 'ground':
        data.qpos[2] += .005
    data.ctrl[:] = stand
    mj.mj_forward(model, data)
    geom_names = [model.geom(i).name or str(i) for i in range(model.ngeom)]
    feet = {model.geom(f'{leg}_foot_proxy').id: i for i, leg in enumerate(LEGS)}
    body_weight = float(np.sum(model.body_mass) * np.linalg.norm(model.opt.gravity))
    single_foot = args.trajectory == 'single-foot'
    controller = None
    if single_foot:
        fold_duration = args.single_foot_timeout
    else:
        _, fold_duration = fold_progress(0., compact - stand, args)
    total = args.settle_seconds + fold_duration + args.hold_seconds
    headers = ['time_s', 'progress', 'root_z_m', 'root_vz_m_s', 'tilt_rad',
               'max_tracking_error_rad', 'max_compact_error_rad', 'max_joint_speed_rad_s',
               'max_torque_nm', 'saturation_fraction', 'shell_normal_force_n',
               'other_floor_normal_force_n', 'total_floor_normal_force_n', 'max_penetration_m']
    headers += [f'{leg}_normal_force_n' for leg in LEGS]
    headers += [f'{leg}_contact_slip_m_s' for leg in LEGS]
    headers += [f'{leg}_foot_clearance_m' for leg in LEGS]
    headers += ['yaw_rad', 'yaw_rate_rad_s', 'support_margin_m', 'active_leg', 'completed_steps']
    records, states, speeds, ctrls, torques, contact_rows = [], [], [], [], [], []
    events, previous_pairs, timers = [], set(), {}
    progress, stop_reasons = 0., []
    sat_duration = np.zeros(model.nu)
    max_sat_duration = sat_duration.copy()
    force_buffer = np.zeros(6)
    csv_file = (folder / 'metrics.csv').open('w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(headers)
    print(f'[physics probe] mode={mode} trajectory={args.trajectory} duration={total:.2f}s', flush=True)
    try:
        for index in range(round(total / dt) + 1):
            # Record the solved state and contacts together. State is never
            # reassigned after reset; only ctrl changes at control boundaries.
            mj.mj_forward(model, data)
            finite = all(np.isfinite(z).all() for z in
                         (data.qpos, data.qvel, data.ctrl, data.actuator_force))
            foot_f, slip, shell_f, other_f, contacts = contact_metrics(
                mj, model, data, floor, feet, geom_names) if finite else (
                    np.zeros(4), np.zeros(4), 0., 0., [])
            pairs = {(c[0], c[1]) for c in contacts}
            for pair in sorted(pairs - previous_pairs):
                events.append({'time_s': float(data.time), 'event': 'contact_onset',
                               'geoms': [geom_names[g] for g in pair]})
            previous_pairs = pairs
            contact_rows.extend((float(data.time), *c) for c in contacts)
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, torso, force_buffer, 0)
            vz = float(force_buffer[5])
            tilt = float(np.arccos(np.clip(data.xmat[torso, 8], -1., 1.)))
            tracking = float(np.max(np.abs(data.qpos[qadr] - data.ctrl)))
            target_error = float(np.max(np.abs(data.qpos[qadr] - compact)))
            saturation = ((data.actuator_force <= .99*limits[:, 0]) |
                          (data.actuator_force >= .99*limits[:, 1]))
            if index:
                sat_duration = np.where(saturation, sat_duration + dt, 0.)
            max_sat_duration = np.maximum(max_sat_duration, sat_duration)
            penetration = max([0.] + [-c[2] for c in contacts])
            total_force = float(foot_f.sum() + shell_f + other_f)
            # Primitive feet are spheres; this is their signed floor clearance.
            clearance = [float(data.geom_xpos[g, 2] - model.geom_size[g, 0]
                               - data.geom_xpos[floor, 2]) for g in feet]
            rotation = data.xmat[torso].reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            yaw_rate = float(force_buffer[2])
            row = [float(data.time), progress, float(data.xpos[torso, 2]), vz, tilt,
                   tracking, target_error, float(np.max(np.abs(data.qvel[vadr]))),
                   float(np.max(np.abs(data.actuator_force))), float(saturation.mean()),
                   shell_f, other_f, total_force, penetration, *foot_f, *slip, *clearance,
                   yaw, yaw_rate, controller.support_margin if controller else 0.,
                   controller.active_leg if controller else -1, controller.completed if controller else 0]
            records.append(row)
            writer.writerow(row)
            states.append(data.qpos.copy()); speeds.append(data.qvel.copy())
            ctrls.append(data.ctrl.copy()); torques.append(data.actuator_force.copy())
            flags = {'upward_velocity': mode == 'ground' and vz > args.max_upward_speed,
                     'tilt': mode == 'ground' and tilt > args.max_tilt,
                     'contact_force': total_force > args.max_force_bodyweights * body_weight,
                     'penetration': penetration > args.max_penetration,
                     'yaw_rate': mode == 'ground' and abs(yaw_rate) > args.max_yaw_rate,
                     'yaw_angle': mode == 'ground' and abs(yaw) > args.max_yaw_angle}
            if single_foot and controller is not None:
                support = [i for i in range(4) if i != controller.active_leg]
                flags['support_slip'] = float(np.max(slip[support])) > args.max_support_slip
            stop_reasons = sustained_triggers(flags, timers, dt if index else 0., args.trigger_seconds)
            if not finite:
                stop_reasons.append('nonfinite')
            if mode == 'fixed' and any(floor in pair for pair in pairs):
                stop_reasons.append('fixture_touches_floor')
            if np.max(sat_duration) >= args.saturation_seconds:
                stop_reasons.append('sustained_torque_saturation')
            if any(w.number for w in data.warning):
                stop_reasons.append('mujoco_warning')
            if controller is not None and controller.failed:
                stop_reasons.append(controller.reason)
            if stop_reasons:
                events.append({'time_s': float(data.time), 'event': 'stop', 'reasons': stop_reasons,
                               'progress': progress, 'phase': 'settle' if data.time < args.settle_seconds else 'fold_or_hold'})
                print(f'[physics probe] STOP t={data.time:.3f}s progress={progress:.3f}: {stop_reasons}', flush=True)
                break
            if index == round(total / dt):
                if single_foot and (controller is None or not controller.done):
                    stop_reasons.append('single_foot_timeout')
                break
            if controller is not None and controller.done:
                break
            if index % control_steps == 0:
                if single_foot:
                    if data.time + 1e-9 >= args.settle_seconds:
                        if controller is None:
                            from curl_robot_2d_mjx.single_foot_compact_probe import SingleFootCompactProbe, SingleFootConfig
                            controller = SingleFootCompactProbe(model, SingleFootConfig(
                                step_m=args.foot_step_m, rounds=args.foot_rounds))
                            controller.reset(data)
                        data.ctrl[:] = controller.step(data, task.control_timestep)
                        progress = controller.completed / (4*args.foot_rounds)
                else:
                    progress, _ = fold_progress(float(data.time) - args.settle_seconds, compact - stand, args)
                    data.ctrl[:] = stand + progress * (compact - stand)
            mj.mj_step(model, data)
    finally:
        csv_file.close()
    rows = np.asarray(records)
    np.savez_compressed(folder / 'trajectory.npz', time=rows[:, 0], qpos=np.asarray(states),
        qvel=np.asarray(speeds), ctrl=np.asarray(ctrls), actuator_force=np.asarray(torques),
        joint_names=np.asarray(names), joint_qpos_indices=qadr, joint_qvel_indices=vadr,
        stand_ctrl=stand, compact_ctrl=compact, metrics=rows, metric_names=np.asarray(headers),
        contacts=np.asarray(contact_rows).reshape(-1, 9), geom_names=np.asarray(geom_names),
        contact_columns=np.asarray(['time_s', 'geom1', 'geom2', 'distance_m', 'normal_force_n',
                                    'tangent_force_n', 'x_m', 'y_m', 'z_m']))
    json_write(folder / 'events.json', events)
    if controller is not None:
        json_write(folder / 'single_foot_events.json', controller.events)
    report = {'mode': mode, 'scope': 'joint tracking fixture only' if mode == 'fixed' else 'ground diagnostic path only',
        'mujoco_version': mj.__version__, 'model_path': str(source), **model_fingerprint(source),
        'physics': asdict(task), 'options': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'state_overwritten_after_reset': False, 'training_or_policy': False,
        'stop_reasons': stop_reasons, 'completed_command_schedule': not stop_reasons and progress >= 1.,
        'ground_transition_certified': False, 'final_time_s': float(data.time), 'final_progress': progress,
        'final_max_compact_error_rad': target_error, 'final_max_tracking_error_rad': tracking,
        'peak_upward_speed_m_s': float(rows[:, 3].max()), 'peak_floor_force_n': float(rows[:, 12].max()),
        'peak_penetration_m': float(rows[:, 13].max()), 'peak_torque_nm': float(rows[:, 8].max()),
        'body_weight_n': body_weight, 'joint_order': names,
        'longest_saturation_s_per_joint': max_sat_duration.tolist(),
        'final_joint_position_rad': data.qpos[qadr].tolist(),
        'final_ctrl_rad': data.ctrl.tolist(), 'compact_ctrl_rad': compact.tolist(),
        'contact_force_note': 'normal magnitudes summed by foot/shell/other, not world vertical components',
        'fixture_support_force_measured': False}
    if single_foot:
        report.update(completed_steps=controller.completed if controller else 0,
                      planned_steps=4*args.foot_rounds,
                      progress_meaning='confirmed single-foot steps / planned steps, NOT compact attainment',
                      final_controller_phase=controller.phase if controller else 'not_started',
                      controller_config=asdict(controller.config) if controller else None)
    report.update(peak_abs_yaw_rad=float(np.max(np.abs(rows[:, headers.index('yaw_rad')]))),
                  peak_abs_yaw_rate_rad_s=float(np.max(np.abs(rows[:, headers.index('yaw_rate_rad_s')]))))
    json_write(folder / 'summary.json', report)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('both', 'fixed', 'ground'), default='both')
    p.add_argument('--trajectory', choices=('incremental', 'smooth', 'single-foot'), default='incremental')
    p.add_argument('--foot-rounds', type=int, default=1)
    for name, default in {'settle_seconds': 1., 'hold_seconds': 1., 'fold_seconds': 5.,
            'increment_rad': .05, 'move_seconds': .25, 'dwell_seconds': .25,
            'fixture_height_m': .8, 'max_upward_speed': .15, 'max_tilt': .35,
            'max_force_bodyweights': 3., 'max_penetration': .004,
            'trigger_seconds': .01, 'saturation_seconds': .2,
            'foot_step_m': .008, 'single_foot_timeout': 30.,
            'max_yaw_rate': .5, 'max_yaw_angle': .15, 'max_support_slip': .05}.items():
        p.add_argument('--' + name.replace('_', '-'), type=float, default=default)
    args = p.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
            p.error(f'{name} must be finite and positive')
    if args.out.exists():
        p.error('use a new output directory')
    if args.foot_rounds < 1:
        p.error('foot-rounds must be positive')
    if args.trajectory == 'single-foot' and args.mode != 'ground':
        p.error('single-foot requires --mode ground')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True)
    modes = ('fixed', 'ground') if args.mode == 'both' else (args.mode,)
    reports = [run(mode, args) for mode in modes]
    json_write(args.out / 'summary.json', {'runs': reports})
    print(f'[physics probe] saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
