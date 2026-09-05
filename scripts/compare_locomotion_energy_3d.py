"""Paired CPU mechanical-work benchmark on the same full CAD robot.

Run from curl_robot_2d: python -m scripts.compare_locomotion_energy_3d.
Absolute work includes braking work; neither metric estimates battery energy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from scripts import evaluate_3d_symmetric_cem_reference as ref

ROOT = Path(__file__).resolve().parents[1]
LEGS = ('front_left', 'front_right', 'rear_left', 'rear_right')
JOINTS = tuple(f'{leg}_{joint}' for leg in LEGS
               for joint in ('hip_abduction', 'hip', 'knee'))


class WalkingPolicy:
    def __init__(self, path, model):
        p = json.loads(path.read_text())
        self.scale = np.asarray(p['action_scale'])
        self.pose = np.asarray(p['default_joint_pos'])
        self.low = np.asarray(p['joint_lower_limits'])
        self.high = np.asarray(p['joint_upper_limits'])
        self.layers = [(np.asarray(l['weights'][0]), np.asarray(l['weights'][1]),
                        l['activation']) for l in p['layers']]
        self.hist = np.zeros((p['observation_history'], 36))
        assert p['in_shape'][1] == self.hist.size
        self.hist[:, 5] = -1
        self.hist[:, 11] = 1
        self.last = np.zeros(12)
        self.qids = [model.joint(n).qposadr[0] for n in JOINTS]
        self.aids = [model.actuator(n + '_servo').id for n in JOINTS]

    def control(self, data, speed):
        rotation = np.zeros(9)
        mj.mju_quat2Mat(rotation, data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ np.array([0., 0., -1.])
        frame = np.concatenate((data.sensor('base_angular_velocity').data,
            gravity, [speed, 0, 0], [0, 0, 1],
            data.qpos[self.qids] - self.pose, self.last))
        self.hist[1:] = self.hist[:-1].copy()
        self.hist[0] = frame
        x = self.hist.reshape(-1)
        for w, b, activation in self.layers:
            x = x @ w + b
            if activation == 'elu':
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
            elif activation == 'tanh':
                x = np.tanh(x)
            elif activation not in ('linear', ''):
                raise ValueError(activation)
        self.last = np.clip(x, -1, 1)
        data.ctrl[self.aids] = np.clip(self.pose + self.scale * self.last,
                                       self.low, self.high)


def summarize(rows, start, dt, mass, gravity):
    selected = [r for r in rows if r['time_s'] >= start - 1e-9]
    if not selected:
        raise ValueError('Empty evaluation interval')
    duration = len(selected) * dt
    dx = sum(r['dx_m'] for r in selected)
    dy = sum(r['dy_m'] for r in selected)
    distance = abs(dx)
    positive = sum(r['positive_power_w'] for r in selected) * dt
    negative = sum(r['negative_power_w'] for r in selected) * dt
    return dict(duration_s=duration, forward_displacement_m=dx,
        lateral_displacement_m=dy, actual_speed_m_s=dx / duration,
        positive_work_j=positive, negative_work_j=negative,
        absolute_work_j=positive + negative,
        positive_power_w=positive / duration,
        absolute_power_w=(positive + negative) / duration,
        positive_j_per_m=positive / distance if distance > .01 else None,
        cot_positive=positive / (mass * gravity * distance) if distance > .01 else None,
        cot_absolute=(positive + negative) / (mass * gravity * distance) if distance > .01 else None,
        min_height_m=min(r['height_m'] for r in selected),
        max_tilt_deg=max(r['tilt_deg'] for r in selected),
        nonfoot_ground_fraction=np.mean([r['nonfoot_ground'] for r in selected]).item(),
        saturation_fraction=np.mean([r['saturation_fraction'] for r in selected]).item())


def run_case(args, mode, speed, label):
    model = mj.MjModel.from_xml_path(str(args.xml))
    # Both modes retain exactly the same geometry, contacts, gains and solver.
    model.opt.timestep = args.dt
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    model.opt.impratio = 10
    model.opt.cone = mj.mjtCone.mjCONE_PYRAMIDAL
    model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_EULERDAMP
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, model.key('stand' if mode == 'walk' else 'compact').id)
    policy = WalkingPolicy(args.policy, model) if mode == 'walk' else None
    if policy:
        data.qpos[policy.qids] = policy.pose
        data.qpos[2] += .0005  # deployment training reset clearance
    mj.mj_forward(model, data)
    config = ref.load_cem_reference(args.controller)
    ref.activate_planar_geometry(ref.PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
    aids = [model.actuator(n + '_servo').id for n in ref.JOINT_NAMES_3D]
    phase = body_phase = 0.
    dt = model.opt.timestep
    repeat = round(.02 / dt)
    if repeat < 1 or not np.isclose(repeat * dt, .02):
        raise ValueError('dt must divide 0.02 s exactly')
    rows = []
    power_error = 0.
    for k in range(round(args.duration / dt)):
        if policy:
            if k % repeat == 0:
                policy.control(data, speed)
        else:
            phase = float(ref.advance_oscillator(np, body_phase, phase, dt, config))
            u = min(k * dt / .25, 1.)
            target = ref.scaled_planar_target(ref.planar_cem_target(phase, config), u*u*(3-2*u))
            data.ctrl[aids] = np.clip(ref.map_planar_to_curl_3d_targets(target),
                model.actuator_ctrlrange[aids, 0], model.actuator_ctrlrange[aids, 1])
        xy = data.qpos[:2].copy()
        # Force and transmission velocity are evaluated at the same state.
        mj.mj_forward(model, data)
        power = data.actuator_force * data.actuator_velocity
        power_error = max(power_error, abs(float(power.sum() - data.qfrc_actuator @ data.qvel)))
        positive = float(np.maximum(power, 0).sum())
        negative = float(np.maximum(-power, 0).sum())
        saturation = float(np.mean(np.abs(data.actuator_force) >= .99 * np.max(np.abs(model.actuator_forcerange), axis=1)))
        nonfoot = False
        for contact in data.contact:
            names = [model.geom(int(i)).name or '' for i in (contact.geom1, contact.geom2)]
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 in bodies:
                robot_name = names[1] if bodies[0] == 0 else names[0]
                if not any(s in robot_name for s in ('foot', 'shank')):
                    nonfoot = True
        mj.mj_step(model, data)
        body_phase += float(data.qvel[4]) * dt
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f'{label}: nonfinite state at step {k}')
        rotation = np.zeros(9)
        mj.mju_quat2Mat(rotation, data.qpos[3:7])
        tilt = float(np.degrees(np.arccos(np.clip(rotation[8], -1, 1))))
        rows.append(dict(time_s=k*dt, dx_m=float(data.qpos[0]-xy[0]),
            dy_m=float(data.qpos[1]-xy[1]), x_m=float(data.qpos[0]),
            y_m=float(data.qpos[1]), height_m=float(data.qpos[2]), tilt_deg=tilt,
            positive_power_w=positive, negative_power_w=negative,
            nonfoot_ground=int(nonfoot), saturation_fraction=saturation))
    with (args.out / (label + '.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    mass = float(model.body_mass.sum())
    gravity = float(np.linalg.norm(model.opt.gravity))
    windows = {name: summarize(rows, start, dt, mass, gravity)
               for name, start in [('full', 0.), ('post_startup', args.warmup)]}
    safe = (windows['post_startup']['min_height_m'] > .079 and
            windows['post_startup']['max_tilt_deg'] < 45) if policy else None
    result = dict(label=label, mode=mode, command_m_s=speed, mass_kg=mass,
        windows=windows, walking_post_startup_posture_ok=safe,
        power_identity_max_error_w=power_error, rolling_turns=body_phase/(2*np.pi) if not policy else None)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml', type=Path, default=ref.DEFAULT_XML_PATH)
    p.add_argument('--controller', type=Path, default=ref.DEFAULT_CONTROLLER_PATH)
    p.add_argument('--policy', type=Path, default=ROOT.parent / 'rollingquad_2_deploy_robust_dr_policy_stable.json')
    p.add_argument('--duration', type=float, default=10)
    p.add_argument('--warmup', type=float, default=2)
    p.add_argument('--dt', type=float, default=.002)
    p.add_argument('--walk-speeds', type=float, nargs='+', default=[.3, .5, .7])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.dt <= 0 or not np.isfinite(args.dt):
        p.error('dt must be positive and finite')
    if not 0 <= args.warmup < args.duration:
        p.error('Require 0 <= warmup < duration')
    if args.out.exists() and any(args.out.iterdir()):
        p.error('Output directory must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(hypothesis='Current rolling reference has lower positive mechanical work per forward metre than walking at similar actual speed.',
        pass_gate='Finite completed rollouts; walking upright; actual speed mismatch <= 10%; lower rolling positive J/m.',
        status='RUNNING', mujoco_version=mj.__version__,
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                [('model', args.xml), ('policy', args.policy), ('controller', args.controller), ('script', Path(__file__))]})
    (args.out/'manifest.json').write_text(json.dumps(metadata, indent=2))
    results = [run_case(args, 'roll', None, 'roll')]
    for i, speed in enumerate(args.walk_speeds):
        results.append(run_case(args, 'walk', speed, f'walk_{i}_{speed:.3f}'))
    metadata['status'] = 'COMPLETED'
    (args.out/'manifest.json').write_text(json.dumps(metadata, indent=2))
    (args.out/'summary.json').write_text(json.dumps(results, indent=2))
    roll_speed = abs(results[0]['windows']['post_startup']['actual_speed_m_s'])
    lines = ['# 三维滚动与行走机械能耗对比', '',
        '同一完整 CAD 模型、伺服器和地面；50 Hz 行走控制。所有结果为确定性单次仿真。', '',
        '下表统计启动后窗口，不能自动视为已达到稳态。正功为各执行器正功率积分，绝对功包含制动负功的绝对值；均不等于电池能耗。', '',
        '| 模式/指令 | 实际速度 m/s | 正功率 W | 正功 J/m | 正功 CoT | 绝对功 CoT |',
        '|---|---:|---:|---:|---:|---:|']
    for result in results:
        w = result['windows']['post_startup']
        fmt = lambda x: 'N/A' if x is None else f'{x:.4f}'
        lines.append('| ' + result['label'] + ' | ' + ' | '.join(fmt(w[k]) for k in
            ('actual_speed_m_s', 'positive_power_w', 'positive_j_per_m', 'cot_positive', 'cot_absolute')) + ' |')
    candidates = [r for r in results[1:] if r['walking_post_startup_posture_ok']]
    lines += ['', '## 比较有效性', '',
        '行走姿态检查：启动后高度 > 0.079 m、最大倾角 < 45°。额外接触比例见 summary.json（foot/shank 接触作为足腿支撑统计，不代表仅足底接触）。',
        '使用前向净位移的绝对值归一化；横向漂移单独记录。未计入行走与滚动之间的切换开销。']
    if candidates and roll_speed > .01:
        best = min(candidates, key=lambda r: abs(abs(r['windows']['post_startup']['actual_speed_m_s'])-roll_speed))
        mismatch = abs(abs(best['windows']['post_startup']['actual_speed_m_s'])-roll_speed)/roll_speed
        lines += ['', f'最接近滚动速度的有效行走工况：{best["label"]}；实际速度差 {mismatch:.1%}。',
            '速度差超过 10% 时，不据此宣称同速节能。']
    else:
        lines += ['', '没有可用于同速比较的有效行走工况，或滚动速度过低。']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
