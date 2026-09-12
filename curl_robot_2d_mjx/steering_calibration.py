"""Validated speed/yaw lookup returning effective normalized steering offsets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def text_sha256(path):
    """Ignore Git's Windows/Linux line-ending conversion, not content changes."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def fit_calibration(records, provenance):
    speeds = sorted({r['speed'] for r in records})
    commands = np.linspace(-.08, .08, 9)
    table, coverage = [], []
    for speed in speeds:
        rows = sorted((r for r in records if r['speed'] == speed), key=lambda r: r['amplitude'])
        center = next(i for i, r in enumerate(rows) if r['amplitude'] == 0.)
        lo = hi = center
        if not rows[center]['stable']:
            raise ValueError(f'Unstable straight reference at speed {speed}')
        # Keep the contiguous stable, strictly monotone branch containing zero.
        while lo > 0 and rows[lo-1]['stable'] and rows[lo-1]['rate_tail'] < rows[lo]['rate_tail']:
            lo -= 1
        while hi+1 < len(rows) and rows[hi+1]['stable'] and rows[hi+1]['rate_tail'] > rows[hi]['rate_tail']:
            hi += 1
        branch = rows[lo:hi+1]
        rates = [r['rate_tail'] for r in branch]
        offsets = [r['amplitude'] for r in branch]
        if rates[0] > commands[0] or rates[-1] < commands[-1]:
            raise ValueError(f'Insufficient measured steering range at speed {speed}: {rates[0]}..{rates[-1]}')
        table.append(np.interp(commands, rates, offsets).tolist())
        coverage.append(dict(speed=speed, minimum_rate=rates[0], maximum_rate=rates[-1]))
    return dict(schema_version=1, output_units='effective_normalized_action_offset',
        speeds_m_s=speeds, yaw_commands_rad_s=commands.tolist(), offsets=table,
        coverage=coverage, provenance=provenance, validation_status='pending',
        note='Bilinear inverse response table; includes small zero-command trim. No residual gain or differential scale is applied after lookup. Queries clamp to the measured command domain; task bounds must be checked before use.')


def calibrated_steering_amplitude(xp, table, forward_command, yaw_command):
    speeds = xp.asarray(table['speeds_m_s'])
    commands = xp.asarray(table['yaw_commands_rad_s'])
    offsets = xp.asarray(table['offsets'])
    speed, yaw = xp.broadcast_arrays(xp.asarray(forward_command), xp.asarray(yaw_command))
    speed = xp.clip(speed, speeds[0], speeds[-1])
    yaw = xp.clip(yaw, commands[0], commands[-1])
    i = xp.clip(xp.searchsorted(speeds, speed, side='right')-1, 0, len(table['speeds_m_s'])-2)
    j = xp.clip(xp.searchsorted(commands, yaw, side='right')-1, 0, len(table['yaw_commands_rad_s'])-2)
    sv = (speed-speeds[i])/(speeds[i+1]-speeds[i])
    sy = (yaw-commands[j])/(commands[j+1]-commands[j])
    lower = offsets[i, j]*(1-sy) + offsets[i, j+1]*sy
    upper = offsets[i+1, j]*(1-sy) + offsets[i+1, j+1]*sy
    return lower*(1-sv) + upper*sv


def calibrated_steering_prior(xp, table, forward_command, yaw_command):
    amplitude = calibrated_steering_amplitude(xp, table, forward_command, yaw_command)
    return amplitude[..., None] * xp.asarray([1., 1., -1., -1., 1., -1., -1., 1.])


def load_steering_calibration(path, *, task, reference, model_path):
    table = json.loads(Path(path).read_text(encoding='utf-8'))
    if table.get('schema_version') != 1 or table.get('validation_status') != 'passed_cpu_holdout':
        raise ValueError('Steering calibration has not passed CPU holdout validation')
    speeds, commands, offsets = [np.asarray(table[k]) for k in ('speeds_m_s', 'yaw_commands_rad_s', 'offsets')]
    if (len(speeds) < 2 or len(commands) < 2 or offsets.shape != (len(speeds), len(commands))
        or not all(np.isfinite(a).all() for a in (speeds, commands, offsets))
        or not (np.diff(speeds) > 0).all() or not (np.diff(commands) > 0).all()
        or not (np.diff(offsets, axis=1) >= 0).all() or np.max(np.abs(offsets)) > .08):
        raise ValueError('Invalid steering calibration grid')
    provenance = table['provenance']
    for source, expected in ((Path(reference.source), provenance['controller_text_sha256']),
                             (Path(model_path), provenance['model_text_sha256'])):
        if text_sha256(source) != expected:
            raise ValueError(f'Steering calibration does not match {source}')
    from curl_robot_2d_mjx.cem_reference import load_cem_reference
    original_reference = load_cem_reference(Path(reference.source))
    for key in ('coefficients', 'oscillator_rate_rad_s', 'oscillator_coupling_per_s',
                'knee_bias_rad', 'minimum_foot_surface_gap_m', 'foot_gap_tracking_margin_m'):
        if getattr(reference, key) != getattr(original_reference, key):
            raise ValueError(f'Steering calibration reference override: {key}')
    if reference.reference_weight != 1.:
        raise ValueError('Steering calibration requires full CEM reference weight')
    for key in ('geometry', 'physics_profile', 'physics_timestep', 'solver_name', 'solver_iterations',
                'solver_ls_iterations', 'integrator_name', 'cone_name', 'jacobian_name',
                'self_collision_enabled', 'geom_friction_scale', 'floor_friction_scale',
                'floor_contact_friction_override', 'body_mass_scale', 'body_mass_left_scale',
                'body_mass_right_scale', 'actuator_gain_scale', 'disable_root_damping',
                'reference_phase_rate_scale', 'action_repeat'):
        if getattr(task, key) != provenance['task'][key]:
            raise ValueError(f'Steering calibration physics mismatch: {key}')
    if task.terrain_enabled or not np.allclose(task.action_scales, provenance['task']['action_scales']):
        raise ValueError('Steering calibration requires flat terrain and matching action scales')
    if not task.forward_command_enabled and task.forward_command_fixed_m_s is None:
        raise ValueError('Steering calibration requires a forward speed command')
    speed_bounds = (task.forward_command_min_m_s, task.forward_command_max_m_s) if task.forward_command_fixed_m_s is None else (task.forward_command_fixed_m_s,)*2
    if speed_bounds[0] < speeds[0]-1e-7 or speed_bounds[1] > speeds[-1]+1e-7:
        raise ValueError('Forward commands exceed the steering calibration domain')
    turn_bound = task.turn_command_max_rad_s if task.turn_command_fixed_rad_s is None else abs(task.turn_command_fixed_rad_s)
    if turn_bound > min(-commands[0], commands[-1])+1e-7:
        raise ValueError('Yaw commands exceed the steering calibration domain')
    return table
