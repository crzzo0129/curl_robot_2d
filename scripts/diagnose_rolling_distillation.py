"""Re-evaluate saved BC and DAgger students; collect traces without training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = args.run / 'distillation.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    saved = config['args']
    if saved['teacher_source'] != 'cem' or saved['deploy_dr'] or saved['terrain_enabled']:
        parser.error('This comparison supports nominal CEM distillation only')
    out = args.out or Path(f"results/rolling_distill_review_{datetime.now():%Y%m%d_%H%M%S}")
    bundle = out.with_name(out.name + '_diagnostics.zip')
    checkpoints = [('bc', args.run/'student_params_before_dagger'), ('dagger', args.run/'student_params')]
    common = [sys.executable, '-u', '-m', 'scripts.train_mjx_3d_roll_distillation',
              '--teacher-source', 'cem', '--eval-only', '--record-diagnostics']
    keys = ('controller', 'geometry', 'preset', 'num_devices', 'envs', 'eval_envs',
            'episode_length', 'snapshot_warmup_min_steps', 'snapshot_warmup_max_steps',
            'snapshot_segment_steps', 'snapshot_pool_refresh_steps',
            'forward_command_min_m_s', 'forward_command_max_m_s',
            'turn_command_min_rad_s', 'turn_command_max_rad_s',
            'turn_command_straight_fraction', 'command_interval_s', 'steering_calibration',
            'seed', 'eval_seed', 'minimum_closed_loop_turns', 'log_every', 'memory_fraction',
            'mujoco_gl', 'reset_pose', 'stand_hold_s', 'stand_to_compact_s')
    for key in keys:
        if saved.get(key) is not None:
            common += ['--'+key.replace('_', '-'), str(saved[key])]
    for key in ('command_conditioned', 'random_cem_snapshots', 'lateral_drift_diagnostic_only'):
        if saved.get(key):
            common.append('--'+key.replace('_', '-'))
    if not saved.get('teacher_explicit_phase_observation', True):
        common.append('--no-teacher-explicit-phase-observation')
    common += ['--hidden-layers', *(str(v) for v in saved['hidden_layers'])]
    # Reproduce the historical reset salt and share real states between policies.
    environment_seed = saved.get('eval_environment_seed')
    if environment_seed is None:
        environment_seed = saved['seed']
    common += ['--eval-environment-seed', str(environment_seed)]
    if saved.get('random_cem_snapshots'):
        common += ['--eval-snapshot-cache', str(out/'evaluation_snapshots.npz')]
    commands = [common + ['--restore-student', str(path), '--out', str(out/name)]
                for name, path in checkpoints]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    for _, path in checkpoints:
        if not path.is_file():
            parser.error(f'Missing checkpoint: {path}')
    if out.exists() or bundle.exists():
        parser.error(f'Output already exists: {out} or {bundle}')
    out.mkdir(parents=True)
    shutil.copy2(config_path, out/'original_distillation.json')
    calibration = saved.get('steering_calibration')
    if calibration:
        shutil.copy2(calibration, out/'steering_calibration.json')
    manifest = {'commands': commands,
        'note': 'Evaluation only; teacher queried at student states, not independently rolled out. Random-snapshot comparisons reuse one persisted physical state/history pool; verify initial_state_sha256 in both command reports.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    status = 0
    for (name, _), command in zip(checkpoints, commands):
        with (out/f'{name}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        if status:
            break
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name != 'evaluation_snapshots.npz':
                archive.write(path, path.relative_to(out.parent))
    print(f'Diagnostics: {bundle}', flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
