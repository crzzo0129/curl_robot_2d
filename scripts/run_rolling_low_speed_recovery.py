"""Cloud continuation: short low-speed DAgger probes, then critic/actor PPO stages."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


TASK_KEYS = (
    'controller', 'geometry', 'forward_command_min_m_s', 'forward_command_max_m_s',
    'turn_command_min_rad_s', 'turn_command_max_rad_s', 'turn_command_straight_fraction',
    'command_interval_s', 'steering_calibration', 'episode_length',
    'snapshot_warmup_min_steps', 'snapshot_warmup_max_steps',
)


def options(saved, keys):
    result = []
    for key in keys:
        if saved.get(key) is not None:
            result += ['--' + key.replace('_', '-'), str(saved[key])]
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(command, log_path):
    print('[cloud command] ' + ' '.join(command), flush=True)
    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', env=env)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise subprocess.CalledProcessError(status, command)


def bundle_reports(out):
    # Exclude checkpoints/RTNeural exports, including nested Brax directories.
    target = out.with_name(out.name + '_diagnostics.zip')
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if (path.is_file() and (path.suffix in ('.json', '.csv', '.log', '.npz', '.py')
                                   or path.name.endswith('_diagnostics.zip'))
                    and not any(p in ('ppo_checkpoint', 'checkpoints', 'ppo_checkpoints')
                                for p in path.relative_to(out).parts)
                    and path.name not in ('student_rtneural.json', 'controller_config.json')):
                archive.write(path, path.relative_to(out.parent))
    temporary.replace(target)
    print(f'[diagnostics] {target}', flush=True)


def distill_command(saved, checkpoint, out, *, seed, evaluation, chunk_steps):
    command = [sys.executable, '-u', '-m', 'scripts.train_mjx_3d_roll_distillation',
               '--teacher-source', 'cem', '--command-conditioned', '--random-cem-snapshots',
               '--restore-student', str(checkpoint), '--out', str(out), '--record-diagnostics']
    command += options(saved, TASK_KEYS + (
        'preset', 'num_devices', 'envs', 'eval_envs', 'eval_seed',
        'minimum_closed_loop_turns', 'memory_fraction', 'mujoco_gl', 'velocity_loss_weight',
    ))
    command += ['--hidden-layers', *map(str, saved['hidden_layers']),
                '--seed', str(seed), '--log-every', '100']
    if not saved.get('teacher_explicit_phase_observation', True):
        command.append('--no-teacher-explicit-phase-observation')
    if evaluation:
        command.append('--eval-only')
    else:
        command += ['--dagger-snapshot-sampling', 'low_speed_focus',
                    '--dagger-steps', str(chunk_steps), '--dagger-learning-rate', '0.00002',
                    '--dagger-teacher-start-probability', '0.10',
                    '--dagger-teacher-end-probability', '0',
                    '--snapshot-pool-refresh-steps', '250']
    return command


def ppo_command(saved, selected, out, stage):
    command = [sys.executable, '-u', '-m', 'scripts.train_mjx_3d_roll_student_dr_ppo',
               str(selected), '--command-conditioned', '--rolling-snapshots']
    command += options(saved, TASK_KEYS + ('preset', 'envs', 'eval_envs', 'memory_fraction', 'mujoco_gl'))
    command += ['--max-devices', str(saved['num_devices']),
                '--hidden-layers', *map(str, saved['hidden_layers']),
                '--minimum-success-turns', str(saved['minimum_closed_loop_turns']),
                '--snapshot-pool-size', '2048', '--eval-snapshot-pool-size', '1024',
                '--snapshot-sampling', 'uniform', '--dr-strength', '0',
                '--observation-noise-scale', '1', '--fixed-eval-observation-noise-scale', '0',
                '--fixed-eval-envs', str(saved['eval_envs']),
                '--fixed-eval-seed', str(saved['eval_seed']), '--seed', str(saved['seed'] + 1000),
                '--student-anchor-weight', '0', '--initial-policy-std', '0.02',
                '--entropy-cost', '0', '--discounting', '0.99', '--reward-scaling', '1',
                '--batch-size', '256', '--num-minibatches', '8', '--unroll-length', '20',
                '--clipping-epsilon', '0.05', '--max-grad-norm', '0.5',
                '--training-metrics-steps', '40960', '--stop-success-drop', '0.05',
                '--out', str(out / stage)]
    if stage == 'critic':
        command += ['--critic-only', '--steps', '409600', '--num-evals', '11',
                    '--learning-rate', '0.0001', '--updates-per-batch', '2']
    else:
        command += ['--restore-ppo', str(out / 'critic' / 'params_final'),
                    '--steps', '819200', '--num-evals', '21', '--updates-per-batch', '1',
                    '--learning-rate', '0.000003', '--learning-rate-schedule', 'adaptive_kl',
                    '--desired-kl', '0.01', '--min-learning-rate', '0.0000001',
                    '--max-learning-rate', '0.000003']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('distill', 'critic', 'actor'))
    parser.add_argument('--source', type=Path,
                        default=Path('results/rolling_distill_calibrated_20260912_034719'))
    parser.add_argument('--out', type=Path, help='reuse this directory for later critic/actor stages')
    parser.add_argument('--chunks', type=int, default=3)
    parser.add_argument('--chunk-steps', type=int, default=500)
    parser.add_argument('--dry-run', action='store_true', help='print commands only; no JAX or simulation')
    args = parser.parse_args()
    if args.chunks < 1 or args.chunk_steps < 500:
        parser.error('chunks must be positive; chunk-steps must be >= 500 for a full student rollout')
    if args.stage != 'distill' and args.out is None:
        parser.error('critic/actor require --out pointing to the completed continuation directory')
    out = (args.out or Path(f'results/rolling_low_speed_{datetime.now():%Y%m%d_%H%M%S}')).resolve()
    if args.stage == 'distill':
        source = args.source.resolve()
        saved = json.loads((source / 'distillation.json').read_text(encoding='utf-8'))['args']
        if (saved['teacher_source'] != 'cem' or not saved['command_conditioned']
                or not saved['random_cem_snapshots'] or not saved.get('steering_calibration')
                or saved['deploy_dr'] or saved['terrain_enabled']
                or saved.get('lateral_drift_diagnostic_only') or saved.get('reset_pose', 'compact') != 'compact'):
            parser.error('Expected calibrated nominal CEM distillation with rolling snapshots and strict lateral gates')
        saved['eval_seed'] = saved.get('eval_seed') if saved.get('eval_seed') is not None else saved['seed'] + 100000
        if saved['episode_length'] != 500 or saved['eval_envs'] < 256:
            parser.error('This recovery profile requires 500-step evaluation and at least 256 eval envs')
        selected = source / 'student_params'
        if args.dry_run:
            commands = [distill_command(saved, selected, out / 'baseline', seed=saved['seed'],
                                       evaluation=True, chunk_steps=args.chunk_steps)]
            for index in range(1, args.chunks + 1):
                chunk = out / f'dagger_{index:02d}'
                commands.append(distill_command(saved, selected, chunk, seed=saved['seed'] + index,
                                                evaluation=False, chunk_steps=args.chunk_steps))
                selected = chunk / 'student_params'
            print(json.dumps({'commands': commands, 'note': 'Later chunks run only while guardrails pass.'}, indent=2))
            return
        for path in (selected, Path(saved['controller']), Path(saved['steering_calibration'])):
            if not path.is_file():
                parser.error(f'Missing input: {path}')
        if out.exists() or out.with_name(out.name + '_diagnostics.zip').exists():
            parser.error(f'Output exists: {out}; choose a new --out')
        out.mkdir(parents=True)
        shutil.copy2(source / 'distillation.json', out / 'source_distillation.json')
        shutil.copy2(saved['steering_calibration'], out / 'steering_calibration.json')
        code_dir = out / 'source'
        code_dir.mkdir()
        for filename in ('scripts/run_rolling_low_speed_recovery.py',
                         'scripts/train_mjx_3d_roll_distillation.py',
                         'curl_robot_2d_mjx/distillation_curriculum.py'):
            shutil.copy2(filename, code_dir / Path(filename).name)
        manifest = {'source': str(source), 'settings': saved, 'chunks': [], 'completed_stages': [],
                    'selected_student': str(selected), 'commands': [],
                    'comparison_note': 'Same eval commands/seed, independently regenerated physical snapshots; '
                                       'not exact state pairing. Selection thresholds are operational guardrails.'}
        try:
            command = distill_command(saved, selected, out / 'baseline', seed=saved['seed'],
                                      evaluation=True, chunk_steps=args.chunk_steps)
            manifest['commands'].append(command)
            write_json(out / 'recovery.json', manifest)
            run(command, out / 'baseline.log')
            baseline = json.loads((out / 'baseline' / 'command_evaluation.json').read_text())
            best = baseline
            current = selected
            from curl_robot_2d_mjx.distillation_curriculum import continuation_decision
            for index in range(1, args.chunks + 1):
                chunk = out / f'dagger_{index:02d}'
                command = distill_command(saved, current, chunk, seed=saved['seed'] + index,
                                          evaluation=False, chunk_steps=args.chunk_steps)
                manifest['commands'].append(command)
                write_json(out / 'recovery.json', manifest)
                run(command, out / f'dagger_{index:02d}.log')
                metrics = json.loads((chunk / 'command_evaluation.json').read_text())
                decision = continuation_decision(baseline, best, metrics)
                manifest['chunks'].append({'directory': str(chunk), 'decision': decision,
                                           'overall': metrics['overall'], 'by_speed': metrics['by_speed']})
                if decision['select']:
                    best = metrics
                    manifest['selected_student'] = str(chunk / 'student_params')
                write_json(out / 'recovery.json', manifest)
                print('[candidate] ' + json.dumps(decision), flush=True)
                if not decision['safe']:
                    break
                current = chunk / 'student_params'
            manifest['completed_stages'].append('distill')
            write_json(out / 'recovery.json', manifest)
            print(f"[selected student] {manifest['selected_student']}", flush=True)
        finally:
            bundle_reports(out)
    else:
        manifest = json.loads((out / 'recovery.json').read_text(encoding='utf-8'))
        saved = manifest['settings']
        prerequisite = 'distill' if args.stage == 'critic' else 'critic'
        if prerequisite not in manifest['completed_stages']:
            parser.error(f'{prerequisite} stage did not complete successfully')
        selected = Path(manifest['selected_student'])
        command = ppo_command(saved, selected, out, args.stage)
        if args.dry_run:
            print(json.dumps(command, indent=2))
            return
        if not selected.is_file():
            parser.error(f'Missing selected student: {selected}')
        if args.stage == 'actor' and not (out / 'critic' / 'params_final').is_file():
            parser.error('Missing completed critic params_final')
        if (out / args.stage).exists() or (out / f'{args.stage}.log').exists():
            parser.error(f'Stage output exists: {out / args.stage}')
        manifest['commands'].append(command)
        write_json(out / 'recovery.json', manifest)
        try:
            run(command, out / f'{args.stage}.log')
            manifest['completed_stages'].append(args.stage)
            write_json(out / 'recovery.json', manifest)
        finally:
            if (out / args.stage).is_dir():
                try:
                    subprocess.run([sys.executable, '-m', 'scripts.collect_rolling_ppo_diagnostics',
                                    str(out / args.stage), '--out', str(out / f'{args.stage}_diagnostics.zip'),
                                    '--log', str(out / f'{args.stage}.log')], check=True)
                finally:
                    bundle_reports(out)
            else:
                bundle_reports(out)


if __name__ == '__main__':
    main()
