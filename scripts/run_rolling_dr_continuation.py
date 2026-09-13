"""Cloud DR continuation from the deployed actor; critic adaptation then conservative PPO."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


def run_logged(command, log):
    print(shlex.join(command), flush=True)
    with log.open('x', encoding='utf-8') as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise subprocess.CalledProcessError(status, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/rolling_low_speed_20260912_065438/actor'))
    parser.add_argument('--step', type=int, default=81920)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--stage', choices=['all', 'critic', 'actor'], default='all')
    parser.add_argument('--dr-strength', type=float, default=0.25)
    parser.add_argument('--lateral-limit-m', type=float, default=0.50)
    parser.add_argument('--critic-steps', type=int, default=204800)
    parser.add_argument('--actor-steps', type=int, default=1966080)
    parser.add_argument('--dry-run', action='store_true', help='print commands only; no JAX, training or output writes')
    args = parser.parse_args()
    if not math.isfinite(args.dr_strength) or not 0 < args.dr_strength <= 1:
        parser.error('--dr-strength must be in (0,1]')
    if not math.isfinite(args.lateral_limit_m) or args.lateral_limit_m <= 0:
        parser.error('--lateral-limit-m must be positive')
    if min(args.critic_steps, args.actor_steps) < 1 or args.step < 0:
        parser.error('steps must be positive and checkpoint step nonnegative')
    if args.stage == 'actor' and args.out is None:
        parser.error('--stage actor requires --out from the completed critic stage')
    source = args.source.resolve()
    checkpoint = source / 'checkpoints' / f'{args.step:012d}'
    student, restore = checkpoint / 'student_params', checkpoint / 'params'
    for path in [source / 'training_config.json', student, restore]:
        if not path.is_file():
            parser.error('Missing source file: ' + str(path))
    saved = json.loads((source / 'training_config.json').read_text())
    if not saved['args'].get('command_conditioned'):
        parser.error('Expected the command-conditioned rolling actor')
    out = (args.out or Path('results/rolling_dr_pd55_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    common = [sys.executable, '-u', '-m', 'scripts.train_mjx_3d_roll_student_dr_ppo', str(student),
        '--geometry', 'rollingquad_2_abd10_no_self_collision', '--command-conditioned', '--rolling-snapshots',
        '--rolling-hip-knee-kp-scale', '1.1', '--terminate-lateral-drift-m', str(args.lateral_limit_m),
        '--forward-command-min-m-s', '0.45', '--forward-command-max-m-s', '0.75',
        '--turn-command-min-rad-s', '0.02', '--turn-command-max-rad-s', '0.07',
        '--turn-command-straight-fraction', '0.4', '--command-interval-s', '10',
        '--snapshot-pool-size', '2048', '--eval-snapshot-pool-size', '1024',
        '--snapshot-warmup-min-steps', '100', '--snapshot-warmup-max-steps', '300',
        '--episode-length', '500', '--minimum-success-turns', str(saved['args'].get('minimum_success_turns', 5)),
        '--preset', 'h200', '--max-devices', '4', '--envs', '2048', '--eval-envs', '256',
        '--batch-size', '256', '--num-minibatches', '8', '--unroll-length', '20',
        '--dr-strength', str(args.dr_strength), '--observation-noise-scale', '1',
        '--student-anchor-weight', '0.01', '--initial-policy-std', '0.02', '--entropy-cost', '0',
        '--discounting', '0.99', '--reward-scaling', '1', '--updates-per-batch', '1',
        '--forward-tracking-weight', '6', '--yaw-tracking-weight', '3',
        '--clipping-epsilon', '0.05', '--max-grad-norm', '0.5',
        '--fixed-eval-envs', '256', '--fixed-eval-observation-noise-scale', '0',
        '--fixed-eval-seed', '20260913', '--seed', '20260913', '--training-metrics-steps', '40960',
        '--stop-success-drop', '0.10']
    # Keep the exact CEM reference/calibration used to train the restored actor.
    for key in ['controller', 'steering_calibration']:
        value = saved['args'].get(key)
        if value:
            if not Path(value).is_file():
                parser.error('Missing original ' + key + ': ' + value)
            common += ['--' + key.replace('_', '-'), value]
    stages = ['critic', 'actor'] if args.stage == 'all' else [args.stage]
    commands = []
    for stage in stages:
        target = out / stage
        if target.exists() or target.with_suffix('.log').exists():
            parser.error('Output already exists: ' + str(target))
        if stage == 'critic':
            options = ['--restore-ppo', str(restore), '--critic-only', '--steps', str(args.critic_steps),
                       '--num-evals', '6', '--learning-rate', '0.0001']
        else:
            critic = out / 'critic/params_final'
            if args.stage == 'actor' and not critic.is_file():
                parser.error('Missing completed critic: ' + str(critic))
            options = ['--restore-ppo', str(critic), '--steps', str(args.actor_steps), '--num-evals', '17',
                       '--learning-rate', '0.000003', '--learning-rate-schedule', 'adaptive_kl',
                       '--desired-kl', '0.01', '--min-learning-rate', '0.0000001', '--max-learning-rate', '0.00001']
        commands.append((stage, common + options + ['--out', str(target)]))
    print('Output:', out)
    print('PPO actor/critic/normalizer restored; optimizer state is reinitialized by the training API.')
    print('Nominal monitor: P=5.5 for hip/knee, P=5 for abduction, D=0.1, lateral=%.2fm.' % args.lateral_limit_m)
    print('Regular PPO evaluation measures DR; fixed evaluation measures nominal physics, not DR robustness.')
    if args.dry_run:
        for _, command in commands:
            print(shlex.join(command))
        return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / ('continuation_' + args.stage + '.json')
    with manifest.open('x', encoding='utf-8') as handle:
        json.dump(dict(source=str(source), source_step=args.step, dr_strength=args.dr_strength,
                       lateral_limit_m=args.lateral_limit_m, commands=commands), handle, indent=2)
    for stage, command in commands:
        try:
            run_logged(command, out / (stage + '.log'))
        finally:
            if (out / stage).is_dir():
                subprocess.run([sys.executable, '-m', 'scripts.collect_rolling_ppo_diagnostics', str(out / stage),
                                '--out', str(out / (stage + '_diagnostics.zip')), '--log', str(out / (stage + '.log'))],
                               check=False)


if __name__ == '__main__':
    main()
