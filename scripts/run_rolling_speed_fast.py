"""Short actor-only speed-tracking continuation from the reviewed PPO checkpoint."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.run_rolling_handoff_tracking import run_logged


def build_command(saved, *, source, step, bank, out, steps, learning_rate, max_learning_rate,
                  exploration_stop_drop=.15, exploration_stop_patience=3,
                  exploration_warmup_steps=245760, no_performance_stop=False):
    previous = saved['args']
    if (previous.get('forward_velocity_frame') != 'heading' or not previous.get('handoff_bank')
            or not previous.get('command_conditioned') or not previous.get('rolling_snapshots')):
        raise ValueError('Expected a completed heading-frame handoff PPO run')
    # Preserve the source recipe's physical/reset/reward/evaluation settings.
    # Only the actor learning-rate schedule and checkpoint objective change.
    keys = ('geometry', 'controller', 'steering_calibration', 'forward_command_min_m_s',
        'forward_command_max_m_s', 'turn_command_min_rad_s', 'turn_command_max_rad_s',
        'turn_command_straight_fraction', 'command_interval_s', 'forward_velocity_frame',
        'handoff_fraction', 'handoff_speed_slew_m_s2', 'handoff_initial_speed_max_m_s',
        'rolling_hip_knee_kp_scale', 'terminate_lateral_drift_m', 'snapshot_pool_size',
        'eval_snapshot_pool_size', 'snapshot_sampling', 'snapshot_warmup_min_steps',
        'snapshot_warmup_max_steps', 'episode_length', 'minimum_success_turns', 'preset',
        'max_devices', 'envs', 'eval_envs', 'batch_size', 'num_minibatches', 'unroll_length',
        'dr_strength', 'observation_noise_scale', 'student_anchor_weight', 'initial_policy_std',
        'entropy_cost', 'discounting', 'reward_scaling', 'updates_per_batch',
        'forward_tracking_weight', 'yaw_tracking_weight', 'clipping_epsilon', 'max_grad_norm',
        'fixed_eval_envs', 'fixed_eval_observation_noise_scale', 'fixed_eval_seed', 'seed',
        'training_metrics_steps')
    checkpoint = source/'checkpoints'/f'{step:012d}'
    # The restored PPO supplies the actor. Keep the original frozen normalizer
    # and anchor reference so changing learning rate does not also change reward.
    anchor_student = previous.get('student', str(checkpoint/'student_params'))
    command = [sys.executable,'-u','-m','scripts.train_mjx_3d_roll_student_dr_ppo',
        str(anchor_student),'--restore-ppo',str(checkpoint/'params'),
        '--command-conditioned','--rolling-snapshots','--handoff-bank',str(bank)]
    for name in keys:
        if previous.get(name) is not None:
            command += ['--'+name.replace('_','-'),str(previous[name])]
    command += ['--bootstrap-on-timeout' if previous.get('bootstrap_on_timeout',True) else '--no-bootstrap-on-timeout',
        '--steps',str(steps),'--num-evals','5',
        '--learning-rate',str(learning_rate),'--learning-rate-schedule','adaptive_kl',
        '--desired-kl',str(previous.get('desired_kl',.01)),
        '--min-learning-rate','0.0000001','--max-learning-rate',str(max_learning_rate),
        '--gradient-diagnostics','--checkpoint-selection','handoff_speed',
        '--tracking-success-drop','0.03',
        '--exploration-stop-drop',str(exploration_stop_drop),
        '--exploration-stop-patience',str(exploration_stop_patience),
        '--exploration-warmup-steps',str(exploration_warmup_steps),'--out',str(out)]
    if no_performance_stop:
        command += ['--no-performance-stop']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/rolling_heading_handoff_20260913_111545/actor'))
    parser.add_argument('--step',type=int,default=737280)
    parser.add_argument('--handoff-bank',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--steps',type=int,default=491520)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--max-learning-rate',type=float,default=1e-5)
    parser.add_argument('--exploration-stop-drop',type=float,default=.15)
    parser.add_argument('--exploration-stop-patience',type=int,default=3)
    parser.add_argument('--exploration-warmup-steps',type=int,default=245760)
    parser.add_argument('--no-performance-stop',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    if args.step < 0 or args.steps < 163840 or args.steps % 40960:
        parser.error('steps must be >=163840 and divisible by 40960; checkpoint step must be nonnegative')
    if not all(math.isfinite(x) for x in (args.learning_rate,args.max_learning_rate)) or not 1e-7 <= args.learning_rate <= args.max_learning_rate:
        parser.error('require 1e-7 <= learning-rate <= max-learning-rate, both finite')
    if (not math.isfinite(args.exploration_stop_drop) or not 0 < args.exploration_stop_drop <= 1
            or args.exploration_stop_patience < 1 or args.exploration_warmup_steps < 0):
        parser.error('exploration drop must be in (0,1], patience positive, warmup nonnegative')
    source = args.source.resolve()
    for file in ('training_config.json',f'checkpoints/{args.step:012d}/params',f'checkpoints/{args.step:012d}/student_params'):
        if not (source/file).is_file(): parser.error('Missing '+str(source/file))
    saved = json.loads((source/'training_config.json').read_text(encoding='utf-8'))
    if saved['args'].get('student') and not Path(saved['args']['student']).is_file():
        parser.error('Missing original anchor student: '+saved['args']['student'])
    bank = (args.handoff_bank or Path(saved['args']['handoff_bank'])).resolve()
    if not bank.is_file(): parser.error('Missing handoff bank: '+str(bank))
    for key in ('controller','steering_calibration'):
        path = saved['args'].get(key)
        if path and not Path(path).is_file(): parser.error('Missing source '+key+': '+path)
    out = (args.out or Path('results/rolling_speed_fast_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if out.exists() or out.with_suffix('.zip').exists(): parser.error('Output exists; choose a new --out')
    try:
        command = build_command(saved,source=source,step=args.step,bank=bank,out=out,
            steps=args.steps,learning_rate=args.learning_rate,max_learning_rate=args.max_learning_rate,
            exploration_stop_drop=args.exploration_stop_drop, exploration_stop_patience=args.exploration_stop_patience,
            exploration_warmup_steps=args.exploration_warmup_steps,no_performance_stop=args.no_performance_stop)
    except ValueError as error:
        parser.error(str(error))
    print('Actor-only continuation; source PPO actor/critic/normalizer restored; optimizer reinitialized.')
    print('Best checkpoint selection keeps the 3 percentage-point survival constraint.')
    print('Performance early stopping disabled.' if args.no_performance_stop else
          f'Exploration: no survival stop before {args.exploration_warmup_steps} steps; '
          f'then require {args.exploration_stop_patience} consecutive evaluations with >{args.exploration_stop_drop:.0%} drop.')
    print('Output:',out,flush=True)
    if args.dry_run:
        print(shlex.join(command))
        return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3')
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests',
                    '-p','test_rolling_speed_fast.py','-v'],check=True)
    out.mkdir(parents=True)
    # The trainer permits an existing directory only when explicitly requested.
    command += ['--allow-existing-output']
    manifest = dict(source=str(source),step=args.step,command=command,bank=str(bank),
        purpose='short actor speed probe; no repeated collection or critic-only warmup')
    (out/'speed_run.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    try:
        try:
            run_logged(command,out/'training.log')
        except subprocess.CalledProcessError:
            if not (out/'stopped.json').is_file(): raise
            print('Stopped by the evaluation guard; preserved all checkpoints.',flush=True)
        history_file = out/'fixed_eval_history.json'
        best_file = out/'best_fixed_checkpoint.json'
        if history_file.is_file() and best_file.is_file():
            history = json.loads(history_file.read_text())
            best = json.loads(best_file.read_text())
            baseline = history[0]['tracking_by_reset_source']['handoff']['windowed_forward_mae_m_s']
            current = best['handoff_tracking']['windowed_forward_mae_m_s']
            result = dict(best_checkpoint=best['checkpoint'],baseline_handoff_window_mae=baseline,
                selected_handoff_window_mae=current,selected_step=best['step'],
                improvement_fraction=(1-current/baseline) if baseline and current is not None else None)
            (out/'speed_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print('[speed result]',json.dumps(result),flush=True)
    finally:
        subprocess.run([sys.executable,'-m','scripts.collect_rolling_ppo_diagnostics',str(out),
            '--out',str(out.with_suffix('.zip')),'--log',str(out/'training.log')],check=True)


if __name__ == '__main__':
    main()
