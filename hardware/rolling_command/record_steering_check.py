"""Record steering diagnostics only. This script never sends robot commands."""
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import shlex
import shutil


TOPICS = [
    '/joy', '/emergency_stop', '/joint_states', '/imu_sensor_broadcaster/imu',
    '/neural_controller_roll/rolling_cmd_vel',
    '/neural_controller_roll/rolling_policy_state',
    '/neural_controller_roll/request_rolling_policy',
    '/neural_controller_roll/enable_policy',
    '/neural_controller_roll/request_roll_to_stand',
    '/neural_controller_roll/observation',
    '/neural_controller_roll/policy_output',
    '/neural_controller_roll/position_command',
    '/neural_controller_roll/imu_latency_seconds',
    '/neural_controller_roll/policy_inference_latency_seconds',
    '/joy_util_node/sequence_state', '/joy_util_node/sequence_detail', '/rosout',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', choices=['straight', 'left', 'right'])
    parser.add_argument('--output-root', type=Path, default=Path('~/bags/rolling_steering'))
    parser.add_argument('--dry-run', action='store_true', help='Print the command without recording')
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = root / f'{stamp}_{args.label}'
    command = ['ros2', 'bag', 'record', '--storage', 'mcap', '--output', str(output), *TOPICS]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if os.name != 'posix' or shutil.which('ros2') is None:
        parser.error('Run on the robot after sourcing its ROS setup.bash')
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'label': args.label,
        'label_is_operator_annotation_not_a_command': True,
        'intended_vx_m_s': 0.6,
        'intended_yaw_rad_s': {'straight': 0.0, 'left': 0.07, 'right': -0.07}[args.label],
        'topics': TOPICS,
        'command': command,
        'note': 'Keep the robot clear of obstacles; annotate any collision or human contact.',
    }
    with output.with_suffix('.trial.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print('Recording only. Operate the robot with the existing gamepad flow. '
          'Ctrl+C stops recording and finalizes the bag.', flush=True)
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
