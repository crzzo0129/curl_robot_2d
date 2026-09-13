"""Cloud CPU collection of actual JSON startup + smooth-handoff reset states."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--startup', type=Path, required=True)
    p.add_argument('--rolling', type=Path, required=True)
    p.add_argument('--stop', type=Path, required=True)
    p.add_argument('--xml', type=Path, default=Path('assets/rollingquad_description_2/mjcf/rollingquad_abd10_no_self_collision.xml'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260914)
    args = p.parse_args()
    if args.count < 8 or args.out.exists():
        p.error('count must be >=8 and output must be new')
    for name in ('startup', 'rolling', 'stop', 'xml'):
        if not getattr(args, name).is_file(): p.error('Missing ' + name)
    from scripts.simulate_rolling_policy_sequence import run, effective_action
    from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D
    policy = json.loads(args.rolling.read_text())
    if (not np.isclose(policy['kp'],5.) or not np.isclose(policy['kd'],.1)
            or not np.allclose(policy.get('kps',[5.]*12),5.)
            or not np.allclose(policy.get('kds',[.1]*12),.1)):
        p.error('This handoff recipe requires the P=5, D=0.1 rolling export')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    random = np.random.default_rng(args.seed)
    log_path = args.out.with_suffix('.log')
    with log_path.open('x', encoding='utf-8') as log:
        for attempt in range(args.count * 3):
            ready = [None]
            age = float(random.uniform(0., .30))
            def capture(*, data, phase, history, previous_target, policy, rows, rolling_phase, healthy):
                if phase not in ('command_ramp', 'rolling') or not healthy: return None
                if ready[0] is None: ready[0] = float(data.time)
                if data.time < ready[0] + age or len(rows) < 51: return None
                recent = rows[-51:]
                h = np.unwrap([r['heading_rad'] for r in recent])
                dx = np.diff([r['x'] for r in recent]); dy = np.diff([r['y'] for r in recent])
                middle = (h[1:] + h[:-1]) / 2
                speed = float(np.sum(dx*np.cos(middle) + dy*np.sin(middle)) /
                              (recent[-1]['time_s'] - recent[0]['time_s']))
                if not .3 <= speed <= 1.2: return None
                # Simulator slot 0 is scratch space for the next frame; slots
                # 1..19 are valid past frames. Reset inserts its fresh frame.
                past = np.concatenate((history[1:], np.zeros((1, 36), dtype=np.float32))).reshape(-1)
                return dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                    time=np.float32(data.time), history=past,
                    previous_action=effective_action(policy, previous_target),
                    speed=np.float32(speed), rolling_phase=np.float32(rolling_phase),
                    trajectory_id=np.int32(attempt))
            config = SimpleNamespace(startup=args.startup, rolling=args.rolling, stop=args.stop, xml=args.xml,
                out=args.out.parent/(args.out.stem+'_failed_'+str(args.seed+attempt)), vx=.6, yaw=0.,
                turn_seconds=3., stop_seconds=1., handoff_mode='smooth', video=False,
                seed=args.seed+attempt, initial_joint_noise_rad=.015, initial_velocity_noise=.05)
            with contextlib.redirect_stdout(log):
                snapshot = run(config, snapshot_callback=capture)
            if snapshot is not None: samples.append(snapshot)
            print(f'[handoff bank] accepted={len(samples)}/{args.count} attempts={attempt+1}', flush=True)
            if len(samples) == args.count: break
    if len(samples) < args.count:
        raise RuntimeError('Too few healthy handoffs; inspect the collector log. No CEM fallback was used.')
    metadata = dict(schema='rolling_handoff_bank_v1', history_order='newest_first_past_frames',
        controller_joint_names=list(CONTROLLER_JOINT_NAMES_3D), control_period_s=.02,
        source='json_stand_to_roll_then_live_blend_and_straight_settle',
        seed=args.seed, count=len(samples), trajectory_split_required=True,
        initial_joint_noise_rad=.015, initial_velocity_noise=.05, capture_after_stable_s=[0,.30],
        files_sha256={k:hashlib.sha256(getattr(args,k).read_bytes()).hexdigest() for k in ('startup','rolling','stop','xml')},
        xml_text_sha256=hashlib.sha256(args.xml.read_text(encoding='utf-8').encode()).hexdigest(),
        controller_metadata={k:policy[k] for k in ('default_joint_pos','action_scale','kp','kd')},
        speed_summary_m_s={k:float(fn([s['speed'] for s in samples])) for k,fn in
                          [('min',np.min),('median',np.median),('max',np.max)]})
    with args.out.open('xb') as f:
        np.savez_compressed(f, **{k:np.stack([s[k] for s in samples]) for k in samples[0]},
                            metadata=np.asarray(json.dumps(metadata)))
    args.out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved', args.out, metadata['speed_summary_m_s'], flush=True)


if __name__ == '__main__': main()
