"""CPU MuJoCo replay of deployed JSON policies and the controller handoff rules."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import mujoco
from scripts.export_rtneural import _activation
from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D
from curl_robot_2d_mjx.reset_grounding import make_floor_clearance
from scripts.rolling_handoff import RollingHandoff, effective_action


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def torso_gyro(model, data, torso):
    spatial = np.zeros(6)
    # mjOBJ_BODY local=1 uses inertial principal axes, not the torso axes.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, torso, spatial, 0)
    return data.xmat[torso].reshape(3, 3).T @ spatial[:3]


class Policy:
    def __init__(self, path):
        self.path = path.resolve()
        self.doc = json.loads(path.read_text(encoding='utf-8'))
        assert self.doc['in_shape'] == [1, 720] and self.doc['out_shape'] == [1, 12]
        self.layers = [(l, [np.asarray(w, dtype=np.float32) for w in l['weights']])
                       for l in self.doc['layers']]
        self.center, self.scale, self.low, self.high = [np.asarray(self.doc[k]) for k in
            ('default_joint_pos', 'action_scale', 'joint_lower_limits', 'joint_upper_limits')]

    def __call__(self, obs):
        x = np.clip(obs, -100., 100.).astype(np.float32)
        for l, weights in self.layers:
            if l['type'] == 'batchnorm':
                gamma, beta, mean, var = weights
                x = (x - mean) * (gamma / np.sqrt(var + np.float32(l['epsilon']))) + beta
            else:
                w, b = weights
                x = _activation(l['activation'], x @ w + b)
        if not np.isfinite(x).all():
            raise RuntimeError('Nonfinite network output')
        return x

    def target(self, action):
        return np.clip(self.center + self.scale * action, self.low, self.high)


def cold_history():
    history = np.zeros((20, 36), dtype=np.float32)
    history[:, 5] = -1
    history[:, 11] = 1
    return history


def remap(history, old, new, last_target, command):
    result = history.copy()
    result[:, 6:9] = command
    result[:, 9:12] = [0, 0, 1]
    result[:, 12:24] += old.center - new.center
    targets = np.clip(old.center + history[:, 24:36] * old.scale, old.low, old.high)
    targets[0] = last_target
    actions = np.zeros((20, 12))
    moving = new.scale != 0
    actions[:, moving] = (targets[:, moving] - new.center[moving]) / new.scale[moving]
    if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.0001:
        return None
    result[:, 24:36] = np.clip(actions, -1, 1)
    return result


def run(args):
    policies = {k: Policy(getattr(args, k)) for k in ('startup', 'rolling', 'stop')}
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    model.opt.iterations, model.opt.ls_iterations, model.opt.timestep = 20, 10, .001
    model.dof_damping[:6] = 0
    data = mujoco.MjData(model)
    joints = np.array([model.joint(n).id for n in CONTROLLER_JOINT_NAMES_3D])
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    ai = np.array([model.actuator(n + '_servo').id for n in CONTROLLER_JOINT_NAMES_3D])
    torso = model.body('torso').id
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    standing = np.asarray([0., .9, 1.15] * 4)
    data.qpos[qi] = standing
    data.qvel[:] = 0
    data.ctrl[ai] = standing
    mujoco.mj_forward(model, data)
    clearance = make_floor_clearance(model, model.geom('floor').id, np)
    data.qpos[2] += max(0., .0005 - float(clearance(data)))
    mujoco.mj_forward(model, data)
    dt = .001
    history = cold_history()
    phase = 'hold'
    last_target = standing.copy()
    request_time = takeover = stop_request = stop_time = None
    previous_pitch = math.atan2(data.xmat[torso].reshape(3, 3)[2, 0], data.xmat[torso].reshape(3, 3)[2, 2])
    rolled = 0.
    events, frames, rows = [], [], []
    next_policy_tick = 500
    policy_ticks = 0
    command = np.array([args.vx, 0., args.yaw])
    stable_duration = longest_stable = 0.
    peak_torque = 0.
    min_candidate_delta = None
    handoff = RollingHandoff()
    startup_history = None
    window_ticks = 0
    max_applied_delta = 0.
    limiter_ticks = 0
    applied_command = np.zeros(3)
    continuous_phases = ('blending', 'settling', 'command_ramp', 'rolling')
    max_seconds = 30.

    def event(name, **extra):
        row = dict(event=name, time_s=round(float(data.time), 6), **extra)
        events.append(row)
        print(json.dumps(row), flush=True)

    def measure():
        R = data.xmat[torso].reshape(3, 3)
        return R, torso_gyro(model, data, torso), math.atan2(R[2, 0], R[2, 2])

    def fill(frame, policy, cmd, R, gyro):
        frame[:3] = gyro
        frame[3:6] = R.T @ np.array([0, 0, -1.])
        frame[6:9] = cmd
        frame[9:12] = [0, 0, 1]
        frame[12:24] = data.qpos[qi] - policy.center

    for tick in range(int(max_seconds / dt)):
        R, gyro, pitch = measure()
        if takeover is not None and stop_request is None and data.time >= takeover + args.turn_seconds - 1e-9:
            stop_request = float(data.time)
            event('roll_to_stand_requested')
        if phase in continuous_phases and stop_request is not None and tick % 2 == 0:
            if abs(wrap(pitch - math.pi / 2)) <= math.radians(5):
                phase, stop_time = 'stop', float(data.time)
                history = cold_history()
                history[0, 24:36] = np.clip((last_target - policies['stop'].center) / policies['stop'].scale, -1, 1)
                next_policy_tick = tick
                event('roll_to_stand_takeover', pitch_deg=math.degrees(pitch))
        if tick == next_policy_tick:
            next_policy_tick += 20
            if phase == 'hold':
                phase = 'startup'
                event('stand_to_roll_enabled')
            if phase == 'startup':
                rolled += wrap(pitch - previous_pitch)
                previous_pitch = pitch
                policy_ticks += 1
                if request_time is None and abs(rolled) >= 2 * math.pi:
                    request_time = float(data.time)
                    event('rolling_requested_after_one_turn', turns=rolled / (2 * math.pi))
                if request_time is not None and policy_ticks >= 50 and abs(gyro[1]) >= .5:
                    candidate_command = [.6, 0., 0.] if args.handoff_mode == 'smooth' else command
                    candidate = remap(history, policies['startup'], policies['rolling'], last_target, candidate_command)
                    if candidate is not None:
                        fill(candidate[0], policies['rolling'], candidate_command, R, gyro)
                        raw = policies['rolling'](candidate.reshape(-1))
                        delta = float(np.max(np.abs(policies['rolling'].target(raw) - last_target)))
                        min_candidate_delta = delta if min_candidate_delta is None else min(delta, min_candidate_delta)
                        valid = RollingHandoff.window(pitch, R[2, 1], gyro[1], delta)
                        window_ticks = window_ticks + 1 if valid else 0
                        if (args.handoff_mode == 'smooth' and window_ticks >= 2) or (args.handoff_mode == 'controller' and delta <= .12) or args.handoff_mode == 'prescribed':
                            startup_history = history.copy()
                            phase, history, takeover = 'rolling', candidate, float(data.time)
                            event('ppo_rolling_takeover', max_target_delta_rad=delta, command=list(candidate_command),
                                  pitch_deg=math.degrees(pitch), per_joint_target_delta_rad=(policies['rolling'].target(raw)-last_target).tolist(),
                                  continuity_gate_bypassed=args.handoff_mode == 'prescribed' and delta > .12)
                    else:
                        window_ticks = 0
                else:
                    window_ticks = 0
                if request_time is not None and takeover is None and data.time - request_time >= 10:
                    event('rolling_request_rejected', min_candidate_target_delta_rad=min_candidate_delta)
                    break
            continuous = phase in continuous_phases
            smooth = continuous and args.handoff_mode == 'smooth'
            if smooth:
                old_stage = handoff.stage
                handoff.tick(RollingHandoff.healthy(R[2, 1], gyro[1]), args.vx, 0. if stop_request is not None else args.yaw)
                phase = handoff.stage
                if handoff.stage != old_stage:
                    event('handoff_stage', stage=handoff.stage)
                if handoff.stop_required and stop_request is None:
                    stop_request = float(data.time)
                    event('handoff_settling_timeout_stop_requested')
                applied_command = np.array([handoff.vx, 0., handoff.yaw])
            else:
                applied_command = command.copy() if continuous else np.zeros(3)
            policy = policies['rolling' if continuous else phase]
            fill(history[0], policy, applied_command, R, gyro)
            raw = policy(history.reshape(-1))
            target = policy.target(raw)
            if smooth:
                if phase == 'blending':
                    fill(startup_history[0], policies['startup'], [0, 0, 0], R, gyro)
                    startup_history[0, 24:36] = effective_action(policies['startup'], last_target)
                    startup_raw = policies['startup'](startup_history.reshape(-1))
                    target = (1-handoff.alpha)*policies['startup'].target(startup_raw) + handoff.alpha*target
                    startup_history = np.roll(startup_history, 1, axis=0)
                limiter_ticks += int(np.max(np.abs(target-last_target)) > .12)
                target = np.clip(target, last_target-.12, last_target+.12)
                if phase == 'blending':
                    startup_history[0, 24:36] = effective_action(policies['startup'], target)
                max_applied_delta = max(max_applied_delta, float(np.max(np.abs(target-last_target))))
            if phase == 'stop':
                cap = np.asarray(policy.doc['target_rate_limits_rad_s']) * policy.doc['target_rate_limit_timestep_s']
                target = np.clip(target, last_target - cap, last_target + cap)
            history = np.roll(history, 1, axis=0)
            history[0, 24:36] = effective_action(policy, target) if smooth else raw
            last_target = target
            data.ctrl[ai] = target
            alpha = handoff.alpha if smooth and phase == 'blending' else 1.
            kp = alpha*policy.doc['kp'] + (1-alpha)*policies['startup'].doc['kp']
            kd = alpha*policy.doc['kd'] + (1-alpha)*policies['startup'].doc['kd']
            model.actuator_gainprm[ai, 0] = kp
            model.actuator_biasprm[ai, 1] = -kp
            model.actuator_biasprm[ai, 2] = -kd
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            event('nonfinite_physics')
            break
        peak_torque = max(peak_torque, float(np.max(np.abs(data.actuator_force[ai]))))
        if tick % 20 == 0:
            R, gyro, pitch = measure()
            tilt = math.acos(np.clip(R[2, 2], -1, 1))
            stable = phase == 'stop' and tilt < .35 and np.max(np.abs(data.qvel[vi])) <= .25 and np.max(np.abs(gyro)) <= .3
            stable_duration = stable_duration + .02 if stable else 0.
            longest_stable = max(longest_stable, stable_duration)
            rows.append(dict(time_s=float(data.time), phase=phase, x=float(data.qpos[0]), y=float(data.qpos[1]),
                             command=applied_command.tolist(), handoff_alpha=handoff.alpha if args.handoff_mode == 'smooth' else None,
                             axis_tilt_deg=math.degrees(math.asin(np.clip(R[2, 1], -1, 1))), roll_rate_rad_s=float(gyro[1]),
                             z=float(data.qpos[2]), pitch_rad=pitch, heading_rad=math.atan2(-R[0, 1], R[1, 1]),
                             upright_tilt_deg=math.degrees(tilt), joint_speed_max=float(np.max(np.abs(data.qvel[vi]))),
                             gyro_max=float(np.max(np.abs(gyro))), stand_stable_s=stable_duration))
            frames.append(data.qpos.copy())
        if tick % 2000 == 0:
            print(f'[simulation {data.time:.1f}s] {phase}', flush=True)
        if stop_time is not None and data.time >= stop_time + args.stop_seconds:
            break
    args.out.mkdir(parents=True, exist_ok=False)
    selected = [r for r in rows if r['phase'] in continuous_phases and r['time_s'] <= (stop_request or math.inf)]
    tracking = None
    if len(selected) > 1:
        yaw_change = float(np.unwrap([r['heading_rad'] for r in selected])[-1] - selected[0]['heading_rad'])
        duration = selected[-1]['time_s'] - selected[0]['time_s']
        tracking = dict(duration_s=duration, mean_world_vx_m_s=(selected[-1]['x']-selected[0]['x'])/duration,
                        mean_yaw_rad_s=yaw_change/duration, heading_change_deg=math.degrees(yaw_change),
                        world_y_change_m=selected[-1]['y']-selected[0]['y'])
    report = dict(events=events, elapsed_s=float(data.time), phase=phase,
        sequence_completed=stop_time is not None, stand_success=longest_stable >= .5,
        longest_stable_stand_s=longest_stable, final=rows[-1], tracking=tracking, peak_torque_nm=peak_torque,
        physics='CPU MuJoCo cg20, 1ms physics, 20ms policy, no self collision, 3Nm model torque limits',
        turn_duration_after_ppo_takeover_s=args.turn_seconds, stop_observation_s=args.stop_seconds,
        handoff_mode=args.handoff_mode,
        observation_gyro_frame='torso: world angular velocity rotated by torso R.T',
        max_applied_rolling_target_delta_rad=max_applied_delta if args.handoff_mode == 'smooth' else None,
        rolling_limiter_ticks=limiter_ticks if args.handoff_mode == 'smooth' else None,
        xml=str(args.xml.resolve()), policy_files={k:dict(path=str(p.path), sha256=hashlib.sha256(p.path.read_bytes()).hexdigest()) for k,p in policies.items()},
        limitations=['Single deterministic nominal episode, not a success-rate estimate.',
                     'Python float32 JSON inference; native controller arithmetic can differ slightly.',
                     'Local JSON replay; compare policy_files hashes with the deployed models.'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.out/'trace.json').write_text(json.dumps(rows)+'\n')
    np.savez_compressed(args.out/'trajectory.npz', qpos=np.asarray(frames), time_s=[r['time_s'] for r in rows])
    print(json.dumps(report, indent=2), flush=True)
    if args.video:
        render(args, model, frames, rows, events)


def render(args, model, frames, rows, events):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=600, width=960)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.1, 115, -27
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    with imageio.get_writer(str(args.out/'sequence.mp4'), fps=25, codec='libx264', quality=8, macro_block_size=2) as writer:
        for i in range(0, len(frames), 2):
            data.qpos[:] = frames[i]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], .13]
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 960, 95), fill=(12, 18, 25))
            r = rows[i]
            vx, _, yaw = r.get('command', [0, 0, 0])
            label = ('PRESCRIBED SIMULATION: hardware continuity gate bypassed'
                     if args.handoff_mode == 'prescribed' else
                     'SIMULATION: phase window / blend / straight settle / command ramp' if args.handoff_mode == 'smooth' else
                     'SIMULATION: legacy controller handoff gate enabled')
            draw.text((15, 8), label, font=font, fill='#ffd077')
            draw.text((15, 36), f"t={r['time_s']:.2f}s | {r['phase']} | vx command {vx:.2f} m/s | yaw command {yaw:+.2f} rad/s", font=font, fill='white')
            draw.text((15, 64), f"pitch={math.degrees(r['pitch_rad']):+.1f} deg | upright tilt={r['upright_tilt_deg']:.1f} deg | stand stable={r['stand_stable_s']:.2f}s", font=font, fill='white')
            writer.append_data(np.asarray(image))
            if i == 0 or i == 2 * ((len(frames)-1)//2):
                image.save(args.out / ('initial.png' if i == 0 else 'final.png'))
    renderer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--startup', type=Path, required=True)
    p.add_argument('--rolling', type=Path, required=True)
    p.add_argument('--stop', type=Path, required=True)
    p.add_argument('--xml', type=Path, default=Path('assets/rollingquad_description_2/mjcf/rollingquad_abd10_no_self_collision.xml'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vx', type=float, default=.6)
    p.add_argument('--yaw', type=float, default=.07)
    p.add_argument('--turn-seconds', type=float, default=7.)
    p.add_argument('--stop-seconds', type=float, default=5.)
    p.add_argument('--video', action='store_true')
    p.add_argument('--handoff-mode', choices=('smooth', 'controller', 'prescribed'), default='smooth',
                   help='smooth: new staged handoff; controller: legacy direct gate; prescribed: simulation-only direct bypass')
    args = p.parse_args()
    if args.out.exists():p.error('Use a new output directory')
    run(args)


if __name__ == '__main__':
    main()
