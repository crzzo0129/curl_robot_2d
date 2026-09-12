"""Collect observation histories using the validated CEM steering calibration.

Uses the calibration's 1 ms physics / 50 Hz control, normalized reference and
action clipping, with no dynamics randomization or neural policy. Commands
drive the simulator but never serve as velocity labels or estimator features.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np

from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceGeometry, advance_oscillator, load_cem_reference, reference_action,
)
from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D, rolling_deploy_frame_3d
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d, disable_rollingquad_self_collision_3d,
    duplicate_planar_action_3d, model_path_3d, reference_startup_scale_3d,
)
from curl_robot_2d_mjx.steering_calibration import load_steering_calibration
from scripts.calibrate_cem_steering import CONTROLLER, GEOMETRY, PATTERN, TASK
from scripts.collect_rolling_velocity_data import (
    ACTION_SCALES, ACTIVE, encode_motor_target, make_command_schedule, speed_statistics,
)

DEFAULT_CALIBRATION = Path("assets/controllers/rollingquad_abd10_high_speed_steering_v1.json")


def load_contract(settings):
    """Validate the actual simulation configuration before looking up a command."""
    task = replace(TASK, forward_command_enabled=True,
                   forward_command_min_m_s=settings["speed_range"][0],
                   forward_command_max_m_s=settings["speed_range"][1],
                   turn_command_enabled=True, turn_command_max_rad_s=settings["turn_range"][1])
    reference = load_cem_reference(Path(settings["controller"]), minimum_residual_gain=.15)
    table = load_steering_calibration(settings["calibration"], task=task, reference=reference,
                                      model_path=settings["model"])
    if mujoco.__version__ != table["provenance"]["mujoco_version"]:
        raise ValueError("MuJoCo version differs from the validated calibration")
    return task, reference, table


class CalibratedMotorTarget:
    """The calibration replay's production reference and two action clips."""

    def __init__(self, reference, compact, lower, upper):
        self.reference, self.compact, self.lower, self.upper = reference, compact, lower, upper
        self.planar = np.array([GEOMETRY.compact_hip_angle, GEOMETRY.compact_knee_angle] * 2)
        self.planar_low = np.array([GEOMETRY.hip.shell_compatible_range[0], GEOMETRY.knee.shell_compatible_range[0]] * 2)
        self.planar_high = np.array([GEOMETRY.hip.shell_compatible_range[1], GEOMETRY.knee.shell_compatible_range[1]] * 2)
        self.scales = np.array([.8, 1.2] * 2)
        self.geometry = CEMReferenceGeometry(torso_length_m=GEOMETRY.torso_length,
            link_length_m=GEOMETRY.edge_length, foot_diameter_m=2 * GEOMETRY.foot_radius,
            upper_link_length_m=GEOMETRY.upper_length, lower_link_length_m=GEOMETRY.lower_length)

    def __call__(self, phase, scale, steering):
        planar = reference_action(np, phase, self.reference, compact_ctrl=self.planar,
            action_scales=self.scales, joint_low=self.planar_low,
            joint_high=self.planar_high, geometry=self.geometry)
        base = np.clip(scale * duplicate_planar_action_3d(np, planar), -1., 1.)
        raw = base + steering * PATTERN
        action = np.clip(raw, -1., 1.)
        target = self.compact.copy()
        target[ACTIVE] += action * ACTION_SCALES[ACTIVE]
        return np.clip(target, self.lower, self.upper), bool(np.any(np.abs(raw) > 1.))


def collect_episode(job):
    settings, episode = job
    task, reference, table = load_contract(settings)
    rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], episode]))
    schedule, interval_steps = make_command_schedule(settings, rng, calibration=table)
    model = mujoco.MjModel.from_xml_path(settings["model"])
    apply_physics_options_3d(model, task)
    disable_rollingquad_self_collision_3d(model)
    data = mujoco.MjData(model)
    key = model.key("compact").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    torso, root, floor = model.body("torso").id, model.joint("root").id, model.geom("floor").id
    if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[root] != torso:
        raise ValueError("expected a free root joint on torso")
    qroot, vroot = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
    actuators = np.array([model.actuator(f"{name}_servo").id for name in CONTROLLER_JOINT_NAMES_3D])
    joints = model.actuator_trnid[actuators, 0]
    qids = model.jnt_qposadr[joints]
    compact = model.key_ctrl[key, actuators].copy()
    low = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[actuators, 0])
    high = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[actuators, 1])
    noise = settings["reset_noise"]
    data.qpos[qids[ACTIVE]] += rng.uniform(-noise, noise, len(ACTIVE))
    data.qvel[vroot + 6:] += rng.uniform(-noise, noise, len(data.qvel) - vroot - 6)
    yaw = float(rng.uniform(-np.pi, np.pi))
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
                      data.qpos[qroot + 3:qroot + 7].copy())
    data.qpos[qroot + 3:qroot + 7] = rotated
    data.ctrl[actuators] = compact
    target_fn = CalibratedMotorTarget(reference, compact, low, high)
    dt = task.physics_timestep * task.action_repeat
    if not np.isclose(dt, 1 / settings["control_hz"]):
        raise ValueError("observation period differs from calibrated control period")
    steps = int(round(settings["duration"] / dt))
    phase = spin = 0.
    previous_action = np.zeros(12, np.float32)
    sensor_noise = settings["observation_noise"]
    bias, sigma = np.zeros(36), np.zeros(36)
    bias[:3] = rng.normal(0, .015 * sensor_noise, 3)
    bias[12:24] = rng.normal(0, .005 * sensor_noise, 12)
    sigma[:3], sigma[3:6], sigma[12:24] = .03 * sensor_noise, .01 * sensor_noise, .002 * sensor_noise
    names = ("frames", "velocity_world", "body_y_world", "position_world", "time_s", "motor_target",
             "cem_phase", "rolling_phase", "command", "command_segment", "command_age_s",
             "speed_command_delta", "turn_command_delta", "target_scale", "steering_amplitude")
    records = {key: [] for key in names}
    saturation_steps = contact_steps = 0
    started = time.perf_counter()
    for step in range(steps):
        segment = min(step // interval_steps, len(schedule) - 1)
        current, previous = schedule[segment], schedule[max(0, segment - 1)]
        command = np.array([current["speed_command"], 0., current["turn_command"]])
        mujoco.mj_forward(model, data)
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all():
            raise FloatingPointError(f"nonfinite state in episode {episode}")
        if not np.isclose(data.time, step * dt, atol=1e-7):
            raise ValueError("observation time drift")
        rotation = data.xmat[torso].reshape(3, 3)
        frame = rolling_deploy_frame_3d(np,
            angular_velocity_body=rotation.T @ data.cvel[torso, :3],
            projected_gravity=rotation.T @ np.array([0., 0., -1.]),
            joint_position_offset=data.qpos[qids] - compact,
            last_action=previous_action, command=command)
        frame = (frame + bias + rng.normal(size=36) * sigma).astype(np.float32)
        frame[3:6] /= max(np.linalg.norm(frame[3:6]), 1e-8)
        values = (frame, data.qvel[vroot:vroot + 3].copy(), rotation[:, 1].copy(),
                  data.qpos[qroot:qroot + 3].copy(), float(data.time), data.ctrl[actuators].copy(),
                  phase, spin, command, segment, (step-current["start_step"]) * dt,
                  current["speed_command"]-previous["speed_command"],
                  current["turn_command"]-previous["turn_command"],
                  current["target_scale"], current["steering_amplitude"])
        for key, value in zip(names, values):
            records[key].append(value)
        for _ in range(task.action_repeat):
            phase = float(advance_oscillator(np, spin, phase, task.physics_timestep, reference))
            scale = reference_startup_scale_3d(np, data.time, task, target_scale=current["target_scale"])
            target, saturated = target_fn(phase, scale, current["steering_amplitude"])
            data.ctrl[actuators] = target
            mujoco.mj_step(model, data)
            spin += float(data.qvel[vroot + 4]) * task.physics_timestep
            saturation_steps += int(saturated)
            contact_steps += int(any(floor not in (int(c.geom1), int(c.geom2)) for c in data.contact))
        previous_action = encode_motor_target(target, compact)
    records = {key: np.asarray(value) for key, value in records.items()}
    records["episode"] = np.full(steps, episode, np.int32)
    elevation = np.arcsin(np.clip(np.abs(records["body_y_world"][:, 2]), 0., 1.))
    stats = speed_statistics(records)
    stable = bool(records["position_world"][:, 2].min() > .025 and
                  records["position_world"][:, 2].max() < .8 and elevation.max() < .5 and
                  stats["mean_after_2s_m_s"] is not None and stats["mean_after_2s_m_s"] > .15)
    if not np.isfinite(np.r_[data.qpos, data.qvel]).all():
        raise FloatingPointError(f"nonfinite final state in episode {episode}")
    summary = dict(episode=episode, command_schedule=schedule, yaw=yaw, stable=stable,
        axis_elevation_max_rad=float(elevation.max()), compact_joint_position=compact.tolist(),
        self_collision_enabled=False, self_contact_fraction=contact_steps / (steps * task.action_repeat),
        action_saturation_fraction=saturation_steps / (steps * task.action_repeat),
        speed_statistics=stats, wall_seconds=time.perf_counter()-started)
    return records, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--controller", type=Path, default=CONTROLLER)
    parser.add_argument("--model", type=Path, default=model_path_3d(TASK.geometry))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--duration", type=float, default=32.)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--command-mode", choices=("random", "scripted"), default="random")
    parser.add_argument("--command-interval", type=float, default=4.)
    parser.add_argument("--speed-range", type=float, nargs=2, default=[.4111, .811])
    parser.add_argument("--turn-range", type=float, nargs=2, default=[.02, .08])
    parser.add_argument("--reset-noise", type=float, default=.005)
    parser.add_argument("--observation-noise", type=float, default=1.)
    args = parser.parse_args(argv)
    if min(args.episodes, args.workers) < 1 or args.seed < 0:
        parser.error("positive episode/worker counts and nonnegative seed required")
    if not np.isfinite([args.duration, args.command_interval, args.reset_noise, args.observation_noise,
                       *args.speed_range, *args.turn_range]).all():
        parser.error("settings must be finite")
    if args.duration <= 2 or args.command_interval < .02 or min(args.reset_noise, args.observation_noise) < 0:
        parser.error("duration >2 s, interval >=.02 s and nonnegative noise required")
    if not 0 < args.turn_range[0] <= args.turn_range[1] or args.speed_range[0] > args.speed_range[1]:
        parser.error("invalid command range")
    if args.out.suffix != ".npz" or args.out.exists():
        parser.error("out must be a new .npz path")
    return args


def main(argv=None):
    args = parse_args(argv)
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v
                for k, v in vars(args).items() if k != "out"}
    settings.update(control_hz=50., source="cem_calibrated")
    task, _, table = load_contract(settings)
    started = time.perf_counter()
    jobs = [(settings, ep) for ep in range(args.episodes)]
    episodes, summaries = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for records, summary in pool.map(collect_episode, jobs):
            episodes.append(records)
            summaries.append(summary)
            print(f"episode {len(episodes)}/{args.episodes}: stable={summary['stable']}, "
                  f"mean_v={summary['speed_statistics']['time_mean_m_s']:.3f}, "
                  f"wall={summary['wall_seconds']:.1f}s", flush=True)
    metadata = dict(schema_version=1, source="cem_calibrated", control_dt=.02, controller_update_dt=.001,
        settings=settings, physics_task=asdict(task), calibration_validation=table["validation"],
        sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.controller, args.model, args.calibration)},
        episodes=summaries, wall_seconds=time.perf_counter()-started, mujoco_version=mujoco.__version__,
        observation_contract=dict(joint_names=list(CONTROLLER_JOINT_NAMES_3D),
            joint_offset_origin="model compact keyframe", action_scales=ACTION_SCALES.tolist(),
            compact_joint_position=summaries[0]["compact_joint_position"],
            last_action="latest physical clipped target encoded about compact"),
        velocity_reference="root free-joint origin, world coordinates",
        limitation="Calibration targets axis heading rate; supervised labels remain actual trajectory rate. Self collision disabled.")
    arrays = {key: np.concatenate([ep[key] for ep in episodes]) for key in episodes[0]}
    metadata["speed_statistics"] = speed_statistics(arrays)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays, metadata_json=np.array(json.dumps(metadata)))
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {args.out}; {args.episodes * args.duration:.1f} simulated s / "
          f"{metadata['wall_seconds']:.1f} wall s; stable={sum(s['stable'] for s in summaries)}/{len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
