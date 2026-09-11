"""Collect independent CPU MuJoCo episodes for the standalone velocity estimator.

Run from project root with python -m scripts.collect_rolling_velocity_data.
The existing phase-locked CEM controller drives the simulator. No neural policy
is loaded. CEM targets update at physics rate and observations sample at 52 Hz.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

# Avoid oversized BLAS pools inside independent simulation workers.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.deployment_rolling_3d import (
    CONTROLLER_JOINT_NAMES_3D, ROLLING_EFFECTIVE_ACTION_INDICES_3D,
    rolling_deploy_frame_3d,
)
from curl_robot_2d_mjx.environment_3d import steering_prior_3d
from scripts import evaluate_3d_symmetric_cem_reference as bridge
from scripts.collect_cem_cycle_data import DEFAULT_CONTROLLER, DEFAULT_XML

ACTION_SCALES = np.tile([0., .8, 1.2], 4)
ACTIVE = np.asarray(ROLLING_EFFECTIVE_ACTION_INDICES_3D)


def cem_motor_target(phase, reference, compact, lower, upper, amplitude, steering):
    """Existing CEM target/projection and validated left/right steering pattern."""
    planar = bridge.planar_cem_target(phase, reference, apply_foot_gap_projection=True)
    planar = bridge.scaled_planar_target(planar, amplitude)
    target = compact.copy()
    target[ACTIVE] = bridge.map_planar_to_curl_3d_targets(planar)
    target[ACTIVE] += steering_prior_3d(np, steering, 1., 1.) * ACTION_SCALES[ACTIVE]
    return np.clip(target, lower, upper)


def encode_motor_target(target, compact):
    """Encode the actual clipped physical command, never an unexecuted action."""
    action = np.zeros(12, np.float32)
    action[ACTIVE] = (target[ACTIVE] - compact[ACTIVE]) / ACTION_SCALES[ACTIVE]
    return action


def speed_statistics(records):
    axis = records["body_y_world"][:, :2]
    length = np.linalg.norm(axis, axis=1)
    valid = length >= .2
    heading = np.column_stack((axis[:, 1], -axis[:, 0])) / np.maximum(length[:, None], 1e-12)
    speed = np.sum(records["velocity_world"][:, :2] * heading, axis=1)
    selected = speed[valid]
    steady = speed[valid & (records["time_s"] >= 2.)]
    return dict(valid_samples=int(valid.sum()),
                instantaneous_min_m_s=float(selected.min()) if selected.size else None,
                instantaneous_max_m_s=float(selected.max()) if selected.size else None,
                time_mean_m_s=float(selected.mean()) if selected.size else None,
                mean_after_2s_m_s=float(steady.mean()) if steady.size else None)


def collect_episode(job):
    settings, episode = job
    rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], episode]))
    controller_path = settings["controllers"][episode % len(settings["controllers"])]
    reference = load_cem_reference(Path(controller_path), reference_weight=1., minimum_residual_gain=0.)
    bridge.activate_planar_geometry(bridge.PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
    model = mujoco.MjModel.from_xml_path(settings["model"])
    bridge.apply_physics_options_3d(model, bridge.physics_profile_3d(settings["physics_profile"]))
    dt = 1.0 / settings["control_hz"]
    substeps = settings["substeps"]
    model.opt.timestep = dt / substeps
    physics_dt = float(model.opt.timestep)

    def object_id(kind, name):
        value = mujoco.mj_name2id(model, kind, name)
        if value < 0:
            raise ValueError(f"missing model object: {name}")
        return value

    torso = object_id(mujoco.mjtObj.mjOBJ_BODY, "torso")
    root = object_id(mujoco.mjtObj.mjOBJ_JOINT, "root")
    if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("root must be a free joint: its translation velocity is world-frame")
    if model.jnt_bodyid[root] != torso:
        raise ValueError("root joint must belong to torso")
    root_qpos, root_dof = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
    key = object_id(mujoco.mjtObj.mjOBJ_KEY, "compact")
    floor = object_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
    actuators = np.array([object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_servo")
                          for name in CONTROLLER_JOINT_NAMES_3D])
    joint_ids = model.actuator_trnid[actuators, 0]
    joint_qpos = model.jnt_qposadr[joint_ids]
    compact = model.key_qpos[key, joint_qpos].copy()
    if not np.allclose(compact[ACTIVE], bridge.map_planar_to_curl_3d_targets(bridge.PLANAR_COMPACT), atol=1e-5):
        raise ValueError("model compact pose does not match the Pupper CEM geometry")
    low = np.maximum(model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[actuators, 0])
    high = np.minimum(model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[actuators, 1])
    strength = settings["dr_strength"]
    friction_scale = float(rng.uniform(1 - .25 * strength, 1 + .25 * strength))
    model.geom_friction[floor, 0] *= friction_scale
    gain_scale = rng.uniform(1 - .15 * strength, 1 + .15 * strength, len(actuators))
    model.actuator_gainprm[actuators, 0] = settings["kp"] * gain_scale
    model.actuator_biasprm[actuators, 1] = -settings["kp"] * gain_scale
    model.actuator_biasprm[actuators, 2] = -settings["kd"]
    model.actuator_forcerange[actuators, 0] = -settings["torque_limit"]
    model.actuator_forcerange[actuators, 1] = settings["torque_limit"]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    # Change world heading without changing the compact pose relative to gravity.
    yaw = float(rng.uniform(-np.pi, np.pi))
    yaw_quat = np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, yaw_quat, data.qpos[root_qpos + 3:root_qpos + 7].copy())
    data.qpos[root_qpos + 3:root_qpos + 7] = rotated
    data.ctrl[actuators] = compact
    data.qvel[:] = 0
    target_scale = float(rng.uniform(*settings["target_scale_range"]))
    phase = rolling_phase = 0.
    previous_action = np.zeros(12, np.float32)
    bias = np.zeros(36, np.float32)
    noise = settings["observation_noise"]
    bias[:3] = rng.normal(0, .015 * noise, 3)
    bias[12:24] = rng.normal(0, .005 * noise, 12)
    sigma = np.zeros(36, np.float32)
    sigma[:3], sigma[3:6], sigma[12:24] = .03 * noise, .01 * noise, .002 * noise
    records = {key: [] for key in ("frames", "velocity_world", "body_y_world", "time_s",
                                   "position_world", "motor_target", "cem_phase", "rolling_phase")}
    # Both signs and nominal runs; offsets are in normalized action space.
    steering_sign = (0, 1, -1)[episode % 3]
    steering = steering_sign * settings["steering_bias"]
    steps = int(round(settings["duration"] / dt))
    self_contact_steps = 0
    start = time.perf_counter()
    for step in range(steps):
        # mj_step's derived kinematics can lag its integrated qpos: refresh before
        # recording observations and root velocity from the SAME simulation time.
        mujoco.mj_forward(model, data)
        if not np.all(np.isfinite(np.r_[data.qpos, data.qvel])):
            raise FloatingPointError(f"nonfinite simulator state in episode {episode}")
        if not np.isclose(data.time, step * dt, atol=1e-7):
            raise RuntimeError("simulator time reset or drifted; refusing corrupted episode")
        rotation = data.xmat[torso].reshape(3, 3)
        frame = rolling_deploy_frame_3d(
            np, angular_velocity_body=rotation.T @ data.cvel[torso, :3],
            projected_gravity=rotation.T @ np.array([0., 0., -1.]),
            joint_position_offset=data.qpos[joint_qpos] - compact,
            last_action=previous_action,
        )
        frame = (frame + bias + rng.normal(size=36) * sigma).astype(np.float32)
        frame[3:6] /= max(np.linalg.norm(frame[3:6]), 1e-8)
        records["frames"].append(frame)
        # Free-joint translational qvel is root-origin velocity, not cvel's
        # COM-based spatial linear component (which would give a wrong label).
        records["velocity_world"].append(data.qvel[root_dof:root_dof + 3].copy())
        records["body_y_world"].append(rotation[:, 1].copy())
        records["position_world"].append(data.qpos[root_qpos:root_qpos + 3].copy())
        records["time_s"].append(float(data.time))
        records["motor_target"].append(data.ctrl[actuators].copy())
        records["cem_phase"].append(phase)
        records["rolling_phase"].append(rolling_phase)
        for _ in range(substeps):
            phase = float(bridge.advance_oscillator(np, rolling_phase, phase, physics_dt,
                                                    reference, rate_scale=settings["phase_rate_scale"]))
            amplitude = bridge.startup_target_scale(
                float(data.time), target_scale=target_scale, startup_scale=0.,
                ramp_duration_s=.25, startup_boost=0., startup_boost_duration_s=.25,
            )
            steering_ramp = np.clip((data.time - 1.) / 1., 0., 1.)
            target = cem_motor_target(phase, reference, compact, low, high,
                                      amplitude, steering * steering_ramp)
            data.ctrl[actuators] = target
            mujoco.mj_step(model, data)
            rolling_phase += float(data.qvel[root_dof + 4]) * physics_dt
            self_contact_steps += int(any(floor not in (int(c.geom1), int(c.geom2))
                                          for c in data.contact))
        previous_action = encode_motor_target(target, compact)
    records = {key: np.asarray(value) for key, value in records.items()}
    records["episode"] = np.full(steps, episode, np.int32)
    return records, dict(episode=episode, controller=controller_path, source="cem", yaw=yaw,
                         friction_scale=friction_scale, steering_bias=steering,
                         target_scale=target_scale, motor_kp_scale=gain_scale.tolist(),
                         compact_joint_position=compact.tolist(),
                         rolling_turns=rolling_phase / (2 * np.pi),
                         self_contact_fraction=self_contact_steps / (steps * substeps),
                         speed_statistics=speed_statistics(records),
                         wall_seconds=time.perf_counter() - start)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, action="append",
                        help="phase_locked_oscillator CEM JSON; repeat for multiple references")
    parser.add_argument("--model", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--control-hz", type=float, default=52.)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dr-strength", type=float, default=.5)
    parser.add_argument("--observation-noise", type=float, default=1.)
    parser.add_argument("--steering-bias", type=float, default=.03)
    parser.add_argument("--target-scale-range", type=float, nargs=2, default=[.45, 1.])
    parser.add_argument("--phase-rate-scale", type=float, default=1.)
    parser.add_argument("--physics-profile", choices=bridge.PHYSICS_PROFILE_NAMES_3D, default="cg20")
    parser.add_argument("--kp", type=float, default=5.)
    parser.add_argument("--kd", type=float, default=.1)
    parser.add_argument("--torque-limit", type=float, default=3.)
    args = parser.parse_args(argv)
    args.controller = args.controller or [DEFAULT_CONTROLLER]
    for name in ("episodes", "duration", "control_hz", "substeps", "workers", "kp", "torque_limit", "phase_rate_scale"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    for name in ("observation_noise", "steering_bias", "dr_strength", "kd"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.dr_strength > 1 or not np.all(np.isfinite(args.target_scale_range)):
        parser.error("dr-strength must be <= 1 and target scales must be finite")
    if not 0 < args.target_scale_range[0] <= args.target_scale_range[1] <= 1:
        parser.error("require 0 < minimum target scale <= maximum <= 1")
    if round(args.duration * args.control_hz) < 2:
        parser.error("duration must allow at least two control steps")
    if args.seed < 0 or args.out.suffix != ".npz":
        parser.error("seed must be nonnegative and output must end in .npz")
    return args


def main(argv=None):
    args = parse_args(argv)
    settings = {key: value for key, value in vars(args).items() if key not in ("controller", "out", "model")}
    settings.update(source="cem", controllers=[str(p.resolve()) for p in args.controller], model=str(args.model.resolve()))
    fingerprints = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*args.controller, args.model]}
    started = time.perf_counter()
    jobs = [(settings, episode) for episode in range(args.episodes)]
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    episodes, summaries = [], []
    try:
        results = pool.map(collect_episode, jobs) if pool else map(collect_episode, jobs)
        for records, summary in results:
            episodes.append(records)
            summaries.append(summary)
            print(f"episode {len(episodes)}/{args.episodes}: scale={summary['target_scale']:.3f}, "
                  f"mean_v={summary['speed_statistics']['time_mean_m_s']:.3f} m/s, "
                  f"rolls={summary['rolling_turns']:.2f}, wall={summary['wall_seconds']:.2f}s", flush=True)
    finally:
        if pool:
            pool.shutdown()
    metadata = dict(schema_version=1, source="cem", control_dt=1 / args.control_hz,
                    controller_update_dt=1 / (args.control_hz * args.substeps),
                    settings=settings, sha256=fingerprints, episodes=summaries,
                    wall_seconds=time.perf_counter() - started,
                    mujoco_version=mujoco.__version__,
                    observation_contract=dict(joint_names=list(CONTROLLER_JOINT_NAMES_3D),
                                              joint_offset_origin="model compact keyframe",
                                              compact_joint_position=summaries[0]["compact_joint_position"],
                                              action_scales=ACTION_SCALES.tolist(),
                                              last_action="latest physical CEM target encoded about compact"),
                    velocity_reference="root free-joint origin, world coordinates",
                    frame_contract="raw 36-value deployment frame, observation before current action")
    arrays = {key: np.concatenate([episode[key] for episode in episodes]) for key in episodes[0]}
    metadata["speed_statistics"] = speed_statistics(arrays)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays, metadata_json=np.array(json.dumps(metadata)))
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    simulated = len(arrays["frames"]) / args.control_hz
    print(f"saved {args.out}; {simulated:.1f} simulated s / {metadata['wall_seconds']:.1f} wall s", flush=True)


if __name__ == "__main__":
    main()
