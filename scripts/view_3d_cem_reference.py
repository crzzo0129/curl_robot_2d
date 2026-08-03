from __future__ import annotations

import argparse
import json
import math
import time

import mujoco
import numpy as np

from scripts.evaluate_3d_symmetric_cem_reference import (
    DEFAULT_CONTROLLER_PATH,
    DEFAULT_XML_PATH,
    map_planar_to_curl_3d_targets,
    planar_cem_target,
    scaled_planar_target,
)
from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.cem_reference import (
    advance_oscillator,
    load_cem_reference,
    wrapped_phase_error,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=DEFAULT_XML_PATH)
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER_PATH)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--initial-phase-rad", type=float, default=0.0)
    parser.add_argument("--phase-rate-scale", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--linear-phase", action="store_true")
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--realtime", type=float, default=1.0)
    parser.add_argument("--camera-distance", type=float, default=0.9)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def _reset(model, data, qpos_indices, actuator_ids, ctrl):
    mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
    data.qpos[qpos_indices] = ctrl
    data.qvel[:] = 0.0
    data.ctrl[actuator_ids] = ctrl
    mujoco.mj_forward(model, data)


def _rolling_axis_tilt(rotation):
    body_y_axis = rotation[:, 1]
    alignment = float(np.clip(abs(body_y_axis[1]), 0.0, 1.0))
    return math.acos(alignment)


def _target_for_phase(phase, config, target_scale, ctrl_low, ctrl_high):
    planar = planar_cem_target(phase, config)
    planar = scaled_planar_target(planar, target_scale)
    return np.clip(map_planar_to_curl_3d_targets(planar), ctrl_low, ctrl_high)


def run(argv=None):
    args = parse_args(argv)
    if args.duration <= 0.0 or args.control_dt <= 0.0 or args.realtime <= 0.0:
        raise SystemExit("--duration, --control-dt and --realtime must be positive")
    if args.kp < 0.0 or args.kd < 0.0 or args.torque_limit <= 0.0:
        raise SystemExit("--kp/--kd must be nonnegative and --torque-limit positive")

    config = load_cem_reference(args.controller)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES_3D])
    qpos_indices = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    actuator_ids = np.asarray(
        [model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D]
    )
    ctrl_low = np.asarray(model.actuator_ctrlrange[actuator_ids, 0], dtype=np.float64)
    ctrl_high = np.asarray(model.actuator_ctrlrange[actuator_ids, 1], dtype=np.float64)
    model.actuator_gainprm[actuator_ids, 0] = args.kp
    model.actuator_biasprm[actuator_ids, 1] = -args.kp
    model.actuator_biasprm[actuator_ids, 2] = -args.kd
    model.actuator_forcerange[actuator_ids, 0] = -args.torque_limit
    model.actuator_forcerange[actuator_ids, 1] = args.torque_limit

    phase = float(args.initial_phase_rad)
    rolling_phase = 0.0
    initial_ctrl = _target_for_phase(phase, config, args.target_scale, ctrl_low, ctrl_high)
    _reset(model, data, qpos_indices, actuator_ids, initial_ctrl)
    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])
    torso_id = model.body("torso").id
    control_repeat = max(1, round(args.control_dt / model.opt.timestep))
    control_dt = control_repeat * float(model.opt.timestep)
    control_steps = max(1, round(args.duration / control_dt))
    tilt_values = []
    saturation_values = []
    phase_rate_values = []

    viewer_context = _null_viewer()
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(model, data)

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = torso_id
            viewer.cam.azimuth = args.azimuth
            viewer.cam.elevation = args.elevation
            viewer.cam.distance = args.camera_distance
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        for step in range(control_steps):
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.perf_counter()
            for _ in range(control_repeat):
                previous_phase = phase
                if not args.linear_phase:
                    phase = float(
                        advance_oscillator(
                            np,
                            rolling_phase,
                            phase,
                            float(model.opt.timestep),
                            config,
                            rate_scale=args.phase_rate_scale,
                        )
                    )
                ctrl = _target_for_phase(
                    phase,
                    config,
                    args.target_scale,
                    ctrl_low,
                    ctrl_high,
                )
                data.ctrl[actuator_ids] = ctrl
                mujoco.mj_step(model, data)
                rolling_phase += float(data.qvel[4]) * float(model.opt.timestep)
                phase_rate_values.append(
                    args.phase_rate_scale * config.oscillator_rate_rad_s
                    if args.linear_phase
                    else (phase - previous_phase) / float(model.opt.timestep)
                )
            rotation = data.xmat[torso_id].reshape(3, 3)
            tilt_values.append(_rolling_axis_tilt(rotation))
            saturation_values.append(
                float(
                    np.mean(
                        np.abs(data.actuator_force[actuator_ids])
                        >= 0.99 * args.torque_limit
                    )
                )
            )
            if args.linear_phase:
                phase += (
                    args.phase_rate_scale
                    * config.oscillator_rate_rad_s
                    * control_dt
                )
            if viewer is not None:
                viewer.sync()
                remaining = control_dt / args.realtime - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)

    elapsed = len(tilt_values) * control_dt
    return {
        "status": "ok",
        "elapsed_s": float(elapsed),
        "distance_x_m": float(data.qpos[0] - start_x),
        "distance_y_m": float(data.qpos[1] - start_y),
        "distance_as_shell_turns": float(
            (data.qpos[0] - start_x)
            / max(2.0 * math.pi * FIXED_PARAMETERS.shell_contact_radius, 1.0e-9)
        ),
        "phase_lock_enabled": not args.linear_phase,
        "reference_turns": float(
            (phase - args.initial_phase_rad) / (2.0 * math.pi)
        ),
        "rolling_phase_turns": float(rolling_phase / (2.0 * math.pi)),
        "phase_error_final_rad": float(
            wrapped_phase_error(np, rolling_phase, phase)
        ),
        "oscillator_rate_mean_rad_s": float(np.mean(phase_rate_values)),
        "rolling_axis_tilt_rms_rad": float(np.sqrt(np.mean(np.square(tilt_values)))),
        "rolling_axis_tilt_max_rad": float(np.max(tilt_values)),
        "torque_saturation_fraction": float(np.mean(saturation_values)),
    }


class _null_viewer:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main(argv=None):
    print(json.dumps(run(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
