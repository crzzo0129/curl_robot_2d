"""CPU smoke test for lifting the 2-D CEM reference to the 3-D curl model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceConfig,
    advance_oscillator,
    load_cem_reference,
    wrapped_phase_error,
)
from curl_robot_2d_mjx.config_3d import (
    PHYSICS_PROFILE_NAMES_3D,
    Rolling3DConfig,
    physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import apply_physics_options_3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_PATH = (
    PROJECT_ROOT
    / "results"
    / "collision_constrained_cem_foot_gap_2mm_short_contact"
    / "best_phase_controller.json"
)
DEFAULT_XML_PATH = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"

PLANAR_COMPACT = np.asarray(
    (
        FIXED_PARAMETERS.compact_hip_angle,
        FIXED_PARAMETERS.compact_knee_angle,
        FIXED_PARAMETERS.compact_hip_angle,
        FIXED_PARAMETERS.compact_knee_angle,
    ),
    dtype=np.float64,
)
PLANAR_JOINT_LOW = np.asarray(
    (
        FIXED_PARAMETERS.hip.shell_compatible_range[0],
        FIXED_PARAMETERS.knee.shell_compatible_range[0],
        FIXED_PARAMETERS.hip.shell_compatible_range[0],
        FIXED_PARAMETERS.knee.shell_compatible_range[0],
    ),
    dtype=np.float64,
)
PLANAR_JOINT_HIGH = np.asarray(
    (
        FIXED_PARAMETERS.hip.shell_compatible_range[1],
        FIXED_PARAMETERS.knee.shell_compatible_range[1],
        FIXED_PARAMETERS.hip.shell_compatible_range[1],
        FIXED_PARAMETERS.knee.shell_compatible_range[1],
    ),
    dtype=np.float64,
)
FOOT_GEOM_NAMES_3D = (
    "front_left_foot_proxy",
    "front_right_foot_proxy",
    "rear_left_foot_proxy",
    "rear_right_foot_proxy",
)


def planar_cem_target(
    phase_rad: float,
    config: CEMReferenceConfig,
    *,
    apply_foot_gap_projection: bool = True,
) -> np.ndarray:
    """Return physical 2-D joint targets: front hip/knee, rear hip/knee."""

    coefficients = np.asarray(config.coefficients, dtype=np.float64)
    sine = coefficients[0::2]
    cosine = coefficients[1::2]
    target = PLANAR_COMPACT + np.asarray(
        (0.0, config.knee_bias_rad, 0.0, config.knee_bias_rad),
        dtype=np.float64,
    )
    target = target + sine * math.sin(phase_rad) + cosine * math.cos(phase_rad)
    if apply_foot_gap_projection and config.minimum_foot_surface_gap_m > 0.0:
        target = _project_to_minimum_foot_gap(target, config)
    return np.clip(target, PLANAR_JOINT_LOW, PLANAR_JOINT_HIGH)


def scaled_planar_target(target: np.ndarray, target_scale: float) -> np.ndarray:
    """Scale the periodic component around the compact 2-D pose."""

    if target_scale < 0.0:
        raise ValueError("target_scale must be nonnegative")
    return PLANAR_COMPACT + target_scale * (np.asarray(target) - PLANAR_COMPACT)


def startup_target_scale(
    elapsed_s: float,
    *,
    target_scale: float,
    startup_boost: float,
    startup_boost_duration_s: float,
) -> float:
    if startup_boost_duration_s <= 0.0:
        boost_decay = 0.0
    else:
        normalized = np.clip(elapsed_s / startup_boost_duration_s, 0.0, 1.0)
        ramp = normalized * normalized * (3.0 - 2.0 * normalized)
        boost_decay = 1.0 - ramp
    return float(target_scale * (1.0 + startup_boost * boost_decay))


def map_planar_to_curl_3d_targets(planar_target: np.ndarray) -> np.ndarray:
    """Copy one 2-D sagittal reference to the left and right side rails."""

    front_hip, front_knee, rear_hip, rear_knee = np.asarray(
        planar_target, dtype=np.float64
    )
    return np.asarray(
        (
            front_hip,
            front_knee,
            front_hip,
            front_knee,
            rear_hip,
            rear_knee,
            rear_hip,
            rear_knee,
        ),
        dtype=np.float64,
    )


def _project_to_minimum_foot_gap(
    target: np.ndarray,
    config: CEMReferenceConfig,
) -> np.ndarray:
    length = FIXED_PARAMETERS.edge_length
    target_distance = (
        2.0 * FIXED_PARAMETERS.foot_radius
        + config.minimum_foot_surface_gap_m
        + config.foot_gap_tracking_margin_m
    )
    projected = np.asarray(target, dtype=np.float64).copy()
    for _ in range(6):
        front_hip, front_knee, rear_hip, rear_knee = projected
        delta_x = FIXED_PARAMETERS.edge_length + length * (
            math.sin(front_hip)
            + math.sin(front_hip - front_knee)
            + math.sin(rear_hip)
            + math.sin(rear_hip - rear_knee)
        )
        delta_z = length * (
            -math.cos(front_hip)
            - math.cos(front_knee - front_hip)
            + math.cos(rear_hip)
            + math.cos(rear_knee - rear_hip)
        )
        distance = math.sqrt(delta_x * delta_x + delta_z * delta_z)
        front_dx = -length * math.cos(front_hip - front_knee)
        front_dz = -length * math.sin(front_knee - front_hip)
        rear_dx = -length * math.cos(rear_hip - rear_knee)
        rear_dz = -length * math.sin(rear_knee - rear_hip)
        front_gradient = (delta_x * front_dx + delta_z * front_dz) / max(
            distance, 1.0e-6
        )
        rear_gradient = (delta_x * rear_dx + delta_z * rear_dz) / max(
            distance, 1.0e-6
        )
        gradient_norm_squared = (
            front_gradient * front_gradient + rear_gradient * rear_gradient
        )
        scale = max(target_distance - distance, 0.0) / max(
            gradient_norm_squared, 1.0e-8
        )
        projected[1] += float(np.clip(scale * front_gradient, -0.20, 0.20))
        projected[3] += float(np.clip(scale * rear_gradient, -0.20, 0.20))
    return projected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER_PATH)
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILE_NAMES_3D,
        default="reference",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--initial-phase-rad", type=float, default=0.0)
    parser.add_argument("--phase-rate-scale", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--startup-target-boost", type=float, default=0.0)
    parser.add_argument(
        "--startup-target-boost-duration-s",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--linear-phase",
        action="store_true",
        help="Use the old fixed-rate phase update for an A/B comparison.",
    )
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-foot-gap-projection",
        action="store_true",
        help="Use the raw sinusoid without the 2-D foot-gap projection.",
    )
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> dict[str, float | str | bool]:
    if args.duration <= 0.0 or args.control_dt <= 0.0:
        raise SystemExit("--duration and --control-dt must be positive")
    if args.kp < 0.0 or args.kd < 0.0 or args.torque_limit <= 0.0:
        raise SystemExit("--kp/--kd must be nonnegative and --torque-limit positive")
    if not math.isfinite(args.phase_rate_scale):
        raise SystemExit("--phase-rate-scale must be finite")
    if not math.isfinite(args.target_scale) or args.target_scale < 0.0:
        raise SystemExit("--target-scale must be nonnegative")
    if (
        not math.isfinite(args.startup_target_boost)
        or args.startup_target_boost < 0.0
    ):
        raise SystemExit("--startup-target-boost must be nonnegative")
    if (
        not math.isfinite(args.startup_target_boost_duration_s)
        or args.startup_target_boost_duration_s <= 0.0
    ):
        raise SystemExit("--startup-target-boost-duration-s must be positive")
    if not args.xml.exists():
        raise SystemExit(f"3-D XML not found: {args.xml}")
    if not args.controller.exists():
        raise SystemExit(f"CEM controller not found: {args.controller}")

    import mujoco

    config = load_cem_reference(args.controller)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    task = physics_profile_3d(args.physics_profile, Rolling3DConfig())
    apply_physics_options_3d(model, task)
    data = mujoco.MjData(model)
    joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES_3D])
    qpos_indices = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    actuator_ids = np.asarray(
        [model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D]
    )
    ctrl_low = np.asarray(model.actuator_ctrlrange[actuator_ids, 0], dtype=np.float64)
    ctrl_high = np.asarray(model.actuator_ctrlrange[actuator_ids, 1], dtype=np.float64)
    torso_body_id = model.body("torso").id
    floor_geom_id = model.geom("floor").id
    foot_geom_ids = {model.geom(name).id for name in FOOT_GEOM_NAMES_3D}
    shell_geom_ids = _shell_geom_ids(model, mujoco)

    model.actuator_gainprm[actuator_ids, 0] = args.kp
    model.actuator_biasprm[actuator_ids, 1] = -args.kp
    model.actuator_biasprm[actuator_ids, 2] = -args.kd
    model.actuator_forcerange[actuator_ids, 0] = -args.torque_limit
    model.actuator_forcerange[actuator_ids, 1] = args.torque_limit

    initial_planar = planar_cem_target(
        args.initial_phase_rad,
        config,
        apply_foot_gap_projection=not args.no_foot_gap_projection,
    )
    initial_planar = scaled_planar_target(
        initial_planar,
        startup_target_scale(
            0.0,
            target_scale=args.target_scale,
            startup_boost=args.startup_target_boost,
            startup_boost_duration_s=args.startup_target_boost_duration_s,
        ),
    )
    initial_ctrl = np.clip(
        map_planar_to_curl_3d_targets(initial_planar),
        ctrl_low,
        ctrl_high,
    )
    _reset_data(model, data, mujoco, qpos_indices, actuator_ids, initial_ctrl)

    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])
    phase = float(args.initial_phase_rad)
    rolling_phase = 0.0
    absolute_rolling_phase = 0.0
    control_repeat = max(1, round(args.control_dt / model.opt.timestep))
    control_dt = control_repeat * float(model.opt.timestep)
    steps = max(1, round(args.duration / control_dt))
    records = []
    nonfinite = False
    for _ in range(steps):
        saturated = []
        shell_contacts = []
        foot_contacts = []
        self_contacts = []
        phase_rates = []
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
            planar = planar_cem_target(
                phase,
                config,
                apply_foot_gap_projection=not args.no_foot_gap_projection,
            )
            planar = scaled_planar_target(
                planar,
                startup_target_scale(
                    float(data.time),
                    target_scale=args.target_scale,
                    startup_boost=args.startup_target_boost,
                    startup_boost_duration_s=args.startup_target_boost_duration_s,
                ),
            )
            ctrl = np.clip(
                map_planar_to_curl_3d_targets(planar),
                ctrl_low,
                ctrl_high,
            )
            data.ctrl[actuator_ids] = ctrl
            mujoco.mj_step(model, data)
            rolling_increment = float(data.qvel[4]) * float(model.opt.timestep)
            rolling_phase += rolling_increment
            absolute_rolling_phase += abs(rolling_increment)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                nonfinite = True
                break
            phase_rates.append(
                (
                    args.phase_rate_scale * config.oscillator_rate_rad_s
                    if args.linear_phase
                    else (phase - previous_phase) / float(model.opt.timestep)
                )
            )
            saturated.append(
                np.mean(
                    np.abs(data.actuator_force[actuator_ids])
                    >= 0.99 * args.torque_limit
                )
            )
            shell_contact, foot_contact, self_contact = _contact_flags(
                data,
                floor_geom_id,
                shell_geom_ids,
                foot_geom_ids,
            )
            shell_contacts.append(shell_contact)
            foot_contacts.append(foot_contact)
            self_contacts.append(self_contact)
        if args.linear_phase:
            phase += (
                args.phase_rate_scale
                * config.oscillator_rate_rad_s
                * control_dt
            )
        rotation = data.xmat[torso_body_id].reshape(3, 3)
        roll, pitch, yaw = _rpy_from_rotation(rotation)
        rolling_axis_tilt = _rolling_axis_tilt(rotation)
        tracking_rmse = float(np.sqrt(np.mean(np.square(ctrl - data.qpos[qpos_indices]))))
        records.append(
            (
                float(data.qpos[0]),
                float(data.qpos[1]),
                float(data.qpos[2]),
                roll,
                pitch,
                yaw,
                rolling_axis_tilt,
                tracking_rmse,
                float(np.mean(saturated)) if saturated else 1.0,
                float(np.mean(shell_contacts)) if shell_contacts else 0.0,
                float(np.mean(foot_contacts)) if foot_contacts else 0.0,
                float(np.mean(self_contacts)) if self_contacts else 0.0,
                phase,
                rolling_phase,
                float(wrapped_phase_error(np, rolling_phase, phase)),
                float(np.mean(phase_rates)) if phase_rates else 0.0,
            )
        )
        if nonfinite:
            break

    values = np.asarray(records, dtype=np.float64)
    elapsed = len(records) * control_dt
    if len(values) == 0:
        raise RuntimeError("simulation produced no records")
    distance_x = float(values[-1, 0] - start_x)
    distance_y = float(values[-1, 1] - start_y)
    status = "failed" if nonfinite else "ok"
    return {
        "status": status,
        "xml": str(args.xml.resolve()),
        "controller": str(args.controller.resolve()),
        "elapsed_s": float(elapsed),
        "control_dt_s": float(control_dt),
        "physics_profile": args.physics_profile,
        "solver": task.solver_name,
        "phase_lock_enabled": not args.linear_phase,
        "reference_turns": float(
            (phase - args.initial_phase_rad) / (2.0 * math.pi)
        ),
        "nominal_reference_turns": float(
            args.phase_rate_scale * config.oscillator_rate_rad_s * elapsed
            / (2.0 * math.pi)
        ),
        "rolling_phase_turns": float(rolling_phase / (2.0 * math.pi)),
        "absolute_rotation_turns": float(
            absolute_rolling_phase / (2.0 * math.pi)
        ),
        "phase_error_final_rad": float(values[-1, 14]),
        "phase_error_rms_rad": float(
            np.sqrt(np.mean(np.square(values[:, 14])))
        ),
        "phase_error_abs_max_rad": float(np.max(np.abs(values[:, 14]))),
        "oscillator_rate_mean_rad_s": float(np.mean(values[:, 15])),
        "oscillator_rate_min_rad_s": float(np.min(values[:, 15])),
        "oscillator_rate_max_rad_s": float(np.max(values[:, 15])),
        "distance_x_m": distance_x,
        "distance_y_m": distance_y,
        "distance_as_shell_turns": float(
            distance_x / max(2.0 * math.pi * FIXED_PARAMETERS.shell_contact_radius, 1.0e-9)
        ),
        "body_z_min_m": float(np.min(values[:, 2])),
        "roll_rms_rad": float(np.sqrt(np.mean(np.square(values[:, 3])))),
        "pitch_rms_rad": float(np.sqrt(np.mean(np.square(values[:, 4])))),
        "yaw_final_rad": float(values[-1, 5]),
        "yaw_abs_max_rad": float(np.max(np.abs(values[:, 5]))),
        "rolling_axis_tilt_rms_rad": float(np.sqrt(np.mean(np.square(values[:, 6])))),
        "rolling_axis_tilt_max_rad": float(np.max(values[:, 6])),
        "tracking_rmse_rad": float(np.mean(values[:, 7])),
        "torque_saturation_fraction": float(np.mean(values[:, 8])),
        "shell_floor_contact_fraction": float(np.mean(values[:, 9])),
        "foot_floor_contact_fraction": float(np.mean(values[:, 10])),
        "self_contact_fraction": float(np.mean(values[:, 11])),
        "nonfinite": bool(nonfinite),
        "target_scale": float(args.target_scale),
        "startup_target_boost": float(args.startup_target_boost),
        "startup_target_boost_duration_s": (
            float(args.startup_target_boost_duration_s)
        ),
        "phase_rate_scale": float(args.phase_rate_scale),
    }


def _reset_data(model, data, mujoco, qpos_indices, actuator_ids, ctrl) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
    data.qpos[qpos_indices] = ctrl
    data.qvel[:] = 0.0
    data.ctrl[actuator_ids] = ctrl
    mujoco.mj_forward(model, data)


def _shell_geom_ids(model, mujoco) -> set[int]:
    geom_ids = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "_shell_" in name:
            geom_ids.add(int(geom_id))
    return geom_ids


def _contact_flags(
    data,
    floor_geom_id: int,
    shell_geom_ids: set[int],
    foot_geom_ids: set[int],
) -> tuple[float, float, float]:
    shell_floor = False
    foot_floor = False
    self_contact = False
    for contact in data.contact:
        if contact.dist > 0.005:
            continue
        pair = {int(contact.geom[0]), int(contact.geom[1])}
        has_floor = floor_geom_id in pair
        shell_floor |= has_floor and bool(pair.intersection(shell_geom_ids))
        foot_floor |= has_floor and bool(pair.intersection(foot_geom_ids))
        self_contact |= not has_floor
    return float(shell_floor), float(foot_floor), float(self_contact)


def _rpy_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw


def _rolling_axis_tilt(rotation: np.ndarray) -> float:
    body_y_axis = rotation[:, 1]
    alignment = float(np.clip(abs(body_y_axis[1]), 0.0, 1.0))
    return math.acos(alignment)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_smoke(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
