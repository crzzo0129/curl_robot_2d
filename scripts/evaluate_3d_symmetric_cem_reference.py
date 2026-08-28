"""CPU smoke test for lifting the 2-D CEM reference to the 3-D curl model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import (
    FIXED_PARAMETERS,
    PUPPER_ORIGINAL_SHELL_60_PARAMETERS,
    REAL_GEOMETRY_PARAMETERS,
    FixedParameters,
)
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
from curl_robot_2d_mjx.environment_3d import (
    DEFAULT_3D_CEM_CONTROLLER,
    ROLLINGQUAD_2_MODEL_PATH_3D,
    apply_physics_options_3d,
    configure_pupper_shell_collisions_3d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_PATH = DEFAULT_3D_CEM_CONTROLLER
DEFAULT_XML_PATH = ROLLINGQUAD_2_MODEL_PATH_3D

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


def activate_planar_geometry(parameters: FixedParameters) -> None:
    """Match lifted-reference kinematics to the selected 3-D geometry."""

    global FIXED_PARAMETERS, PLANAR_COMPACT, PLANAR_JOINT_LOW, PLANAR_JOINT_HIGH
    FIXED_PARAMETERS = parameters
    PLANAR_COMPACT = np.asarray(
        (
            parameters.compact_hip_angle,
            parameters.compact_knee_angle,
            parameters.compact_hip_angle,
            parameters.compact_knee_angle,
        ),
        dtype=np.float64,
    )
    PLANAR_JOINT_LOW = np.asarray(
        (
            parameters.hip.shell_compatible_range[0],
            parameters.knee.shell_compatible_range[0],
            parameters.hip.shell_compatible_range[0],
            parameters.knee.shell_compatible_range[0],
        ),
        dtype=np.float64,
    )
    PLANAR_JOINT_HIGH = np.asarray(
        (
            parameters.hip.shell_compatible_range[1],
            parameters.knee.shell_compatible_range[1],
            parameters.hip.shell_compatible_range[1],
            parameters.knee.shell_compatible_range[1],
        ),
        dtype=np.float64,
    )


def override_reference_foot_gap(config, minimum_gap_mm, tracking_margin_mm):
    """Apply optional playback-only foot clearance using active geometry."""

    if minimum_gap_mm is None and tracking_margin_mm is None:
        return config
    minimum_gap_m = (
        config.minimum_foot_surface_gap_m
        if minimum_gap_mm is None
        else minimum_gap_mm / 1000.0
    )
    tracking_margin_m = (
        config.foot_gap_tracking_margin_m
        if tracking_margin_mm is None
        else tracking_margin_mm / 1000.0
    )
    separated = replace(
        FIXED_PARAMETERS,
        compact_foot_surface_gap=minimum_gap_m,
    )
    return replace(
        config,
        minimum_foot_surface_gap_m=minimum_gap_m,
        foot_gap_tracking_margin_m=tracking_margin_m,
        knee_bias_rad=(
            separated.compact_knee_angle
            - FIXED_PARAMETERS.compact_knee_angle
        ),
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
    startup_scale: float | None,
    ramp_duration_s: float,
    startup_boost: float,
    startup_boost_duration_s: float,
) -> float:
    start_scale = target_scale if startup_scale is None else startup_scale
    normalized = np.clip(elapsed_s / ramp_duration_s, 0.0, 1.0)
    ramp = normalized * normalized * (3.0 - 2.0 * normalized)
    ramped_scale = start_scale + (target_scale - start_scale) * ramp
    if startup_boost_duration_s <= 0.0:
        boost_decay = 0.0
    else:
        normalized = np.clip(elapsed_s / startup_boost_duration_s, 0.0, 1.0)
        ramp = normalized * normalized * (3.0 - 2.0 * normalized)
        boost_decay = 1.0 - ramp
    return float(ramped_scale * (1.0 + startup_boost * boost_decay))


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
    upper = FIXED_PARAMETERS.upper_length
    lower = FIXED_PARAMETERS.lower_length
    target_distance = (
        2.0 * FIXED_PARAMETERS.foot_radius
        + config.minimum_foot_surface_gap_m
        + config.foot_gap_tracking_margin_m
    )
    projected = np.asarray(target, dtype=np.float64).copy()
    for _ in range(6):
        front_hip, front_knee, rear_hip, rear_knee = projected
        delta_x = FIXED_PARAMETERS.torso_length + (
            upper * math.sin(front_hip)
            + lower * math.sin(front_hip - front_knee)
            + upper * math.sin(rear_hip)
            + lower * math.sin(rear_hip - rear_knee)
        )
        delta_z = (
            -upper * math.cos(front_hip)
            - lower * math.cos(front_knee - front_hip)
            + upper * math.cos(rear_hip)
            + lower * math.cos(rear_knee - rear_hip)
        )
        distance = math.sqrt(delta_x * delta_x + delta_z * delta_z)
        front_dx = -lower * math.cos(front_hip - front_knee)
        front_dz = -lower * math.sin(front_knee - front_hip)
        rear_dx = -lower * math.cos(rear_hip - rear_knee)
        rear_dz = -lower * math.sin(rear_knee - rear_hip)
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
        "--geometry",
        choices=("baseline", "real", "pupper60", "rollingquad_2"),
        default="rollingquad_2",
    )
    parser.add_argument("--minimum-foot-gap-mm", type=float, default=None)
    parser.add_argument("--foot-gap-tracking-margin-mm", type=float, default=None)
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
    parser.add_argument(
        "--startup-target-scale",
        type=float,
        default=0.0,
        help=(
            "Periodic-reference scale at reset. The 0 default reproduces "
            "the 2-D controller's compact-to-reference 0.25 s ramp."
        ),
    )
    parser.add_argument(
        "--target-ramp-duration-s",
        type=float,
        default=0.25,
    )
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
    parser.add_argument(
        "--diagnose-self-collision",
        action="store_true",
        help=(
            "Replay each simulated pose through a shadow MuJoCo model with "
            "robot-robot collision enabled. This measures potential CAD "
            "self-contact without changing the reference trajectory."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--joint-plot",
        type=Path,
        default=None,
        help="Save actual/commanded angle curves for every hinge joint.",
    )
    parser.add_argument(
        "--joint-series-out",
        type=Path,
        default=None,
        help="Save the joint-angle time series as CSV.",
    )
    parser.add_argument(
        "--no-foot-gap-projection",
        action="store_true",
        help="Use the raw sinusoid without the 2-D foot-gap projection.",
    )
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    activate_planar_geometry({
        "baseline": FixedParameters(),
        "real": REAL_GEOMETRY_PARAMETERS,
        "pupper60": PUPPER_ORIGINAL_SHELL_60_PARAMETERS,
        "rollingquad_2": PUPPER_ORIGINAL_SHELL_60_PARAMETERS,
    }[args.geometry])
    if args.duration <= 0.0 or args.control_dt <= 0.0:
        raise SystemExit("--duration and --control-dt must be positive")
    if args.kp < 0.0 or args.kd < 0.0 or args.torque_limit <= 0.0:
        raise SystemExit("--kp/--kd must be nonnegative and --torque-limit positive")
    if not math.isfinite(args.phase_rate_scale):
        raise SystemExit("--phase-rate-scale must be finite")
    if not math.isfinite(args.target_scale) or args.target_scale < 0.0:
        raise SystemExit("--target-scale must be nonnegative")
    if args.startup_target_scale is not None:
        if (
            not math.isfinite(args.startup_target_scale)
            or args.startup_target_scale < 0.0
        ):
            raise SystemExit("--startup-target-scale must be nonnegative")
    if (
        not math.isfinite(args.target_ramp_duration_s)
        or args.target_ramp_duration_s <= 0.0
    ):
        raise SystemExit("--target-ramp-duration-s must be positive")
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

    config = override_reference_foot_gap(
        load_cem_reference(args.controller),
        args.minimum_foot_gap_mm,
        args.foot_gap_tracking_margin_mm,
    )
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    if args.geometry == "pupper60":
        configure_pupper_shell_collisions_3d(model, enabled=True)
    task = physics_profile_3d(args.physics_profile, Rolling3DConfig())
    apply_physics_options_3d(model, task)
    data = mujoco.MjData(model)
    joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES_3D])
    qpos_indices = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    actuator_ids = np.asarray(
        [model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D]
    )
    measured_joint_ids = np.asarray(
        [
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ],
        dtype=np.int32,
    )
    measured_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        or f"joint_{joint_id}"
        for joint_id in measured_joint_ids
    ]
    measured_qpos_indices = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in measured_joint_ids],
        dtype=np.int32,
    )
    measured_actuator_ids = np.asarray(
        [model.actuator(f"{name}_servo").id for name in measured_joint_names],
        dtype=np.int32,
    )
    ctrl_low = np.asarray(model.actuator_ctrlrange[actuator_ids, 0], dtype=np.float64)
    ctrl_high = np.asarray(model.actuator_ctrlrange[actuator_ids, 1], dtype=np.float64)
    torso_body_id = model.body("torso").id
    floor_geom_id = model.geom("floor").id
    foot_geom_ids = {model.geom(name).id for name in FOOT_GEOM_NAMES_3D}
    shell_geom_ids = _shell_geom_ids(model, mujoco, foot_geom_ids)
    self_collision_physics_enabled = _robot_self_collision_enabled(
        model, floor_geom_id
    )
    diagnostic_model = None
    diagnostic_data = None
    diagnostic_floor_geom_id = None
    if args.diagnose_self_collision:
        diagnostic_model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
        if args.geometry == "pupper60":
            configure_pupper_shell_collisions_3d(diagnostic_model, enabled=True)
        apply_physics_options_3d(diagnostic_model, task)
        diagnostic_floor_geom_id = diagnostic_model.geom("floor").id
        _enable_robot_self_collision(
            diagnostic_model, diagnostic_floor_geom_id
        )
        diagnostic_data = mujoco.MjData(diagnostic_model)

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
            startup_scale=args.startup_target_scale,
            ramp_duration_s=args.target_ramp_duration_s,
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
    joint_times = []
    actual_joint_angles = []
    commanded_joint_angles = []
    self_contact_pair_steps: dict[str, int] = {}
    self_contact_pair_max_penetration: dict[str, float] = {}
    potential_self_contact_fractions = []
    potential_self_contact_pair_steps: dict[str, int] = {}
    potential_self_contact_pair_max_penetration: dict[str, float] = {}
    nonfinite = False
    for _ in range(steps):
        saturated = []
        shell_contacts = []
        foot_contacts = []
        self_contacts = []
        potential_self_contacts = []
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
                    startup_scale=args.startup_target_scale,
                    ramp_duration_s=args.target_ramp_duration_s,
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
            active_pairs: dict[str, float] = {}
            for contact in data.contact:
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                if floor_geom_id in (geom1, geom2):
                    continue
                names = sorted(
                    (
                        mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, geom1
                        )
                        or f"geom_{geom1}",
                        mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, geom2
                        )
                        or f"geom_{geom2}",
                    )
                )
                pair_name = "__".join(names)
                active_pairs[pair_name] = max(
                    active_pairs.get(pair_name, 0.0),
                    max(-float(contact.dist), 0.0),
                )
            for pair_name, penetration in active_pairs.items():
                self_contact_pair_steps[pair_name] = (
                    self_contact_pair_steps.get(pair_name, 0) + 1
                )
                self_contact_pair_max_penetration[pair_name] = max(
                    self_contact_pair_max_penetration.get(pair_name, 0.0),
                    penetration,
                )
            if diagnostic_data is not None:
                diagnostic_data.qpos[:] = data.qpos
                diagnostic_data.qvel[:] = data.qvel
                diagnostic_data.time = data.time
                mujoco.mj_forward(diagnostic_model, diagnostic_data)
                potential_self_contacts.append(
                    _contact_flags(
                        diagnostic_data,
                        diagnostic_floor_geom_id,
                        set(),
                        set(),
                    )[2]
                )
                diagnostic_active_pairs: dict[str, float] = {}
                for contact in diagnostic_data.contact:
                    geom1 = int(contact.geom1)
                    geom2 = int(contact.geom2)
                    if diagnostic_floor_geom_id in (geom1, geom2):
                        continue
                    names = sorted(
                        (
                            mujoco.mj_id2name(
                                diagnostic_model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                geom1,
                            )
                            or f"geom_{geom1}",
                            mujoco.mj_id2name(
                                diagnostic_model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                geom2,
                            )
                            or f"geom_{geom2}",
                        )
                    )
                    pair_name = "__".join(names)
                    diagnostic_active_pairs[pair_name] = max(
                        diagnostic_active_pairs.get(pair_name, 0.0),
                        max(-float(contact.dist), 0.0),
                    )
                for pair_name, penetration in diagnostic_active_pairs.items():
                    potential_self_contact_pair_steps[pair_name] = (
                        potential_self_contact_pair_steps.get(pair_name, 0) + 1
                    )
                    potential_self_contact_pair_max_penetration[pair_name] = max(
                        potential_self_contact_pair_max_penetration.get(
                            pair_name, 0.0
                        ),
                        penetration,
                    )
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
        joint_times.append(float(data.time))
        if potential_self_contacts:
            potential_self_contact_fractions.append(
                float(np.mean(potential_self_contacts))
            )
        actual_joint_angles.append(data.qpos[measured_qpos_indices].copy())
        commanded_joint_angles.append(data.ctrl[measured_actuator_ids].copy())
        if nonfinite:
            break

    values = np.asarray(records, dtype=np.float64)
    elapsed = len(records) * control_dt
    if len(values) == 0:
        raise RuntimeError("simulation produced no records")
    joint_times_array = np.asarray(joint_times, dtype=np.float64)
    actual_joint_angles_array = np.asarray(actual_joint_angles, dtype=np.float64)
    commanded_joint_angles_array = np.asarray(
        commanded_joint_angles, dtype=np.float64
    )
    joint_ranges_deg = {}
    for column, (joint_id, name) in enumerate(
        zip(measured_joint_ids, measured_joint_names, strict=True)
    ):
        actual_deg = np.rad2deg(actual_joint_angles_array[:, column])
        commanded_deg = np.rad2deg(commanded_joint_angles_array[:, column])
        limited = bool(model.jnt_limited[joint_id])
        limit_deg = (
            np.rad2deg(model.jnt_range[joint_id]).tolist() if limited else None
        )
        joint_ranges_deg[name] = {
            "actual_min": float(np.min(actual_deg)),
            "actual_max": float(np.max(actual_deg)),
            "actual_peak_to_peak": float(np.ptp(actual_deg)),
            "commanded_min": float(np.min(commanded_deg)),
            "commanded_max": float(np.max(commanded_deg)),
            "xml_limit": limit_deg,
        }

    if args.joint_series_out is not None:
        _write_joint_series(
            args.joint_series_out,
            joint_times_array,
            measured_joint_names,
            actual_joint_angles_array,
            commanded_joint_angles_array,
        )
    if args.joint_plot is not None:
        _plot_joint_angles(
            args.joint_plot,
            joint_times_array,
            measured_joint_names,
            measured_joint_ids,
            actual_joint_angles_array,
            commanded_joint_angles_array,
            model,
        )
    distance_x = float(values[-1, 0] - start_x)
    distance_y = float(values[-1, 1] - start_y)
    status = "failed" if nonfinite else "ok"
    return {
        "status": status,
        "geometry": args.geometry,
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
            distance_x
            / max(
                2.0 * math.pi * FIXED_PARAMETERS.shell_contact_radius,
                1.0e-9,
            )
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
        "self_collision_physics_enabled": bool(self_collision_physics_enabled),
        "self_contact_pairs": {
            pair_name: {
                "duration_s": steps * float(model.opt.timestep),
                "maximum_penetration_m": self_contact_pair_max_penetration[
                    pair_name
                ],
            }
            for pair_name, steps in sorted(self_contact_pair_steps.items())
        },
        "potential_self_contact_fraction": (
            None
            if not args.diagnose_self_collision
            else float(np.mean(potential_self_contact_fractions))
        ),
        "potential_self_contact_pairs": {
            pair_name: {
                "duration_s": steps * float(model.opt.timestep),
                "maximum_penetration_m": (
                    potential_self_contact_pair_max_penetration[pair_name]
                ),
            }
            for pair_name, steps in sorted(
                potential_self_contact_pair_steps.items()
            )
        },
        "nonfinite": bool(nonfinite),
        "target_scale": float(args.target_scale),
        "startup_target_scale": (
            None
            if args.startup_target_scale is None
            else float(args.startup_target_scale)
        ),
        "target_ramp_duration_s": float(args.target_ramp_duration_s),
        "startup_target_boost": float(args.startup_target_boost),
        "startup_target_boost_duration_s": (
            float(args.startup_target_boost_duration_s)
        ),
        "phase_rate_scale": float(args.phase_rate_scale),
        "joint_angle_ranges_deg": joint_ranges_deg,
        "joint_plot": (
            None if args.joint_plot is None else str(args.joint_plot.resolve())
        ),
        "joint_series": (
            None
            if args.joint_series_out is None
            else str(args.joint_series_out.resolve())
        ),
    }


def _write_joint_series(
    path: Path,
    times: np.ndarray,
    joint_names: list[str],
    actual_angles: np.ndarray,
    commanded_angles: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [times]
    header = ["time_s"]
    for column, name in enumerate(joint_names):
        columns.extend(
            (
                np.rad2deg(actual_angles[:, column]),
                np.rad2deg(commanded_angles[:, column]),
            )
        )
        header.extend((f"{name}_actual_deg", f"{name}_commanded_deg"))
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def _plot_joint_angles(
    path: Path,
    times: np.ndarray,
    joint_names: list[str],
    joint_ids: np.ndarray,
    actual_angles: np.ndarray,
    commanded_angles: np.ndarray,
    model,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    columns = 3
    rows = math.ceil(len(joint_names) / columns)
    cell_width, cell_height = 500, 310
    top_margin = 80
    image = Image.new(
        "RGB", (columns * cell_width, top_margin + rows * cell_height), "white"
    )
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        panel_font = ImageFont.truetype("arial.ttf", 16)
        label_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        title_font = panel_font = label_font = ImageFont.load_default()

    actual_color = (0, 104, 181)
    commanded_color = (229, 107, 31)
    limit_color = (115, 115, 115)
    grid_color = (220, 224, 228)
    draw.text((25, 15), "3-D rolling joint angles", fill=(25, 25, 25), font=title_font)
    draw.line((780, 29, 820, 29), fill=actual_color, width=3)
    draw.text((828, 20), "actual", fill=(30, 30, 30), font=label_font)
    for x in range(930, 970, 10):
        draw.line((x, 29, min(x + 6, 970), 29), fill=commanded_color, width=2)
    draw.text((978, 20), "commanded", fill=(30, 30, 30), font=label_font)
    draw.line((1120, 29, 1160, 29), fill=limit_color, width=1)
    draw.text((1168, 20), "XML limit", fill=(30, 30, 30), font=label_font)

    time_min = float(times[0])
    time_max = float(times[-1])
    time_span = max(time_max - time_min, 1.0e-9)
    for column, (joint_id, name) in enumerate(
        zip(joint_ids, joint_names, strict=True)
    ):
        row, grid_column = divmod(column, columns)
        origin_x = grid_column * cell_width
        origin_y = top_margin + row * cell_height
        left, right = origin_x + 65, origin_x + cell_width - 20
        top, bottom = origin_y + 38, origin_y + cell_height - 48
        actual_deg = np.rad2deg(actual_angles[:, column])
        commanded_deg = np.rad2deg(commanded_angles[:, column])
        y_values = [float(np.min(actual_deg)), float(np.max(actual_deg))]
        y_values.extend((float(np.min(commanded_deg)), float(np.max(commanded_deg))))
        if model.jnt_limited[joint_id]:
            limit_low, limit_high = np.rad2deg(model.jnt_range[joint_id])
            y_values.extend((float(limit_low), float(limit_high)))
        y_min, y_max = min(y_values), max(y_values)
        padding = max(4.0, 0.06 * max(y_max - y_min, 1.0))
        y_min -= padding
        y_max += padding
        y_span = max(y_max - y_min, 1.0e-9)

        def x_pixel(time: float) -> int:
            return round(left + (time - time_min) / time_span * (right - left))

        def y_pixel(angle: float) -> int:
            return round(bottom - (angle - y_min) / y_span * (bottom - top))

        for tick in range(5):
            fraction = tick / 4.0
            y = round(top + fraction * (bottom - top))
            value = y_max - fraction * y_span
            draw.line((left, y, right, y), fill=grid_color, width=1)
            draw.text((origin_x + 5, y - 7), f"{value:6.1f}", fill=(80, 80, 80), font=label_font)
        for tick in range(6):
            fraction = tick / 5.0
            x = round(left + fraction * (right - left))
            draw.line((x, top, x, bottom), fill=grid_color, width=1)
            draw.text(
                (x - 13, bottom + 7),
                f"{time_min + fraction * time_span:.1f}",
                fill=(80, 80, 80),
                font=label_font,
            )
        if model.jnt_limited[joint_id]:
            for limit in (float(limit_low), float(limit_high)):
                y = y_pixel(limit)
                for x in range(left, right, 8):
                    draw.line((x, y, min(x + 4, right), y), fill=limit_color, width=1)
        commanded_points = [
            (x_pixel(float(time)), y_pixel(float(angle)))
            for time, angle in zip(times, commanded_deg, strict=True)
        ]
        for start in range(0, len(commanded_points) - 1, 8):
            end = min(start + 5, len(commanded_points) - 1)
            draw.line(commanded_points[start : end + 1], fill=commanded_color, width=2)
        actual_points = [
            (x_pixel(float(time)), y_pixel(float(angle)))
            for time, angle in zip(times, actual_deg, strict=True)
        ]
        draw.line(actual_points, fill=actual_color, width=3)
        draw.rectangle((left, top, right, bottom), outline=(100, 100, 100), width=1)
        draw.text(
            (left, origin_y + 8),
            name.replace("_", " "),
            fill=(25, 25, 25),
            font=panel_font,
        )
        draw.text((right - 58, bottom + 28), "time (s)", fill=(60, 60, 60), font=label_font)
        draw.text((origin_x + 5, top - 2), "deg", fill=(60, 60, 60), font=label_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _reset_data(model, data, mujoco, qpos_indices, actuator_ids, ctrl) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
    data.qpos[qpos_indices] = ctrl
    data.qvel[:] = 0.0
    data.ctrl[actuator_ids] = ctrl
    mujoco.mj_forward(model, data)


def _shell_geom_ids(model, mujoco, foot_geom_ids: set[int]) -> set[int]:
    geom_ids = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "_shell_" in name:
            geom_ids.add(int(geom_id))
    if geom_ids:
        return geom_ids

    # rollingquad_2 exports each physical shell as part of its CAD link mesh,
    # rather than as a separately named analytic ``*_shell_*`` geom.  Report
    # every non-foot robot geom as shell/body contact while preserving the
    # four lower-leg/foot meshes as the existing foot-contact category.
    floor_geom_id = model.geom("floor").id
    for geom_id in range(model.ngeom):
        if geom_id == floor_geom_id or geom_id in foot_geom_ids:
            continue
        if int(model.geom_bodyid[geom_id]) != 0:
            geom_ids.add(int(geom_id))
    return geom_ids


def _robot_geom_ids(model, floor_geom_id: int) -> list[int]:
    floor_body_id = int(model.geom_bodyid[floor_geom_id])
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) != floor_body_id
    ]


def _robot_self_collision_enabled(model, floor_geom_id: int) -> bool:
    geom_ids = _robot_geom_ids(model, floor_geom_id)
    for index, geom1 in enumerate(geom_ids):
        for geom2 in geom_ids[index + 1 :]:
            if int(model.geom_bodyid[geom1]) == int(model.geom_bodyid[geom2]):
                continue
            if (
                int(model.geom_contype[geom1])
                & int(model.geom_conaffinity[geom2])
                or int(model.geom_contype[geom2])
                & int(model.geom_conaffinity[geom1])
            ):
                return True
    return False


def _enable_robot_self_collision(model, floor_geom_id: int) -> None:
    # Bit 1 preserves floor contact; bit 2 enables robot-robot contact.  This
    # is used only by the shadow diagnostic model, never by the rollout model.
    for geom_id in _robot_geom_ids(model, floor_geom_id):
        model.geom_contype[geom_id] = 2
        model.geom_conaffinity[geom_id] = 3


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
