"""Dependency-light configuration for 3-D curl rolling MJX tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


PHYSICS_PROFILE_NAMES_3D = (
    "reference",
    "newton4",
    "newton8",
    "cg12",
    "cg20",
)

GEOMETRY_NAMES_3D = (
    "baseline",
    "real",
    "pupper_open60",
    "rollingquad_2",
)


@dataclass(frozen=True)
class Rolling3DConfig:
    """Task constants for the first 3-D CEM-reference rolling env."""

    geometry: str = "rollingquad_2"
    physics_profile: str = "reference"
    physics_timestep: float = 0.001
    solver_name: str = "newton"
    integrator_name: str = "implicitfast"
    cone_name: str = "elliptic"
    jacobian_name: str = "dense"
    geom_friction_scale: float = 1.0
    floor_friction_scale: float = 1.0
    floor_contact_friction_override: bool = False
    body_mass_scale: float = 1.0
    body_mass_left_scale: float = 1.0
    body_mass_right_scale: float = 1.0
    actuator_gain_scale: float = 1.0
    action_repeat: int = 20
    episode_length: int = 500
    action_scales: tuple[float, ...] = (
        0.8,
        1.2,
        0.8,
        1.2,
        0.8,
        1.2,
        0.8,
        1.2,
    )
    startup_action_ramp_s: float = 0.25
    reset_joint_noise_rad: float = 0.005
    reset_velocity_noise: float = 0.005
    reset_root_velocity_noise: float = 0.0
    reset_pair_differential_scale: float | None = None
    reset_axis_tilt_noise_rad: float = 0.0
    reference_phase_rate_scale: float = 1.0
    reference_action_scale: float = 1.0
    reference_ramp_start_scale: float | None = 0.0
    reference_ramp_duration_s: float = 0.25
    reference_startup_boost: float = 0.0
    reference_startup_boost_duration_s: float = 0.25
    residual_pair_differential_scale: float | None = None
    lateral_reflex_gain: float = 0.0
    lateral_reflex_position_gain: float = 2.0
    lateral_reflex_velocity_gain: float = 2.0
    lateral_command_enabled: bool = False
    lateral_command_max: float = 0.15
    lateral_command_probability: float = 0.20
    lateral_command_error_limit: float = 0.20
    lateral_command_fixed: float | None = None
    explicit_phase_observation: bool = False
    disable_root_damping: bool = True

    terminate_root_z_min: float | None = 0.025
    terminate_root_z_low_duration_s: float = 0.20
    terminate_root_z_max: float = 0.80
    terminate_lateral_drift_m: float = 0.20
    terminate_axis_tilt_rad: float = 0.50
    terminate_axis_tilt_duration_s: float = 0.10
    terminate_forbidden_depth_m: float = 0.004
    terminate_forbidden_contact_duration_s: float = 0.20

    solver_iterations: int = 20
    solver_ls_iterations: int = 10

    @property
    def control_timestep(self) -> float:
        return self.physics_timestep * self.action_repeat


def validate_3d_config(config: Rolling3DConfig) -> None:
    if config.geometry not in GEOMETRY_NAMES_3D:
        raise ValueError(
            f"unknown 3-D geometry: {config.geometry!r}; "
            f"expected one of {GEOMETRY_NAMES_3D}"
        )
    if len(config.action_scales) != 8:
        raise ValueError("3-D action_scales must contain 8 values")
    if any(not math.isfinite(value) or value <= 0.0 for value in config.action_scales):
        raise ValueError("3-D action scales must be finite and positive")
    if config.action_repeat < 1 or config.episode_length < 1:
        raise ValueError("action_repeat and episode_length must be positive")
    for value, name in (
        (config.geom_friction_scale, "geom_friction_scale"),
        (config.floor_friction_scale, "floor_friction_scale"),
        (config.body_mass_scale, "body_mass_scale"),
        (config.body_mass_left_scale, "body_mass_left_scale"),
        (config.body_mass_right_scale, "body_mass_right_scale"),
        (config.actuator_gain_scale, "actuator_gain_scale"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for value, name in (
        (config.reset_joint_noise_rad, "reset_joint_noise_rad"),
        (config.reset_velocity_noise, "reset_velocity_noise"),
        (config.reset_root_velocity_noise, "reset_root_velocity_noise"),
        (config.reset_axis_tilt_noise_rad, "reset_axis_tilt_noise_rad"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.reset_pair_differential_scale is not None:
        if (
            not math.isfinite(config.reset_pair_differential_scale)
            or not 0.0 <= config.reset_pair_differential_scale <= 1.0
        ):
            raise ValueError(
                "reset_pair_differential_scale must be in [0, 1]"
            )
    if not math.isfinite(config.reference_phase_rate_scale):
        raise ValueError("reference_phase_rate_scale must be finite")
    if (
        not math.isfinite(config.reference_action_scale)
        or config.reference_action_scale <= 0.0
    ):
        raise ValueError("reference_action_scale must be finite and positive")
    if config.reference_ramp_start_scale is not None:
        if (
            not math.isfinite(config.reference_ramp_start_scale)
            or config.reference_ramp_start_scale < 0.0
        ):
            raise ValueError(
                "reference_ramp_start_scale must be finite and nonnegative"
            )
    _validate_positive_duration(
        config.reference_ramp_duration_s,
        "reference_ramp_duration_s",
    )
    if (
        not math.isfinite(config.reference_startup_boost)
        or config.reference_startup_boost < 0.0
    ):
        raise ValueError("reference_startup_boost must be finite and nonnegative")
    _validate_positive_duration(
        config.reference_startup_boost_duration_s,
        "reference_startup_boost_duration_s",
    )
    if config.residual_pair_differential_scale is not None:
        if (
            not math.isfinite(config.residual_pair_differential_scale)
            or not 0.0 <= config.residual_pair_differential_scale <= 1.0
        ):
            raise ValueError(
                "residual_pair_differential_scale must be in [0, 1]"
            )
    for value, name in (
        (config.lateral_reflex_gain, "lateral_reflex_gain"),
        (config.lateral_reflex_position_gain, "lateral_reflex_position_gain"),
        (config.lateral_reflex_velocity_gain, "lateral_reflex_velocity_gain"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.lateral_reflex_gain < 0.0:
        raise ValueError("lateral_reflex_gain must be nonnegative")
    for value, name in (
        (config.lateral_command_max, "lateral_command_max"),
        (config.lateral_command_probability, "lateral_command_probability"),
        (config.lateral_command_error_limit, "lateral_command_error_limit"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.lateral_command_max < 0.0:
        raise ValueError("lateral_command_max must be nonnegative")
    if not 0.0 <= config.lateral_command_probability <= 1.0:
        raise ValueError(
            "lateral_command_probability must be in [0, 1]"
        )
    if config.lateral_command_error_limit <= 0.0:
        raise ValueError("lateral_command_error_limit must be positive")
    if config.lateral_command_fixed is not None and not math.isfinite(
        config.lateral_command_fixed
    ):
        raise ValueError("lateral_command_fixed must be finite")
    if not isinstance(config.explicit_phase_observation, bool):
        raise ValueError("explicit_phase_observation must be boolean")
    if not isinstance(config.floor_contact_friction_override, bool):
        raise ValueError("floor_contact_friction_override must be boolean")
    if config.terminate_root_z_min is not None:
        if (
            not math.isfinite(config.terminate_root_z_min)
            or config.terminate_root_z_min < 0.0
        ):
            raise ValueError("terminate_root_z_min must be finite and nonnegative")
        _validate_positive_duration(
            config.terminate_root_z_low_duration_s,
            "terminate_root_z_low_duration_s",
        )
    for value, name in (
        (config.terminate_root_z_max, "terminate_root_z_max"),
        (config.terminate_lateral_drift_m, "terminate_lateral_drift_m"),
        (config.terminate_axis_tilt_rad, "terminate_axis_tilt_rad"),
        (config.terminate_forbidden_depth_m, "terminate_forbidden_depth_m"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    _validate_positive_duration(
        config.terminate_axis_tilt_duration_s,
        "terminate_axis_tilt_duration_s",
    )
    _validate_positive_duration(
        config.terminate_forbidden_contact_duration_s,
        "terminate_forbidden_contact_duration_s",
    )


def smoothstep_ramp(xp, elapsed_s, duration_s: float):
    if duration_s <= 0.0:
        return xp.ones_like(elapsed_s)
    normalized = xp.clip(elapsed_s / duration_s, 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def physics_profile_3d(
    name: str,
    config: Rolling3DConfig | None = None,
) -> Rolling3DConfig:
    base = config or Rolling3DConfig()
    if name == "reference":
        return replace(
            base,
            physics_profile="reference",
            physics_timestep=0.001,
            solver_name="newton",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=20,
            solver_ls_iterations=10,
        )
    if name == "newton4":
        return replace(
            base,
            physics_profile="newton4",
            physics_timestep=0.001,
            solver_name="newton",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=4,
            solver_ls_iterations=4,
        )
    if name == "newton8":
        return replace(
            base,
            physics_profile="newton8",
            physics_timestep=0.001,
            solver_name="newton",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=8,
            solver_ls_iterations=8,
        )
    if name == "cg12":
        return replace(
            base,
            physics_profile="cg12",
            physics_timestep=0.001,
            solver_name="cg",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=12,
            solver_ls_iterations=6,
        )
    if name == "cg20":
        return replace(
            base,
            physics_profile="cg20",
            physics_timestep=0.001,
            solver_name="cg",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=20,
            solver_ls_iterations=10,
        )
    raise ValueError(f"unknown 3-D physics profile: {name}")


def _validate_positive_duration(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
