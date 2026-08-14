"""Dependency-light configuration for the 3-D walking MJX task."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from curl_robot_2d.parameters import (
    FIXED_PARAMETERS,
    PUPPER_ORIGINAL_SHELL_60_PARAMETERS,
    REAL_GEOMETRY_PARAMETERS,
)


WALKING_GEOMETRY_NAMES_3D = ("pupper_open60",)


WALKING_PHYSICS_PROFILE_NAMES_3D = (
    "accurate",
    "newton4",
    "cg12",
)


@dataclass(frozen=True)
class Walking3DConfig:
    """Task constants for reference-free 3-D locomotion PPO."""

    physics_profile: str = "accurate"
    geometry: str = "pupper_open60"
    physics_timestep: float = 0.001
    solver_name: str = "newton"
    integrator_name: str = "implicitfast"
    cone_name: str = "elliptic"
    jacobian_name: str = "dense"
    geom_friction_scale: float = 1.0
    body_mass_scale: float = 1.0
    body_mass_left_scale: float = 1.0
    body_mass_right_scale: float = 1.0
    action_repeat: int = 20
    episode_length: int = 500
    reset_keyframe_name: str = "stand"
    desired_speed_m_s: float = 0.20
    command_forward_velocity_range_m_s: tuple[float, float] = (-0.10, 0.35)
    command_lateral_velocity_range_m_s: tuple[float, float] = (-0.15, 0.15)
    command_yaw_rate_range_rad_s: tuple[float, float] = (-0.60, 0.60)
    command_resample_time_s: float = 4.0
    command_deadband_probability: float = 0.10
    nominal_root_height_m: float = (
        PUPPER_ORIGINAL_SHELL_60_PARAMETERS.stand_3d_root_height
    )
    foot_radius_m: float = PUPPER_ORIGINAL_SHELL_60_PARAMETERS.foot_radius
    action_scales: tuple[float, ...] = (
        0.10, 0.40, 0.55,
        0.10, 0.40, 0.55,
        0.10, 0.40, 0.55,
        0.10, 0.40, 0.55,
    )
    startup_action_ramp_s: float = 0.50
    reset_joint_noise_rad: float = 0.015
    reset_velocity_noise: float = 0.05
    reset_root_xy_velocity_noise_m_s: float = 0.15
    reset_root_yaw_rate_noise_rad_s: float = 0.20
    observation_noise_enabled: bool = True
    observation_noise_level: float = 1.0
    observation_noise_linear_velocity_m_s: float = 0.10
    observation_noise_angular_velocity_rad_s: float = 0.20
    observation_noise_gravity: float = 0.05
    observation_noise_joint_position_rad: float = 0.01
    observation_noise_joint_velocity_rad_s: float = 1.50
    soft_joint_limit_fraction: float = 0.90
    disable_root_damping: bool = True

    terminate_root_z_min: float = 0.145
    terminate_root_z_low_duration_s: float = 0.08
    terminate_root_z_max: float = 0.46
    terminate_upright_tilt_rad: float = 0.72
    terminate_upright_tilt_duration_s: float = 0.08
    diagnostic_lateral_drift_m: float = 1.50
    terminate_airborne_duration_s: float = 0.14
    terminate_nonfoot_depth_m: float = 0.004
    terminate_nonfoot_contact_duration_s: float = 0.06
    terminate_self_contact_depth_m: float = 0.004
    terminate_self_contact_duration_s: float = 0.08

    solver_iterations: int = 20
    solver_ls_iterations: int = 10

    @property
    def control_timestep(self) -> float:
        return self.physics_timestep * self.action_repeat


def validate_walking_3d_config(config: Walking3DConfig) -> None:
    if config.geometry not in WALKING_GEOMETRY_NAMES_3D:
        raise ValueError(
            f"unknown walking 3-D geometry: {config.geometry}"
        )
    if len(config.action_scales) != 12:
        raise ValueError("walking 3-D action_scales must contain 12 values")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in config.action_scales
    ):
        raise ValueError("walking 3-D action scales must be finite and positive")
    if config.action_repeat < 1 or config.episode_length < 1:
        raise ValueError("action_repeat and episode_length must be positive")
    if not config.reset_keyframe_name:
        raise ValueError("reset_keyframe_name must not be empty")
    for value, name in (
        (config.physics_timestep, "physics_timestep"),
        (config.geom_friction_scale, "geom_friction_scale"),
        (config.body_mass_scale, "body_mass_scale"),
        (config.body_mass_left_scale, "body_mass_left_scale"),
        (config.body_mass_right_scale, "body_mass_right_scale"),
        (config.desired_speed_m_s, "desired_speed_m_s"),
        (config.nominal_root_height_m, "nominal_root_height_m"),
        (config.foot_radius_m, "foot_radius_m"),
        (config.terminate_root_z_min, "terminate_root_z_min"),
        (config.terminate_root_z_max, "terminate_root_z_max"),
        (config.terminate_upright_tilt_rad, "terminate_upright_tilt_rad"),
        (
            config.diagnostic_lateral_drift_m,
            "diagnostic_lateral_drift_m",
        ),
        (config.terminate_nonfoot_depth_m, "terminate_nonfoot_depth_m"),
        (config.terminate_self_contact_depth_m, "terminate_self_contact_depth_m"),
    ):
        _validate_positive(value, name)
    for limits, name in (
        (config.command_forward_velocity_range_m_s, "forward command range"),
        (config.command_lateral_velocity_range_m_s, "lateral command range"),
        (config.command_yaw_rate_range_rad_s, "yaw command range"),
    ):
        if len(limits) != 2 or not all(math.isfinite(x) for x in limits):
            raise ValueError(f"{name} must contain two finite values")
        if limits[1] < limits[0]:
            raise ValueError(f"{name} must be ordered")
    _validate_positive(config.command_resample_time_s, "command_resample_time_s")
    if not 0.0 <= config.command_deadband_probability <= 1.0:
        raise ValueError("command_deadband_probability must be in [0, 1]")
    for value, name in (
        (config.observation_noise_level, "observation_noise_level"),
        (
            config.observation_noise_linear_velocity_m_s,
            "observation_noise_linear_velocity_m_s",
        ),
        (
            config.observation_noise_angular_velocity_rad_s,
            "observation_noise_angular_velocity_rad_s",
        ),
        (config.observation_noise_gravity, "observation_noise_gravity"),
        (
            config.observation_noise_joint_position_rad,
            "observation_noise_joint_position_rad",
        ),
        (
            config.observation_noise_joint_velocity_rad_s,
            "observation_noise_joint_velocity_rad_s",
        ),
    ):
        _validate_nonnegative(value, name)
    if config.terminate_root_z_min >= config.terminate_root_z_max:
        raise ValueError("root-z termination bounds must be ordered")
    if not 0.0 < config.soft_joint_limit_fraction <= 1.0:
        raise ValueError("soft_joint_limit_fraction must be in (0, 1]")
    for value, name in (
        (config.startup_action_ramp_s, "startup_action_ramp_s"),
        (config.terminate_root_z_low_duration_s, "terminate_root_z_low_duration_s"),
        (
            config.terminate_upright_tilt_duration_s,
            "terminate_upright_tilt_duration_s",
        ),
        (config.terminate_airborne_duration_s, "terminate_airborne_duration_s"),
        (
            config.terminate_nonfoot_contact_duration_s,
            "terminate_nonfoot_contact_duration_s",
        ),
        (
            config.terminate_self_contact_duration_s,
            "terminate_self_contact_duration_s",
        ),
    ):
        _validate_nonnegative(value, name)
def smoothstep_ramp(xp, elapsed_s, duration_s: float):
    if duration_s <= 0.0:
        return xp.ones_like(elapsed_s)
    normalized = xp.clip(elapsed_s / duration_s, 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def walking_physics_profile_3d(
    name: str,
    config: Walking3DConfig | None = None,
) -> Walking3DConfig:
    base = walking_geometry_config_3d(config or Walking3DConfig())
    if name == "accurate":
        return replace(
            base,
            physics_profile="accurate",
            physics_timestep=0.001,
            solver_name="newton",
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
            action_repeat=20,
            solver_iterations=4,
            solver_ls_iterations=4,
        )
    if name == "cg12":
        return replace(
            base,
            physics_profile="cg12",
            physics_timestep=0.001,
            solver_name="cg",
            action_repeat=20,
            solver_iterations=12,
            solver_ls_iterations=6,
        )
    raise ValueError(f"unknown walking 3-D physics profile: {name}")


def walking_geometry_config_3d(config: Walking3DConfig) -> Walking3DConfig:
    """Bind geometry-dependent task measurements to the selected MJCF."""

    geometry = {
        "fixed": FIXED_PARAMETERS,
        "real": REAL_GEOMETRY_PARAMETERS,
        "pupper_open60": PUPPER_ORIGINAL_SHELL_60_PARAMETERS,
    }.get(config.geometry)
    if geometry is None:
        raise ValueError(f"unknown walking 3-D geometry: {config.geometry}")
    return replace(
        config,
        nominal_root_height_m=geometry.stand_3d_root_height,
        foot_radius_m=geometry.foot_radius,
    )


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
