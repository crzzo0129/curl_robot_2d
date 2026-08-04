"""Dependency-light configuration for 3-D curl rolling MJX tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


PHYSICS_PROFILE_NAMES_3D = (
    "reference",
    "newton4",
    "cg12",
)


@dataclass(frozen=True)
class Rolling3DConfig:
    """Task constants for the first 3-D CEM-reference rolling env."""

    physics_profile: str = "reference"
    physics_timestep: float = 0.001
    solver_name: str = "newton"
    integrator_name: str = "implicitfast"
    cone_name: str = "elliptic"
    jacobian_name: str = "dense"
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
    reference_phase_rate_scale: float = 1.0
    residual_pair_differential_scale: float | None = None
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
    if len(config.action_scales) != 8:
        raise ValueError("3-D action_scales must contain 8 values")
    if any(not math.isfinite(value) or value <= 0.0 for value in config.action_scales):
        raise ValueError("3-D action scales must be finite and positive")
    if config.action_repeat < 1 or config.episode_length < 1:
        raise ValueError("action_repeat and episode_length must be positive")
    if not math.isfinite(config.reference_phase_rate_scale):
        raise ValueError("reference_phase_rate_scale must be finite")
    if config.residual_pair_differential_scale is not None:
        if (
            not math.isfinite(config.residual_pair_differential_scale)
            or not 0.0 <= config.residual_pair_differential_scale <= 1.0
        ):
            raise ValueError(
                "residual_pair_differential_scale must be in [0, 1]"
            )
    if not isinstance(config.explicit_phase_observation, bool):
        raise ValueError("explicit_phase_observation must be boolean")
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
    raise ValueError(f"unknown 3-D physics profile: {name}")


def _validate_positive_duration(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
