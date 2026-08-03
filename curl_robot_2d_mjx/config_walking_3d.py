"""Dependency-light configuration for the 3-D walking MJX task."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from curl_robot_2d.parameters import FIXED_PARAMETERS


WALKING_PHYSICS_PROFILE_NAMES_3D = (
    "reference",
    "newton4",
    "cg12",
)


@dataclass(frozen=True)
class WalkingReference3DConfig:
    """Open-loop foot reference for the mirrored curl leg mechanism."""

    frequency_hz: float = 0.70
    duty_factor: float = 0.90
    step_length_m: float = 0.040
    foot_lift_m: float = 0.010
    body_height_m: float = (
        FIXED_PARAMETERS.walk_root_height - FIXED_PARAMETERS.foot_radius
    )
    fore_aft_center_m: float = 0.0
    # Left/right legs stay synchronized because each side is a duplicated
    # sagittal chain without a hip-abduction degree of freedom.
    phase_offsets: tuple[float, ...] = (0.0, 0.0, 0.5, 0.5)
    initial_phase_fraction: float = 0.0
    upper_length_m: float = FIXED_PARAMETERS.upper_length
    lower_length_m: float = FIXED_PARAMETERS.lower_length
    foot_radius_m: float = FIXED_PARAMETERS.foot_radius
    hip_range: tuple[float, float] = (
        FIXED_PARAMETERS.hip.shell_compatible_range
    )
    knee_range: tuple[float, float] = (
        FIXED_PARAMETERS.knee.shell_compatible_range
    )

    @property
    def desired_speed_m_s(self) -> float:
        return self.step_length_m * self.frequency_hz

    @property
    def root_height_m(self) -> float:
        return self.body_height_m + self.foot_radius_m


@dataclass(frozen=True)
class Walking3DConfig:
    """Task constants for residual PPO around a morphology-aware gait."""

    physics_profile: str = "reference"
    physics_timestep: float = 0.001
    solver_name: str = "newton"
    integrator_name: str = "implicitfast"
    cone_name: str = "elliptic"
    jacobian_name: str = "dense"
    action_repeat: int = 20
    episode_length: int = 500
    reset_keyframe_name: str = "walk"
    reference: WalkingReference3DConfig = field(
        default_factory=WalkingReference3DConfig
    )
    action_scales: tuple[float, ...] = (
        0.25,
        0.35,
        0.25,
        0.35,
        0.25,
        0.35,
        0.25,
        0.35,
    )
    residual_gain: float = 0.65
    # The XML walk keyframe currently has only its front feet on the floor.
    # Anneal this to zero after the keyframe itself has been optimized.
    reset_reference_weight: float = 1.0
    startup_reference_ramp_s: float = 0.20
    startup_action_ramp_s: float = 0.25
    reset_joint_noise_rad: float = 0.008
    reset_velocity_noise: float = 0.008
    disable_root_damping: bool = True

    terminate_root_z_min: float = 0.145
    terminate_root_z_low_duration_s: float = 0.08
    terminate_root_z_max: float = 0.46
    terminate_upright_tilt_rad: float = 0.72
    terminate_upright_tilt_duration_s: float = 0.08
    terminate_lateral_drift_m: float = 0.25
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
    reference = config.reference
    if len(config.action_scales) != 8:
        raise ValueError("walking 3-D action_scales must contain 8 values")
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
        (config.residual_gain, "residual_gain"),
        (config.terminate_root_z_min, "terminate_root_z_min"),
        (config.terminate_root_z_max, "terminate_root_z_max"),
        (config.terminate_upright_tilt_rad, "terminate_upright_tilt_rad"),
        (config.terminate_lateral_drift_m, "terminate_lateral_drift_m"),
        (config.terminate_nonfoot_depth_m, "terminate_nonfoot_depth_m"),
        (config.terminate_self_contact_depth_m, "terminate_self_contact_depth_m"),
    ):
        _validate_positive(value, name)
    if not 0.0 <= config.reset_reference_weight <= 1.0:
        raise ValueError("reset_reference_weight must be between 0 and 1")
    if config.terminate_root_z_min >= config.terminate_root_z_max:
        raise ValueError("root-z termination bounds must be ordered")
    for value, name in (
        (config.startup_reference_ramp_s, "startup_reference_ramp_s"),
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
    if len(reference.phase_offsets) != 4:
        raise ValueError("walking reference phase_offsets must contain 4 values")
    if any(not math.isfinite(value) for value in reference.phase_offsets):
        raise ValueError("walking reference phase offsets must be finite")
    if not 0.0 <= reference.initial_phase_fraction < 1.0:
        raise ValueError(
            "reference.initial_phase_fraction must be in [0, 1)"
        )
    for value, name in (
        (reference.frequency_hz, "reference.frequency_hz"),
        (reference.step_length_m, "reference.step_length_m"),
        (reference.body_height_m, "reference.body_height_m"),
        (reference.upper_length_m, "reference.upper_length_m"),
        (reference.lower_length_m, "reference.lower_length_m"),
        (reference.foot_radius_m, "reference.foot_radius_m"),
    ):
        _validate_positive(value, name)
    _validate_nonnegative(reference.foot_lift_m, "reference.foot_lift_m")
    if not 0.5 < reference.duty_factor < 1.0:
        raise ValueError("reference.duty_factor must be between 0.5 and 1")
    if not reference.hip_range[0] < reference.hip_range[1]:
        raise ValueError("reference hip range must be ordered")
    if not reference.knee_range[0] < reference.knee_range[1]:
        raise ValueError("reference knee range must be ordered")


def smoothstep_ramp(xp, elapsed_s, duration_s: float):
    if duration_s <= 0.0:
        return xp.ones_like(elapsed_s)
    normalized = xp.clip(elapsed_s / duration_s, 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def walking_physics_profile_3d(
    name: str,
    config: Walking3DConfig | None = None,
) -> Walking3DConfig:
    base = config or Walking3DConfig()
    if name == "reference":
        return replace(
            base,
            physics_profile="reference",
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


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
