"""Dependency-light configuration for the 3-D roll-to-walk transition task."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import math


TRANSITION_GEOMETRY_NAMES_3D = ("pupper_open60",)
TRANSITION_PHYSICS_PROFILE_NAMES_3D = ("accurate", "newton4", "cg12")
TRANSITION_CURRICULUM_STAGE_NAMES_3D = (
    "deploy_near_stand",
    "deploy_capture",
    "brake_low",
    "brake_full",
)

TRANSITION_ACTION_SIZE_3D = 12
TRANSITION_ACTOR_OBSERVATION_SIZE_3D = 66
TRANSITION_CRITIC_OBSERVATION_SIZE_3D = 86


class TransitionMode3D(IntEnum):
    """Internal modes owned by one transition policy."""

    BRAKE = 0
    DEPLOY = 1
    STABILIZE = 2
    READY = 3


@dataclass(frozen=True)
class Transition3DConfig:
    """Task, curriculum, and deployment-gate constants.

    The actor always produces twelve normalized joint-position residuals.  The
    environment changes only the reference center: compact while braking,
    compact->park->stand while deploying, and stand while stabilizing.
    """

    geometry: str = "pupper_open60"
    physics_profile: str = "newton4"
    curriculum_stage: str = "deploy_near_stand"
    physics_timestep: float = 0.001
    action_repeat: int = 20
    episode_length: int = 350
    solver_name: str = "newton"
    integrator_name: str = "implicitfast"
    cone_name: str = "elliptic"
    jacobian_name: str = "dense"
    solver_iterations: int = 4
    solver_ls_iterations: int = 4
    geom_friction_scale: float = 1.0
    floor_friction_scale: float = 1.0
    floor_contact_friction_override: bool = False
    body_mass_scale: float = 1.0
    body_mass_left_scale: float = 1.0
    body_mass_right_scale: float = 1.0
    disable_root_damping: bool = True

    # Residual action authority, ordered (abduction, hip, knee) x four legs.
    action_scales: tuple[float, ...] = (
        0.12, 0.35, 0.45,
        0.12, 0.35, 0.45,
        0.12, 0.35, 0.45,
        0.12, 0.35, 0.45,
    )

    brake_timeout_s: float = 2.5
    deploy_duration_s: float = 1.8
    deploy_park_fraction: float = 0.55
    deploy_gate_linear_speed_m_s: float = 0.35
    deploy_gate_angular_speed_rad_s: float = 2.5
    deploy_gate_tilt_rad: float = 1.20
    deploy_gate_hold_s: float = 0.08
    stabilize_min_s: float = 0.25
    ready_hold_s: float = 0.40
    ready_linear_speed_m_s: float = 0.12
    ready_angular_speed_rad_s: float = 0.45
    ready_upright_tilt_rad: float = 0.22
    ready_joint_error_rad: float = 0.20
    ready_root_height_min_m: float = 0.145
    ready_root_height_max_m: float = 0.235
    ready_min_foot_contacts: int = 3
    failure_root_height_min_m: float = 0.09
    failure_root_height_max_m: float = 0.55
    failure_nonfinite: bool = True

    # Reset distributions are overwritten by transition_curriculum_config_3d.
    reset_start_mode: int = int(TransitionMode3D.DEPLOY)
    reset_compact_fraction_range: tuple[float, float] = (0.0, 0.20)
    reset_joint_noise_rad: float = 0.02
    reset_linear_speed_m_s: float = 0.05
    reset_angular_speed_rad_s: float = 0.15
    reset_tilt_rad: float = 0.08
    reset_roll_phase_range_rad: tuple[float, float] = (-0.15, 0.15)

    observation_noise_enabled: bool = True
    observation_noise_level: float = 1.0
    observation_noise_velocity: float = 0.05
    observation_noise_gravity: float = 0.02
    observation_noise_joint_position: float = 0.01
    observation_noise_joint_velocity: float = 0.30

    @property
    def control_timestep(self) -> float:
        return self.physics_timestep * self.action_repeat


@dataclass(frozen=True)
class ReadyToWalkSample3D:
    linear_speed_m_s: float
    angular_speed_rad_s: float
    upright_tilt_rad: float
    joint_error_rad: float
    root_height_m: float
    foot_contacts: int
    finite: bool = True


def smoothstep01(value):
    """C1 interpolation that works with Python, NumPy, and JAX scalars."""

    clipped = min(max(float(value), 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def ready_to_walk_reasons_3d(
    sample: ReadyToWalkSample3D,
    config: Transition3DConfig | None = None,
) -> tuple[str, ...]:
    """Return failed READY gate names for deployment/runtime diagnostics."""

    task = config or Transition3DConfig()
    reasons: list[str] = []
    if not sample.finite:
        reasons.append("nonfinite")
    if sample.linear_speed_m_s > task.ready_linear_speed_m_s:
        reasons.append("linear_speed")
    if sample.angular_speed_rad_s > task.ready_angular_speed_rad_s:
        reasons.append("angular_speed")
    if sample.upright_tilt_rad > task.ready_upright_tilt_rad:
        reasons.append("upright_tilt")
    if sample.joint_error_rad > task.ready_joint_error_rad:
        reasons.append("joint_error")
    if not (
        task.ready_root_height_min_m
        <= sample.root_height_m
        <= task.ready_root_height_max_m
    ):
        reasons.append("root_height")
    if sample.foot_contacts < task.ready_min_foot_contacts:
        reasons.append("foot_contacts")
    return tuple(reasons)


def is_ready_to_walk_3d(
    sample: ReadyToWalkSample3D,
    config: Transition3DConfig | None = None,
) -> bool:
    return not ready_to_walk_reasons_3d(sample, config)


def transition_curriculum_config_3d(
    stage: str,
    config: Transition3DConfig | None = None,
) -> Transition3DConfig:
    """Apply the backward curriculum from easy deployment to full braking."""

    base = config or Transition3DConfig()
    if stage == "deploy_near_stand":
        result = replace(
            base,
            curriculum_stage=stage,
            reset_start_mode=int(TransitionMode3D.DEPLOY),
            reset_compact_fraction_range=(0.0, 0.20),
            reset_linear_speed_m_s=0.05,
            reset_angular_speed_rad_s=0.15,
            reset_tilt_rad=0.08,
            reset_roll_phase_range_rad=(-0.15, 0.15),
        )
    elif stage == "deploy_capture":
        result = replace(
            base,
            curriculum_stage=stage,
            reset_start_mode=int(TransitionMode3D.DEPLOY),
            reset_compact_fraction_range=(0.20, 0.85),
            reset_linear_speed_m_s=0.20,
            reset_angular_speed_rad_s=1.25,
            reset_tilt_rad=0.55,
            reset_roll_phase_range_rad=(-0.80, 0.80),
        )
    elif stage == "brake_low":
        result = replace(
            base,
            curriculum_stage=stage,
            reset_start_mode=int(TransitionMode3D.BRAKE),
            reset_compact_fraction_range=(0.85, 1.0),
            reset_linear_speed_m_s=0.35,
            reset_angular_speed_rad_s=3.5,
            reset_tilt_rad=1.0,
            reset_roll_phase_range_rad=(-math.pi, math.pi),
        )
    elif stage == "brake_full":
        result = replace(
            base,
            curriculum_stage=stage,
            reset_start_mode=int(TransitionMode3D.BRAKE),
            reset_compact_fraction_range=(0.95, 1.0),
            reset_linear_speed_m_s=0.70,
            reset_angular_speed_rad_s=7.0,
            reset_tilt_rad=math.pi,
            reset_roll_phase_range_rad=(-math.pi, math.pi),
        )
    else:
        raise ValueError(f"unknown transition curriculum stage: {stage}")
    validate_transition_config_3d(result)
    return result


def transition_physics_profile_3d(
    name: str,
    config: Transition3DConfig | None = None,
) -> Transition3DConfig:
    base = config or Transition3DConfig()
    if name == "accurate":
        return replace(
            base, physics_profile=name, solver_name="newton",
            solver_iterations=20, solver_ls_iterations=10,
        )
    if name == "newton4":
        return replace(
            base, physics_profile=name, solver_name="newton",
            solver_iterations=4, solver_ls_iterations=4,
        )
    if name == "cg12":
        return replace(
            base, physics_profile=name, solver_name="cg",
            solver_iterations=12, solver_ls_iterations=6,
        )
    raise ValueError(f"unknown transition physics profile: {name}")


def validate_transition_config_3d(config: Transition3DConfig) -> None:
    if config.geometry not in TRANSITION_GEOMETRY_NAMES_3D:
        raise ValueError(f"unknown transition geometry: {config.geometry}")
    if config.curriculum_stage not in TRANSITION_CURRICULUM_STAGE_NAMES_3D:
        raise ValueError(
            f"unknown transition curriculum stage: {config.curriculum_stage}"
        )
    if len(config.action_scales) != TRANSITION_ACTION_SIZE_3D:
        raise ValueError("transition action_scales must contain 12 values")
    positive = (
        config.physics_timestep,
        config.action_repeat,
        config.episode_length,
        config.brake_timeout_s,
        config.deploy_duration_s,
        config.deploy_gate_hold_s,
        config.stabilize_min_s,
        config.ready_hold_s,
    )
    if any(not math.isfinite(float(value)) or value <= 0 for value in positive):
        raise ValueError("transition timing and step values must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in config.action_scales):
        raise ValueError("transition action scales must be finite and positive")
    if not 0.0 < config.deploy_park_fraction < 1.0:
        raise ValueError("deploy_park_fraction must lie in (0, 1)")
    if config.ready_root_height_min_m >= config.ready_root_height_max_m:
        raise ValueError("READY root-height bounds must be ordered")
    if not 1 <= config.ready_min_foot_contacts <= 4:
        raise ValueError("ready_min_foot_contacts must be in [1, 4]")
    low, high = config.reset_compact_fraction_range
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("reset_compact_fraction_range must lie in [0, 1]")

