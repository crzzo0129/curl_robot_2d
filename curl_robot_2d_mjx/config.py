"""Dependency-light configuration for nominal-COM rolling RL."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


PHYSICS_PROFILE_NAMES = (
    "reference",
    "newton4",
    "cg12",
)


@dataclass(frozen=True)
class NominalRLConfig:
    """Task constants shared by smoke tests and PPO training.

    COM, mass, inertia, friction, gains and torque limits all come directly
    from ``assets/curl_robot_2d.xml``.  This first RL stage deliberately does
    not randomize any of them.
    """

    physics_profile: str = "reference"
    physics_timestep: float = 0.001
    solver_name: str = "newton"
    integrator_name: str = "implicitfast"
    cone_name: str = "elliptic"
    jacobian_name: str = "dense"
    model_xml: str | None = None
    action_repeat: int = 20
    episode_length: int = 500
    action_scales: tuple[float, float, float, float] = (
        0.8,
        1.2,
        0.8,
        1.2,
    )
    startup_action_ramp_s: float = 0.25
    reset_joint_noise_rad: float = 0.01
    reset_velocity_noise: float = 0.01
    # Optional periodic command phase for a reference-free feed-forward policy.
    # This adds only sin/cos(clock_phase) to the observation; it does not add
    # CEM actions, coefficients, or a reference trajectory.
    policy_clock_rate_rad_s: float | None = None
    disturbance_root_x_velocity_m_s: float = 0.0
    disturbance_root_pitch_velocity_rad_s: float = 0.0
    disturbance_probability: float = 1.0
    disturbance_level_scales: tuple[float, ...] = (1.0,)
    disturbance_level_probabilities: tuple[float, ...] = (1.0,)
    disturbance_backward_probability: float = 0.5
    disturbance_min_step: int = 100
    disturbance_max_step: int = 400
    # The CPU CEM and release evaluators model the planar root as free and
    # remove the XML's small numerical damping at runtime.
    disable_root_damping: bool = True
    # MJX JAX lacks cylinder-box collision.  The real-geometry model has two
    # knee-motor-cylinder/torso-box candidate pairs, so training may replace
    # only those collision proxies with same-size capsules in memory.  The XML
    # and authoritative CPU evaluation model remain unchanged.
    mjx_compatible_collision_proxies: bool = False

    # Torso root height is phase dependent during valid planar rolling.  The
    # collision-constrained CEM baseline reaches 0.0437 m once per turn, so a
    # fixed lower-height termination would reject known-good behavior.
    terminate_root_z_min: float | None = None
    terminate_root_z_low_duration_s: float = 0.30
    # A low root is valid while the robot is actively rolling.  Treat it as
    # unrecoverable only when conservative rolling progress has also stalled.
    terminate_stuck_root_z_max: float | None = None
    terminate_stuck_progress_window_s: float = 1.0
    terminate_stuck_min_progress_rad: float = 0.20
    terminate_stuck_duration_s: float = 3.0
    terminate_stuck_grace_s: float = 1.50
    tail_progress_window_s: float = 2.0
    terminate_root_z_max: float | None = 0.70
    maximum_foot_center_distance_m: float | None = 0.28
    terminate_leg_crossing: bool = True

    # MJX-JAX is much faster with a small Newton iteration count.  The final
    # policy must still be replayed in the unmodified 20/10 CPU MuJoCo model.
    solver_iterations: int = 20
    solver_ls_iterations: int = 10

    @property
    def control_timestep(self) -> float:
        return self.physics_timestep * self.action_repeat


def validate_nominal_rl_config(config: NominalRLConfig) -> None:
    """Reject disturbance settings that cannot produce one valid impulse."""

    amplitudes = (
        config.disturbance_root_x_velocity_m_s,
        config.disturbance_root_pitch_velocity_rad_s,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in amplitudes):
        raise ValueError("disturbance velocity limits must be finite and nonnegative")
    probabilities = (
        config.disturbance_probability,
        config.disturbance_backward_probability,
    )
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise ValueError("disturbance probabilities must be finite and in [0, 1]")
    level_scales = config.disturbance_level_scales
    level_probabilities = config.disturbance_level_probabilities
    if not level_scales or len(level_scales) != len(level_probabilities):
        raise ValueError(
            "disturbance level scales and probabilities must have the same "
            "nonzero length"
        )
    if any(
        not math.isfinite(value) or value <= 0.0 for value in level_scales
    ):
        raise ValueError("disturbance level scales must be finite and positive")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in level_probabilities
    ):
        raise ValueError(
            "disturbance level probabilities must be finite and positive"
        )
    if not math.isclose(sum(level_probabilities), 1.0, abs_tol=1.0e-6):
        raise ValueError("disturbance level probabilities must sum to 1")
    if config.disturbance_min_step < 0:
        raise ValueError("disturbance_min_step must be nonnegative")
    if config.disturbance_max_step < config.disturbance_min_step:
        raise ValueError(
            "disturbance_max_step must be at least disturbance_min_step"
        )
    if config.policy_clock_rate_rad_s is not None and (
        not math.isfinite(config.policy_clock_rate_rad_s)
        or config.policy_clock_rate_rad_s <= 0.0
    ):
        raise ValueError(
            "policy_clock_rate_rad_s must be finite and positive"
        )
    if config.terminate_root_z_min is not None:
        if (
            not math.isfinite(config.terminate_root_z_min)
            or config.terminate_root_z_min < 0.0
        ):
            raise ValueError("terminate_root_z_min must be finite and nonnegative")
        if (
            not math.isfinite(config.terminate_root_z_low_duration_s)
            or config.terminate_root_z_low_duration_s <= 0.0
        ):
            raise ValueError(
                "terminate_root_z_low_duration_s must be finite and positive"
            )
    if config.terminate_root_z_max is not None and (
        not math.isfinite(config.terminate_root_z_max)
        or config.terminate_root_z_max <= 0.0
    ):
        raise ValueError("terminate_root_z_max must be finite and positive")
    if config.terminate_stuck_root_z_max is not None:
        if (
            not math.isfinite(config.terminate_stuck_root_z_max)
            or config.terminate_stuck_root_z_max <= 0.0
        ):
            raise ValueError(
                "terminate_stuck_root_z_max must be finite and positive"
            )
        positive_stuck_values = (
            config.terminate_stuck_progress_window_s,
            config.terminate_stuck_duration_s,
        )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in positive_stuck_values
        ):
            raise ValueError(
                "stuck progress window and duration must be finite and positive"
            )
        if (
            not math.isfinite(config.terminate_stuck_min_progress_rad)
            or config.terminate_stuck_min_progress_rad < 0.0
        ):
            raise ValueError(
                "terminate_stuck_min_progress_rad must be finite and nonnegative"
            )
        if (
            not math.isfinite(config.terminate_stuck_grace_s)
            or config.terminate_stuck_grace_s
            < config.terminate_stuck_progress_window_s
        ):
            raise ValueError(
                "terminate_stuck_grace_s must be finite and at least the "
                "progress window"
            )
    if (
        not math.isfinite(config.tail_progress_window_s)
        or config.tail_progress_window_s <= 0.0
        or config.tail_progress_window_s
        > config.episode_length * config.control_timestep
    ):
        raise ValueError(
            "tail_progress_window_s must fit inside the episode"
        )
    if config.maximum_foot_center_distance_m is not None and (
        not math.isfinite(config.maximum_foot_center_distance_m)
        or config.maximum_foot_center_distance_m <= 0.0
    ):
        raise ValueError(
            "maximum_foot_center_distance_m must be finite and positive"
        )
    if (
        config.disturbance_probability > 0.0
        and any(value > 0.0 for value in amplitudes)
    ):
        if config.disturbance_max_step >= config.episode_length:
            raise ValueError(
                "disturbance_max_step must be smaller than episode_length "
                "when disturbances are enabled"
            )


def smoothstep_ramp(xp, elapsed_s, duration_s: float):
    """Blend from zero to one without a velocity jump at either endpoint."""

    if duration_s <= 0.0:
        return xp.ones_like(elapsed_s)
    normalized = xp.clip(elapsed_s / duration_s, 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def advance_policy_clock(xp, phase, rate_rad_s: float, timestep_s: float):
    """Advance an independent periodic policy phase by one control step."""

    return xp.mod(
        phase + rate_rad_s * timestep_s,
        2.0 * xp.pi,
    )


def physics_profile(
    name: str,
    config: NominalRLConfig | None = None,
) -> NominalRLConfig:
    """Apply a measured MJX physics profile without changing the XML."""

    base = config or NominalRLConfig()
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
    raise ValueError(f"unknown physics profile: {name}")
