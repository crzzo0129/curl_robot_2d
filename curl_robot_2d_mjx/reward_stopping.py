"""Dependency-light reward and observation helpers for 2-D braking residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math


STOPPING_REWARD_TERM_NAMES = (
    "target_progress",
    "speed_tracking",
    "linear_speed",
    "phase_error",
    "overshoot",
    "action_rate",
    "residual_action",
    "torque",
    "internal_contact",
    "torso_contact",
    "success",
    "failure",
    "timeout",
)


@dataclass(frozen=True)
class StoppingRewardConfig:
    target_progress: float = 4.0
    speed_tracking: float = 0.7
    linear_speed: float = 0.6
    phase_error: float = 0.08
    overshoot: float = 4.0
    action_rate: float = 0.04
    residual_action: float = 0.01
    torque: float = 0.04
    internal_contact: float = 0.20
    torso_contact: float = 8.0
    success: float = 25.0
    failure: float = 20.0
    timeout: float = 5.0


@dataclass(frozen=True)
class StoppingTaskConfig:
    park_phase_rad: float = 0.0
    maximum_deceleration_rad_s2: float = 8.0
    braking_margin_rad: float = math.radians(20.0)
    phase_tolerance_rad: float = math.radians(5.0)
    linear_speed_tolerance_m_s: float = 0.03
    angular_speed_tolerance_rad_s: float = 0.10
    nominal_roll_rate_rad_s: float = 2.0 * math.pi * 0.40
    desired_speed_gain_per_s: float = 1.5
    maximum_duration_s: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            self.maximum_deceleration_rad_s2,
            self.phase_tolerance_rad,
            self.linear_speed_tolerance_m_s,
            self.angular_speed_tolerance_rad_s,
            self.nominal_roll_rate_rad_s,
            self.desired_speed_gain_per_s,
            self.maximum_duration_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("stopping task thresholds must be finite and positive")
        if not math.isfinite(self.braking_margin_rad) or self.braking_margin_rad < 0.0:
            raise ValueError("braking margin must be finite and nonnegative")


def select_reachable_target_phase_xp(
    xp,
    body_phase,
    angular_speed,
    config: StoppingTaskConfig,
):
    """JAX/NumPy-compatible forward target occurrence selection."""

    tau = 2.0 * xp.pi
    base_distance = xp.mod(config.park_phase_rad - body_phase, tau)
    required = (
        xp.square(xp.abs(angular_speed))
        / (2.0 * config.maximum_deceleration_rad_s2)
        + config.braking_margin_rad
    )
    turns = xp.ceil(xp.maximum(required - base_distance, 0.0) / tau)
    distance = base_distance + turns * tau
    return body_phase + distance, distance


def desired_braking_speed(xp, target_remaining_rad, config: StoppingTaskConfig):
    """Speed envelope that smoothly approaches zero at the selected target."""

    return xp.clip(
        config.desired_speed_gain_per_s * xp.maximum(target_remaining_rad, 0.0),
        0.0,
        config.nominal_roll_rate_rad_s,
    )


def stopping_observation_features(
    xp,
    *,
    body_phase,
    target_phase,
    initial_distance,
    linear_speed,
    angular_speed,
    elapsed_s,
    config: StoppingTaskConfig,
):
    """Return bounded task features appended to the rolling observation."""

    remaining = target_phase - body_phase
    progress = 1.0 - remaining / xp.maximum(initial_distance, 1.0e-6)
    desired_speed = desired_braking_speed(xp, remaining, config)
    return xp.stack(
        (
            xp.sin(body_phase),
            xp.cos(body_phase),
            xp.sin(remaining),
            xp.cos(remaining),
            xp.clip(remaining / (2.0 * xp.pi), -2.0, 2.0),
            xp.clip(progress, -1.0, 2.0),
            xp.clip(linear_speed / 0.5, -4.0, 4.0),
            xp.clip(angular_speed / config.nominal_roll_rate_rad_s, -4.0, 4.0),
            desired_speed / config.nominal_roll_rate_rad_s,
            xp.clip(elapsed_s / config.maximum_duration_s, 0.0, 2.0),
        )
    )


def stopping_reward_terms(xp, config: StoppingRewardConfig, values):
    """Compute named dense braking rewards from scalar transition values."""

    terms = {
        "target_progress": config.target_progress * values["target_progress"],
        "speed_tracking": -config.speed_tracking * values["speed_error_sq"],
        "linear_speed": -config.linear_speed * values["linear_speed_sq"],
        "phase_error": -config.phase_error * values["phase_error_sq"],
        "overshoot": -config.overshoot * values["overshoot"],
        "action_rate": -config.action_rate * values["action_rate_sq"],
        "residual_action": -config.residual_action * values["residual_action_sq"],
        "torque": -config.torque * values["torque_sq"],
        "internal_contact": -config.internal_contact * values["internal_contact"],
        "torso_contact": -config.torso_contact * values["torso_contact"],
        "success": config.success * values["success"],
        "failure": -config.failure * values["failure"],
        "timeout": -config.timeout * values["timeout"],
    }
    return {name: xp.asarray(terms[name]) for name in STOPPING_REWARD_TERM_NAMES}
