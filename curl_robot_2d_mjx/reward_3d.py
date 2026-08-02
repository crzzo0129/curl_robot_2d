"""Reward terms for the first 3-D curl rolling MJX environment."""

from __future__ import annotations

from dataclasses import dataclass


REWARD_3D_TERM_NAMES = (
    "roll_progress",
    "roll_mismatch",
    "backward",
    "lateral_velocity",
    "lateral_drift",
    "axis_tilt",
    "action_rate",
    "residual_action",
    "torque",
    "collision",
    "termination",
    "early_termination",
)


@dataclass(frozen=True)
class Rolling3DRewardConfig:
    progress_clip_rad: float = 0.25
    roll_progress: float = 6.0
    roll_mismatch: float = 0.8
    backward: float = 1.5
    lateral_velocity: float = 2.0
    lateral_drift: float = 1.0
    axis_tilt: float = 8.0
    action_rate: float = 0.03
    residual_action: float = 0.02
    torque: float = 0.01

    allowed_foot_penetration_m: float = 0.0005
    foot_contact_event: float = 2.0
    foot_contact_time: float = 4.0
    allowed_excess_integral: float = 8000.0
    maximum_allowed_excess: float = 2000.0
    forbidden_contact_time: float = 4.0
    forbidden_penetration_integral: float = 20000.0
    maximum_forbidden_penetration: float = 2500.0
    cross_side_foot_contact: float = 30.0

    termination: float = 20.0
    severe_extra_termination: float = 20.0
    nonfinite_termination: float = 80.0
    early_termination_scale: float = 1.0


def conservative_rolling_potential(xp, cumulative_rotation, cumulative_translation):
    return xp.minimum(cumulative_rotation, cumulative_translation)


def reward_terms_3d(xp, config: Rolling3DRewardConfig, inputs):
    clipped_progress = xp.clip(
        inputs["conservative_progress"],
        -config.progress_clip_rad,
        config.progress_clip_rad,
    )
    collision_cost = (
        config.forbidden_contact_time
        * inputs["control_dt"]
        * inputs["forbidden_active"]
        + config.forbidden_penetration_integral
        * inputs["forbidden_depth"]
        * inputs["control_dt"]
        + config.maximum_forbidden_penetration
        * inputs["forbidden_max_increment"]
        + config.foot_contact_event
        * inputs["same_side_foot_contact_start"]
        + config.foot_contact_time
        * inputs["control_dt"]
        * inputs["same_side_foot_contact_active"]
        + config.allowed_excess_integral
        * inputs["same_side_foot_excess"]
        * inputs["control_dt"]
        + config.maximum_allowed_excess
        * inputs["same_side_foot_max_increment"]
        + config.cross_side_foot_contact
        * inputs["cross_side_foot_contact"]
    )
    severe_penalty = config.termination + (
        config.severe_extra_termination * inputs["failure_severe"]
    )
    terminal_penalty = xp.where(
        inputs["failure_nonfinite"] > 0.0,
        config.nonfinite_termination,
        severe_penalty,
    )
    return {
        "roll_progress": config.roll_progress * clipped_progress,
        "roll_mismatch": -config.roll_mismatch * inputs["mismatch_progress"],
        "backward": -config.backward * inputs["backward_progress"],
        "lateral_velocity": (
            -config.lateral_velocity * inputs["lateral_velocity_squared"]
        ),
        "lateral_drift": -config.lateral_drift * inputs["lateral_drift_abs"],
        "axis_tilt": -config.axis_tilt * inputs["axis_tilt_squared"],
        "action_rate": -config.action_rate * inputs["action_rate"],
        "residual_action": (
            -config.residual_action * inputs["residual_action_cost"]
        ),
        "torque": -config.torque * inputs["torque_cost"],
        "collision": -collision_cost,
        "termination": -terminal_penalty * inputs["failed"],
        "early_termination": (
            -terminal_penalty
            * config.early_termination_scale
            * inputs["remaining_fraction"]
            * inputs["failed"]
        ),
    }
