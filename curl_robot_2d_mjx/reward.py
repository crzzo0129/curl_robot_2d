"""Reward computation kept separate from MJX physics and termination logic."""

from __future__ import annotations

from curl_robot_2d_mjx.reward_config import RollingRewardConfig


REWARD_TERM_NAMES = (
    "roll_progress",
    "roll_mismatch",
    "backward",
    "action_rate",
    "torque",
    "airborne",
    "foot_gap",
    "collision",
    "termination",
    "early_termination",
)


def conservative_rolling_potential(
    xp, cumulative_phase, cumulative_translation
):
    """Progress that can only grow after both rotation and translation."""

    return xp.minimum(cumulative_phase, cumulative_translation)


def reward_terms(xp, config: RollingRewardConfig, inputs):
    """Return independently logged scalar reward terms."""

    clipped_progress = xp.clip(
        inputs["conservative_progress"],
        -config.progress_clip_rad,
        config.progress_clip_rad,
    )
    collision_cost = (
        config.forbidden_contact_time
        * inputs["control_dt"]
        * (inputs["forbidden_count"] > 0).astype(xp.float32)
        + config.forbidden_penetration_integral
        * inputs["forbidden_depth"]
        * inputs["control_dt"]
        + config.maximum_forbidden_penetration
        * inputs["forbidden_max_increment"]
        + config.allowed_excess_integral
        * inputs["allowed_excess"]
        * inputs["control_dt"]
        + config.maximum_allowed_excess
        * inputs["allowed_max_increment"]
        + config.leg_crossing * inputs["leg_crossing"]
    )
    return {
        "roll_progress": config.roll_progress * clipped_progress,
        "roll_mismatch": (
            -config.roll_mismatch * inputs["mismatch_progress"]
        ),
        "backward": -config.backward * inputs["backward"],
        "action_rate": -config.action_rate * inputs["action_rate"],
        "torque": -config.torque * inputs["torque_cost"],
        "airborne": -config.airborne * inputs["airborne"],
        "foot_gap": (
            -config.foot_gap
            * xp.maximum(
                inputs["foot_distance"] - config.foot_gap_threshold_m,
                0.0,
            )
        ),
        "collision": -collision_cost,
        "termination": -config.termination * inputs["failed"],
        "early_termination": (
            -config.termination
            * config.early_termination_scale
            * inputs["remaining_fraction"]
            * inputs["failed"]
        ),
    }
