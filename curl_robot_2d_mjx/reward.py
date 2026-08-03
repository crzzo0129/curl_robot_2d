"""Reward computation kept separate from MJX physics and termination logic."""

from __future__ import annotations

from curl_robot_2d_mjx.reward_config import RollingRewardConfig


REWARD_TERM_NAMES = (
    "roll_progress",
    "phase_progress",
    "translation_progress",
    "roll_mismatch",
    "backward",
    "action_rate",
    "residual_action",
    "torque",
    "airborne",
    "stuck",
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


def stuck_termination_state(
    xp,
    *,
    root_z,
    rolling_window_progress,
    previous_count,
    step_count,
    root_z_max,
    minimum_progress_rad,
    grace_steps,
    termination_steps,
):
    """Track continuous low-and-stalled behavior after a startup grace period."""

    eligible = (step_count >= grace_steps) & (root_z < root_z_max)
    progress_scale = xp.maximum(
        xp.asarray(minimum_progress_rad), 1.0e-6
    )
    deficit = xp.where(
        eligible,
        xp.clip(
            (minimum_progress_rad - rolling_window_progress)
            / progress_scale,
            0.0,
            1.0,
        ),
        0.0,
    )
    active = deficit > 0.0
    count = xp.where(active, previous_count + 1, xp.asarray(0, dtype=xp.int32))
    return active, deficit, count, count >= termination_steps


def reward_terms(xp, config: RollingRewardConfig, inputs):
    """Return independently logged scalar reward terms."""

    clipped_progress = xp.clip(
        inputs["conservative_progress"],
        -config.progress_clip_rad,
        config.progress_clip_rad,
    )
    clipped_phase_progress = xp.clip(
        inputs["phase_progress"],
        -config.progress_clip_rad,
        config.progress_clip_rad,
    )
    clipped_translation_progress = xp.clip(
        inputs["translation_progress"],
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
        + config.foot_contact_event
        * inputs["allowed_contact_start"]
        + config.foot_contact_time
        * inputs["control_dt"]
        * inputs["allowed_contact_active"]
        + config.allowed_excess_integral
        * inputs["allowed_excess"]
        * inputs["control_dt"]
        + config.maximum_allowed_excess
        * inputs["allowed_max_increment"]
        + config.leg_crossing * inputs["leg_crossing"]
    )
    extra_termination_penalty = xp.maximum(
        config.root_low_extra_termination * inputs["failure_root_low"],
        config.stuck_extra_termination * inputs["failure_stuck"],
    )
    termination_penalty = config.termination + extra_termination_penalty
    return {
        "roll_progress": config.roll_progress * clipped_progress,
        "phase_progress": (
            config.phase_progress * clipped_phase_progress
        ),
        "translation_progress": (
            config.translation_progress * clipped_translation_progress
        ),
        "roll_mismatch": (
            -config.roll_mismatch * inputs["mismatch_progress"]
        ),
        "backward": -config.backward * inputs["backward"],
        "action_rate": -config.action_rate * inputs["action_rate"],
        "residual_action": (
            -config.residual_action * inputs["residual_action_cost"]
        ),
        "torque": -config.torque * inputs["torque_cost"],
        "airborne": -config.airborne * inputs["airborne"],
        "stuck": -config.stuck * inputs["stuck_deficit"],
        "foot_gap": (
            -config.foot_gap
            * xp.maximum(
                inputs["foot_distance"] - config.foot_gap_threshold_m,
                0.0,
            )
        ),
        "collision": -collision_cost,
        "termination": -termination_penalty * inputs["failed"],
        "early_termination": (
            -termination_penalty
            * config.early_termination_scale
            * inputs["remaining_fraction"]
            * inputs["failed"]
        ),
    }
