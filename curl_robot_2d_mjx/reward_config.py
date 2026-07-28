"""Frequently tuned reward settings for planar rolling RL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingRewardConfig:
    """All reward weights and reward-only thresholds in one editable place."""

    progress_clip_rad: float = 0.25
    roll_progress: float = 5.0
    roll_mismatch: float = 0.5
    backward: float = 2.0
    action_rate: float = 0.02
    residual_action: float = 0.0
    torque: float = 0.01
    airborne: float = 0.15
    foot_gap: float = 5.0
    foot_gap_threshold_m: float = 0.20
    termination: float = 5.0
    early_termination_scale: float = 1.0

    # Collision reward terms mirror the collision-constrained CEM objective.
    allowed_foot_penetration_m: float = 0.0005
    forbidden_contact_time: float = 6.0
    forbidden_penetration_integral: float = 20000.0
    maximum_forbidden_penetration: float = 2500.0
    allowed_excess_integral: float = 12000.0
    maximum_allowed_excess: float = 2500.0
    leg_crossing: float = 100.0
