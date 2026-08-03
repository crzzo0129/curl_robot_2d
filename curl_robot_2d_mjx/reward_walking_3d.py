"""Reward terms for the 3-D walking MJX environment."""

from __future__ import annotations

from dataclasses import dataclass


WALKING_REWARD_TERM_NAMES_3D = (
    "alive",
    "velocity_tracking",
    "forward_progress",
    "upright",
    "height",
    "heading",
    "lateral",
    "contact_schedule",
    "swing_clearance",
    "joint_tracking",
    "action_rate",
    "residual_action",
    "torque",
    "collision",
    "termination",
    "early_termination",
)


@dataclass(frozen=True)
class Walking3DRewardConfig:
    alive: float = 0.05
    velocity_tracking: float = 1.40
    velocity_tracking_sigma_m_s: float = 0.040
    forward_progress: float = 0.40
    upright: float = 0.45
    upright_sigma_rad: float = 0.30
    height: float = 0.50
    height_sigma_m: float = 0.050
    heading: float = 0.10
    heading_sigma_rad: float = 0.50
    lateral_velocity: float = 0.12
    lateral_drift: float = 0.18
    lateral_velocity_sigma_m_s: float = 0.20
    lateral_drift_sigma_m: float = 0.10
    stance_miss: float = 0.18
    swing_contact: float = 0.22
    swing_clearance: float = 0.12
    joint_tracking: float = 0.06
    action_rate: float = 0.015
    residual_action: float = 0.004
    torque: float = 0.008

    nonfoot_contact: float = 1.5
    nonfoot_depth: float = 350.0
    self_contact: float = 0.8
    self_contact_depth: float = 250.0

    termination: float = 25.0
    severe_extra_termination: float = 15.0
    nonfinite_termination: float = 80.0
    early_termination_scale: float = 1.0


def reward_terms_walking_3d(xp, config: Walking3DRewardConfig, inputs):
    velocity_score = xp.exp(
        -xp.square(
            inputs["forward_velocity_error"]
            / config.velocity_tracking_sigma_m_s
        )
    )
    upright_score = xp.exp(
        -xp.square(inputs["upright_tilt"] / config.upright_sigma_rad)
    )
    height_cost = xp.square(
        inputs["root_height_error"] / config.height_sigma_m
    )
    heading_cost = xp.square(
        inputs["heading_error"] / config.heading_sigma_rad
    )
    lateral_cost = (
        config.lateral_velocity
        * xp.square(
            inputs["lateral_velocity"]
            / config.lateral_velocity_sigma_m_s
        )
        + config.lateral_drift
        * xp.square(
            inputs["lateral_drift"] / config.lateral_drift_sigma_m
        )
    )
    collision_cost = (
        config.nonfoot_contact * inputs["nonfoot_contact_active"]
        + config.nonfoot_depth * inputs["nonfoot_depth"]
        + config.self_contact * inputs["self_contact_active"]
        + config.self_contact_depth * inputs["self_contact_depth"]
    )
    terminal_penalty = xp.where(
        inputs["failure_nonfinite"] > 0.0,
        config.nonfinite_termination,
        config.termination
        + config.severe_extra_termination * inputs["failure_severe"],
    )
    return {
        "alive": config.alive * (1.0 - inputs["failed"]),
        "velocity_tracking": config.velocity_tracking * velocity_score,
        "forward_progress": (
            config.forward_progress * inputs["normalized_forward_velocity"]
        ),
        "upright": config.upright * upright_score,
        "height": -config.height * height_cost,
        "heading": -config.heading * heading_cost,
        "lateral": -lateral_cost,
        "contact_schedule": -(
            config.stance_miss * inputs["stance_miss_fraction"]
            + config.swing_contact * inputs["swing_contact_fraction"]
        ),
        "swing_clearance": (
            -config.swing_clearance * inputs["swing_clearance_cost"]
        ),
        "joint_tracking": (
            -config.joint_tracking * inputs["joint_tracking_cost"]
        ),
        "action_rate": -config.action_rate * inputs["action_rate_cost"],
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
