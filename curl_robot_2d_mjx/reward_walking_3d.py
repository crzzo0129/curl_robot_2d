"""Reference-free reward terms for direct 3-D locomotion training."""

from __future__ import annotations

from dataclasses import dataclass


WALKING_REWARD_TERM_NAMES_3D = (
    "alive",
    "velocity_tracking",
    "yaw_rate_tracking",
    "forward_progress",
    "upright",
    "height",
    "heading",
    "lateral",
    "vertical_velocity",
    "angular_velocity",
    "foot_air_time",
    "swing_clearance",
    "foot_slip",
    "action_rate",
    "action_magnitude",
    "joint_velocity",
    "joint_limits",
    "stand_still",
    "torque",
    "collision",
    "termination",
    "early_termination",
)


@dataclass(frozen=True)
class Walking3DRewardConfig:
    """Scales for command tracking plus generic locomotion regularization."""

    alive: float = 0.0
    velocity_tracking: float = 2.00
    velocity_tracking_sigma_m_s: float = 0.20
    yaw_rate_tracking: float = 0.75
    yaw_rate_tracking_sigma_rad_s: float = 0.35
    forward_progress: float = 0.0
    upright: float = 0.50
    upright_sigma_rad: float = 0.30
    height: float = 0.0
    height_sigma_m: float = 0.050
    heading: float = 0.0
    heading_sigma_rad: float = 0.50
    lateral_velocity: float = 0.10
    lateral_drift: float = 0.0
    lateral_velocity_sigma_m_s: float = 0.20
    lateral_drift_sigma_m: float = 0.10
    vertical_velocity: float = 0.05
    vertical_velocity_sigma_m_s: float = 0.20
    angular_velocity: float = 0.08
    angular_velocity_sigma_rad_s: float = 0.75

    foot_air_time: float = 0.15
    foot_air_time_threshold_s: float = 0.08
    foot_air_time_cap_s: float = 0.30
    swing_clearance: float = 0.05
    swing_clearance_m: float = 0.015
    foot_slip: float = 0.05
    foot_slip_sigma_m_s: float = 0.15

    action_rate: float = 0.020
    action_magnitude: float = 0.004
    joint_velocity: float = 0.005
    joint_velocity_sigma_rad_s: float = 8.0
    joint_limits: float = 0.10
    stand_still: float = 0.20
    torque: float = 0.010

    nonfoot_contact: float = 2.0
    nonfoot_depth: float = 350.0
    self_contact: float = 0.8
    self_contact_depth: float = 250.0

    termination: float = 25.0
    severe_extra_termination: float = 15.0
    nonfinite_termination: float = 80.0
    early_termination_scale: float = 1.0


def reward_terms_walking_3d(xp, config: Walking3DRewardConfig, inputs):
    planar_velocity_error = (
        inputs["planar_velocity_error_norm"]
        if "planar_velocity_error_norm" in inputs
        else xp.abs(inputs["forward_velocity_error"])
    )
    velocity_score = xp.exp(
        -xp.square(
            planar_velocity_error / config.velocity_tracking_sigma_m_s
        )
    )
    yaw_rate_error = inputs.get("yaw_rate_error", xp.asarray(0.0))
    yaw_rate_score = xp.exp(
        -xp.square(
            yaw_rate_error / config.yaw_rate_tracking_sigma_rad_s
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
    vertical_velocity_cost = xp.square(
        inputs["vertical_velocity"] / config.vertical_velocity_sigma_m_s
    )
    angular_velocity_cost = (
        inputs["roll_pitch_angular_velocity_squared"]
        / (config.angular_velocity_sigma_rad_s**2)
    )
    foot_slip_cost = (
        inputs["foot_slip_velocity_squared"]
        / (config.foot_slip_sigma_m_s**2)
    )
    joint_velocity_cost = (
        inputs["joint_velocity_squared"]
        / (config.joint_velocity_sigma_rad_s**2)
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
        "velocity_tracking": (
            config.velocity_tracking * velocity_score * upright_score
        ),
        "yaw_rate_tracking": config.yaw_rate_tracking * yaw_rate_score,
        "forward_progress": (
            config.forward_progress
            * inputs["normalized_forward_velocity"]
            * upright_score
        ),
        "upright": config.upright * upright_score,
        "height": -config.height * height_cost,
        "heading": -config.heading * heading_cost,
        "lateral": -lateral_cost,
        "vertical_velocity": (
            -config.vertical_velocity * vertical_velocity_cost
        ),
        "angular_velocity": (
            -config.angular_velocity * angular_velocity_cost
        ),
        "foot_air_time": (
            config.foot_air_time
            * inputs["foot_air_time_reward"]
            * inputs.get("locomotion_active", 1.0)
        ),
        "swing_clearance": (
            -config.swing_clearance * inputs["swing_clearance_cost"]
        ),
        "foot_slip": -config.foot_slip * foot_slip_cost,
        "action_rate": -config.action_rate * inputs["action_rate_cost"],
        "action_magnitude": (
            -config.action_magnitude * inputs["action_magnitude_cost"]
        ),
        "joint_velocity": (
            -config.joint_velocity * joint_velocity_cost
        ),
        "joint_limits": (
            -config.joint_limits * inputs["joint_limit_cost"]
        ),
        "stand_still": (
            -config.stand_still * inputs.get("stand_still_cost", 0.0)
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
