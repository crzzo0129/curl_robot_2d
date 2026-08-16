"""Reference-free reward terms for direct 3-D locomotion training."""

from __future__ import annotations

from dataclasses import dataclass


WALKING_REWARD_TERM_NAMES_3D = (
    "alive",
    "velocity_tracking",
    "overspeed",
    "yaw_rate_tracking",
    "forward_progress",
    "upright",
    "stagnation",
    "height",
    "heading",
    "lateral",
    "vertical_velocity",
    "angular_velocity",
    "orientation",
    "angular_momentum",
    "foot_air_time",
    "gait_contact",
    "swing_clearance",
    "foot_clearance",
    "foot_slip",
    "soft_landing",
    "action_rate",
    "action_magnitude",
    "joint_velocity",
    "joint_acceleration",
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
    velocity_tracking_vertical_weight: float = 0.0
    velocity_tracking_upright_gate: float = 1.0
    overspeed: float = 0.0
    overspeed_margin_m_s: float = 0.05
    overspeed_scale_m_s: float = 0.15
    yaw_rate_tracking: float = 0.75
    yaw_rate_tracking_sigma_rad_s: float = 0.35
    yaw_rate_tracking_roll_pitch_weight: float = 0.0
    forward_progress: float = 0.0
    upright: float = 0.50
    upright_sigma_rad: float = 0.30
    stagnation: float = 0.0
    stagnation_window_s: float = 1.0
    stagnation_min_progress_m: float = 0.05
    upright_stagnation_gate: float = 0.0
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
    orientation: float = 0.0
    angular_momentum: float = 0.0

    foot_air_time: float = 0.15
    foot_air_time_threshold_s: float = 0.08
    foot_air_time_cap_s: float = 0.30
    gait_contact: float = 0.0
    swing_clearance: float = 0.05
    swing_clearance_m: float = 0.015
    swing_clearance_target_sigma_m: float = 0.0075
    swing_clearance_target_tracking: float = 0.0
    swing_clearance_speed_m_s: float = 0.10
    foot_clearance: float = 0.0
    foot_clearance_target_m: float = 0.025
    foot_slip: float = 0.05
    foot_slip_sigma_m_s: float = 0.15
    soft_landing: float = 0.0
    soft_landing_velocity_m_s: float = 0.50

    action_rate: float = 0.020
    action_magnitude: float = 0.004
    joint_velocity: float = 0.005
    joint_velocity_sigma_rad_s: float = 8.0
    joint_acceleration: float = 0.0
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
    overspeed_scale = max(config.overspeed_scale_m_s, 1.0e-6)
    planar_velocity_error = (
        inputs["planar_velocity_error_norm"]
        if "planar_velocity_error_norm" in inputs
        else xp.abs(inputs["forward_velocity_error"])
    )
    velocity_error_squared = (
        xp.square(planar_velocity_error)
        + config.velocity_tracking_vertical_weight
        * xp.square(inputs.get("vertical_velocity", xp.asarray(0.0)))
    )
    velocity_score = xp.exp(
        -velocity_error_squared / (config.velocity_tracking_sigma_m_s**2)
    )
    overspeed_cost = xp.square(
        xp.clip(
            inputs.get("overspeed", xp.asarray(0.0)),
            0.0,
            overspeed_scale,
        )
        / overspeed_scale
    )
    yaw_rate_error = inputs.get("yaw_rate_error", xp.asarray(0.0))
    yaw_error_squared = (
        xp.square(yaw_rate_error)
        + config.yaw_rate_tracking_roll_pitch_weight
        * inputs.get(
            "roll_pitch_angular_velocity_squared", xp.asarray(0.0)
        )
    )
    yaw_rate_score = xp.exp(
        -yaw_error_squared / (config.yaw_rate_tracking_sigma_rad_s**2)
    )
    upright_score = xp.exp(
        -xp.square(inputs["upright_tilt"] / config.upright_sigma_rad)
    )
    velocity_gate_strength = xp.clip(
        xp.asarray(config.velocity_tracking_upright_gate), 0.0, 1.0
    )
    velocity_upright_gate = (
        1.0
        - velocity_gate_strength
        + velocity_gate_strength * upright_score
    )
    stagnation_fraction = xp.clip(
        inputs.get("stagnation_fraction", xp.asarray(0.0)), 0.0, 1.0
    )
    upright_progress_gate = xp.clip(
        1.0 - config.upright_stagnation_gate * stagnation_fraction,
        0.0,
        1.0,
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
            config.velocity_tracking * velocity_score * velocity_upright_gate
        ),
        "overspeed": -config.overspeed * overspeed_cost,
        "yaw_rate_tracking": config.yaw_rate_tracking * yaw_rate_score,
        "forward_progress": (
            config.forward_progress
            * inputs["normalized_forward_velocity"]
            * upright_score
        ),
        "upright": config.upright * upright_score * upright_progress_gate,
        "stagnation": -config.stagnation * stagnation_fraction,
        "height": -config.height * height_cost,
        "heading": -config.heading * heading_cost,
        "lateral": -lateral_cost,
        "vertical_velocity": (
            -config.vertical_velocity * vertical_velocity_cost
        ),
        "angular_velocity": (
            -config.angular_velocity * angular_velocity_cost
        ),
        "orientation": (
            -config.orientation
            * inputs.get("projected_gravity_xy_squared", 0.0)
        ),
        "angular_momentum": (
            -config.angular_momentum
            * inputs.get("angular_momentum_squared", 0.0)
        ),
        "foot_air_time": (
            config.foot_air_time
            * inputs["foot_air_time_reward"]
            * inputs.get("locomotion_active", 1.0)
        ),
        "gait_contact": (
            config.gait_contact
            * inputs.get("gait_contact_reward", 0.0)
            * inputs.get("locomotion_active", 1.0)
        ),
        "swing_clearance": (
            config.swing_clearance
            * inputs["swing_clearance_reward"]
            * inputs.get("locomotion_active", 1.0)
        ),
        "foot_clearance": (
            -config.foot_clearance
            * inputs.get("foot_clearance_cost", 0.0)
            * inputs.get("locomotion_active", 1.0)
        ),
        "foot_slip": -config.foot_slip * foot_slip_cost,
        "soft_landing": (
            -config.soft_landing * inputs.get("soft_landing_cost", 0.0)
        ),
        "action_rate": -config.action_rate * inputs["action_rate_cost"],
        "action_magnitude": (
            -config.action_magnitude * inputs["action_magnitude_cost"]
        ),
        "joint_velocity": (
            -config.joint_velocity * joint_velocity_cost
        ),
        "joint_acceleration": (
            -config.joint_acceleration
            * inputs.get("joint_acceleration_squared", 0.0)
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
