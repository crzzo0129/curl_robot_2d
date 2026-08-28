"""Dense, mode-conditioned rewards for the 3-D transition policy."""

from __future__ import annotations

from dataclasses import dataclass


TRANSITION_REWARD_TERM_NAMES_3D = (
    "brake_speed",
    "brake_progress",
    "deploy_pose",
    "deploy_progress",
    "upright",
    "height",
    "support",
    "stabilize",
    "ready",
    "action_rate",
    "action_magnitude",
    "joint_velocity",
    "foot_slip",
    "impact",
    "nonfoot_contact",
    "termination",
)


@dataclass(frozen=True)
class Transition3DRewardConfig:
    brake_speed: float = 2.5
    brake_speed_sigma: float = 1.25
    brake_progress: float = 1.0
    deploy_pose: float = 2.0
    deploy_pose_sigma_rad: float = 0.35
    deploy_progress: float = 1.5
    upright: float = 1.5
    upright_sigma_rad: float = 0.45
    height: float = 0.6
    height_sigma_m: float = 0.06
    support: float = 0.8
    stabilize: float = 1.5
    ready: float = 12.0
    action_rate: float = 0.03
    action_magnitude: float = 0.005
    joint_velocity: float = 0.01
    joint_velocity_sigma_rad_s: float = 8.0
    foot_slip: float = 0.10
    foot_slip_sigma_m_s: float = 0.25
    impact: float = 0.02
    impact_scale_n: float = 80.0
    nonfoot_contact: float = 0.15
    termination: float = 20.0


def reward_terms_transition_3d(
    xp,
    config: Transition3DRewardConfig,
    inputs,
):
    """Return named terms; ``xp`` may be numpy or jax.numpy."""

    brake = inputs["mode_brake"]
    deploy = inputs["mode_deploy"]
    stabilize = inputs["mode_stabilize"]
    speed = inputs["combined_speed"]
    previous_speed = inputs["previous_combined_speed"]
    pose_error = inputs["reference_pose_error_rms"]
    previous_pose_error = inputs["previous_reference_pose_error_rms"]
    upright_tilt = inputs["upright_tilt"]
    root_height_error = inputs["root_height_error"]
    support_fraction = inputs["support_fraction"]

    brake_score = xp.exp(-xp.square(speed / config.brake_speed_sigma))
    brake_delta = xp.clip(previous_speed - speed, -1.0, 1.0)
    deploy_score = xp.exp(
        -xp.square(pose_error / config.deploy_pose_sigma_rad)
    )
    deploy_delta = xp.clip(previous_pose_error - pose_error, -1.0, 1.0)
    upright_score = xp.exp(
        -xp.square(upright_tilt / config.upright_sigma_rad)
    )
    height_score = xp.exp(
        -xp.square(root_height_error / config.height_sigma_m)
    )
    stable_score = (
        deploy_score * upright_score * height_score * support_fraction
    )
    return {
        "brake_speed": config.brake_speed * brake * brake_score,
        "brake_progress": config.brake_progress * brake * brake_delta,
        "deploy_pose": config.deploy_pose * deploy * deploy_score,
        "deploy_progress": config.deploy_progress * deploy * deploy_delta,
        "upright": config.upright * (deploy + stabilize) * upright_score,
        "height": config.height * (deploy + stabilize) * height_score,
        "support": config.support * (deploy + stabilize) * support_fraction,
        "stabilize": config.stabilize * stabilize * stable_score,
        "ready": config.ready * inputs["newly_ready"],
        "action_rate": -config.action_rate * inputs["action_rate_squared"],
        "action_magnitude": (
            -config.action_magnitude * inputs["action_squared"]
        ),
        "joint_velocity": -config.joint_velocity * (
            inputs["joint_velocity_squared"]
            / (config.joint_velocity_sigma_rad_s**2)
        ),
        "foot_slip": -config.foot_slip * (
            inputs["foot_slip_velocity_squared"]
            / (config.foot_slip_sigma_m_s**2)
        ),
        "impact": -config.impact * xp.square(
            inputs["contact_force_peak_n"] / config.impact_scale_n
        ),
        "nonfoot_contact": (
            -config.nonfoot_contact * inputs["nonfoot_contact_count"]
        ),
        "termination": -config.termination * inputs["failed"],
    }

