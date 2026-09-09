"""Dense, mode-conditioned rewards for the 3-D transition policy."""

from __future__ import annotations

from dataclasses import dataclass, replace


TRANSITION_REWARD_TERM_NAMES_3D = (
    "brake_speed",
    "brake_progress",
    "brake_capture",
    "deploy_pose",
    "deploy_progress",
    "upright",
    "height",
    "support",
    "stabilize",
    "stabilize_pose",
    "ready",
    "action_rate",
    "target_rate",
    "reference_tracking",
    "touchdown_speed",
    "target_acceleration",
    "hold_joint_motion",
    "hold_body_motion",
    "hold_foot_slip",
    "deploy_instability",
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
    brake_capture: float = 1.5
    deploy_pose: float = 2.0
    deploy_pose_sigma_rad: float = 0.35
    deploy_progress: float = 1.5
    upright: float = 1.5
    upright_sigma_rad: float = 0.45
    height: float = 0.6
    height_sigma_m: float = 0.06
    support: float = 0.8
    stabilize: float = 1.5
    stabilize_pose: float = 2.0
    stabilize_speed_sigma: float = 0.20
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
    target_rate: float = 0.0
    target_rate_sigma_rad_s: float = 10.0
    smooth_stand: bool = False
    deploy_only_smoothing: bool = False
    target_acceleration: float = 0.0
    target_acceleration_sigma_rad_s2: float = 500.0
    hold_target_rate: float = 0.0
    hold_joint_velocity: float = 0.0
    deploy_instability: float = 0.0
    reference_tracking: float = 0.0
    reference_tracking_sigma_rad: float = 0.30
    touchdown_speed: float = 0.0
    touchdown_speed_sigma_m_s: float = 0.5
    hold_body_motion: float = 0.0
    hold_body_speed_sigma: float = 0.20
    hold_foot_slip: float = 0.0


def smooth_stand_reward_config_3d():
    """Opt-in absolute-policy refinement; preserve the existing action ABI."""
    return Transition3DRewardConfig(
        smooth_stand=True, brake_speed=0.0, brake_progress=0.0,
        brake_capture=0.0, stabilize=4.0, stabilize_pose=0.5,
        stabilize_speed_sigma=0.15, action_rate=0.10,
        target_rate=0.10, joint_velocity=0.05, foot_slip=0.40,
        nonfoot_contact=0.30, impact=0.0,
    )


def smooth_deploy_reward_config_3d():
    """V2: finite smoothing window, then Stand tracking rather than idle bonus."""
    return replace(smooth_stand_reward_config_3d(),
                   deploy_only_smoothing=True, target_rate=1.0,
                   action_rate=0.5, joint_velocity=0.20, foot_slip=0.8,
                   stabilize_pose=4.0)


def deploy_window_fraction_3d(xp, step_count, control_dt, duration):
    """Overlap of this control interval with [handoff, handoff + duration)."""
    return xp.clip((duration - step_count * control_dt) / control_dt, 0.0, 1.0)


def smooth_deploy_v3_reward_config_3d():
    return replace(smooth_deploy_reward_config_3d(),
                   target_rate=2.0, joint_velocity=0.40,
                   target_acceleration=0.20, hold_target_rate=0.5,
                   hold_joint_velocity=0.20, deploy_instability=1.0)


def reward_terms_roll_to_stand_3d(xp, config, inputs):
    """Dynamic recovery; low speed is rewarded only with upright foot support.

    Retain the metric ABI of the legacy task, without its brake-first rewards.
    Normal shell contacts during recovery carry no constant collision penalty.
    """
    terms = reward_terms_transition_3d(xp, config, inputs)
    pose = xp.exp(-xp.square(inputs["reference_pose_error_rms"] /
                            config.deploy_pose_sigma_rad))
    upright = xp.exp(-xp.square(inputs["upright_tilt"] / config.upright_sigma_rad))
    height = xp.exp(-xp.square(inputs["root_height_error"] / config.height_sigma_m))
    foot_support = inputs["support_fraction"] / (1.0 + inputs["nonfoot_contact_count"])
    standing = pose * upright * height * foot_support
    zero = xp.asarray(0.0)
    terms.update(brake_speed=zero, brake_progress=zero, brake_capture=zero,
                 stabilize_pose=zero, nonfoot_contact=zero,
                 upright=config.upright * upright,
                 height=config.height * upright * height,
                 support=config.support * upright * height * foot_support,
                 stabilize=config.stabilize * standing * xp.exp(-xp.square(
                     inputs["combined_speed"] / config.stabilize_speed_sigma)))
    if config.smooth_stand:
        # Keep recovery free to rotate; apply holding costs near supported upright poses.
        hold_gate = upright * height * inputs["support_fraction"]
        terms["stabilize_pose"] = -config.stabilize_pose * hold_gate * xp.minimum(
            xp.square(inputs["reference_pose_error_rms"] / config.deploy_pose_sigma_rad), 4.0)
        terms["nonfoot_contact"] = -config.nonfoot_contact * hold_gate * inputs["nonfoot_contact_count"]
    if config.deploy_only_smoothing:
        window = inputs["deploy_window_fraction"]
        for name in ("stabilize", "action_rate", "target_rate", "joint_velocity", "foot_slip"):
            terms[name] *= window
        # After deployment, these become non-positive tracking costs. A near-Stand
        # timeout cannot collect a perpetual posture/support living bonus.
        for name, maximum in (("deploy_pose", config.deploy_pose),
                              ("upright", config.upright),
                              ("height", config.height), ("support", config.support)):
            terms[name] -= (1.0 - window) * maximum
        # Track the requested pose even if the robot avoids the supported-upright gate.
        terms["stabilize_pose"] = -(1.0 - window) * config.stabilize_pose * xp.minimum(
            xp.square(inputs["reference_pose_error_rms"] / config.deploy_pose_sigma_rad), 4.0)
        # Penalize target reversals even in flight; no foot-contact gate.
        terms["target_acceleration"] = -config.target_acceleration * (
            inputs.get("target_acceleration_squared", 0.0)
            / config.target_acceleration_sigma_rad_s2**2)
        terms["hold_joint_motion"] = -(1.0 - window) * (
            config.hold_target_rate * inputs.get("target_rate_squared", 0.0)
            / config.target_rate_sigma_rad_s**2
            + config.hold_joint_velocity * inputs["joint_velocity_squared"]
            / config.joint_velocity_sigma_rad_s**2)
        # Post-deploy costs, not a living bonus. Do not gate on READY or support:
        # a policy must not avoid these costs by remaining just outside the gate.
        terms["hold_body_motion"] = -(1.0 - window) * config.hold_body_motion * xp.minimum(
            xp.square(inputs["combined_speed"] / config.hold_body_speed_sigma), 4.0)
        terms["hold_foot_slip"] = -(1.0 - window) * config.hold_foot_slip * (
            inputs["foot_slip_velocity_squared"] / config.foot_slip_sigma_m_s**2)
        # Near upright, suppress residual body motion; preserve initial rolling momentum.
        terms["deploy_instability"] = -config.deploy_instability * window * upright * (
            1.0 - xp.exp(-xp.square(inputs["combined_speed"] / config.stabilize_speed_sigma)))
    terms["reference_tracking"] = -config.reference_tracking * (
        inputs.get("executed_reference_error_squared", 0.0) / config.reference_tracking_sigma_rad**2)
    terms["touchdown_speed"] = -config.touchdown_speed * (
        inputs.get("touchdown_downward_speed_squared", 0.0) / config.touchdown_speed_sigma_m_s**2)
    return terms


def guided_absolute_reward_config_3d():
    return replace(smooth_deploy_v3_reward_config_3d(), reference_tracking=2.0)


def guided_landing_reward_config_3d():
    """Small refinement of guided_absolute; touchdown velocity is a proxy, not force."""
    return replace(guided_absolute_reward_config_3d(),
                   reference_tracking=2.5, target_acceleration=0.30,
                   joint_velocity=0.50, hold_joint_velocity=0.30,
                   touchdown_speed=0.5)


def guided_hold_reward_config_3d():
    """Preserve guided_absolute deployment rewards, refine only the hold phase."""
    return replace(guided_absolute_reward_config_3d(),
                   hold_target_rate=0.75, hold_joint_velocity=0.30,
                   hold_body_motion=0.50, hold_foot_slip=0.20)


def touchdown_downward_speed_squared(xp, previous_contact, contact,
                                     previous_velocity, interval_velocity):
    """20 ms contact-onset proxy; retain pre-impact speed after foot arrest."""
    onset = (contact > 0) & (previous_contact <= 0)
    downward = xp.maximum(0.0, -xp.minimum(previous_velocity[:, 2], interval_velocity[:, 2]))
    return xp.mean(onset * xp.square(downward))


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
    support_quality = support_fraction / (1.0 + inputs["nonfoot_contact_count"])
    standing_quality = deploy_score * support_quality
    stable_score = (standing_quality * upright_score * height_score
                    * xp.exp(-xp.square(speed / config.stabilize_speed_sigma)))
    # DEPLOY still needs shaping before feet land. In STABILIZE, belly/shell
    # support with an upright torso must not collect the standing bonuses.
    pose_support_gate = deploy + stabilize * standing_quality
    return {
        "brake_speed": config.brake_speed * brake * brake_score,
        "brake_progress": config.brake_progress * brake * brake_delta,
        # Low speed alone could reward stopping upside-down indefinitely.
        "brake_capture": config.brake_capture * brake * brake_score * upright_score,
        "deploy_pose": config.deploy_pose * deploy * deploy_score,
        "deploy_progress": config.deploy_progress * deploy * deploy_delta,
        "upright": config.upright * pose_support_gate * upright_score,
        "height": config.height * pose_support_gate * height_score,
        "support": config.support * (deploy * support_fraction
                                     + stabilize * standing_quality * height_score),
        "stabilize": config.stabilize * stabilize * stable_score,
        "stabilize_pose": -config.stabilize_pose * stabilize * xp.minimum(
            xp.square(pose_error / config.deploy_pose_sigma_rad), 1.0),
        "ready": config.ready * inputs["newly_ready"],
        "action_rate": -config.action_rate * inputs["action_rate_squared"],
        "target_rate": -config.target_rate * (
            inputs.get("target_rate_squared", 0.0) / config.target_rate_sigma_rad_s**2),
        "target_acceleration": xp.asarray(0.0),
        "reference_tracking": xp.asarray(0.0),
        "touchdown_speed": xp.asarray(0.0),
        "hold_joint_motion": xp.asarray(0.0),
        "hold_body_motion": xp.asarray(0.0),
        "hold_foot_slip": xp.asarray(0.0),
        "deploy_instability": xp.asarray(0.0),
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
