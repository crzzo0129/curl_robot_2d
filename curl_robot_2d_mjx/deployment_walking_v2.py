"""Dependency-light helpers for the deploy-interface walking V2 policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET


DEPLOY_SINGLE_OBSERVATION_SIZE = 36
DEPLOY_OBSERVATION_HISTORY = 20
DEPLOY_OBSERVATION_SIZE = (
    DEPLOY_SINGLE_OBSERVATION_SIZE * DEPLOY_OBSERVATION_HISTORY
)
DEPLOY_ACTION_SIZE = 12
LEFT_RIGHT_LEG_PERMUTATION = (1, 0, 3, 2)


def ground_only_collision_xml(xml):
    """Disable robot self-contact while preserving every ground contact geom."""

    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no <worldbody>")
    for body in worldbody.iter("body"):
        for geom in body.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "1")

    contact = root.find("contact")
    if contact is not None:
        for pair in list(contact.findall("pair")):
            contact.remove(pair)
    return ET.tostring(root, encoding="unicode")


def straight_speed_terms(
    xp,
    command_speed,
    forward_speed,
    *,
    error_weight,
    progress_weight,
    minimum_scale=0.10,
):
    """Bounded shaping that makes standing worse than tracking a slow command."""

    scale = xp.maximum(xp.abs(command_speed), minimum_scale)
    normalized_error = (forward_speed - command_speed) / scale
    error_penalty = error_weight * xp.clip(
        xp.square(normalized_error), 0.0, 4.0
    )
    progress_reward = progress_weight * xp.clip(
        forward_speed / scale, -1.0, 1.0
    )
    return error_penalty, progress_reward


def bounded_normalized_square(xp, value, scale, maximum=4.0):
    """Dimensionless squared error with an explicit physical tolerance."""

    return xp.clip(xp.square(value / scale), 0.0, maximum)


def smooth_normalized_square(xp, value, scale):
    """Pseudo-Huber error that stays quadratic near zero without a flat cap."""

    normalized = value / scale
    return 2.0 * (xp.sqrt(1.0 + xp.square(normalized)) - 1.0)


def quaternion_yaw(xp, quaternion):
    """Return planar yaw for a MuJoCo wxyz quaternion."""

    w, x, y, z = (quaternion[index] for index in range(4))
    return xp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def signed_cross_track_error(xp, position_xy, origin_xy, heading):
    """Return signed world displacement normal to the reference heading."""

    displacement = position_xy - origin_xy
    return (
        -xp.sin(heading) * displacement[0]
        + xp.cos(heading) * displacement[1]
    )


def mirror_deploy_action(xp, action):
    """Swap left/right legs in canonical FL, FR, RL, RR action order."""

    action = xp.asarray(action)
    legs = xp.reshape(action, action.shape[:-1] + (4, 3))
    mirrored = legs[..., xp.asarray(LEFT_RIGHT_LEG_PERMUTATION), :]
    return xp.reshape(mirrored, action.shape)


def mirror_deploy_observation(xp, observation):
    """Reflect every 36-value frame in a 20-frame deploy observation."""

    observation = xp.asarray(observation)
    if observation.shape[-1] != DEPLOY_OBSERVATION_SIZE:
        raise ValueError(
            "deploy observation must contain "
            f"{DEPLOY_OBSERVATION_SIZE} values"
        )
    frames = xp.reshape(
        observation,
        observation.shape[:-1]
        + (DEPLOY_OBSERVATION_HISTORY, DEPLOY_SINGLE_OBSERVATION_SIZE),
    )
    mirrored = xp.concatenate(
        (
            frames[..., 0:3] * xp.asarray((-1.0, 1.0, -1.0)),
            frames[..., 3:6] * xp.asarray((1.0, -1.0, 1.0)),
            frames[..., 6:9] * xp.asarray((1.0, -1.0, -1.0)),
            frames[..., 9:12] * xp.asarray((1.0, -1.0, 1.0)),
            mirror_deploy_action(xp, frames[..., 12:24]),
            mirror_deploy_action(xp, frames[..., 24:36]),
        ),
        axis=-1,
    )
    return xp.reshape(mirrored, observation.shape)


@dataclass(frozen=True)
class DeployEvaluation:
    """Fixed-command physical metrics used to select checkpoints."""

    survived_fraction: float
    command_speed_m_s: float
    forward_speed_m_s: float
    lateral_speed_m_s: float
    yaw_rate_rad_s: float
    heading_change_rad: float
    front_hip_amplitude_ratio: float
    front_hip_measured_amplitude_ratio: float = 1.0
    front_rear_pose_rms_rad: float = 0.0
    stand_action_delta_rms: float = 0.0
    stand_joint_velocity_rms_rad_s: float = 0.0
    stand_body_angular_velocity_rms_rad_s: float = 0.0
    stand_action_rms: float = 0.0
    front_rear_leg_length_error_m: float = 0.0
    stand_body_linear_velocity_rms_m_s: float = 0.0
    stand_height_std_m: float = 0.0
    lateral_drift_m: float = 0.0


def deploy_checkpoint_rank(report: DeployEvaluation) -> tuple[float, ...]:
    """Lexicographic rank that prioritizes survival and straight tracking."""

    values = (
        report.survived_fraction,
        report.command_speed_m_s,
        report.forward_speed_m_s,
        report.lateral_speed_m_s,
        report.yaw_rate_rad_s,
        report.heading_change_rad,
        report.front_hip_amplitude_ratio,
        report.front_hip_measured_amplitude_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        return (float("-inf"),) * 10
    speed_error = abs(report.forward_speed_m_s - report.command_speed_m_s)
    action_amplitude_error = abs(report.front_hip_amplitude_ratio - 1.0)
    measured_amplitude_error = abs(
        report.front_hip_measured_amplitude_ratio - 1.0
    )
    completed = float(report.survived_fraction >= 0.999)
    direction_ok = float(
        abs(report.heading_change_rad) <= 0.20
        and abs(report.yaw_rate_rad_s) <= 0.08
    )
    amplitude_ok = float(measured_amplitude_error <= 0.20)
    speed_ok = float(speed_error <= 0.05)
    return (
        completed,
        direction_ok,
        amplitude_ok,
        speed_ok,
        report.survived_fraction,
        -abs(report.heading_change_rad),
        -abs(report.yaw_rate_rad_s),
        -speed_error,
        -measured_amplitude_error,
        -action_amplitude_error,
    )


def deploy_checkpoint_rank_v3(report: DeployEvaluation) -> tuple[float, ...]:
    """Rank V3 checkpoints with posture and stand smoothness gates."""

    base_rank = deploy_checkpoint_rank(report)
    extra_values = (
        report.front_rear_pose_rms_rad,
        report.stand_action_delta_rms,
        report.stand_joint_velocity_rms_rad_s,
        report.stand_body_angular_velocity_rms_rad_s,
        report.stand_action_rms,
    )
    if not all(math.isfinite(value) for value in extra_values):
        return (float("-inf"),) * 17

    posture_ok = float(report.front_rear_pose_rms_rad <= 0.10)
    stand_smooth_ok = float(
        report.stand_action_delta_rms <= 0.05
        and report.stand_joint_velocity_rms_rad_s <= 0.20
        and report.stand_body_angular_velocity_rms_rad_s <= 0.10
    )
    return (
        *base_rank[:4],
        posture_ok,
        stand_smooth_ok,
        base_rank[4],
        -report.front_rear_pose_rms_rad,
        -report.stand_body_angular_velocity_rms_rad_s,
        -report.stand_joint_velocity_rms_rad_s,
        -report.stand_action_delta_rms,
        -report.stand_action_rms,
        *base_rank[5:],
    )


def deploy_checkpoint_rank_v4(
    reports: tuple[DeployEvaluation, ...] | list[DeployEvaluation],
) -> tuple[float, ...]:
    """Rank a multi-speed V4 suite by its worst physical failure mode."""

    reports = tuple(reports)
    if not reports:
        return (float("-inf"),) * 22
    values = tuple(
        value
        for report in reports
        for value in (
            report.survived_fraction,
            report.command_speed_m_s,
            report.forward_speed_m_s,
            report.lateral_speed_m_s,
            report.yaw_rate_rad_s,
            report.heading_change_rad,
            report.lateral_drift_m,
            report.front_hip_amplitude_ratio,
            report.front_hip_measured_amplitude_ratio,
            report.front_rear_pose_rms_rad,
            report.front_rear_leg_length_error_m,
            report.stand_action_delta_rms,
            report.stand_joint_velocity_rms_rad_s,
            report.stand_body_angular_velocity_rms_rad_s,
            report.stand_body_linear_velocity_rms_m_s,
            report.stand_action_rms,
            report.stand_height_std_m,
        )
    )
    if not all(math.isfinite(value) for value in values):
        return (float("-inf"),) * 22

    speed_errors = tuple(
        abs(report.forward_speed_m_s - report.command_speed_m_s)
        for report in reports
    )
    measured_amplitude_errors = tuple(
        abs(report.front_hip_measured_amplitude_ratio - 1.0)
        for report in reports
    )
    action_amplitude_errors = tuple(
        abs(report.front_hip_amplitude_ratio - 1.0)
        for report in reports
    )
    completed = float(
        all(report.survived_fraction >= 0.999 for report in reports)
    )
    direction_ok = float(
        all(
            abs(report.heading_change_rad) <= 0.20
            and abs(report.yaw_rate_rad_s) <= 0.08
            for report in reports
        )
    )
    lateral_drift_ok = float(
        all(abs(report.lateral_drift_m) <= 0.12 for report in reports)
    )
    speed_ok = float(all(error <= 0.06 for error in speed_errors))
    amplitude_ok = float(
        all(error <= 0.25 for error in measured_amplitude_errors)
    )
    leg_length_ok = float(
        max(report.front_rear_leg_length_error_m for report in reports)
        <= 0.012
    )
    posture_ok = float(
        max(report.front_rear_pose_rms_rad for report in reports) <= 0.12
    )
    stand_smooth_ok = float(
        max(report.stand_action_delta_rms for report in reports) <= 0.03
        and max(
            report.stand_joint_velocity_rms_rad_s for report in reports
        )
        <= 0.20
        and max(
            report.stand_body_angular_velocity_rms_rad_s
            for report in reports
        )
        <= 0.10
        and max(
            report.stand_body_linear_velocity_rms_m_s
            for report in reports
        )
        <= 0.03
        and max(report.stand_height_std_m for report in reports) <= 0.008
    )
    return (
        completed,
        direction_ok,
        lateral_drift_ok,
        speed_ok,
        amplitude_ok,
        leg_length_ok,
        posture_ok,
        stand_smooth_ok,
        min(report.survived_fraction for report in reports),
        -max(abs(report.lateral_drift_m) for report in reports),
        -max(abs(report.heading_change_rad) for report in reports),
        -max(abs(report.yaw_rate_rad_s) for report in reports),
        -max(abs(report.lateral_speed_m_s) for report in reports),
        -max(speed_errors),
        -sum(speed_errors) / len(speed_errors),
        -max(
            report.front_rear_leg_length_error_m for report in reports
        ),
        -max(report.front_rear_pose_rms_rad for report in reports),
        -max(
            report.stand_body_angular_velocity_rms_rad_s
            for report in reports
        ),
        -max(
            report.stand_body_linear_velocity_rms_m_s
            for report in reports
        ),
        -max(
            report.stand_joint_velocity_rms_rad_s for report in reports
        ),
        -max(report.stand_action_delta_rms for report in reports),
        -max(
            max(measured_amplitude_errors),
            max(action_amplitude_errors),
        ),
    )
