"""Evaluation gates and failure classification for rolling-stop episodes."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StopEpisodeMetrics:
    survived: bool
    numerical_failure: bool
    stop_time_s: float
    extra_distance_m: float
    final_linear_speed_m_s: float
    final_angular_speed_rad_s: float
    final_phase_error_rad: float
    final_pose_rms_error_rad: float
    settled_duration_s: float
    grounded_feet: int
    torso_contact_total_s: float
    hold_internal_contact_total_s: float
    leg_crossing: bool
    lateral_drift_m: float
    maximum_torque_nm: float


@dataclass(frozen=True)
class StopEvaluationGate:
    maximum_stop_time_s: float = 3.0
    maximum_extra_distance_m: float = 1.25
    maximum_final_linear_speed_m_s: float = 0.03
    maximum_final_angular_speed_rad_s: float = 0.10
    maximum_phase_error_rad: float = math.radians(10.0)
    maximum_pose_rms_error_rad: float = math.radians(5.0)
    minimum_settled_duration_s: float = 2.0
    minimum_grounded_feet: int = 2
    maximum_torso_contact_s: float = 0.0
    maximum_lateral_drift_m: float = 0.20
    maximum_torque_nm: float = 6.0
    require_zero_internal_contact_in_hold: bool = True


def stop_failure_reasons(
    metrics: StopEpisodeMetrics,
    gate: StopEvaluationGate,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if metrics.numerical_failure:
        reasons.append("numerical_failure")
    if not metrics.survived:
        reasons.append("did_not_survive")
    if metrics.stop_time_s > gate.maximum_stop_time_s:
        reasons.append("stop_timeout")
    if metrics.extra_distance_m > gate.maximum_extra_distance_m:
        reasons.append("excess_stop_distance")
    if abs(metrics.final_linear_speed_m_s) > gate.maximum_final_linear_speed_m_s:
        reasons.append("linear_speed")
    if abs(metrics.final_angular_speed_rad_s) > gate.maximum_final_angular_speed_rad_s:
        reasons.append("angular_speed")
    if abs(metrics.final_phase_error_rad) > gate.maximum_phase_error_rad:
        reasons.append("parking_phase")
    if metrics.final_pose_rms_error_rad > gate.maximum_pose_rms_error_rad:
        reasons.append("parking_pose")
    if metrics.settled_duration_s < gate.minimum_settled_duration_s:
        reasons.append("hold_duration")
    if metrics.grounded_feet < gate.minimum_grounded_feet:
        reasons.append("insufficient_support")
    if metrics.torso_contact_total_s > gate.maximum_torso_contact_s:
        reasons.append("torso_contact")
    if (
        gate.require_zero_internal_contact_in_hold
        and metrics.hold_internal_contact_total_s > 0.0
    ):
        reasons.append("internal_contact")
    if metrics.leg_crossing:
        reasons.append("leg_crossing")
    if abs(metrics.lateral_drift_m) > gate.maximum_lateral_drift_m:
        reasons.append("lateral_drift")
    if metrics.maximum_torque_nm > gate.maximum_torque_nm:
        reasons.append("torque_limit")
    return tuple(reasons)


def stop_episode_succeeded(
    metrics: StopEpisodeMetrics,
    gate: StopEvaluationGate,
) -> bool:
    return not stop_failure_reasons(metrics, gate)


def stop_selection_key(
    metrics: StopEpisodeMetrics,
    gate: StopEvaluationGate,
) -> tuple[float, ...]:
    """Lexicographic checkpoint key; higher tuples are better."""

    succeeded = stop_episode_succeeded(metrics, gate)
    safe = metrics.survived and not metrics.numerical_failure
    return (
        float(safe),
        float(succeeded),
        -float(len(stop_failure_reasons(metrics, gate))),
        metrics.settled_duration_s,
        -abs(metrics.final_linear_speed_m_s),
        -abs(metrics.final_angular_speed_rad_s),
        -metrics.stop_time_s,
        -metrics.extra_distance_m,
        -metrics.maximum_torque_nm,
    )


@dataclass(frozen=True)
class ParkPoseStaticMetrics:
    """Metrics measured while holding a keyframe under gravity."""

    survived: bool
    numerical_failure: bool
    duration_s: float
    final_linear_speed_m_s: float
    final_angular_speed_rad_s: float
    final_torso_tilt_rad: float
    maximum_torso_tilt_rad: float
    final_joint_pose_rms_error_rad: float
    grounded_feet: int
    internal_contact_total_s: float
    torso_ground_contact_total_s: float
    torso_internal_contact_total_s: float
    lateral_drift_m: float
    minimum_root_height_m: float
    maximum_torque_nm: float


@dataclass(frozen=True)
class ParkPoseStaticGate:
    maximum_final_linear_speed_m_s: float = 0.03
    maximum_final_angular_speed_rad_s: float = 0.10
    maximum_final_torso_tilt_rad: float = math.radians(5.0)
    maximum_joint_pose_rms_error_rad: float = math.radians(5.0)
    minimum_grounded_feet: int = 4
    maximum_internal_contact_s: float = 0.0
    maximum_torso_contact_s: float = 0.0
    maximum_lateral_drift_m: float = 0.02
    minimum_root_height_m: float = 0.10
    maximum_torque_nm: float = 2.1


def park_pose_failure_reasons(
    metrics: ParkPoseStaticMetrics,
    gate: ParkPoseStaticGate,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if metrics.numerical_failure:
        reasons.append("numerical_failure")
    if not metrics.survived:
        reasons.append("did_not_survive")
    if abs(metrics.final_linear_speed_m_s) > gate.maximum_final_linear_speed_m_s:
        reasons.append("linear_speed")
    if abs(metrics.final_angular_speed_rad_s) > gate.maximum_final_angular_speed_rad_s:
        reasons.append("angular_speed")
    if abs(metrics.final_torso_tilt_rad) > gate.maximum_final_torso_tilt_rad:
        reasons.append("torso_tilt")
    if metrics.final_joint_pose_rms_error_rad > gate.maximum_joint_pose_rms_error_rad:
        reasons.append("parking_pose")
    if metrics.grounded_feet < gate.minimum_grounded_feet:
        reasons.append("insufficient_support")
    if metrics.internal_contact_total_s > gate.maximum_internal_contact_s:
        reasons.append("internal_contact")
    if (
        metrics.torso_ground_contact_total_s > gate.maximum_torso_contact_s
        or metrics.torso_internal_contact_total_s > gate.maximum_torso_contact_s
    ):
        reasons.append("torso_contact")
    if abs(metrics.lateral_drift_m) > gate.maximum_lateral_drift_m:
        reasons.append("lateral_drift")
    if metrics.minimum_root_height_m < gate.minimum_root_height_m:
        reasons.append("root_height")
    if metrics.maximum_torque_nm > gate.maximum_torque_nm:
        reasons.append("torque_limit")
    return tuple(reasons)


def park_pose_succeeded(
    metrics: ParkPoseStaticMetrics,
    gate: ParkPoseStaticGate,
) -> bool:
    return not park_pose_failure_reasons(metrics, gate)
