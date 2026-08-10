"""Training-backend-independent rolling-stop task primitives.

The functions in this module deliberately avoid importing JAX or MuJoCo.
Array-producing helpers accept an ``xp`` module so NumPy and JAX callers use
the same phase, state-machine, and reference-scheduling contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import math


TAU = 2.0 * math.pi


class StopMode(IntEnum):
    ROLL = 0
    BRAKE_ALIGN = 1
    PARK_DEPLOY = 2
    HOLD = 3


@dataclass(frozen=True)
class ParkPose:
    """Desired orientation and joints for the final supported posture."""

    joint_targets_rad: tuple[float, ...]
    root_pitch_rad: float = 0.0
    foot_down_phase_rad: float = 0.0
    required_grounded_feet: int = 2

    def __post_init__(self) -> None:
        if not self.joint_targets_rad:
            raise ValueError("park pose requires at least one joint target")
        if not all(math.isfinite(value) for value in self.joint_targets_rad):
            raise ValueError("park joint targets must be finite")
        if not math.isfinite(self.root_pitch_rad):
            raise ValueError("park root pitch must be finite")
        if not math.isfinite(self.foot_down_phase_rad):
            raise ValueError("park phase must be finite")
        if self.required_grounded_feet < 1:
            raise ValueError("required grounded feet must be positive")


@dataclass(frozen=True)
class StopTaskConfig:
    """Mode transition, reference scheduling, and success thresholds."""

    direction: float = 1.0
    maximum_brake_duration_s: float = 3.0
    deploy_duration_s: float = 1.0
    required_hold_duration_s: float = 2.0
    phase_tolerance_rad: float = math.radians(10.0)
    joint_pose_rms_tolerance_rad: float = math.radians(5.0)
    linear_speed_tolerance_m_s: float = 0.03
    angular_speed_tolerance_rad_s: float = 0.10
    root_pitch_tolerance_rad: float = math.radians(5.0)
    maximum_brake_deceleration_rad_s2: float = 8.0
    brake_phase_margin_rad: float = math.radians(20.0)
    forbid_torso_contact: bool = True
    forbid_internal_contact: bool = True

    def __post_init__(self) -> None:
        if self.direction not in (-1.0, 1.0):
            raise ValueError("stop direction must be -1 or +1")
        for name in (
            "maximum_brake_duration_s",
            "deploy_duration_s",
            "required_hold_duration_s",
            "phase_tolerance_rad",
            "joint_pose_rms_tolerance_rad",
            "linear_speed_tolerance_m_s",
            "angular_speed_tolerance_rad_s",
            "root_pitch_tolerance_rad",
            "maximum_brake_deceleration_rad_s2",
            "brake_phase_margin_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class StopState:
    """Small state carried alongside the physics pipeline state."""

    mode: StopMode = StopMode.ROLL
    stop_command_time_s: float = math.inf
    mode_start_time_s: float = 0.0
    brake_start_phase_rad: float = 0.0
    target_phase_unwrapped_rad: float = 0.0
    initial_phase_distance_rad: float = 0.0
    settled_duration_s: float = 0.0

    @property
    def stop_requested(self) -> bool:
        return self.mode != StopMode.ROLL


@dataclass(frozen=True)
class StopTransitionInput:
    time_s: float
    body_phase_unwrapped_rad: float
    stop_command: bool
    linear_speed_m_s: float
    angular_speed_rad_s: float
    joint_pose_rms_error_rad: float
    root_pitch_error_rad: float
    grounded_feet: int
    torso_contact: bool = False
    internal_contact: bool = False


@dataclass(frozen=True)
class StopReferenceSchedule:
    mode: StopMode
    rolling_reference_scale: float
    parking_reference_blend: float
    target_phase_error_rad: float
    time_since_stop_s: float
    time_in_mode_s: float


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap one scalar angle to [-pi, pi)."""

    return (angle_rad + math.pi) % TAU - math.pi


def forward_phase_delta(
    current_phase_rad: float,
    target_phase_rad: float,
    direction: float = 1.0,
) -> float:
    """Return the nonnegative same-direction distance to target phase."""

    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1 or +1")
    return ((target_phase_rad - current_phase_rad) * direction) % TAU


def select_target_phase_unwrapped(
    current_phase_unwrapped_rad: float,
    park_phase_rad: float,
    direction: float = 1.0,
) -> tuple[float, float]:
    """Select the nearest same-direction occurrence of the parking phase."""

    distance = forward_phase_delta(
        current_phase_unwrapped_rad,
        park_phase_rad,
        direction,
    )
    return current_phase_unwrapped_rad + direction * distance, distance


def required_braking_phase_distance(
    angular_speed_rad_s: float,
    maximum_deceleration_rad_s2: float,
    margin_rad: float,
) -> float:
    """Estimate the phase needed to brake under constant angular deceleration."""

    if not math.isfinite(maximum_deceleration_rad_s2) or maximum_deceleration_rad_s2 <= 0.0:
        raise ValueError("maximum deceleration must be finite and positive")
    if not math.isfinite(margin_rad) or margin_rad < 0.0:
        raise ValueError("braking margin must be finite and nonnegative")
    omega = abs(float(angular_speed_rad_s))
    return omega * omega / (2.0 * maximum_deceleration_rad_s2) + margin_rad


def select_reachable_target_phase_unwrapped(
    current_phase_unwrapped_rad: float,
    park_phase_rad: float,
    required_distance_rad: float,
    direction: float = 1.0,
) -> tuple[float, float]:
    """Select the first park-phase occurrence far enough away for braking."""

    if required_distance_rad < 0.0 or not math.isfinite(required_distance_rad):
        raise ValueError("required phase distance must be finite and nonnegative")
    target, distance = select_target_phase_unwrapped(
        current_phase_unwrapped_rad, park_phase_rad, direction
    )
    if distance + 1.0e-12 < required_distance_rad:
        turns = math.ceil((required_distance_rad - distance) / TAU)
        distance += turns * TAU
        target += direction * turns * TAU
    return target, distance


def smoothstep01(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _within_settled_thresholds(
    inputs: StopTransitionInput,
    pose: ParkPose,
    config: StopTaskConfig,
) -> bool:
    return (
        abs(inputs.linear_speed_m_s) <= config.linear_speed_tolerance_m_s
        and abs(inputs.angular_speed_rad_s)
        <= config.angular_speed_tolerance_rad_s
        and inputs.joint_pose_rms_error_rad
        <= config.joint_pose_rms_tolerance_rad
        and abs(inputs.root_pitch_error_rad) <= config.root_pitch_tolerance_rad
        and inputs.grounded_feet >= pose.required_grounded_feet
        and (not config.forbid_torso_contact or not inputs.torso_contact)
        and (not config.forbid_internal_contact or not inputs.internal_contact)
    )


def advance_stop_state(
    state: StopState,
    inputs: StopTransitionInput,
    pose: ParkPose,
    config: StopTaskConfig,
    timestep_s: float,
) -> StopState:
    """Advance the irreversible ROLL->BRAKE->DEPLOY->HOLD machine."""

    if timestep_s <= 0.0:
        raise ValueError("timestep must be positive")
    if state.mode == StopMode.ROLL:
        if not inputs.stop_command:
            return state
        required_distance = required_braking_phase_distance(
            inputs.angular_speed_rad_s,
            config.maximum_brake_deceleration_rad_s2,
            config.brake_phase_margin_rad,
        )
        target, distance = select_reachable_target_phase_unwrapped(
            inputs.body_phase_unwrapped_rad,
            pose.foot_down_phase_rad,
            required_distance,
            config.direction,
        )
        return StopState(
            mode=StopMode.BRAKE_ALIGN,
            stop_command_time_s=inputs.time_s,
            mode_start_time_s=inputs.time_s,
            brake_start_phase_rad=inputs.body_phase_unwrapped_rad,
            target_phase_unwrapped_rad=target,
            initial_phase_distance_rad=distance,
            settled_duration_s=0.0,
        )

    if state.mode == StopMode.BRAKE_ALIGN:
        phase_error = (
            state.target_phase_unwrapped_rad
            - inputs.body_phase_unwrapped_rad
        )
        ready = (
            abs(phase_error) <= config.phase_tolerance_rad
            and abs(inputs.linear_speed_m_s)
            <= config.linear_speed_tolerance_m_s
            and abs(inputs.angular_speed_rad_s)
            <= config.angular_speed_tolerance_rad_s
            and inputs.joint_pose_rms_error_rad
            <= config.joint_pose_rms_tolerance_rad
            and inputs.grounded_feet >= pose.required_grounded_feet
            and (not config.forbid_torso_contact or not inputs.torso_contact)
            and (not config.forbid_internal_contact or not inputs.internal_contact)
        )
        if ready:
            return replace(
                state,
                mode=StopMode.PARK_DEPLOY,
                mode_start_time_s=inputs.time_s,
            )
        # Missing a deploy window is not terminal: aim at the next occurrence.
        directed_error = phase_error * config.direction
        if directed_error < -config.phase_tolerance_rad:
            return replace(
                state,
                target_phase_unwrapped_rad=(
                    state.target_phase_unwrapped_rad + config.direction * TAU
                ),
                initial_phase_distance_rad=(
                    abs(phase_error) + TAU
                ),
            )
        return state

    if state.mode == StopMode.PARK_DEPLOY:
        deploy_elapsed = max(inputs.time_s - state.mode_start_time_s, 0.0)
        pose_ready = (
            inputs.joint_pose_rms_error_rad
            <= config.joint_pose_rms_tolerance_rad
            and inputs.grounded_feet >= pose.required_grounded_feet
            and abs(inputs.linear_speed_m_s)
            <= config.linear_speed_tolerance_m_s
            and abs(inputs.angular_speed_rad_s)
            <= config.angular_speed_tolerance_rad_s
        )
        if deploy_elapsed >= config.deploy_duration_s and pose_ready:
            return replace(
                state,
                mode=StopMode.HOLD,
                mode_start_time_s=inputs.time_s,
                settled_duration_s=0.0,
            )
        return state

    settled = (
        state.settled_duration_s + timestep_s
        if _within_settled_thresholds(inputs, pose, config)
        else 0.0
    )
    return replace(state, settled_duration_s=settled)


def stop_succeeded(state: StopState, config: StopTaskConfig) -> bool:
    return (
        state.mode == StopMode.HOLD
        and state.settled_duration_s >= config.required_hold_duration_s
    )


def reference_schedule(
    state: StopState,
    *,
    time_s: float,
    body_phase_unwrapped_rad: float,
    config: StopTaskConfig,
) -> StopReferenceSchedule:
    """Return smooth rolling and parking reference weights for one step."""

    if state.mode == StopMode.ROLL:
        return StopReferenceSchedule(
            mode=state.mode,
            rolling_reference_scale=1.0,
            parking_reference_blend=0.0,
            target_phase_error_rad=0.0,
            time_since_stop_s=0.0,
            time_in_mode_s=max(time_s - state.mode_start_time_s, 0.0),
        )

    time_since_stop = max(time_s - state.stop_command_time_s, 0.0)
    time_in_mode = max(time_s - state.mode_start_time_s, 0.0)
    phase_error = state.target_phase_unwrapped_rad - body_phase_unwrapped_rad
    if state.mode == StopMode.BRAKE_ALIGN:
        initial = max(state.initial_phase_distance_rad, 1.0e-6)
        phase_scale = smoothstep01(abs(phase_error) / initial)
        time_scale = 1.0 - smoothstep01(
            time_since_stop / config.maximum_brake_duration_s
        )
        rolling_scale = min(phase_scale, time_scale)
        parking_blend = 0.0
    elif state.mode == StopMode.PARK_DEPLOY:
        rolling_scale = 0.0
        parking_blend = smoothstep01(time_in_mode / config.deploy_duration_s)
    else:
        rolling_scale = 0.0
        parking_blend = 1.0
    return StopReferenceSchedule(
        mode=state.mode,
        rolling_reference_scale=rolling_scale,
        parking_reference_blend=parking_blend,
        target_phase_error_rad=phase_error,
        time_since_stop_s=time_since_stop,
        time_in_mode_s=time_in_mode,
    )


def blend_joint_reference(
    xp,
    rolling_joint_targets,
    compact_joint_targets,
    parking_joint_targets,
    schedule: StopReferenceSchedule,
):
    """Brake toward compact, then deploy from compact to the parking pose."""

    rolling = xp.asarray(rolling_joint_targets)
    compact = xp.asarray(compact_joint_targets)
    parking = xp.asarray(parking_joint_targets)
    braking = (
        compact
        + schedule.rolling_reference_scale * (rolling - compact)
    )
    return braking + schedule.parking_reference_blend * (parking - braking)


def stop_observation_features(
    xp,
    state: StopState,
    schedule: StopReferenceSchedule,
    body_phase_rad,
    stop_command,
):
    """Return fixed-width command/state features for a future policy."""

    mode_one_hot = xp.asarray([float(state.mode == mode) for mode in StopMode])
    phase_error = schedule.target_phase_error_rad
    dtype = xp.asarray(body_phase_rad).dtype
    scalars = xp.stack(
        (
            xp.asarray(stop_command, dtype=dtype),
            xp.sin(body_phase_rad),
            xp.cos(body_phase_rad),
            xp.sin(xp.asarray(phase_error, dtype=dtype)),
            xp.cos(xp.asarray(phase_error, dtype=dtype)),
            xp.asarray(schedule.time_since_stop_s, dtype=dtype),
            xp.asarray(schedule.time_in_mode_s, dtype=dtype),
            xp.asarray(schedule.rolling_reference_scale, dtype=dtype),
            xp.asarray(schedule.parking_reference_blend, dtype=dtype),
        )
    )
    return xp.concatenate((mode_one_hot.astype(dtype), scalars))
