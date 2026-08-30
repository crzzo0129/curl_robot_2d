"""Real-controller observation and action contract for 3-D rolling.

The Pupper neural controller consumes 36 values per policy update and keeps
the newest frame first in a configurable history buffer.  Rolling uses the
same controller ABI as walking, but the student predicts the complete motor
command produced by the CEM reference plus the residual teacher.
"""

from __future__ import annotations


CONTROLLER_LEGS_3D = (
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)
CONTROLLER_JOINT_NAMES_3D = tuple(
    f"{leg}_{joint}"
    for leg in CONTROLLER_LEGS_3D
    for joint in ("hip_abduction", "hip", "knee")
)
# The hardware URDF calls hip/abduction/knee motors _1/_2/_3.  neural_controller
# deliberately lists _2/_1/_3 so its tensor order remains abd/hip/knee.
HARDWARE_CONTROLLER_JOINT_NAMES_3D = tuple(
    name
    for leg in ("front_l", "front_r", "back_l", "back_r")
    for name in (f"leg_{leg}_2", f"leg_{leg}_1", f"leg_{leg}_3")
)

ROLLING_EFFECTIVE_ACTION_INDICES_3D = (1, 2, 4, 5, 7, 8, 10, 11)
ROLLING_CONTROLLER_ACTION_SIZE_3D = 12
ROLLING_DEPLOY_SINGLE_OBSERVATION_SIZE_3D = 36
ROLLING_DEPLOY_OBSERVATION_HISTORY_3D = 20
ROLLING_DEPLOY_OBSERVATION_SIZE_3D = (
    ROLLING_DEPLOY_SINGLE_OBSERVATION_SIZE_3D
    * ROLLING_DEPLOY_OBSERVATION_HISTORY_3D
)

# config_rollingquad_2.yaml: the controller manager runs at 520 Hz and the
# neural controller holds each policy action for 10 manager updates.  The IMU
# broadcaster's independent 260 Hz rate is not the policy rate.
HARDWARE_CONTROL_LOOP_HZ_3D = 520.0
HARDWARE_REPEAT_ACTION_3D = 10
HARDWARE_POLICY_FREQUENCY_HZ_3D = (
    HARDWARE_CONTROL_LOOP_HZ_3D / HARDWARE_REPEAT_ACTION_3D
)
HARDWARE_IMU_PUBLISH_FREQUENCY_HZ_3D = 260.0


def effective_action_to_controller_action_3d(xp, effective_action):
    """Embed FL/FR/RL/RR hip+knee actions in the 12-motor ABI."""

    indices = xp.asarray(ROLLING_EFFECTIVE_ACTION_INDICES_3D)
    output_shape = effective_action.shape[:-1] + (
        ROLLING_CONTROLLER_ACTION_SIZE_3D,
    )
    output = xp.zeros(output_shape, dtype=effective_action.dtype)
    if hasattr(output, "at"):
        return output.at[..., indices].set(effective_action)
    output[..., ROLLING_EFFECTIVE_ACTION_INDICES_3D] = effective_action
    return output


def controller_action_to_effective_action_3d(xp, controller_action):
    """Extract the eight rolling hip+knee channels from a controller action."""

    return xp.take(
        controller_action,
        xp.asarray(ROLLING_EFFECTIVE_ACTION_INDICES_3D),
        axis=-1,
    )


def rolling_deploy_frame_3d(
    xp,
    *,
    angular_velocity_body,
    projected_gravity,
    joint_position_offset,
    last_action,
    command=None,
    desired_world_z=None,
):
    """Build one raw 36-value frame in neural_controller.cpp order."""

    if command is None:
        command = xp.zeros_like(angular_velocity_body)
    else:
        command = xp.broadcast_to(command, angular_velocity_body.shape)
    if desired_world_z is None:
        desired_world_z = xp.broadcast_to(
            xp.asarray(
                (0.0, 0.0, 1.0), dtype=angular_velocity_body.dtype
            ),
            angular_velocity_body.shape,
        )
    else:
        desired_world_z = xp.broadcast_to(
            desired_world_z, angular_velocity_body.shape
        )
    return xp.concatenate(
        (
            angular_velocity_body,
            projected_gravity,
            command,
            desired_world_z,
            joint_position_offset,
            last_action,
        ),
        axis=-1,
    )


def initial_rolling_deploy_history_3d(xp, *, dtype=None):
    """Match neural_controller::on_activate history initialization."""

    dtype = dtype or xp.float32
    frame = xp.zeros(
        (ROLLING_DEPLOY_SINGLE_OBSERVATION_SIZE_3D,), dtype=dtype
    )
    if hasattr(frame, "at"):
        frame = frame.at[5].set(-1.0).at[11].set(1.0)
    else:
        frame[5] = -1.0
        frame[11] = 1.0
    return xp.tile(frame, ROLLING_DEPLOY_OBSERVATION_HISTORY_3D)


def push_rolling_deploy_frame_3d(xp, history, frame):
    """Insert the newest frame first, matching the C++ rotate operation."""

    return xp.concatenate(
        (
            frame,
            history[
                ..., :-ROLLING_DEPLOY_SINGLE_OBSERVATION_SIZE_3D
            ],
        ),
        axis=-1,
    )
