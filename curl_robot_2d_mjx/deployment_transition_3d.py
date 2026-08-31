"""Transition Actor ABI shared with the current ROS2 neural_controller.

Only this 36-value sensor/command frame enters the Actor. Mode, simulation
velocity, root height, contacts and success counters belong to the Critic.
"""

from curl_robot_2d_mjx.deployment_rolling_3d import (
    rolling_deploy_frame_3d,
    initial_rolling_deploy_history_3d,
    push_rolling_deploy_frame_3d,
    HARDWARE_CONTROLLER_JOINT_NAMES_3D,
    CONTROLLER_JOINT_NAMES_3D,
    HARDWARE_POLICY_FREQUENCY_HZ_3D,
)


def transition_controller_frame_3d(xp, *, angular_velocity_body,
                                   projected_gravity, joint_position_offset,
                                   last_action):
    """Same field meanings as Walking: stop cmd=(0,0,0), desired z=(0,0,1).

    Stop selects this policy outside the observation. No mode or phase is
    smuggled into the command fields and no qvel difference is appended.
    """
    return rolling_deploy_frame_3d(
        xp, angular_velocity_body=angular_velocity_body,
        projected_gravity=projected_gravity,
        joint_position_offset=joint_position_offset, last_action=last_action,
    )


def initial_transition_history_3d(xp, *, dtype=None):
    return initial_rolling_deploy_history_3d(xp, dtype=dtype)


def push_transition_frame_3d(xp, history, frame, observation_limit=100.0):
    """Newest first, clipped before inference just like the C++ controller."""
    return xp.clip(push_rolling_deploy_frame_3d(xp, history, frame),
                   -observation_limit, observation_limit)


def transition_controller_metadata_3d(model, config):
    """Exportable metadata, NOT a live controller reconfiguration command."""
    import numpy as np
    from curl_robot_2d_mjx.transition_initialization_3d import walking_start_state_3d
    target = walking_start_state_3d(model, config)["ctrl"]
    ids = [model.joint(name).id for name in CONTROLLER_JOINT_NAMES_3D]
    low, high = model.jnt_range[ids].T
    scale = config.action_range_fraction * np.maximum(high - target, target - low)
    kp, kd = model.actuator_gainprm[:, 0], -model.actuator_biasprm[:, 2]
    # Current C++ uses set_param_from_json_scalar for kp/kd (unlike scales).
    if not np.allclose(kp, kp[0]) or not np.allclose(kd, kd[0]):
        raise ValueError("current neural_controller JSON requires uniform scalar kp/kd")
    return {
        "contract_version": "transition_neural_controller_36x20_v3",
        "use_imu": True, "control_orientation": False,
        "observation_history": 20, "single_observation_size": 36,
        "observation_limit": config.observation_limit,
        "policy_frequency_hz": 1.0 / config.control_timestep,
        "expected_hardware_policy_frequency_hz": HARDWARE_POLICY_FREQUENCY_HZ_3D,
        "activation": "elu", "actor_output": "tanh_location",
        "action_scale": scale.tolist(), "default_joint_pos": target.tolist(),
        "joint_lower_limits": low.tolist(), "joint_upper_limits": high.tolist(),
        "joint_names": list(HARDWARE_CONTROLLER_JOINT_NAMES_3D),
        "kp": float(kp[0]), "kd": float(kd[0]),
        "transition_cmd_vel": [0.0, 0.0, 0.0],
        "desired_world_z": [0.0, 0.0, 1.0],
        "live_takeover_requires_hot_switch": True,
    }
