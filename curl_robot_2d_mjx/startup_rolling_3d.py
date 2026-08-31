"""Standing startup helpers shared by MJX training and CPU verification.

Only joint targets are interpolated after reset, never the simulated state.
The teacher uses this trajectory prior; direct-action students must learn it.
"""

from dataclasses import replace

import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, smoothstep_ramp, validate_3d_config


def add_stand_startup_arguments(parser):
    parser.add_argument("--reset-pose", choices=("compact", "stand"), default="compact",
                        help="stand includes a physical stand-to-compact startup in the episode")
    parser.add_argument("--stand-hold-s", type=float, default=0.2)
    parser.add_argument("--stand-to-compact-s", type=float, default=1.0)


def with_stand_startup(task, args):
    result = replace(task, reset_pose=args.reset_pose, stand_hold_s=args.stand_hold_s,
                     stand_to_compact_s=args.stand_to_compact_s)
    validate_3d_config(result)
    return result


def rolling_elapsed_3d(xp, elapsed_s, task):
    if task.reset_pose == "compact":
        return elapsed_s  # Preserve legacy timing exactly.
    return xp.maximum(elapsed_s - task.rolling_start_time_s, 0.0)


def residual_elapsed_3d(xp, elapsed_s, task):
    # Let PPO correct the fold itself, not only the motion after it.  Do not
    # reset this ramp at the handoff, which would abruptly remove corrections.
    if task.reset_pose == "compact":
        return elapsed_s
    return xp.maximum(elapsed_s - task.stand_hold_s, 0.0)


def stand_startup_action_3d(xp, elapsed_s, stand_action, task):
    if task.reset_pose == "compact":
        return xp.zeros_like(stand_action)
    blend = smoothstep_ramp(xp, elapsed_s - task.stand_hold_s, task.stand_to_compact_s)
    return (1.0 - blend) * stand_action


def compose_startup_action_3d(xp, reference_action, startup_action, policy_action,
                             *, reference_weight, residual_gain, ramp, direct):
    if direct:
        # No hidden startup controller, reference or ramp on the student's output.
        return xp.clip(policy_action, -1.0, 1.0)
    # Do not attenuate the stand pose when a curriculum lowers reference_weight.
    return xp.clip(startup_action + reference_weight * (reference_action - startup_action)
                   + ramp * residual_gain * policy_action, -1.0, 1.0)


def reset_pose_arrays_3d(model, task: Rolling3DConfig):
    """Return reset qpos and normalized startup action, mapping joints by name.

    Fail rather than silently clipping a stand pose not representable by the
    existing 8-D rolling action ABI (including its locked abduction joints).
    """
    compact = model.key("compact")
    reset = model.key(task.reset_pose)
    if task.reset_pose == "compact":
        return reset.qpos.copy(), np.zeros(8)
    actuators = np.asarray([model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D])
    joint_ids = model.actuator_trnid[:, 0]
    addresses = model.jnt_qposadr[joint_ids]
    np.testing.assert_allclose(reset.qpos[addresses], reset.ctrl, atol=1e-8,
                               err_msg="stand qpos and ctrl disagree by joint name")
    locked = np.ones(model.nu, dtype=bool)
    locked[actuators] = False
    np.testing.assert_allclose(reset.ctrl[locked], compact.ctrl[locked], atol=1e-8,
                               err_msg="stand changes locked rolling joints")
    action = (reset.ctrl[actuators] - compact.ctrl[actuators]) / np.asarray(task.action_scales)
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1.0 + 1e-8):
        raise ValueError("stand is outside the existing compact-centered action range")
    limits = model.actuator_ctrlrange
    limited = model.actuator_ctrllimited.astype(bool)
    if np.any(limited & ((reset.ctrl < limits[:, 0]) | (reset.ctrl > limits[:, 1]))):
        raise ValueError("stand exceeds actuator position limits")
    return reset.qpos.copy(), action
