"""Front/back policy consistency for the 720-observation deploy actor.

Geometry and array helpers require no JAX. The training-loss factory imports
JAX only when enabled by the trainer. Reflection is x -> -x in the torso frame,
not a 180-degree yaw rotation: FL <-> RL and FR <-> RR.
"""

import functools
import math
from types import FunctionType, SimpleNamespace

import numpy as np


LEG_ORDER = ("front_left", "front_right", "rear_left", "rear_right")
JOINT_KINDS = ("hip_abduction", "hip", "knee")
JOINT_PERM = (6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5)
POLAR_SIGN = (-1, 1, 1)
AXIAL_SIGN = (1, -1, -1)
COMMAND_SIGN = (-1, 1, -1)  # vx, vy, wz
FRAME_SIZE = 36
HISTORY = 20


def mirror_action(xp, action):
    if action.shape[-1] != 12:
        raise ValueError("front/back symmetry expects 12 canonical joint actions")
    return xp.take(action, xp.asarray(JOINT_PERM), axis=-1)


def mirror_observation(xp, observation):
    """Mirror physical observations BEFORE the actor's usual normalization.

    Angular velocity is an axial vector; gravity and desired vertical are
    polar vectors. Mirror every historical command/joint/action, retaining
    the order of history frames. Joint offsets need no affine correction:
    defaults and action scales match between paired legs (checked at startup).
    """
    if observation.shape[-1] != FRAME_SIZE * HISTORY:
        raise ValueError("front/back symmetry expects the 36 x 20 deploy layout")
    frames = observation.reshape(observation.shape[:-1] + (HISTORY, FRAME_SIZE))
    reflected = xp.concatenate([
        frames[..., :3] * xp.asarray(AXIAL_SIGN, dtype=frames.dtype),
        frames[..., 3:6] * xp.asarray(POLAR_SIGN, dtype=frames.dtype),
        frames[..., 6:9] * xp.asarray(COMMAND_SIGN, dtype=frames.dtype),
        frames[..., 9:12] * xp.asarray(POLAR_SIGN, dtype=frames.dtype),
        mirror_action(xp, frames[..., 12:24]),
        mirror_action(xp, frames[..., 24:36]),
    ], axis=-1)
    return reflected.reshape(observation.shape)


def consistency_statistics(xp, observation, action, mirrored_observation_action):
    """Compare policies on straight-command samples; no invented PPO returns."""
    cmd = observation[..., 6:9]  # newest frame, physical command units
    mask = ((xp.abs(cmd[..., 0]) > 0.05)
            & (xp.abs(cmd[..., 1]) < 0.05)
            & (xp.abs(cmd[..., 2]) < 0.15))
    error = xp.mean(xp.square(
        mirrored_observation_action - mirror_action(xp, action)), axis=-1)
    mse = xp.sum(xp.where(mask, error, 0.0)) / xp.maximum(xp.sum(mask), 1)
    fraction = xp.mean(mask.astype(observation.dtype))
    return mse, fraction


def make_symmetry_loss(base_loss, weight, *, array_module=None):
    """Add a differentiable actor-only regularizer to the installed PPO loss."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("front/back symmetry weight must be finite and nonnegative")
    if array_module is None:
        import jax.numpy as jp
    else:
        jp = array_module  # NumPy permits static loss-wiring checks without JAX.

    @functools.wraps(base_loss)
    def loss(params, normalizer_params, data, rng, ppo_network, **kwargs):
        total, metrics = base_loss(
            params, normalizer_params, data, rng, ppo_network=ppo_network, **kwargs)
        obs = data.observation
        apply = ppo_network.policy_network.apply
        distribution = ppo_network.parametric_action_distribution
        action = distribution.mode(apply(normalizer_params, params.policy, obs))
        reflected_action = distribution.mode(apply(
            normalizer_params, params.policy, mirror_observation(jp, obs)))
        # Both predictions receive gradients; task PPO remains the anchor.
        # Neither critic targets nor behaviour log-probabilities are mirrored.
        mse, fraction = consistency_statistics(jp, obs, action, reflected_action)
        regularizer = weight * mse
        total = total + regularizer
        return total, {
            **metrics,
            "total_loss": total,
            "fb_symmetry_loss": regularizer,
            "fb_symmetry_action_rmse": jp.sqrt(mse),
            "fb_symmetry_fraction": fraction,
        }

    return loss


def bind_ppo_loss(train_fn, loss_fn):
    """Bind a local loss namespace without editing/monkeypatching Brax.

    Brax versions used by this project bind ppo_losses.compute_ppo_loss inside
    train. Clone only that function's globals so callbacks and other trainers
    continue to see the original library. Fail explicitly on an unsupported
    trainer implementation instead of silently omitting the regularizer.
    """
    if (not isinstance(train_fn, FunctionType)
            or "ppo_losses" not in train_fn.__code__.co_names):
        raise RuntimeError("Brax train does not expose ppo_losses; symmetry adapter needs updating")
    namespace = dict(train_fn.__globals__)
    losses = namespace.get("ppo_losses")
    if losses is None or not callable(getattr(losses, "compute_ppo_loss", None)):
        raise RuntimeError("Brax PPO loss binding is incompatible with front/back symmetry")
    namespace["ppo_losses"] = SimpleNamespace(
        **{**vars(losses), "compute_ppo_loss": loss_fn})
    bound = FunctionType(train_fn.__code__, namespace, train_fn.__name__,
                         train_fn.__defaults__, train_fn.__closure__)
    bound.__kwdefaults__ = train_fn.__kwdefaults__
    return functools.update_wrapper(bound, train_fn)


def with_front_back_symmetry(train_fn, weight):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("front/back symmetry weight must be finite and nonnegative")
    if weight == 0:
        return train_fn
    losses = getattr(train_fn, "__globals__", {}).get("ppo_losses")
    if losses is None or not callable(getattr(losses, "compute_ppo_loss", None)):
        raise RuntimeError("Cannot locate the installed Brax PPO loss for symmetry")
    return bind_ppo_loss(train_fn, make_symmetry_loss(losses.compute_ppo_loss, weight))


def audit_front_back_mapping(model, default_pose, action_scale, ctrl_lo, ctrl_hi):
    """Check the model/actor mapping with native kinematics, not simulation.

    This verifies the coordinate convention, not exact dynamic symmetry of
    CAD inertias, contacts or a particular DR realization. The loss is soft.
    """
    import mujoco

    for label, values in (("default pose", default_pose), ("action scale", action_scale),
                          ("lower limits", ctrl_lo), ("upper limits", ctrl_hi)):
        values = np.asarray(values)
        if values.shape != (12,) or not np.allclose(
                values, mirror_action(np, values), atol=1e-6, rtol=0):
            raise ValueError(f"front/back {label} does not match the symmetry mapping")
    joint_ids = np.array([model.joint(f"{leg}_{kind}").id
                          for leg in LEG_ORDER for kind in JOINT_KINDS])
    addresses = model.jnt_qposadr[joint_ids]
    site_ids = [model.site(f"{leg}_foot_site").id for leg in LEG_ORDER]
    torso_id = model.body("torso").id
    stand_id = model.key("stand").id
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    data.qpos[addresses] = np.asarray(default_pose)
    mujoco.mj_kinematics(model, data)
    axes = data.xaxis[joint_ids] @ data.xmat[torso_id].reshape(3, 3)
    alignment = np.sum(axes[np.asarray(JOINT_PERM)] * axes * AXIAL_SIGN, axis=-1)
    if np.min(alignment) < 0.999:
        raise ValueError("front/back joint-axis signs no longer match the actor mapping")
    rng = np.random.default_rng(7)
    max_foot_error = 0.0
    for action in np.vstack([np.zeros((1, 12)), rng.uniform(-0.8, 0.8, (8, 12))]):
        feet = []
        for candidate in (action, mirror_action(np, action)):
            data.qpos[addresses] = np.asarray(default_pose) + candidate * np.asarray(action_scale)
            mujoco.mj_kinematics(model, data)
            feet.append((data.site_xpos[site_ids] - data.xpos[torso_id])
                        @ data.xmat[torso_id].reshape(3, 3))
        error = feet[1] - feet[0][[2, 3, 0, 1]] * POLAR_SIGN
        max_foot_error = max(max_foot_error, float(np.max(np.linalg.norm(error, axis=-1))))
    if max_foot_error > 0.003:
        raise ValueError(f"front/back foot mapping discrepancy {max_foot_error*1000:.2f} mm exceeds 3 mm")
    return {"axis_alignment_min": float(np.min(alignment)),
            "sampled_foot_error_m": max_foot_error}
