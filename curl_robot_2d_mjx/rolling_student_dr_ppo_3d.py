"""Dependency-light contracts for rolling Student DR PPO."""

from __future__ import annotations

from collections.abc import Mapping
import math

from curl_robot_2d_mjx.deployment_rolling_3d import (
    ROLLING_EFFECTIVE_ACTION_INDICES_3D,
)


ROLLING_STUDENT_PPO_ACTION_SIZE_3D = 8
ROLLING_STUDENT_PPO_CRITIC_OBSERVATION_SIZE_3D = 65


def tanh_normal_scale_logit_3d(initial_std: float) -> float:
    """Inverse of softplus(scale_logit) + 0.001 used by Brax."""

    if not math.isfinite(initial_std) or initial_std <= 0.001:
        raise ValueError("initial policy std must be finite and greater than 0.001")
    adjusted = initial_std - 0.001
    return adjusted if adjusted > 20.0 else math.log(math.expm1(adjusted))


def _params_mapping(tree):
    if not isinstance(tree, Mapping) or "params" not in tree:
        raise ValueError("expected Flax parameters with a 'params' mapping")
    params = tree["params"]
    if not isinstance(params, Mapping):
        raise ValueError("Flax 'params' must be a mapping")
    return params


def initialize_ppo_actor_from_student_3d(
    xp,
    ppo_params,
    student_params,
    *,
    hidden_layers=(512, 256, 128),
    initial_std=0.05,
):
    """Copy a deterministic 12-output Student into an 8-action PPO actor."""

    ppo = _params_mapping(ppo_params)
    student = _params_mapping(student_params)
    result = {name: dict(value) for name, value in ppo.items()}
    for index, width in enumerate(hidden_layers):
        name = f"hidden_{index}"
        if name not in student or name not in ppo:
            raise ValueError(f"missing compatible hidden layer: {name}")
        source = student[name]
        target = ppo[name]
        if (
            source["kernel"].shape != target["kernel"].shape
            or source["bias"].shape != target["bias"].shape
            or source["bias"].shape != (width,)
        ):
            raise ValueError(f"Student/PPO hidden layer mismatch: {name}")
        result[name] = {
            **target,
            "kernel": xp.asarray(source["kernel"]),
            "bias": xp.asarray(source["bias"]),
        }

    if "location" not in student:
        raise ValueError("Student is missing its location output layer")
    head_name = f"hidden_{len(hidden_layers)}"
    if head_name not in ppo:
        raise ValueError(f"PPO actor is missing its output layer: {head_name}")
    source_head = student["location"]
    target_head = ppo[head_name]
    indices = xp.asarray(ROLLING_EFFECTIVE_ACTION_INDICES_3D)
    source_kernel = xp.take(source_head["kernel"], indices, axis=1)
    source_bias = xp.take(source_head["bias"], indices, axis=0)
    action_size = ROLLING_STUDENT_PPO_ACTION_SIZE_3D
    if (
        target_head["kernel"].shape
        != (source_kernel.shape[0], 2 * action_size)
        or target_head["bias"].shape != (2 * action_size,)
    ):
        raise ValueError("PPO actor must have an 8-location/8-scale tanh head")
    scale_logit = tanh_normal_scale_logit_3d(initial_std)
    result[head_name] = {
        **target_head,
        "kernel": xp.concatenate(
            (source_kernel, xp.zeros_like(source_kernel)), axis=1
        ),
        "bias": xp.concatenate(
            (
                source_bias,
                xp.full((action_size,), scale_logit, dtype=source_bias.dtype),
            )
        ),
    }
    return {**ppo_params, "params": result}


def expand_ppo_actor_to_controller_3d(xp, ppo_params):
    """Convert an 8-action tanh-normal PPO actor to a 12-output actor tree."""

    params = _params_mapping(ppo_params)
    dense_names = sorted(
        (name for name in params if name.startswith("hidden_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    if not dense_names:
        raise ValueError("PPO actor has no hidden layers")
    head_name = dense_names[-1]
    head = params[head_name]
    action_size = ROLLING_STUDENT_PPO_ACTION_SIZE_3D
    if head["bias"].shape != (2 * action_size,):
        raise ValueError("expected an 8-action tanh-normal PPO output head")
    location_kernel = head["kernel"][:, :action_size]
    location_bias = head["bias"][:action_size]
    controller_kernel = xp.zeros(
        (location_kernel.shape[0], 12), dtype=location_kernel.dtype
    )
    controller_bias = xp.zeros((12,), dtype=location_bias.dtype)
    indices = xp.asarray(ROLLING_EFFECTIVE_ACTION_INDICES_3D)
    if hasattr(controller_kernel, "at"):
        controller_kernel = controller_kernel.at[:, indices].set(location_kernel)
        controller_bias = controller_bias.at[indices].set(location_bias)
    else:
        controller_kernel[:, ROLLING_EFFECTIVE_ACTION_INDICES_3D] = location_kernel
        controller_bias[list(ROLLING_EFFECTIVE_ACTION_INDICES_3D)] = location_bias
    result = {name: dict(value) for name, value in params.items()}
    result[head_name] = {
        **head,
        "kernel": controller_kernel,
        "bias": controller_bias,
    }
    return {**ppo_params, "params": result}
