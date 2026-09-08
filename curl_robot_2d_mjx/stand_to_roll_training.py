"""Dependency-light BC and PPO contracts for stand-to-roll training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import math

import numpy as np

from curl_robot_2d_mjx.deployment_rolling_3d import (
    ROLLING_DEPLOY_OBSERVATION_HISTORY_3D,
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
    rolling_deploy_frame_3d,
)


STAND_TO_ROLL_ACTION_SIZE = 12
STAND_TO_ROLL_HIDDEN_LAYERS = (512, 256, 128)
BC_CONTRACT_VERSION = 2


def action_center_and_scale(config) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical FL/FR/RL/RR ABD/hip/knee controller vectors."""

    center = np.asarray(
        (0.0, config.action_center_hip_rad, config.action_center_knee_rad) * 4,
        dtype=np.float32,
    )
    scale = np.asarray(
        (
            config.action_scale_abduction_rad,
            config.action_scale_hip_rad,
            config.action_scale_knee_rad,
        )
        * 4,
        dtype=np.float32,
    )
    return center, scale


def build_cem_bc_dataset(
    npz_path: Path,
    *,
    controller_qpos_indices,
    action_center,
    action_scale,
    controller_actuator_indices=None,
    history: int = ROLLING_DEPLOY_OBSERVATION_HISTORY_3D,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the CEM rollout into deploy-observation/action BC pairs.

    The output observation layout is exactly train_ppo_deploy's newest-first
    36 x 20 history.  CEM phase and privileged velocities are never exposed to
    the actor.
    """

    if history != ROLLING_DEPLOY_OBSERVATION_HISTORY_3D:
        raise ValueError("stand-to-roll BC requires a 20-frame history")
    indices = np.asarray(controller_qpos_indices, dtype=np.int64)
    center = np.asarray(action_center, dtype=np.float64)
    scale = np.asarray(action_scale, dtype=np.float64)
    if indices.shape != (12,) or center.shape != (12,) or scale.shape != (12,):
        raise ValueError("joint indices, action center and scale must be 12-vectors")
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("action scales must be finite and positive")

    with np.load(npz_path) as data:
        required = {"qpos", "orientation", "angular_velocity", "joint_target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"CEM BC data is missing keys: {sorted(missing)}")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        orientation = np.asarray(data["orientation"], dtype=np.float64)
        angular_world = np.asarray(data["angular_velocity"], dtype=np.float64)
        target = np.asarray(data["joint_target"], dtype=np.float64)
        # The legacy collector stores free-joint LOCAL angular velocity under
        # angular_velocity. qvel is authoritative for those recordings.
        angular_body_recorded = (
            np.asarray(data["qvel"], dtype=np.float64)[:, 3:6]
            if "qvel" in data.files else None
        )
    if controller_actuator_indices is not None:
        target = target[:, np.asarray(controller_actuator_indices, dtype=np.int64)]

    n = qpos.shape[0]
    if (
        qpos.ndim != 2
        or orientation.shape != (n, 3, 3)
        or angular_world.shape != (n, 3)
        or target.shape != (n, 12)
        or n <= history
    ):
        raise ValueError("invalid CEM BC array shapes")
    if indices.min() < 0 or indices.max() >= qpos.shape[1]:
        raise ValueError("controller qpos index is outside the CEM qpos array")

    angular_body = np.einsum("nji,nj->ni", orientation, angular_world)
    if angular_body_recorded is not None:
        angular_body = angular_body_recorded
    gravity_world = np.broadcast_to(np.asarray((0.0, 0.0, -1.0)), (n, 3))
    projected_gravity = np.einsum("nji,nj->ni", orientation, gravity_world)
    action = np.clip((target - center) / scale, -1.0, 1.0)
    # Row i is recorded AFTER executing action i. Predict action i+1.
    previous_action = action
    frames = rolling_deploy_frame_3d(
        np,
        angular_velocity_body=angular_body,
        projected_gravity=projected_gravity,
        joint_position_offset=qpos[:, indices] - center,
        last_action=previous_action,
    )
    # For sample i, history is [frame_i, frame_(i-1), ...], matching the C++
    # std::rotate buffer and train_ppo_deploy._push.
    observations = np.stack(
        [frames[i - np.arange(history)].reshape(-1) for i in range(history - 1, n - 1)]
    )
    actions = action[history:]
    observations = observations.astype(np.float32)
    actions = actions.astype(np.float32)
    if observations.shape[1] != ROLLING_DEPLOY_OBSERVATION_SIZE_3D:
        raise AssertionError("deploy history width drifted from 720")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("CEM BC dataset contains non-finite values")
    return observations, actions


def observation_normalizer(observations: np.ndarray) -> dict[str, np.ndarray]:
    """Return a stable mean/std mapping suitable for BC initialization."""

    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 720:
        raise ValueError("observations must have shape (N, 720)")
    return {
        "mean": observations.mean(axis=0),
        "std": np.maximum(observations.std(axis=0), np.tile(
            np.asarray([0.2] * 3 + [0.1] * 3 + [1.0] * 6
                       + [0.1] * 12 + [0.1] * 12, dtype=np.float32), 20)),
        "clip": np.asarray(5.0, dtype=np.float32),
        "contract_version": np.asarray(BC_CONTRACT_VERSION),
    }


def preprocess_observation(xp, observation, normalizer):
    return xp.clip((observation - normalizer["mean"]) / normalizer["std"],
                   -normalizer["clip"], normalizer["clip"])


def tanh_normal_scale_logit(initial_std: float) -> float:
    """Inverse of Brax's softplus(logit)+0.001 scale parameterization."""

    if not math.isfinite(initial_std) or initial_std <= 0.001:
        raise ValueError("initial std must be finite and greater than 0.001")
    adjusted = initial_std - 0.001
    return adjusted if adjusted > 20.0 else math.log(math.expm1(adjusted))


def initialize_ppo_actor_from_bc(
    xp,
    ppo_params,
    bc_params,
    *,
    hidden_layers=STAND_TO_ROLL_HIDDEN_LAYERS,
    initial_std=0.05,
):
    """Copy a deterministic BC actor into a Brax tanh-normal policy head."""

    if not isinstance(ppo_params, Mapping) or "params" not in ppo_params:
        raise ValueError("invalid PPO actor parameter tree")
    if not isinstance(bc_params, Mapping) or "params" not in bc_params:
        raise ValueError("invalid BC actor parameter tree")
    ppo = ppo_params["params"]
    bc = bc_params["params"]
    result = {name: dict(value) for name, value in ppo.items()}
    for index, width in enumerate(hidden_layers):
        name = f"hidden_{index}"
        if name not in ppo or name not in bc:
            raise ValueError(f"missing shared BC/PPO layer: {name}")
        if ppo[name]["kernel"].shape != bc[name]["kernel"].shape:
            raise ValueError(f"BC/PPO kernel mismatch: {name}")
        if bc[name]["bias"].shape != (width,):
            raise ValueError(f"BC hidden width mismatch: {name}")
        result[name] = {
            **ppo[name],
            "kernel": xp.asarray(bc[name]["kernel"]),
            "bias": xp.asarray(bc[name]["bias"]),
        }
    bc_head = bc.get("location")
    ppo_head_name = f"hidden_{len(hidden_layers)}"
    if bc_head is None or ppo_head_name not in ppo:
        raise ValueError("missing BC or PPO action head")
    if bc_head["bias"].shape != (STAND_TO_ROLL_ACTION_SIZE,):
        raise ValueError("BC action head must have 12 outputs")
    zeros = xp.zeros_like(bc_head["kernel"])
    scale = xp.full(
        (STAND_TO_ROLL_ACTION_SIZE,),
        tanh_normal_scale_logit(initial_std),
        dtype=bc_head["bias"].dtype,
    )
    result[ppo_head_name] = {
        **ppo[ppo_head_name],
        "kernel": xp.concatenate((xp.asarray(bc_head["kernel"]), zeros), axis=1),
        "bias": xp.concatenate((xp.asarray(bc_head["bias"]), scale)),
    }
    return {**ppo_params, "params": result}
