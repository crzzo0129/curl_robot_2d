"""CPU-testable Walking target and unmodified ROLL takeover state contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig
from curl_robot_2d_mjx.environment_3d import model_path_3d
from curl_robot_2d_mjx.environment_walking_3d import WALKING_JOINT_NAMES_3D


def walking_start_state_3d(model, config: Transition3DConfig):
    """Match DeployEnv.reset without importing its JAX training module.

    MJCF qpos order is NOT policy order on rollingquad_2. All joint values
    and checks are resolved by name; ctrl remains in actuator/policy order.
    """
    indices = np.asarray([
        model.jnt_qposadr[model.joint(name).id]
        for name in WALKING_JOINT_NAMES_3D
    ], dtype=np.int32)
    key = model.key(config.walking_start_keyframe).id
    qpos = model.key_qpos[key].copy()
    ctrl = model.key_ctrl[key].copy()
    if not np.allclose(qpos[indices], ctrl, atol=1e-6):
        raise ValueError("Walking keyframe qpos and named actuator targets differ")
    qpos[:2] = 0.0
    qpos[2] += config.walking_start_height_offset_m
    qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    return {"qpos": qpos, "qvel": np.zeros(model.nv), "ctrl": ctrl}


def transition_target_ctrl_3d(xp, action, nominal, low, high, fraction=1.0):
    """Exact post-fade C++ position mapping (raw action, fixed scale, limits)."""
    scale = xp.maximum(high - nominal, nominal - low)
    return xp.clip(nominal + fraction * action * scale, low, high)


def transition_action_from_ctrl_3d(xp, ctrl, nominal, low, high, fraction=1.0):
    """Represent the last ROLL servo command in Transition action coordinates."""
    delta = ctrl - nominal
    scale = fraction * xp.maximum(high - nominal, nominal - low)
    return xp.clip(delta / xp.maximum(scale, 1e-8), -1.0, 1.0)


def snapshot_metadata_3d(model, config):
    return {
        "schema_version": np.asarray(1),
        "geometry": np.asarray(config.geometry),
        "model_xml_sha256": np.asarray(hashlib.sha256(
            model_path_3d(config.geometry).read_bytes()).hexdigest()),
        "qpos_joint_names": np.asarray([
            model.joint(i).name for i in range(model.njnt)
        ]),
        "actuator_names": np.asarray([
            model.actuator(i).name for i in range(model.nu)
        ]),
    }


def save_roll_snapshots_3d(path, model, config, *, qpos, qvel, ctrl,
                           time_s, episode_id, source_policy):
    """Export arrays sampled from a frozen ROLL policy, before any braking.

    Caller owns trajectory collection. Velocities must be measured simulator
    qvel; qpos-only videos cannot be converted into faithful takeover states.
    """
    if not source_policy:
        raise ValueError("source_policy provenance is required")
    arrays = {
        **snapshot_metadata_3d(model, config),
        "qpos": np.asarray(qpos), "qvel": np.asarray(qvel),
        "ctrl": np.asarray(ctrl), "time_s": np.asarray(time_s),
        "episode_id": np.asarray(episode_id),
        "source_policy": np.asarray(source_policy),
    }
    validate_roll_snapshots_3d(arrays, model, config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No implicit filename extension and no overwrite of an existing bank.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def validate_roll_snapshots_3d(arrays, model, config):
    expected = snapshot_metadata_3d(model, config)
    for key, value in expected.items():
        if key not in arrays or not np.array_equal(arrays[key], value):
            raise ValueError(f"ROLL snapshot model/order mismatch: {key}")
    if "source_policy" not in arrays or not str(arrays["source_policy"].item()):
        raise ValueError("ROLL snapshots require source_policy provenance")
    count = len(arrays.get("qpos", ()))
    if count == 0:
        raise ValueError("ROLL snapshot bank is empty")
    for key, width in (("qpos", model.nq), ("qvel", model.nv), ("ctrl", model.nu)):
        if key not in arrays or np.shape(arrays[key]) != (count, width):
            raise ValueError(f"ROLL snapshots require {key} shape ({count}, {width})")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"nonfinite ROLL snapshot {key}")
    if not np.allclose(np.linalg.norm(arrays["qpos"][:, 3:7], axis=1), 1.0,
                       atol=1e-4, rtol=0):
        raise ValueError("ROLL snapshot quaternion must be normalized")
    for key in ("time_s", "episode_id"):
        if key not in arrays or np.shape(arrays[key]) != (count,):
            raise ValueError(f"ROLL snapshots require {key} shape ({count},)")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"nonfinite ROLL snapshot {key}")


def load_roll_snapshots_3d(path, model, config):
    """Select later/lower-speed real states; NEVER zero or scale their qvel."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_roll_snapshots_3d(arrays, model, config)
    count = len(arrays["qpos"])
    keep = np.ones(count, dtype=bool)
    # Tail selection is within each trajectory; 'later' does not imply 'slow'.
    if config.snapshot_tail_fraction < 1.0:
        for episode in np.unique(arrays["episode_id"]):
            indices = np.flatnonzero(arrays["episode_id"] == episode)
            times = arrays["time_s"][indices]
            threshold = times.max() - config.snapshot_tail_fraction * (
                times.max() - times.min())
            keep[indices] &= times >= threshold
    for bound, section in (
        (config.snapshot_max_linear_speed_m_s, slice(0, 3)),
        (config.snapshot_max_angular_speed_rad_s, slice(3, 6)),
    ):
        if bound is not None:
            keep &= np.linalg.norm(arrays["qvel"][:, section], axis=1) <= bound
    if not keep.any():
        raise ValueError("no ROLL snapshots satisfy tail/speed filters; collect "
                         "suitable states, do not artificially slow the saved qvel")
    return {
        key: arrays[key][keep].copy()
        for key in ("qpos", "qvel", "ctrl", "time_s", "episode_id")
    }


def collect_roll_snapshots_3d(env, policy, path, *, source_policy,
                             config=None, seed=0, episodes=8,
                             steps_per_episode=500, warmup_steps=100,
                             sample_every=5):
    """Sample a loaded, frozen Brax ROLL policy on cloud; never apply a brake.

    ``policy(obs, key) -> (action, extras)`` uses the native ROLL contract.
    The snapshot contains resulting full model ctrl, NOT its 8-D policy action.
    Offline snapshots restore physical state, not exact solver warm-starts;
    live takeover should instead call Transition.reset_from_roll_state.
    """
    config = config or Transition3DConfig()
    if env.config.geometry != config.geometry:
        raise ValueError("ROLL collector and Transition must use the same geometry")
    if episodes < 1 or steps_per_episode < 1 or sample_every < 1:
        raise ValueError("collection counts must be positive")
    if not 0 <= warmup_steps < steps_per_episode:
        raise ValueError("warmup must be shorter than the rollout")
    if Path(path).exists():
        raise FileExistsError(path)
    import jax
    reset, step, inference = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
    rows = {key: [] for key in ("qpos", "qvel", "ctrl", "time_s", "episode_id")}
    for episode in range(episodes):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), episode)
        state = reset(key)
        for index in range(steps_per_episode):
            action, _ = inference(state.obs, jax.random.fold_in(key, index + 1))
            state = step(state, action)
            if bool(jax.device_get(state.done)):
                break
            if index + 1 < warmup_steps or (index + 1 - warmup_steps) % sample_every:
                continue
            data = jax.device_get(state.pipeline_state)
            for name in ("qpos", "qvel", "ctrl"):
                rows[name].append(np.asarray(getattr(data, name)).copy())
            rows["time_s"].append(float(data.time))
            rows["episode_id"].append(episode)
    save_roll_snapshots_3d(path, env.mj_model, config,
                          source_policy=source_policy, **rows)
    return {"samples": len(rows["qpos"]), "path": str(Path(path).resolve()),
            "source_policy": source_policy, "external_braking": False}
