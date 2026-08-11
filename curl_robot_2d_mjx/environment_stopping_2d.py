"""Snapshot-reset MJX environment for feedback braking residual training.

The frozen rolling reference remains active.  The policy outputs only four
normalized joint residuals and episodes terminate as soon as the strict
deploy-entry set is reached.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig, reference_action
from curl_robot_2d_mjx.config import NominalRLConfig
from curl_robot_2d_mjx.environment import make_brax_env
from curl_robot_2d_mjx.reward_stopping import (
    STOPPING_REWARD_TERM_NAMES,
    StoppingRewardConfig,
    StoppingTaskConfig,
    desired_braking_speed,
    select_reachable_target_phase_xp,
    stopping_observation_features,
    stopping_reward_terms,
)


def scaled_reference_frequency(
    reference: CEMReferenceConfig,
    frequency_hz: float,
) -> CEMReferenceConfig:
    """Scale both native rate and phase-lock correction like the CPU runner."""

    native_hz = reference.oscillator_rate_rad_s / (2.0 * np.pi)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("reference frequency must be finite and positive")
    scale = frequency_hz / native_hz
    return replace(
        reference,
        oscillator_rate_rad_s=reference.oscillator_rate_rad_s * scale,
        oscillator_coupling_per_s=reference.oscillator_coupling_per_s * scale,
    )


def _load_snapshot_arrays(
    path: Path,
    *,
    maximum_initial_angular_speed_rad_s: float | None = None,
) -> dict[str, np.ndarray]:
    source = np.load(Path(path).expanduser().resolve())
    required = ("qpos", "qvel", "ctrl", "oscillator_phase_rad", "episode_time_s")
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"snapshot dataset is missing: {', '.join(missing)}")
    valid = np.ones(len(source["qpos"]), dtype=bool)
    if "contact_features" in source:
        contact = source["contact_features"]
        valid &= (contact[:, 2] == 0) & (contact[:, 3] == 0)
    if maximum_initial_angular_speed_rad_s is not None:
        if (
            not np.isfinite(maximum_initial_angular_speed_rad_s)
            or maximum_initial_angular_speed_rad_s <= 0.0
        ):
            raise ValueError("maximum initial angular speed must be positive")
        valid &= (
            np.abs(source["qvel"][:, 2])
            <= maximum_initial_angular_speed_rad_s
        )
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("snapshot dataset has no safe reset states")
    return {
        name: np.asarray(source[name][indices], dtype=np.float32)
        for name in required
    }


def make_stopping_brax_env(
    config: NominalRLConfig,
    *,
    cem_reference: CEMReferenceConfig,
    snapshots: Path,
    stopping_config: StoppingTaskConfig | None = None,
    reward_config: StoppingRewardConfig | None = None,
    maximum_initial_angular_speed_rad_s: float | None = None,
    seed: int = 0,
):
    """Create a Brax environment that starts directly from rolling snapshots."""

    import jax
    import jax.numpy as jp
    import mujoco
    from brax.envs.base import Env
    from mujoco import mjx

    stop = stopping_config or StoppingTaskConfig()
    reward_settings = reward_config or StoppingRewardConfig()
    arrays = _load_snapshot_arrays(
        snapshots,
        maximum_initial_angular_speed_rad_s=(
            maximum_initial_angular_speed_rad_s
        ),
    )
    base = make_brax_env(config, cem_reference=cem_reference, seed=seed)
    if arrays["qpos"].shape[1] != base.mj_model.nq:
        raise ValueError("snapshot qpos width does not match the MuJoCo model")
    if arrays["qvel"].shape[1] != base.mj_model.nv:
        raise ValueError("snapshot qvel width does not match the MuJoCo model")

    qpos_table = jp.asarray(arrays["qpos"])
    qvel_table = jp.asarray(arrays["qvel"])
    ctrl_table = jp.asarray(arrays["ctrl"])
    oscillator_table = jp.asarray(arrays["oscillator_phase_rad"])
    time_table = jp.asarray(arrays["episode_time_s"])
    snapshot_count = int(len(arrays["qpos"]))

    torso_geom_id = int(
        mujoco.mj_name2id(
            base.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "torso_proxy"
        )
    )
    if torso_geom_id < 0:
        raise ValueError("missing MuJoCo geom: torso_proxy")

    class CurlRobot2DStoppingEnv(Env):
        def __init__(self):
            self.base = base
            self.config = config
            self.stopping_config = stop
            self.reward_config = reward_settings
            self.cem_reference = cem_reference
            self.model_path = base.model_path
            self.mj_model = base.mj_model
            self.mjx_model = base.mjx_model

        @property
        def observation_size(self):
            return base.observation_size + 10

        @property
        def action_size(self):
            return 4

        @property
        def backend(self):
            return "mjx"

        def _zero_metrics(self):
            zero = jp.zeros((), dtype=jp.float32)
            return {
                "reward": zero,
                "reward_total": zero,
                **{f"reward_{name}": zero for name in STOPPING_REWARD_TERM_NAMES},
                "stop_success": zero,
                "target_remaining_rad": zero,
                "phase_error_rad": zero,
                "linear_speed_m_s": zero,
                "angular_speed_rad_s": zero,
                "desired_angular_speed_rad_s": zero,
                "residual_action_rms": zero,
                "normalized_torque_rms": zero,
                "forbidden_contact_count": zero,
                "torso_contact": zero,
                "failed": zero,
                "timeout": zero,
            }

        def _torso_ground_contact(self, data):
            geom1, geom2, distance = base._contact_arrays(data)
            valid = (geom1 >= 0) & (geom2 >= 0) & (distance <= 0.0)
            ground = (geom1 == base.floor_geom_id) | (geom2 == base.floor_geom_id)
            torso = (geom1 == torso_geom_id) | (geom2 == torso_geom_id)
            return jp.any(valid & ground & torso)

        def _task_observation(self, data, info):
            elapsed = info["step_count"].astype(jp.float32) * config.control_timestep
            return stopping_observation_features(
                jp,
                body_phase=data.qpos[base.root_pitch_qpos],
                target_phase=info["stop_target_phase"],
                initial_distance=info["stop_initial_distance"],
                linear_speed=data.qvel[base.root_x_dof],
                angular_speed=data.qvel[base.root_pitch_dof],
                elapsed_s=elapsed,
                config=stop,
            )

        def reset(self, rng):
            base_state = base.reset(rng)
            sample_key = jax.random.fold_in(rng, seed + 73_001)
            index = jax.random.randint(sample_key, (), 0, snapshot_count)
            qpos = qpos_table[index]
            qvel = qvel_table[index]
            ctrl = ctrl_table[index]
            oscillator = oscillator_table[index]
            data = base_state.pipeline_state.replace(
                qpos=qpos, qvel=qvel, ctrl=ctrl, time=time_table[index]
            )
            data = mjx.forward(base.mjx_model, data)
            phase = data.qpos[base.root_pitch_qpos]
            angular_speed = data.qvel[base.root_pitch_dof]
            target_phase, initial_distance = select_reachable_target_phase_xp(
                jp, phase, angular_speed, stop
            )
            normalized_ctrl = jp.clip(
                (ctrl - base.compact_ctrl) / base.action_scales, -1.0, 1.0
            )
            cem_action = reference_action(
                jp,
                oscillator,
                cem_reference,
                compact_ctrl=base.compact_ctrl,
                action_scales=base.action_scales,
                joint_low=base.joint_low,
                joint_high=base.joint_high,
            )
            contacts = base._contact_metrics(data)
            info = {
                **base_state.info,
                "initial_phase": phase,
                "initial_root_x": data.qpos[base.root_x_qpos],
                "previous_phase": phase,
                "previous_root_x": data.qpos[base.root_x_qpos],
                "last_action": normalized_ctrl,
                "last_policy_action": jp.zeros(4, dtype=jp.float32),
                "last_reference_action": cem_action,
                "oscillator_phase": oscillator,
                "step_count": jp.asarray(0, dtype=jp.int32),
                "stop_target_phase": target_phase,
                "stop_initial_distance": initial_distance,
                "stop_previous_remaining": target_phase - phase,
                "stop_snapshot_index": index.astype(jp.int32),
            }
            base_obs = base._observation(
                data,
                normalized_ctrl,
                contacts,
                reference_action_value=cem_action,
                oscillator_phase=oscillator,
                action_ramp=jp.ones((), dtype=jp.float32),
            )
            obs = jp.concatenate((base_obs, self._task_observation(data, info)))
            return base_state.replace(
                pipeline_state=data,
                obs=jp.nan_to_num(obs),
                reward=jp.zeros((), dtype=jp.float32),
                done=jp.zeros((), dtype=jp.float32),
                metrics=self._zero_metrics(),
                info=info,
            )

        def step(self, state, action):
            base_next = base.step(state, action)
            data = base_next.pipeline_state
            contacts = base._contact_metrics(data)
            phase = data.qpos[base.root_pitch_qpos]
            linear_speed = data.qvel[base.root_x_dof]
            angular_speed = data.qvel[base.root_pitch_dof]
            remaining = state.info["stop_target_phase"] - phase
            previous_remaining = state.info["stop_previous_remaining"]
            desired_speed = desired_braking_speed(jp, remaining, stop)
            torso_contact = self._torso_ground_contact(data)
            internal_contact = contacts["forbidden_count"] > 0
            grounded = contacts["ground_count"] > 0
            phase_ready = jp.abs(remaining) <= stop.phase_tolerance_rad
            speed_ready = (
                (jp.abs(linear_speed) <= stop.linear_speed_tolerance_m_s)
                & (jp.abs(angular_speed) <= stop.angular_speed_tolerance_rad_s)
            )
            success = (
                phase_ready & speed_ready & grounded
                & (~torso_contact) & (~internal_contact)
            )
            base_failed = base_next.metrics["failed"] > 0
            elapsed = base_next.info["step_count"].astype(jp.float32) * config.control_timestep
            timeout = (
                (base_next.metrics["timeout"] > 0)
                | (elapsed >= stop.maximum_duration_s)
            ) & (~success)
            failure = base_failed | torso_contact
            normalized_torque = data.actuator_force / jp.maximum(base.force_limits, 1.0e-6)
            action_rate_sq = jp.mean(
                jp.square(action - state.info["last_policy_action"])
            )
            raw_terms = stopping_reward_terms(
                jp,
                reward_settings,
                {
                    "target_progress": jp.abs(previous_remaining) - jp.abs(remaining),
                    "speed_error_sq": jp.square(angular_speed - desired_speed),
                    "linear_speed_sq": jp.square(linear_speed),
                    "phase_error_sq": jp.square(remaining),
                    "overshoot": jp.maximum(-remaining, 0.0),
                    "action_rate_sq": action_rate_sq,
                    "residual_action_sq": jp.mean(jp.square(action)),
                    "torque_sq": jp.mean(jp.square(normalized_torque)),
                    "internal_contact": internal_contact.astype(jp.float32),
                    "torso_contact": torso_contact.astype(jp.float32),
                    "success": success.astype(jp.float32),
                    "failure": failure.astype(jp.float32),
                    "timeout": timeout.astype(jp.float32),
                },
            )
            reward = jp.nan_to_num(sum(raw_terms.values()), nan=-reward_settings.failure)
            done = (success | failure | timeout).astype(jp.float32)
            info = {
                **base_next.info,
                "stop_target_phase": state.info["stop_target_phase"],
                "stop_initial_distance": state.info["stop_initial_distance"],
                "stop_previous_remaining": remaining,
                "stop_snapshot_index": state.info["stop_snapshot_index"],
            }
            obs = jp.concatenate((base_next.obs, self._task_observation(data, info)))
            metrics = {
                "reward": reward,
                "reward_total": reward,
                **{f"reward_{name}": value for name, value in raw_terms.items()},
                "stop_success": success.astype(jp.float32),
                "target_remaining_rad": remaining,
                "phase_error_rad": jp.abs(remaining),
                "linear_speed_m_s": jp.abs(linear_speed),
                "angular_speed_rad_s": jp.abs(angular_speed),
                "desired_angular_speed_rad_s": desired_speed,
                "residual_action_rms": jp.sqrt(jp.mean(jp.square(action))),
                "normalized_torque_rms": jp.sqrt(jp.mean(jp.square(normalized_torque))),
                "forbidden_contact_count": contacts["forbidden_count"],
                "torso_contact": torso_contact.astype(jp.float32),
                "failed": failure.astype(jp.float32),
                "timeout": timeout.astype(jp.float32),
            }
            return base_next.replace(
                obs=jp.nan_to_num(obs), reward=reward, done=done,
                metrics={name: jp.nan_to_num(value) for name, value in metrics.items()},
                info=info,
            )

    return CurlRobot2DStoppingEnv()
