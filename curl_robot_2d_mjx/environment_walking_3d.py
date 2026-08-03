"""Brax-compatible MJX environment for 3-D curl robot walking."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from curl_robot_2d.model_3d import FOOT_SITE_NAMES_3D, JOINT_NAMES_3D
from curl_robot_2d_mjx.config_walking_3d import (
    Walking3DConfig,
    WalkingReference3DConfig,
    smoothstep_ramp,
    validate_walking_3d_config,
)
from curl_robot_2d_mjx.environment_3d import apply_physics_options_3d
from curl_robot_2d_mjx.reward_walking_3d import (
    WALKING_REWARD_TERM_NAMES_3D,
    Walking3DRewardConfig,
    reward_terms_walking_3d,
)
from curl_robot_2d_mjx.walking_reference_3d import walking_reference_3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WALKING_MODEL_PATH_3D = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"
WALKING_ACTION_SIZE_3D = 8
WALKING_OBSERVATION_SIZE_3D = 74
FOOT_GEOM_NAMES_3D = (
    "front_left_foot_proxy",
    "front_right_foot_proxy",
    "rear_left_foot_proxy",
    "rear_right_foot_proxy",
)
EXPECTED_WALKING_JOINT_AXES_3D = {
    "front_left_hip": (0.0, -1.0, 0.0),
    "front_left_knee": (0.0, 1.0, 0.0),
    "front_right_hip": (0.0, -1.0, 0.0),
    "front_right_knee": (0.0, 1.0, 0.0),
    "rear_left_hip": (0.0, 1.0, 0.0),
    "rear_left_knee": (0.0, -1.0, 0.0),
    "rear_right_hip": (0.0, 1.0, 0.0),
    "rear_right_knee": (0.0, -1.0, 0.0),
}


def validate_walking_morphology_3d(
    model,
    reference: WalkingReference3DConfig | None = None,
) -> None:
    """Reject models that do not match the mirrored planar-leg convention."""

    import mujoco

    geometry = reference or WalkingReference3DConfig()
    if tuple(JOINT_NAMES_3D) != tuple(EXPECTED_WALKING_JOINT_AXES_3D):
        raise ValueError("unexpected 3-D walking joint order")
    for joint_name, expected_axis in EXPECTED_WALKING_JOINT_AXES_3D.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise ValueError(f"missing walking joint: {joint_name}")
        if not np.allclose(model.jnt_axis[joint_id], expected_axis, atol=1e-8):
            raise ValueError(
                f"walking joint axis changed for {joint_name}: "
                f"{model.jnt_axis[joint_id]}"
            )
    for prefix in ("front_left", "front_right", "rear_left", "rear_right"):
        thigh_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_thigh"
        )
        shank_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_shank"
        )
        foot_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_foot_proxy"
        )
        if min(thigh_id, shank_id, foot_id) < 0:
            raise ValueError(f"incomplete walking leg chain: {prefix}")
        if not np.isclose(
            abs(model.body_pos[shank_id, 2]),
            geometry.upper_length_m,
            atol=1.0e-8,
        ):
            raise ValueError(f"walking upper-link length changed: {prefix}")
        if not np.isclose(
            abs(model.geom_pos[foot_id, 2]),
            geometry.lower_length_m,
            atol=1.0e-8,
        ):
            raise ValueError(f"walking lower-link length changed: {prefix}")
        if not np.isclose(
            model.geom_size[foot_id, 0],
            geometry.foot_radius_m,
            atol=1.0e-8,
        ):
            raise ValueError(f"walking foot radius changed: {prefix}")
        thigh_position = model.body_pos[thigh_id]
        expected_x_sign = 1.0 if prefix.startswith("front") else -1.0
        expected_y_sign = 1.0 if prefix.endswith("left") else -1.0
        if thigh_position[0] * expected_x_sign <= 0.0:
            raise ValueError(f"walking front/rear layout changed: {prefix}")
        if thigh_position[1] * expected_y_sign <= 0.0:
            raise ValueError(f"walking left/right rail layout changed: {prefix}")


def _load_walking_dependencies_3d():
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "3-D walking MJX dependencies are unavailable. Install "
            "requirements-mjx.txt on the Linux GPU instance."
        ) from exc
    return jax, jp, mujoco, mjx, Env, State


def make_brax_walking_env_3d(
    config: Walking3DConfig | None = None,
    *,
    reward_config: Walking3DRewardConfig | None = None,
    seed: int = 0,
):
    """Create the morphology-aware 3-D walking residual environment."""

    task = config or Walking3DConfig()
    validate_walking_3d_config(task)
    reward_settings = reward_config or Walking3DRewardConfig()
    jax, jp, mujoco, mjx, Env, State = _load_walking_dependencies_3d()

    class CurlRobot3DWalkingMJXEnv(Env):
        def __init__(self):
            self.config = task
            self.reward_config = reward_settings
            self.seed = seed
            self.mj_model = mujoco.MjModel.from_xml_path(
                str(WALKING_MODEL_PATH_3D)
            )
            validate_walking_morphology_3d(self.mj_model, task.reference)
            apply_physics_options_3d(self.mj_model, task)
            self.cpu_data = mujoco.MjData(self.mj_model)
            self.mjx_model = mjx.put_model(self.mj_model)
            self.base_data = mjx.put_data(self.mj_model, self.cpu_data)

            def object_id(object_type, name):
                value = mujoco.mj_name2id(self.mj_model, object_type, name)
                if value < 0:
                    raise ValueError(f"missing MuJoCo object: {name}")
                return int(value)

            key_id = object_id(
                mujoco.mjtObj.mjOBJ_KEY, task.reset_keyframe_name
            )
            self.reset_qpos = jp.asarray(self.mj_model.key_qpos[key_id])
            self.reset_ctrl = jp.asarray(self.mj_model.key_ctrl[key_id])
            self.action_scales = jp.asarray(task.action_scales)
            self.torso_body_id = object_id(
                mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
            self.floor_geom_id = object_id(
                mujoco.mjtObj.mjOBJ_GEOM, "floor"
            )
            self.foot_geom_id_values = tuple(
                object_id(mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in FOOT_GEOM_NAMES_3D
            )
            self.foot_geom_ids = jp.asarray(
                self.foot_geom_id_values, dtype=jp.int32
            )
            self.foot_site_ids = jp.asarray(
                [
                    object_id(mujoco.mjtObj.mjOBJ_SITE, name)
                    for name in FOOT_SITE_NAMES_3D
                ],
                dtype=jp.int32,
            )
            joint_ids = [
                object_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES_3D
            ]
            self.joint_qpos_indices = jp.asarray(
                [
                    int(self.mj_model.jnt_qposadr[joint_id])
                    for joint_id in joint_ids
                ],
                dtype=jp.int32,
            )
            self.joint_dof_indices = jp.asarray(
                [
                    int(self.mj_model.jnt_dofadr[joint_id])
                    for joint_id in joint_ids
                ],
                dtype=jp.int32,
            )
            self.joint_low = jp.asarray(
                [self.mj_model.jnt_range[joint_id, 0] for joint_id in joint_ids]
            )
            self.joint_high = jp.asarray(
                [self.mj_model.jnt_range[joint_id, 1] for joint_id in joint_ids]
            )
            self.initial_phase = jp.asarray(
                2.0 * math.pi * task.reference.initial_phase_fraction,
                dtype=jp.float32,
            )
            self.initial_reference = walking_reference_3d(
                jp, self.initial_phase, task.reference
            )
            self.startup_ctrl = jp.clip(
                self.reset_ctrl
                + task.reset_reference_weight
                * (
                    self.initial_reference["joint_targets"]
                    - self.reset_ctrl
                ),
                self.joint_low,
                self.joint_high,
            )
            actuator_ids = [
                object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_servo")
                for name in JOINT_NAMES_3D
            ]
            self.actuator_ids = jp.asarray(actuator_ids, dtype=jp.int32)
            self.force_limits = jp.asarray(
                np.abs(self.mj_model.actuator_forcerange[actuator_ids, 1])
            )
            self.root_low_steps = _duration_to_steps(
                task.terminate_root_z_low_duration_s, task.control_timestep
            )
            self.upright_tilt_steps = _duration_to_steps(
                task.terminate_upright_tilt_duration_s,
                task.control_timestep,
            )
            self.airborne_steps = _duration_to_steps(
                task.terminate_airborne_duration_s, task.control_timestep
            )
            self.nonfoot_contact_steps = _duration_to_steps(
                task.terminate_nonfoot_contact_duration_s,
                task.control_timestep,
            )
            self.self_contact_steps = _duration_to_steps(
                task.terminate_self_contact_duration_s,
                task.control_timestep,
            )

        @property
        def observation_size(self):
            return WALKING_OBSERVATION_SIZE_3D

        @property
        def action_size(self):
            return WALKING_ACTION_SIZE_3D

        @property
        def backend(self):
            return "mjx"

        def _zero_metrics(self):
            zero = jp.zeros((), dtype=jp.float32)
            return {
                "reward": zero,
                "reward_total": zero,
                **{
                    f"reward_{name}": zero
                    for name in WALKING_REWARD_TERM_NAMES_3D
                },
                "forward_velocity_m_s": zero,
                "forward_progress_m": zero,
                "velocity_error_m_s": zero,
                "root_x_m": zero,
                "root_y_m": zero,
                "root_z_m": zero,
                "root_height_error_m": zero,
                "lateral_drift_m": zero,
                "lateral_velocity_m_s": zero,
                "upright_tilt_rad": zero,
                "heading_error_rad": zero,
                "foot_contact_count": zero,
                "stance_miss_fraction": zero,
                "swing_contact_fraction": zero,
                "swing_clearance_cost": zero,
                "nonfoot_ground_contact_count": zero,
                "nonfoot_ground_depth_m": zero,
                "self_contact_count": zero,
                "self_contact_depth_m": zero,
                "airborne_active": zero,
                "airborne_step_count": zero,
                "root_low_step_count": zero,
                "upright_tilt_step_count": zero,
                "nonfoot_contact_step_count": zero,
                "self_contact_step_count": zero,
                "joint_tracking_rms": zero,
                "residual_action_rms": zero,
                "action_rate_rms": zero,
                "normalized_torque_rms": zero,
                "reference_blend": zero,
                "startup_action_ramp": zero,
                "oscillator_phase_rad": zero,
                "desired_speed_m_s": zero,
                "reset_reference_weight": zero,
                "failed": zero,
                "timeout": zero,
                "failure_nonfinite": zero,
                "failure_nonfinite_action": zero,
                "failure_nonfinite_physics": zero,
                "failure_root_low": zero,
                "failure_root_high": zero,
                "failure_upright_tilt": zero,
                "failure_lateral_drift": zero,
                "failure_airborne": zero,
                "failure_nonfoot_depth": zero,
                "failure_nonfoot_contact": zero,
                "failure_self_contact_depth": zero,
                "failure_self_contact": zero,
            }

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            joint_key, velocity_key = jax.random.split(rng, 2)
            joint_noise = jax.random.uniform(
                joint_key,
                shape=(WALKING_ACTION_SIZE_3D,),
                minval=-task.reset_joint_noise_rad,
                maxval=task.reset_joint_noise_rad,
            )
            velocity_noise = jax.random.uniform(
                velocity_key,
                shape=(self.mj_model.nv,),
                minval=-task.reset_velocity_noise,
                maxval=task.reset_velocity_noise,
            )
            start_target = jp.clip(
                self.startup_ctrl + joint_noise,
                self.joint_low,
                self.joint_high,
            )
            qpos = self.reset_qpos.at[self.joint_qpos_indices].set(start_target)
            qpos = qpos.at[0].set(0.0)
            qpos = qpos.at[1].set(0.0)
            data = self.base_data.replace(
                qpos=qpos,
                qvel=velocity_noise,
                ctrl=start_target,
            )
            data = mjx.forward(self.mjx_model, data)
            contacts = self._contact_metrics(data)
            body = self._body_metrics(data)
            phase = self.initial_phase
            reference = self.initial_reference
            info = {
                "initial_root_x": data.qpos[0],
                "initial_root_y": data.qpos[1],
                "previous_root_x": data.qpos[0],
                "last_policy_action": jp.zeros(
                    (WALKING_ACTION_SIZE_3D,), dtype=jp.float32
                ),
                "last_target": start_target,
                "oscillator_phase": phase,
                "root_low_step_count": jp.asarray(0, dtype=jp.int32),
                "upright_tilt_step_count": jp.asarray(0, dtype=jp.int32),
                "airborne_step_count": jp.asarray(0, dtype=jp.int32),
                "nonfoot_contact_step_count": jp.asarray(0, dtype=jp.int32),
                "self_contact_step_count": jp.asarray(0, dtype=jp.int32),
                "step_count": jp.asarray(0, dtype=jp.int32),
            }
            observation = self._observation(
                data,
                contacts,
                body,
                initial_root_y=data.qpos[1],
                policy_action=info["last_policy_action"],
                reference_target=start_target,
                stance=reference["stance"],
                oscillator_phase=phase,
                reference_blend=jp.zeros((), dtype=jp.float32),
                action_ramp=jp.zeros((), dtype=jp.float32),
            )
            return State(
                data,
                jp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0),
                jp.zeros((), dtype=jp.float32),
                jp.zeros((), dtype=jp.float32),
                metrics=self._zero_metrics(),
                info=info,
            )

        def step(self, state, action):
            action_finite = jp.all(jp.isfinite(action))
            policy_action = jp.nan_to_num(
                jp.clip(action, -1.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            control_dt = task.control_timestep
            elapsed_s = state.info["step_count"].astype(jp.float32) * control_dt
            reference_blend = smoothstep_ramp(
                jp, elapsed_s, task.startup_reference_ramp_s
            )
            action_ramp = smoothstep_ramp(
                jp, elapsed_s, task.startup_action_ramp_s
            )
            reference = walking_reference_3d(
                jp, state.info["oscillator_phase"], task.reference
            )
            reference_target = jp.clip(
                self.startup_ctrl
                + reference_blend
                * (reference["joint_targets"] - self.startup_ctrl),
                self.joint_low,
                self.joint_high,
            )
            target = jp.clip(
                reference_target
                + action_ramp
                * task.residual_gain
                * policy_action
                * self.action_scales,
                self.joint_low,
                self.joint_high,
            )
            data = state.pipeline_state.replace(ctrl=target)

            def physics_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            candidate_data = jax.lax.scan(
                physics_step, data, (), length=task.action_repeat
            )[0]
            _, _, candidate_contact_distance = self._contact_arrays(
                candidate_data
            )
            physics_finite = (
                jp.all(jp.isfinite(candidate_data.qpos))
                & jp.all(jp.isfinite(candidate_data.qvel))
                & jp.all(jp.isfinite(candidate_data.qacc))
                & jp.all(jp.isfinite(candidate_data.actuator_force))
                & jp.all(jp.isfinite(candidate_data.xpos))
                & jp.all(jp.isfinite(candidate_data.xmat))
                & jp.all(jp.isfinite(candidate_data.site_xpos))
                & jp.all(jp.isfinite(candidate_contact_distance))
            )
            transition_finite = action_finite & physics_finite
            data = jax.lax.cond(
                transition_finite,
                lambda _: candidate_data,
                lambda _: state.pipeline_state,
                operand=None,
            )
            policy_action = jp.where(
                transition_finite,
                policy_action,
                state.info["last_policy_action"],
            )
            target = jp.where(
                transition_finite, target, state.info["last_target"]
            )
            oscillator_phase = jp.where(
                transition_finite,
                jp.mod(
                    state.info["oscillator_phase"]
                    + 2.0 * math.pi * task.reference.frequency_hz * control_dt,
                    2.0 * math.pi,
                ),
                state.info["oscillator_phase"],
            )
            contacts = self._contact_metrics(data)
            body = self._body_metrics(data)
            root_x, root_y, root_z = data.qpos[:3]
            forward_velocity = data.qvel[0]
            lateral_velocity = data.qvel[1]
            forward_progress = root_x - state.info["previous_root_x"]
            lateral_drift = root_y - state.info["initial_root_y"]
            root_height_error = root_z - task.reference.root_height_m
            desired_speed = task.reference.desired_speed_m_s

            foot_contact = contacts["foot_ground"] > 0.0
            stance = reference["stance"]
            stance_count = jp.maximum(jp.sum(stance), 1)
            swing = ~stance
            swing_count = jp.maximum(jp.sum(swing), 1)
            stance_miss_fraction = (
                jp.sum(stance & (~foot_contact)).astype(jp.float32)
                / stance_count.astype(jp.float32)
            )
            swing_contact_fraction = (
                jp.sum(swing & foot_contact).astype(jp.float32)
                / swing_count.astype(jp.float32)
            )
            foot_clearance = jp.maximum(
                data.site_xpos[self.foot_site_ids, 2]
                - task.reference.foot_radius_m,
                0.0,
            )
            desired_clearance = 0.75 * reference["foot_lift_m"]
            clearance_shortfall = jp.maximum(
                desired_clearance - foot_clearance, 0.0
            )
            swing_clearance_cost = (
                jp.sum(
                    swing
                    * jp.square(
                        clearance_shortfall
                        / max(task.reference.foot_lift_m, 1.0e-4)
                    )
                )
                / swing_count.astype(jp.float32)
            )
            joint_position = data.qpos[self.joint_qpos_indices]
            normalized_joint_error = (
                joint_position - reference_target
            ) / self.action_scales
            joint_tracking_cost = jp.mean(jp.square(normalized_joint_error))
            action_rate_cost = jp.mean(
                jp.square(policy_action - state.info["last_policy_action"])
            )
            residual_action_cost = jp.mean(jp.square(policy_action))
            normalized_torque = (
                data.actuator_force[self.actuator_ids]
                / jp.maximum(self.force_limits, 1.0e-6)
            )
            torque_cost = jp.mean(jp.square(normalized_torque))

            root_low_active = root_z < task.terminate_root_z_min
            upright_tilt_active = (
                body["upright_tilt"] > task.terminate_upright_tilt_rad
            )
            airborne_active = contacts["foot_ground_count"] < 0.5
            nonfoot_active = contacts["nonfoot_ground_count"] > 0.0
            self_contact_active = contacts["self_contact_count"] > 0.0
            root_low_step_count = _next_active_count(
                jp, root_low_active, state.info["root_low_step_count"]
            )
            upright_tilt_step_count = _next_active_count(
                jp,
                upright_tilt_active,
                state.info["upright_tilt_step_count"],
            )
            airborne_step_count = _next_active_count(
                jp, airborne_active, state.info["airborne_step_count"]
            )
            nonfoot_contact_step_count = _next_active_count(
                jp,
                nonfoot_active,
                state.info["nonfoot_contact_step_count"],
            )
            self_contact_step_count = _next_active_count(
                jp,
                self_contact_active,
                state.info["self_contact_step_count"],
            )

            failure_nonfinite_action = ~action_finite
            failure_nonfinite_physics = action_finite & (~physics_finite)
            failure_nonfinite = ~transition_finite
            failure_root_low = root_low_step_count >= self.root_low_steps
            failure_root_high = root_z > task.terminate_root_z_max
            failure_upright_tilt = (
                upright_tilt_step_count >= self.upright_tilt_steps
            )
            failure_lateral_drift = (
                jp.abs(lateral_drift) > task.terminate_lateral_drift_m
            )
            failure_airborne = airborne_step_count >= self.airborne_steps
            failure_nonfoot_depth = (
                contacts["nonfoot_ground_depth"]
                > task.terminate_nonfoot_depth_m
            )
            failure_nonfoot_contact = (
                nonfoot_contact_step_count >= self.nonfoot_contact_steps
            )
            failure_self_contact_depth = (
                contacts["self_contact_depth"]
                > task.terminate_self_contact_depth_m
            )
            failure_self_contact = (
                self_contact_step_count >= self.self_contact_steps
            )
            failed_bool = (
                failure_nonfinite
                | failure_root_low
                | failure_root_high
                | failure_upright_tilt
                | failure_lateral_drift
                | failure_airborne
                | failure_nonfoot_depth
                | failure_nonfoot_contact
                | failure_self_contact_depth
                | failure_self_contact
            )
            failure_severe = failed_bool & (~failure_nonfinite)
            step_count = state.info["step_count"] + 1
            timeout_bool = step_count >= task.episode_length
            done = (failed_bool | timeout_bool).astype(jp.float32)
            remaining_fraction = jp.maximum(
                task.episode_length - step_count, 0
            ).astype(jp.float32) / max(task.episode_length - 1, 1)

            raw_reward_terms = reward_terms_walking_3d(
                jp,
                reward_settings,
                {
                    "forward_velocity_error": (
                        forward_velocity - desired_speed
                    ),
                    "normalized_forward_velocity": jp.clip(
                        forward_velocity / max(desired_speed, 1.0e-4),
                        -1.0,
                        1.5,
                    ),
                    "upright_tilt": body["upright_tilt"],
                    "root_height_error": root_height_error,
                    "heading_error": body["heading_error"],
                    "lateral_velocity": lateral_velocity,
                    "lateral_drift": lateral_drift,
                    "stance_miss_fraction": (
                        reference_blend * stance_miss_fraction
                    ),
                    "swing_contact_fraction": (
                        reference_blend * swing_contact_fraction
                    ),
                    "swing_clearance_cost": (
                        reference_blend * swing_clearance_cost
                    ),
                    "joint_tracking_cost": joint_tracking_cost,
                    "action_rate_cost": action_rate_cost,
                    "residual_action_cost": residual_action_cost,
                    "torque_cost": torque_cost,
                    "nonfoot_contact_active": nonfoot_active.astype(jp.float32),
                    "nonfoot_depth": contacts["nonfoot_ground_depth"],
                    "self_contact_active": self_contact_active.astype(jp.float32),
                    "self_contact_depth": contacts["self_contact_depth"],
                    "failed": failed_bool.astype(jp.float32),
                    "failure_severe": failure_severe.astype(jp.float32),
                    "failure_nonfinite": failure_nonfinite.astype(jp.float32),
                    "remaining_fraction": remaining_fraction,
                },
            )
            raw_reward_terms = {
                name: (
                    value
                    if name in ("termination", "early_termination")
                    else jp.where(failure_nonfinite, 0.0, value)
                )
                for name, value in raw_reward_terms.items()
            }
            reward = jp.nan_to_num(
                sum(raw_reward_terms.values()),
                nan=-reward_settings.nonfinite_termination,
                posinf=-reward_settings.nonfinite_termination,
                neginf=-reward_settings.nonfinite_termination,
            )
            rewards = {
                f"reward_{name}": jp.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                )
                for name, value in raw_reward_terms.items()
            }

            next_elapsed_s = step_count.astype(jp.float32) * control_dt
            next_reference_blend = smoothstep_ramp(
                jp, next_elapsed_s, task.startup_reference_ramp_s
            )
            next_action_ramp = smoothstep_ramp(
                jp, next_elapsed_s, task.startup_action_ramp_s
            )
            next_reference = walking_reference_3d(
                jp, oscillator_phase, task.reference
            )
            next_reference_target = jp.clip(
                self.startup_ctrl
                + next_reference_blend
                * (next_reference["joint_targets"] - self.startup_ctrl),
                self.joint_low,
                self.joint_high,
            )
            info = {
                **state.info,
                "previous_root_x": root_x,
                "last_policy_action": policy_action,
                "last_target": target,
                "oscillator_phase": oscillator_phase,
                "root_low_step_count": root_low_step_count,
                "upright_tilt_step_count": upright_tilt_step_count,
                "airborne_step_count": airborne_step_count,
                "nonfoot_contact_step_count": nonfoot_contact_step_count,
                "self_contact_step_count": self_contact_step_count,
                "step_count": step_count,
            }
            metrics = {
                "reward": reward,
                "reward_total": reward,
                **rewards,
                "forward_velocity_m_s": forward_velocity,
                "forward_progress_m": forward_progress,
                "velocity_error_m_s": forward_velocity - desired_speed,
                "root_x_m": root_x,
                "root_y_m": root_y,
                "root_z_m": root_z,
                "root_height_error_m": root_height_error,
                "lateral_drift_m": lateral_drift,
                "lateral_velocity_m_s": lateral_velocity,
                "upright_tilt_rad": body["upright_tilt"],
                "heading_error_rad": body["heading_error"],
                "foot_contact_count": contacts["foot_ground_count"],
                "stance_miss_fraction": stance_miss_fraction,
                "swing_contact_fraction": swing_contact_fraction,
                "swing_clearance_cost": swing_clearance_cost,
                "nonfoot_ground_contact_count": contacts[
                    "nonfoot_ground_count"
                ],
                "nonfoot_ground_depth_m": contacts["nonfoot_ground_depth"],
                "self_contact_count": contacts["self_contact_count"],
                "self_contact_depth_m": contacts["self_contact_depth"],
                "airborne_active": airborne_active.astype(jp.float32),
                "airborne_step_count": airborne_step_count.astype(jp.float32),
                "root_low_step_count": root_low_step_count.astype(jp.float32),
                "upright_tilt_step_count": upright_tilt_step_count.astype(
                    jp.float32
                ),
                "nonfoot_contact_step_count": nonfoot_contact_step_count.astype(
                    jp.float32
                ),
                "self_contact_step_count": self_contact_step_count.astype(
                    jp.float32
                ),
                "joint_tracking_rms": jp.sqrt(joint_tracking_cost),
                "residual_action_rms": jp.sqrt(residual_action_cost),
                "action_rate_rms": jp.sqrt(action_rate_cost),
                "normalized_torque_rms": jp.sqrt(torque_cost),
                "reference_blend": reference_blend,
                "startup_action_ramp": action_ramp,
                "oscillator_phase_rad": oscillator_phase,
                "desired_speed_m_s": jp.asarray(desired_speed),
                "reset_reference_weight": jp.asarray(
                    task.reset_reference_weight
                ),
                "failed": failed_bool.astype(jp.float32),
                "timeout": timeout_bool.astype(jp.float32),
                "failure_nonfinite": failure_nonfinite.astype(jp.float32),
                "failure_nonfinite_action": failure_nonfinite_action.astype(
                    jp.float32
                ),
                "failure_nonfinite_physics": failure_nonfinite_physics.astype(
                    jp.float32
                ),
                "failure_root_low": failure_root_low.astype(jp.float32),
                "failure_root_high": failure_root_high.astype(jp.float32),
                "failure_upright_tilt": failure_upright_tilt.astype(jp.float32),
                "failure_lateral_drift": failure_lateral_drift.astype(jp.float32),
                "failure_airborne": failure_airborne.astype(jp.float32),
                "failure_nonfoot_depth": failure_nonfoot_depth.astype(jp.float32),
                "failure_nonfoot_contact": failure_nonfoot_contact.astype(
                    jp.float32
                ),
                "failure_self_contact_depth": failure_self_contact_depth.astype(
                    jp.float32
                ),
                "failure_self_contact": failure_self_contact.astype(jp.float32),
            }
            observation = self._observation(
                data,
                contacts,
                body,
                initial_root_y=state.info["initial_root_y"],
                policy_action=policy_action,
                reference_target=next_reference_target,
                stance=next_reference["stance"],
                oscillator_phase=oscillator_phase,
                reference_blend=next_reference_blend,
                action_ramp=next_action_ramp,
            )
            metrics = {
                name: jp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                for name, value in metrics.items()
            }
            return State(
                data,
                jp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0),
                reward,
                done,
                metrics=metrics,
                info=info,
            )

        def _contact_arrays(self, data):
            contact = data.contact
            if hasattr(contact, "geom1"):
                return contact.geom1, contact.geom2, contact.dist
            return contact.geom[:, 0], contact.geom[:, 1], contact.dist

        def _geom_in_ids(self, geom, ids):
            return jp.any(geom[:, None] == ids[None, :], axis=1)

        def _contact_metrics(self, data):
            geom1, geom2, distance = self._contact_arrays(data)
            valid = (geom1 >= 0) & (geom2 >= 0) & (distance <= 0.0)
            ground = valid & (
                (geom1 == self.floor_geom_id)
                | (geom2 == self.floor_geom_id)
            )
            geom1_foot = self._geom_in_ids(geom1, self.foot_geom_ids)
            geom2_foot = self._geom_in_ids(geom2, self.foot_geom_ids)
            foot_ground = jp.stack(
                [
                    jp.any(
                        ground
                        & ((geom1 == foot_id) | (geom2 == foot_id))
                    ).astype(jp.float32)
                    for foot_id in self.foot_geom_id_values
                ]
            )
            nonfoot_ground = ground & (~geom1_foot) & (~geom2_foot)
            self_contact = valid & (~ground)
            return {
                "foot_ground": foot_ground,
                "foot_ground_count": jp.sum(foot_ground),
                "nonfoot_ground_count": jp.sum(nonfoot_ground).astype(
                    jp.float32
                ),
                "nonfoot_ground_depth": jp.max(
                    jp.where(nonfoot_ground, -distance, 0.0)
                ),
                "self_contact_count": jp.sum(self_contact).astype(jp.float32),
                "self_contact_depth": jp.max(
                    jp.where(self_contact, -distance, 0.0)
                ),
            }

        def _body_metrics(self, data):
            rotation = jp.reshape(data.xmat[self.torso_body_id], (3, 3))
            body_x_axis = rotation[:, 0]
            body_y_axis = rotation[:, 1]
            body_z_axis = rotation[:, 2]
            upright_tilt = jp.arccos(jp.clip(body_z_axis[2], -1.0, 1.0))
            heading_error = jp.arctan2(body_x_axis[1], body_x_axis[0])
            return {
                "body_x_axis": body_x_axis,
                "body_y_axis": body_y_axis,
                "body_z_axis": body_z_axis,
                "upright_tilt": upright_tilt,
                "heading_error": heading_error,
            }

        def _observation(
            self,
            data,
            contacts,
            body,
            *,
            initial_root_y,
            policy_action,
            reference_target,
            stance,
            oscillator_phase,
            reference_blend,
            action_ramp,
        ):
            joint_position = data.qpos[self.joint_qpos_indices]
            joint_velocity = data.qvel[self.joint_dof_indices]
            foot_height = (
                data.site_xpos[self.foot_site_ids, 2]
                - task.reference.foot_radius_m
            )
            return jp.concatenate(
                (
                    jp.asarray((data.qpos[2], data.qpos[1] - initial_root_y)),
                    body["body_x_axis"],
                    body["body_y_axis"],
                    body["body_z_axis"],
                    data.qvel[:3],
                    data.qvel[3:6],
                    joint_position,
                    joint_velocity,
                    policy_action,
                    reference_target,
                    joint_position - reference_target,
                    contacts["foot_ground"],
                    foot_height,
                    stance.astype(jp.float32),
                    jp.asarray(
                        (
                            jp.sin(oscillator_phase),
                            jp.cos(oscillator_phase),
                            reference_blend,
                            action_ramp,
                            task.reference.desired_speed_m_s,
                        )
                    ),
                )
            )

    return CurlRobot3DWalkingMJXEnv()


def _duration_to_steps(duration_s: float, control_timestep: float) -> int:
    return max(1, int(np.ceil(duration_s / control_timestep)))


def _next_active_count(xp, active, previous_count):
    return xp.where(
        active,
        previous_count + 1,
        xp.asarray(0, dtype=xp.int32),
    )
