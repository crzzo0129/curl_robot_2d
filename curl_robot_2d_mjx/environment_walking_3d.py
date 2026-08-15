"""Brax-compatible MJX environment for reference-free 3-D walking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from curl_robot_2d.model_3d import FOOT_SITE_NAMES_3D
from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS
from curl_robot_2d_mjx.config_walking_3d import (
    Walking3DConfig,
    validate_walking_3d_config,
    walking_geometry_config_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d,
    geometry_parameters_3d,
    model_path_3d,
)
from curl_robot_2d_mjx.reward_walking_3d import (
    WALKING_REWARD_TERM_NAMES_3D,
    Walking3DRewardConfig,
    reward_terms_walking_3d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WALKING_MODEL_PATH_3D = (
    PROJECT_ROOT
    / "assets"
    / "curl_robot_3d_pupper_r127p5_open60_width120.xml"
)


def freejoint_body_velocity_3d(xp, rotation, qvel):
    """Return torso-frame linear and angular velocity for a free joint.

    MuJoCo stores free-joint translation velocity in the world frame but its
    rotational velocity in the body's local frame.
    """

    return rotation.T @ qvel[:3], qvel[3:6]


def normalized_command_progress_3d(xp, planar_velocity, command):
    """Target-relative directional progress, saturated at commanded speed."""

    command_speed = xp.linalg.norm(command[:2])
    command_direction = command[:2] / xp.maximum(command_speed, 1.0e-6)
    directional_speed = xp.dot(planar_velocity[:2], command_direction)
    progress_ratio = directional_speed / xp.maximum(command_speed, 1.0e-6)
    return xp.where(
        command_speed > 0.05,
        xp.clip(progress_ratio, 0.0, 1.0),
        xp.asarray(0.0),
    )
WALKING_JOINT_NAMES_3D = tuple(
    f"{leg}_{joint}"
    for leg in ("front_left", "front_right", "rear_left", "rear_right")
    for joint in ("hip_abduction", "hip", "knee")
)
WALKING_ACTION_SIZE_3D = 12
WALKING_OBSERVATION_SIZE_3D = 48
FOOT_GEOM_NAMES_3D = (
    "front_left_foot_proxy",
    "front_right_foot_proxy",
    "rear_left_foot_proxy",
    "rear_right_foot_proxy",
)
EXPECTED_WALKING_JOINT_AXES_3D = {
    "front_left_hip_abduction": (1.0, 0.0, 0.0),
    "front_left_hip": (0.0, -1.0, 0.0),
    "front_left_knee": (0.0, 1.0, 0.0),
    "front_right_hip_abduction": (-1.0, 0.0, 0.0),
    "front_right_hip": (0.0, -1.0, 0.0),
    "front_right_knee": (0.0, 1.0, 0.0),
    "rear_left_hip_abduction": (1.0, 0.0, 0.0),
    "rear_left_hip": (0.0, 1.0, 0.0),
    "rear_left_knee": (0.0, -1.0, 0.0),
    "rear_right_hip_abduction": (-1.0, 0.0, 0.0),
    "rear_right_hip": (0.0, 1.0, 0.0),
    "rear_right_knee": (0.0, -1.0, 0.0),
}


def validate_walking_morphology_3d(
    model, geometry=PUPPER_ORIGINAL_SHELL_60_PARAMETERS
) -> None:
    """Reject models that do not match the mirrored planar-leg convention."""

    import mujoco

    if tuple(WALKING_JOINT_NAMES_3D) != tuple(EXPECTED_WALKING_JOINT_AXES_3D):
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
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_servo"
        )
        if actuator_id < 0:
            raise ValueError(f"missing walking actuator: {joint_name}_servo")
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(
                f"walking actuator is bound to wrong joint: {joint_name}_servo"
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
            geometry.upper_length,
            atol=1e-8,
        ):
            raise ValueError(f"walking upper-link length changed: {prefix}")
        if not np.isclose(
            abs(model.geom_pos[foot_id, 2]),
            geometry.lower_length,
            atol=1e-8,
        ):
            raise ValueError(f"walking lower-link length changed: {prefix}")
        if not np.isclose(
            model.geom_size[foot_id, 0],
            geometry.foot_radius,
            atol=1e-8,
        ):
            raise ValueError(f"walking foot radius changed: {prefix}")
        hip_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{prefix}_hip_abduction_body",
        )
        thigh_position = (
            model.body_pos[hip_body_id]
            if hip_body_id >= 0
            else model.body_pos[thigh_id]
        )
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
    """Create a direct-action walking task with no gait trajectory."""

    task = walking_geometry_config_3d(config or Walking3DConfig())
    validate_walking_3d_config(task)
    reward_settings = reward_config or Walking3DRewardConfig()
    jax, jp, mujoco, mjx, Env, State = _load_walking_dependencies_3d()

    class CurlRobot3DWalkingMJXEnv(Env):
        def __init__(self):
            self.config = task
            self.reward_config = reward_settings
            self.seed = seed
            self.model_path = model_path_3d(task.geometry)
            self.geometry_parameters = geometry_parameters_3d(task.geometry)
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
            validate_walking_morphology_3d(
                self.mj_model, self.geometry_parameters
            )
            apply_physics_options_3d(self.mj_model, task)
            self.cpu_data = mujoco.MjData(self.mj_model)
            # Brax's DomainRandomizationVmapWrapper requires the physics model
            # under ``env.sys`` and temporarily replaces it for every vmapped
            # environment.  All dynamics calls below must therefore read
            # ``self.sys`` rather than retaining a separate model reference.
            self.sys = mjx.put_model(self.mj_model)
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
            self.nominal_ctrl = jp.asarray(self.mj_model.key_ctrl[key_id])
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
                for name in WALKING_JOINT_NAMES_3D
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
            joint_mid = 0.5 * (self.joint_low + self.joint_high)
            soft_half_range = (
                0.5
                * (self.joint_high - self.joint_low)
                * task.soft_joint_limit_fraction
            )
            self.soft_joint_low = joint_mid - soft_half_range
            self.soft_joint_high = joint_mid + soft_half_range
            self.soft_joint_margin = jp.maximum(
                self.soft_joint_low - self.joint_low,
                self.joint_high - self.soft_joint_high,
            )
            actuator_ids = [
                object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_servo")
                for name in WALKING_JOINT_NAMES_3D
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
                "vertical_velocity_m_s": zero,
                "roll_pitch_angular_velocity_rms": zero,
                "root_x_m": zero,
                "root_y_m": zero,
                "root_z_m": zero,
                "root_height_error_m": zero,
                "lateral_drift_m": zero,
                "lateral_velocity_m_s": zero,
                "upright_tilt_rad": zero,
                "heading_error_rad": zero,
                "foot_contact_count": zero,
                "foot_air_time_reward": zero,
                "swing_clearance_cost": zero,
                "foot_slip_rms_m_s": zero,
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
                "action_rms": zero,
                "action_rate_rms": zero,
                "joint_velocity_rms_rad_s": zero,
                "joint_limit_cost": zero,
                "normalized_torque_rms": zero,
                "desired_speed_m_s": zero,
                "command_forward_velocity_m_s": zero,
                "command_lateral_velocity_m_s": zero,
                "command_yaw_rate_rad_s": zero,
                "planar_velocity_error_m_s": zero,
                "yaw_rate_error_rad_s": zero,
                "failed": zero,
                "timeout": zero,
                "failure_nonfinite": zero,
                "failure_nonfinite_action": zero,
                "failure_nonfinite_physics": zero,
                "failure_root_low": zero,
                "failure_root_high": zero,
                "failure_upright_tilt": zero,
                "lateral_drift_exceeded": zero,
                "failure_airborne": zero,
                "failure_nonfoot_depth": zero,
                "failure_nonfoot_contact": zero,
                "failure_self_contact_depth": zero,
                "failure_self_contact": zero,
            }

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            joint_key, velocity_key, root_velocity_key, command_key = (
                jax.random.split(rng, 4)
            )
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
            velocity_noise = velocity_noise.at[:2].set(
                jax.random.uniform(
                    root_velocity_key,
                    shape=(2,),
                    minval=-task.reset_root_xy_velocity_noise_m_s,
                    maxval=task.reset_root_xy_velocity_noise_m_s,
                )
            )
            velocity_noise = velocity_noise.at[5].set(
                jax.random.uniform(
                    jax.random.fold_in(root_velocity_key, 1),
                    shape=(),
                    minval=-task.reset_root_yaw_rate_noise_rad_s,
                    maxval=task.reset_root_yaw_rate_noise_rad_s,
                )
            )
            command = self._sample_command(command_key)
            start_target = jp.clip(
                self.nominal_ctrl + joint_noise,
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
            data = mjx.forward(self.sys, data)
            contacts = self._contact_metrics(data)
            body = self._body_metrics(data)
            foot_position = data.site_xpos[self.foot_site_ids]
            last_foot_contact = contacts["foot_ground"] > 0.0
            info = {
                "initial_root_y": data.qpos[1],
                "previous_root_x": data.qpos[0],
                "previous_foot_position": foot_position,
                "last_foot_contact": last_foot_contact,
                "foot_air_time": jp.zeros((4,), dtype=jp.float32),
                "last_policy_action": jp.zeros(
                    (WALKING_ACTION_SIZE_3D,), dtype=jp.float32
                ),
                "last_target": start_target,
                "root_low_step_count": jp.asarray(0, dtype=jp.int32),
                "upright_tilt_step_count": jp.asarray(0, dtype=jp.int32),
                "airborne_step_count": jp.asarray(0, dtype=jp.int32),
                "nonfoot_contact_step_count": jp.asarray(0, dtype=jp.int32),
                "self_contact_step_count": jp.asarray(0, dtype=jp.int32),
                "step_count": jp.asarray(0, dtype=jp.int32),
                "command": command,
                "command_step_count": jp.asarray(0, dtype=jp.int32),
                "time_out": jp.zeros((), dtype=jp.float32),
                "rng": rng,
            }
            observation = self._observation(
                data,
                contacts,
                body,
                initial_root_y=data.qpos[1],
                policy_action=info["last_policy_action"],
                command=command,
                noise_key=jax.random.fold_in(rng, 99),
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
            step_rng = jax.random.fold_in(
                state.info["rng"], state.info["step_count"]
            )
            command_steps = max(
                1, int(round(task.command_resample_time_s / task.control_timestep))
            )
            resample_command = state.info["command_step_count"] >= command_steps
            command = jax.lax.cond(
                resample_command,
                lambda key: self._sample_command(key),
                lambda key: state.info["command"],
                step_rng,
            )
            action_finite = jp.all(jp.isfinite(action))
            policy_action = jp.nan_to_num(
                jp.clip(action, -1.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            control_dt = task.control_timestep
            target = jp.clip(
                self.nominal_ctrl
                + policy_action * self.action_scales,
                self.joint_low,
                self.joint_high,
            )
            data = state.pipeline_state.replace(ctrl=target)

            def physics_step(carry, _):
                return mjx.step(self.sys, carry), None

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
            contacts = self._contact_metrics(data)
            body = self._body_metrics(data)
            rotation = jp.reshape(data.xmat[self.torso_body_id], (3, 3))
            body_linear_velocity, body_angular_velocity = (
                freejoint_body_velocity_3d(jp, rotation, data.qvel)
            )
            root_x, root_y, root_z = data.qpos[:3]
            forward_velocity = body_linear_velocity[0]
            lateral_velocity = body_linear_velocity[1]
            vertical_velocity = body_linear_velocity[2]
            yaw_rate = body_angular_velocity[2]
            planar_velocity_error = jp.linalg.norm(
                body_linear_velocity[:2] - command[:2]
            )
            roll_pitch_angular_velocity_squared = jp.mean(
                jp.square(body_angular_velocity[:2])
            )
            forward_progress = root_x - state.info["previous_root_x"]
            lateral_drift = root_y - state.info["initial_root_y"]
            root_height_error = root_z - task.nominal_root_height_m

            foot_contact = contacts["foot_ground"] > 0.0
            foot_position = data.site_xpos[self.foot_site_ids]
            foot_height = jp.maximum(
                foot_position[:, 2] - task.foot_radius_m, 0.0
            )
            swing = ~foot_contact
            swing_count = jp.maximum(jp.sum(swing), 1)
            clearance_shortfall = jp.maximum(
                reward_settings.swing_clearance_m - foot_height, 0.0
            )
            swing_clearance_cost = (
                jp.sum(
                    swing
                    * jp.square(
                        clearance_shortfall
                        / max(reward_settings.swing_clearance_m, 1.0e-4)
                    )
                )
                / swing_count.astype(jp.float32)
            )
            foot_velocity_xy = (
                foot_position[:, :2]
                - state.info["previous_foot_position"][:, :2]
            ) / control_dt
            foot_slip_velocity_squared = (
                jp.sum(
                    foot_contact
                    * jp.sum(jp.square(foot_velocity_xy), axis=1)
                )
                / jp.maximum(jp.sum(foot_contact), 1).astype(jp.float32)
            )
            air_time_at_touchdown = state.info["foot_air_time"] + control_dt
            touchdown = foot_contact & (~state.info["last_foot_contact"])
            air_time_span = max(
                reward_settings.foot_air_time_cap_s
                - reward_settings.foot_air_time_threshold_s,
                1.0e-4,
            )
            foot_air_time_reward = jp.mean(
                jp.where(
                    touchdown,
                    jp.clip(
                        air_time_at_touchdown
                        - reward_settings.foot_air_time_threshold_s,
                        0.0,
                        air_time_span,
                    )
                    / air_time_span,
                    0.0,
                )
            )
            foot_air_time = jp.where(
                foot_contact,
                0.0,
                air_time_at_touchdown,
            )

            joint_position = data.qpos[self.joint_qpos_indices]
            joint_velocity = data.qvel[self.joint_dof_indices]
            action_rate_cost = jp.mean(
                jp.square(policy_action - state.info["last_policy_action"])
            )
            action_magnitude_cost = jp.mean(jp.square(policy_action))
            joint_velocity_squared = jp.mean(jp.square(joint_velocity))
            lower_violation = jp.maximum(
                self.soft_joint_low - joint_position, 0.0
            )
            upper_violation = jp.maximum(
                joint_position - self.soft_joint_high, 0.0
            )
            normalized_limit_violation = (
                lower_violation + upper_violation
            ) / jp.maximum(self.soft_joint_margin, 1.0e-4)
            joint_limit_cost = jp.mean(
                jp.square(normalized_limit_violation)
            )
            command_is_still = jp.linalg.norm(command) < 0.05
            stand_still_cost = jp.where(
                command_is_still,
                jp.mean(jp.square(joint_position - self.nominal_ctrl)),
                0.0,
            )
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
            lateral_drift_exceeded = (
                jp.abs(lateral_drift) > task.diagnostic_lateral_drift_m
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
                | failure_airborne
                | failure_nonfoot_depth
                | failure_nonfoot_contact
                | failure_self_contact_depth
                | failure_self_contact
            )
            failure_severe = failed_bool & (~failure_nonfinite)
            step_count = state.info["step_count"] + 1
            timeout_bool = step_count >= task.episode_length
            # Brax PPO bootstraps healthy time limits via info["time_out"],
            # while genuine failures remain ordinary terminal transitions.
            done = (failed_bool | timeout_bool).astype(jp.float32)
            remaining_fraction = jp.maximum(
                task.episode_length - step_count, 0
            ).astype(jp.float32) / max(task.episode_length - 1, 1)

            raw_reward_terms = reward_terms_walking_3d(
                jp,
                reward_settings,
                {
                    "planar_velocity_error_norm": planar_velocity_error,
                    "yaw_rate_error": yaw_rate - command[2],
                    "normalized_forward_velocity": (
                        normalized_command_progress_3d(
                            jp, body_linear_velocity[:2], command
                        )
                    ),
                    "upright_tilt": body["upright_tilt"],
                    "root_height_error": root_height_error,
                    "heading_error": body["heading_error"],
                    "lateral_velocity": lateral_velocity,
                    "lateral_drift": lateral_drift,
                    "vertical_velocity": vertical_velocity,
                    "roll_pitch_angular_velocity_squared": (
                        roll_pitch_angular_velocity_squared
                    ),
                    "foot_air_time_reward": foot_air_time_reward,
                    "locomotion_active": (
                        jp.linalg.norm(command[:2]) > 0.05
                    ).astype(jp.float32),
                    "swing_clearance_cost": swing_clearance_cost,
                    "foot_slip_velocity_squared": (
                        foot_slip_velocity_squared
                    ),
                    "action_rate_cost": action_rate_cost,
                    "action_magnitude_cost": action_magnitude_cost,
                    "joint_velocity_squared": joint_velocity_squared,
                    "joint_limit_cost": joint_limit_cost,
                    "stand_still_cost": stand_still_cost,
                    "torque_cost": torque_cost,
                    "nonfoot_contact_active": nonfoot_active.astype(
                        jp.float32
                    ),
                    "nonfoot_depth": contacts["nonfoot_ground_depth"],
                    "self_contact_active": self_contact_active.astype(
                        jp.float32
                    ),
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

            info = {
                **state.info,
                "previous_root_x": root_x,
                "previous_foot_position": foot_position,
                "last_foot_contact": foot_contact,
                "foot_air_time": foot_air_time,
                "last_policy_action": policy_action,
                "last_target": target,
                "root_low_step_count": root_low_step_count,
                "upright_tilt_step_count": upright_tilt_step_count,
                "airborne_step_count": airborne_step_count,
                "nonfoot_contact_step_count": nonfoot_contact_step_count,
                "self_contact_step_count": self_contact_step_count,
                "step_count": step_count,
                "command": command,
                "command_step_count": jp.where(
                    resample_command,
                    jp.asarray(0, dtype=jp.int32),
                    state.info["command_step_count"] + 1,
                ),
                "time_out": timeout_bool.astype(jp.float32),
            }
            metrics = {
                "reward": reward,
                "reward_total": reward,
                **rewards,
                "forward_velocity_m_s": forward_velocity,
                "forward_progress_m": forward_progress,
                "velocity_error_m_s": forward_velocity - command[0],
                "vertical_velocity_m_s": vertical_velocity,
                "roll_pitch_angular_velocity_rms": jp.sqrt(
                    roll_pitch_angular_velocity_squared
                ),
                "root_x_m": root_x,
                "root_y_m": root_y,
                "root_z_m": root_z,
                "root_height_error_m": root_height_error,
                "lateral_drift_m": lateral_drift,
                "lateral_velocity_m_s": lateral_velocity,
                "upright_tilt_rad": body["upright_tilt"],
                "heading_error_rad": body["heading_error"],
                "foot_contact_count": contacts["foot_ground_count"],
                "foot_air_time_reward": foot_air_time_reward,
                "swing_clearance_cost": swing_clearance_cost,
                "foot_slip_rms_m_s": jp.sqrt(foot_slip_velocity_squared),
                "nonfoot_ground_contact_count": contacts[
                    "nonfoot_ground_count"
                ],
                "nonfoot_ground_depth_m": contacts[
                    "nonfoot_ground_depth"
                ],
                "self_contact_count": contacts["self_contact_count"],
                "self_contact_depth_m": contacts["self_contact_depth"],
                "airborne_active": airborne_active.astype(jp.float32),
                "airborne_step_count": airborne_step_count.astype(jp.float32),
                "root_low_step_count": root_low_step_count.astype(jp.float32),
                "upright_tilt_step_count": upright_tilt_step_count.astype(
                    jp.float32
                ),
                "nonfoot_contact_step_count": (
                    nonfoot_contact_step_count.astype(jp.float32)
                ),
                "self_contact_step_count": self_contact_step_count.astype(
                    jp.float32
                ),
                "action_rms": jp.sqrt(action_magnitude_cost),
                "action_rate_rms": jp.sqrt(action_rate_cost),
                "joint_velocity_rms_rad_s": jp.sqrt(joint_velocity_squared),
                "joint_limit_cost": joint_limit_cost,
                "normalized_torque_rms": jp.sqrt(torque_cost),
                "desired_speed_m_s": command[0],
                "command_forward_velocity_m_s": command[0],
                "command_lateral_velocity_m_s": command[1],
                "command_yaw_rate_rad_s": command[2],
                "planar_velocity_error_m_s": planar_velocity_error,
                "yaw_rate_error_rad_s": jp.abs(yaw_rate - command[2]),
                "failed": failed_bool.astype(jp.float32),
                "timeout": timeout_bool.astype(jp.float32),
                "failure_nonfinite": failure_nonfinite.astype(jp.float32),
                "failure_nonfinite_action": (
                    failure_nonfinite_action.astype(jp.float32)
                ),
                "failure_nonfinite_physics": (
                    failure_nonfinite_physics.astype(jp.float32)
                ),
                "failure_root_low": failure_root_low.astype(jp.float32),
                "failure_root_high": failure_root_high.astype(jp.float32),
                "failure_upright_tilt": failure_upright_tilt.astype(
                    jp.float32
                ),
                "lateral_drift_exceeded": lateral_drift_exceeded.astype(
                    jp.float32
                ),
                "failure_airborne": failure_airborne.astype(jp.float32),
                "failure_nonfoot_depth": failure_nonfoot_depth.astype(
                    jp.float32
                ),
                "failure_nonfoot_contact": failure_nonfoot_contact.astype(
                    jp.float32
                ),
                "failure_self_contact_depth": (
                    failure_self_contact_depth.astype(jp.float32)
                ),
                "failure_self_contact": failure_self_contact.astype(
                    jp.float32
                ),
            }
            observation = self._observation(
                data,
                contacts,
                body,
                initial_root_y=state.info["initial_root_y"],
                policy_action=policy_action,
                command=command,
                noise_key=jax.random.fold_in(step_rng, 99),
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
                        ground & ((geom1 == foot_id) | (geom2 == foot_id))
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
            command,
            noise_key,
        ):
            joint_position = data.qpos[self.joint_qpos_indices]
            joint_velocity = data.qvel[self.joint_dof_indices]
            rotation = jp.reshape(data.xmat[self.torso_body_id], (3, 3))
            body_linear_velocity, body_angular_velocity = (
                freejoint_body_velocity_3d(jp, rotation, data.qvel)
            )
            projected_gravity = rotation.T @ jp.asarray((0.0, 0.0, -1.0))
            observation_scale = jp.concatenate(
                (
                    jp.full((3,), task.observation_scale_linear_velocity),
                    jp.full((3,), task.observation_scale_angular_velocity),
                    jp.full((3,), task.observation_scale_projected_gravity),
                    jp.asarray(
                        (
                            task.observation_scale_command_linear_velocity,
                            task.observation_scale_command_linear_velocity,
                            task.observation_scale_command_yaw_rate,
                        )
                    ),
                    jp.full((12,), task.observation_scale_joint_position),
                    jp.full((12,), task.observation_scale_joint_velocity),
                    jp.full((12,), task.observation_scale_previous_action),
                )
            )
            physical_observation = jp.concatenate(
                (
                    body_linear_velocity,
                    body_angular_velocity,
                    projected_gravity,
                    command,
                    joint_position - self.nominal_ctrl,
                    joint_velocity,
                    policy_action,
                )
            )
            if not task.observation_noise_enabled:
                return physical_observation * observation_scale
            noise_scale = task.observation_noise_level * jp.concatenate(
                (
                    jp.full((3,), task.observation_noise_linear_velocity_m_s),
                    jp.full((3,), task.observation_noise_angular_velocity_rad_s),
                    jp.full((3,), task.observation_noise_gravity),
                    jp.zeros((3,)),
                    jp.full((12,), task.observation_noise_joint_position_rad),
                    jp.full((12,), task.observation_noise_joint_velocity_rad_s),
                    jp.zeros((12,)),
                )
            )
            noise = jax.random.uniform(
                noise_key,
                shape=(WALKING_OBSERVATION_SIZE_3D,),
                minval=-1.0,
                maxval=1.0,
            )
            return (physical_observation + noise * noise_scale) * observation_scale

        def _sample_command(self, rng):
            forward_key, lateral_key, yaw_key, stop_key = jax.random.split(rng, 4)
            command = jp.asarray(
                (
                    jax.random.uniform(
                        forward_key,
                        minval=task.command_forward_velocity_range_m_s[0],
                        maxval=task.command_forward_velocity_range_m_s[1],
                    ),
                    jax.random.uniform(
                        lateral_key,
                        minval=task.command_lateral_velocity_range_m_s[0],
                        maxval=task.command_lateral_velocity_range_m_s[1],
                    ),
                    jax.random.uniform(
                        yaw_key,
                        minval=task.command_yaw_rate_range_rad_s[0],
                        maxval=task.command_yaw_rate_range_rad_s[1],
                    ),
                )
            )
            stopped = jax.random.bernoulli(
                stop_key, task.command_deadband_probability
            )
            return jp.where(stopped, jp.zeros_like(command), command)

    return CurlRobot3DWalkingMJXEnv()


def _duration_to_steps(duration_s: float, control_timestep: float) -> int:
    return max(1, int(np.ceil(duration_s / control_timestep)))


def _next_active_count(xp, active, previous_count):
    return xp.where(
        active,
        previous_count + 1,
        xp.asarray(0, dtype=xp.int32),
    )
