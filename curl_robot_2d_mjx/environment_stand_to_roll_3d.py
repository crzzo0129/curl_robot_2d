"""Pure-policy stand-to-roll MJX environment with fixed teacher shaping.

The actor controls every physics step.  CEM is never evaluated as a controller:
its recorded orbit is used only for distance shaping and the capture milestone.
The actor observation is the deployable 36 x 20 contract from
``scripts/train_ppo_deploy.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.cem_matcher import build_reference_dict, cem_match_xp
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.config_stand_to_roll import (
    StandToRollConfig,
    validate_stand_to_roll_config,
)
from curl_robot_2d_mjx.deployment_rolling_3d import (
    CONTROLLER_JOINT_NAMES_3D,
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
    initial_rolling_deploy_history_3d,
    push_rolling_deploy_frame_3d,
    rolling_deploy_frame_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d,
    disable_rollingquad_self_collision_3d,
    geometry_parameters_3d,
    model_path_3d,
)
from curl_robot_2d_mjx.stand_to_roll_training import action_center_and_scale


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_SIZE_STAND_TO_ROLL = 12
OBSERVATION_SIZE_STAND_TO_ROLL = ROLLING_DEPLOY_OBSERVATION_SIZE_3D
ABDUCTION_ACTION_INDICES = (0, 3, 6, 9)


def _load_dependencies():
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError("stand-to-roll MJX dependencies are unavailable") from exc
    return jax, jp, mujoco, mjx, Env, State


def make_stand_to_roll_env_3d(
    config: StandToRollConfig | None = None,
    *,
    matcher_npz: Path | None = None,
    seed: int = 0,
):
    """Build the one-policy stand-to-roll environment."""

    config = config or StandToRollConfig()
    validate_stand_to_roll_config(config)
    jax, jp, mujoco, mjx, Env, State = _load_dependencies()
    physics = physics_profile_3d(
        config.physics_profile,
        Rolling3DConfig(
            geometry=config.geometry, episode_length=config.episode_length
        ),
    )
    matcher_path = Path(matcher_npz or (
        PROJECT_ROOT / "results" / "cem_cycle_data" / "cem_cycles.npz"
    ))
    if not matcher_path.is_file():
        raise FileNotFoundError(f"CEM matcher data not found: {matcher_path}")

    from curl_robot_2d_mjx.cem_matcher import CEMStateMatcher

    matcher = CEMStateMatcher(matcher_path, num_bins=200)
    reference = {
        name: jp.asarray(value)
        for name, value in build_reference_dict(matcher).items()
    }

    class StandToRollEnv(Env):
        def __init__(self):
            self.config = config
            self.seed = seed
            self.geometry_parameters = geometry_parameters_3d(config.geometry)
            self.model_path = model_path_3d(config.geometry)
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
            apply_physics_options_3d(self.mj_model, physics)
            if not config.self_collision_enabled:
                disable_rollingquad_self_collision_3d(self.mj_model)
            self.cpu_data = mujoco.MjData(self.mj_model)
            self.mjx_model = mjx.put_model(self.mj_model)
            self.base_data = mjx.put_data(self.mj_model, self.cpu_data)

            def object_id(obj_type, name):
                value = mujoco.mj_name2id(self.mj_model, obj_type, name)
                if value < 0:
                    raise ValueError(f"missing MuJoCo object: {name}")
                return int(value)

            self.torso_body_id = object_id(mujoco.mjtObj.mjOBJ_BODY, "torso")
            self.floor_geom_id = object_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
            qpos_indices = []
            dof_indices = []
            joint_ids = []
            for name in CONTROLLER_JOINT_NAMES_3D:
                joint_id = object_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                joint_ids.append(joint_id)
                qpos_indices.append(int(self.mj_model.jnt_qposadr[joint_id]))
                dof_indices.append(int(self.mj_model.jnt_dofadr[joint_id]))
            self.controller_qpos_indices = jp.asarray(qpos_indices, dtype=jp.int32)
            self.controller_dof_indices = jp.asarray(dof_indices, dtype=jp.int32)
            joint_ids = np.asarray(joint_ids)
            self.joint_low = jp.asarray(self.mj_model.jnt_range[joint_ids, 0])
            self.joint_high = jp.asarray(self.mj_model.jnt_range[joint_ids, 1])
            center, scale = action_center_and_scale(config)
            self.action_center = jp.asarray(center)
            self.action_scale = jp.asarray(scale)

            compact_id = object_id(mujoco.mjtObj.mjOBJ_KEY, "compact")
            stand_id = object_id(mujoco.mjtObj.mjOBJ_KEY, "stand")
            compact_qpos = np.array(self.mj_model.key_qpos[compact_id], copy=True)
            stand_qpos = np.array(self.mj_model.key_qpos[stand_id], copy=True)
            stand_qpos[
                np.asarray(qpos_indices)[np.asarray(ABDUCTION_ACTION_INDICES)]
            ] = 0.0
            self.compact_qpos = jp.asarray(compact_qpos)
            self.stand_qpos = jp.asarray(stand_qpos)
            self.compact_joint_position = self.compact_qpos[
                self.controller_qpos_indices
            ]

            self.force_limits = jp.asarray(
                np.maximum(np.abs(self.mj_model.actuator_forcerange[:, 1]), 1e-6)
            )
            self.rolling_radius = self.geometry_parameters.shell_contact_radius
            self.physics_timestep = float(self.mj_model.opt.timestep)
            self.action_repeat = max(
                1, round(config.control_timestep / self.physics_timestep)
            )
            self.capture_sustain_steps = max(
                1, round(config.capture_sustain_s / config.control_timestep)
            )
            self.frame_sigma = jp.concatenate(
                (
                    jp.full((3,), config.observation_noise_angular_velocity_rad_s),
                    jp.full((3,), config.observation_noise_gravity),
                    jp.zeros((6,)),
                    jp.full((12,), config.observation_noise_joint_position_rad),
                    jp.zeros((12,)),
                )
            )

        @property
        def observation_size(self):
            return OBSERVATION_SIZE_STAND_TO_ROLL

        @property
        def action_size(self):
            return ACTION_SIZE_STAND_TO_ROLL

        @property
        def backend(self):
            return "mjx"

        @property
        def sys(self):
            return self.mjx_model

        @sys.setter
        def sys(self, value):
            self.mjx_model = value

        def _body_rotation(self, data):
            return jp.reshape(data.xmat[self.torso_body_id], (3, 3))

        def _frame(self, data, last_action, noise_key):
            rotation = self._body_rotation(data)
            angular_world = data.cvel[self.torso_body_id, :3]
            angular_body = rotation.T @ angular_world
            gravity = rotation.T @ jp.asarray((0.0, 0.0, -1.0))
            joint_offset = (
                data.qpos[self.controller_qpos_indices] - self.action_center
            )
            frame = rolling_deploy_frame_3d(
                jp,
                angular_velocity_body=angular_body,
                projected_gravity=gravity,
                joint_position_offset=joint_offset,
                last_action=last_action,
            )
            if config.observation_noise_enabled:
                frame = frame + self.frame_sigma * jax.random.normal(
                    noise_key, frame.shape
                )
            return jp.clip(
                jp.nan_to_num(frame),
                -config.observation_limit,
                config.observation_limit,
            )

        def _match(self, data):
            # Raw MuJoCo storage order matches the qpos/qvel arrays used to fit
            # cem_matcher.py.  Actor observations remain in controller order.
            return cem_match_xp(
                jp,
                reference,
                data.qpos[7:19],
                data.qvel[6:18],
                self._body_rotation(data),
                data.qvel[3:6],
            )

        def _contact_arrays(self, data):
            if hasattr(data.contact, "geom1"):
                return data.contact.geom1, data.contact.geom2, data.contact.dist
            return data.contact.geom[:, 0], data.contact.geom[:, 1], data.contact.dist

        def _forbidden_contact(self, data):
            geom1, geom2, dist = self._contact_arrays(data)
            valid = (geom1 >= 0) & (geom2 >= 0) & (dist <= 0.0)
            floor = (geom1 == self.floor_geom_id) | (geom2 == self.floor_geom_id)
            return jp.any(valid & (~floor))

        def _axis_tilt(self, data):
            body_y = self._body_rotation(data)[:, 1]
            return jp.arccos(jp.clip(jp.abs(body_y[1]), 0.0, 1.0))

        def _zero_metrics(self, distance, alpha, root_z):
            zero = jp.zeros((), dtype=jp.float32)
            return {
                "reward": zero,
                "roll_progress": zero,
                "cem_progress": zero,
                "cem_orbit": jp.exp(-distance),
                "compact_progress": zero,
                "capture_bonus": zero,
                "captured": zero,
                "capture_time_s": zero,
                "cem_distance": distance,
                "reset_alpha": alpha,
                "root_z_m": root_z,
                "axis_tilt_rad": zero,
                "forbidden_contact": zero,
                "failed": zero,
                "timeout": zero,
            }

        def reset(self, rng):
            alpha_key, joint_key, velocity_key, obs_key, next_rng = jax.random.split(
                jax.random.fold_in(rng, self.seed), 5
            )
            alpha = jax.random.uniform(
                alpha_key,
                (),
                minval=config.reset_alpha_min,
                maxval=config.reset_alpha_max,
            )
            qpos = (1.0 - alpha) * self.compact_qpos + alpha * self.stand_qpos
            quat = qpos[3:7] / jp.maximum(jp.linalg.norm(qpos[3:7]), 1e-6)
            qpos = qpos.at[3:7].set(quat)
            joint_noise = jax.random.uniform(
                joint_key,
                (12,),
                minval=-config.reset_joint_noise_rad,
                maxval=config.reset_joint_noise_rad,
            )
            joints = jp.clip(
                qpos[self.controller_qpos_indices] + joint_noise,
                self.joint_low,
                self.joint_high,
            )
            qpos = qpos.at[self.controller_qpos_indices].set(joints)
            qvel = jax.random.uniform(
                velocity_key,
                (self.mj_model.nv,),
                minval=-config.reset_velocity_noise_rad_s,
                maxval=config.reset_velocity_noise_rad_s,
            )
            data = self.base_data.replace(qpos=qpos, qvel=qvel, ctrl=joints)
            data = mjx.forward(self.mjx_model, data)
            zero_action = jp.zeros((12,), dtype=jp.float32)
            history = initial_rolling_deploy_history_3d(jp)
            history = push_rolling_deploy_frame_3d(
                jp, history, self._frame(data, zero_action, obs_key)
            )
            _, distance = self._match(data)
            compact_distance = jp.sqrt(
                jp.mean(jp.square(joints - self.compact_joint_position))
            )
            info = {
                "rng": next_rng,
                "alpha": alpha,
                "history": history,
                "last_action": zero_action,
                "initial_root_x": data.qpos[0],
                "previous_roll_potential": jp.zeros(()),
                "cumulative_rotation": jp.zeros(()),
                "previous_cem_distance": distance,
                "previous_compact_distance": compact_distance,
                "capture_sustain_count": jp.zeros((), dtype=jp.int32),
                "captured": jp.asarray(False),
                "capture_bonus_given": jp.asarray(False),
                "step_count": jp.zeros((), dtype=jp.int32),
            }
            zero = jp.zeros((), dtype=jp.float32)
            return State(
                data,
                history,
                zero,
                zero,
                metrics=self._zero_metrics(distance, alpha, data.qpos[2]),
                info=info,
            )

        def step(self, state, action):
            obs_key, next_rng = jax.random.split(state.info["rng"])
            action = jp.nan_to_num(
                jp.clip(action, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0
            )
            target = jp.clip(
                self.action_center + self.action_scale * action,
                self.joint_low,
                self.joint_high,
            )

            def physics_step(data, _):
                return mjx.step(self.mjx_model, data.replace(ctrl=target)), None

            candidate, _ = jax.lax.scan(
                physics_step, state.pipeline_state, (), length=self.action_repeat
            )
            finite = (
                jp.all(jp.isfinite(candidate.qpos))
                & jp.all(jp.isfinite(candidate.qvel))
            )
            data = jax.lax.cond(
                finite, lambda _: candidate, lambda _: state.pipeline_state, None
            )

            _, distance = self._match(data)
            joints = data.qpos[self.controller_qpos_indices]
            compact_distance = jp.sqrt(
                jp.mean(jp.square(joints - self.compact_joint_position))
            )
            cumulative_rotation = (
                state.info["cumulative_rotation"]
                + data.qvel[4] * config.control_timestep
            )
            translation = (
                data.qpos[0] - state.info["initial_root_x"]
            ) / self.rolling_radius
            roll_potential = jp.minimum(cumulative_rotation, translation)
            roll_progress = roll_potential - state.info["previous_roll_potential"]
            cem_progress = state.info["previous_cem_distance"] - distance
            compact_progress = (
                state.info["previous_compact_distance"] - compact_distance
            )

            near = distance < config.capture_d_threshold
            forward = data.qvel[4] > config.capture_omega_min_rad_s
            sustain_count = jp.where(
                near & forward,
                state.info["capture_sustain_count"] + 1,
                jp.zeros((), dtype=jp.int32),
            )
            newly_captured = (
                (~state.info["captured"])
                & (sustain_count >= self.capture_sustain_steps)
            )
            captured = state.info["captured"] | newly_captured
            capture_bonus = jp.where(
                newly_captured & (~state.info["capture_bonus_given"]),
                config.reward_capture_bonus,
                0.0,
            )

            forbidden = self._forbidden_contact(data)
            action_rate = jp.mean(jp.square(action - state.info["last_action"]))
            torque_cost = jp.mean(
                jp.square(data.actuator_force / self.force_limits)
            )
            span = jp.maximum(self.joint_high - self.joint_low, 1e-6)
            joint_center = 0.5 * (self.joint_high + self.joint_low)
            normalized_joint = 2.0e0 * (joints - joint_center) / span
            joint_limit_cost = jp.mean(
                jp.square(jp.maximum(jp.abs(normalized_joint) - 0.95, 0.0))
            )
            cem_orbit = jp.exp(-distance)
            reward = (
                config.reward_roll_progress * roll_progress
                + config.reward_cem_progress * cem_progress
                + config.reward_cem_orbit * cem_orbit
                + config.reward_compact_progress * compact_progress
                + capture_bonus
                - config.reward_action_rate * action_rate
                - config.reward_torque * torque_cost
                - config.reward_joint_limit * joint_limit_cost
                - config.reward_forbidden_collision * forbidden.astype(jp.float32)
            )

            axis_tilt = self._axis_tilt(data)
            failed = (
                (~finite)
                | forbidden
                | (data.qpos[2] > config.terminate_root_z_max_m)
                | (jp.abs(data.qpos[1]) > config.terminate_lateral_m)
                | (axis_tilt > config.terminate_axis_tilt_rad)
            )
            step_count = state.info["step_count"] + 1
            timeout = step_count >= config.episode_length
            done = (failed | timeout).astype(jp.float32)
            history = push_rolling_deploy_frame_3d(
                jp, state.info["history"], self._frame(data, action, obs_key)
            )
            info = {
                **state.info,
                "rng": next_rng,
                "history": history,
                "last_action": action,
                "previous_roll_potential": roll_potential,
                "cumulative_rotation": cumulative_rotation,
                "previous_cem_distance": distance,
                "previous_compact_distance": compact_distance,
                "capture_sustain_count": sustain_count,
                "captured": captured,
                "capture_bonus_given": state.info["capture_bonus_given"]
                | newly_captured,
                "step_count": step_count,
            }
            metrics = {
                "reward": reward,
                "roll_progress": roll_progress,
                "cem_progress": cem_progress,
                "cem_orbit": cem_orbit,
                "compact_progress": compact_progress,
                "capture_bonus": capture_bonus,
                "captured": newly_captured.astype(jp.float32),
                "capture_time_s": newly_captured.astype(jp.float32)
                * step_count.astype(jp.float32)
                * config.control_timestep,
                "cem_distance": distance,
                "reset_alpha": state.info["alpha"],
                "root_z_m": data.qpos[2],
                "axis_tilt_rad": axis_tilt,
                "forbidden_contact": forbidden.astype(jp.float32),
                "failed": failed.astype(jp.float32),
                "timeout": timeout.astype(jp.float32),
            }
            return State(data, history, reward, done, metrics=metrics, info=info)

    return StandToRollEnv()
