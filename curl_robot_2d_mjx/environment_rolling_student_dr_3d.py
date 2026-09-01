"""Asymmetric rolling environment for reward-driven Student DR PPO."""

from __future__ import annotations

from curl_robot_2d_mjx.deployment_rolling_3d import (
    CONTROLLER_JOINT_NAMES_3D,
    ROLLING_CONTROLLER_ACTION_SIZE_3D,
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
    effective_action_to_controller_action_3d,
    initial_rolling_deploy_history_3d,
    push_rolling_deploy_frame_3d,
    rolling_deploy_frame_3d,
)
from curl_robot_2d_mjx.rolling_student_dr_ppo_3d import (
    ROLLING_STUDENT_PPO_ACTION_SIZE_3D,
    ROLLING_STUDENT_PPO_CRITIC_OBSERVATION_SIZE_3D,
)


def make_rolling_student_dr_env_3d(
    base_env,
    deploy_settings,
    *,
    student_anchor_policy,
    student_anchor_weight: float,
    observation_noise_scale: float,
    minimum_success_turns: float = 5.0,
):
    """Wrap the direct rolling task with real observations and deploy effects."""

    import jax
    import jax.numpy as jp
    import mujoco
    from brax.envs.base import Env

    if student_anchor_weight < 0.0:
        raise ValueError("student_anchor_weight must be nonnegative")
    if observation_noise_scale < 0.0:
        raise ValueError("observation_noise_scale must be nonnegative")
    if minimum_success_turns < 0.0:
        raise ValueError("minimum_success_turns must be nonnegative")

    controller_qpos_indices = []
    for name in CONTROLLER_JOINT_NAMES_3D:
        joint_id = mujoco.mj_name2id(
            base_env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"missing deployment joint: {name}")
        controller_qpos_indices.append(
            int(base_env.mj_model.jnt_qposadr[joint_id])
        )
    compact_key_id = mujoco.mj_name2id(
        base_env.mj_model, mujoco.mjtObj.mjOBJ_KEY, "compact"
    )
    if compact_key_id < 0:
        raise ValueError("missing required compact keyframe")
    compact_position = jp.asarray(
        base_env.mj_model.key_qpos[compact_key_id][controller_qpos_indices]
    )
    controller_qpos_indices = jp.asarray(controller_qpos_indices)
    frame_sigma = jp.concatenate(
        (
            jp.full((3,), 0.20),
            jp.full((3,), 0.05),
            jp.zeros((6,)),
            jp.full((12,), 0.01),
            jp.zeros((12,)),
        )
    )
    latency_probabilities = jp.asarray(
        deploy_settings.action_latency_probabilities
    )

    class RollingStudentDREnv(Env):
        def __init__(self):
            self.base_env = base_env
            self.config = base_env.config
            self.mj_model = base_env.mj_model
            self.torso_body_id = base_env.torso_body_id
            self.floor_geom_id = base_env.floor_geom_id

        @property
        def sys(self):
            return self.base_env.sys

        @sys.setter
        def sys(self, value):
            self.base_env.sys = value

        @property
        def observation_size(self):
            return {
                "state": ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
                "privileged_state": (
                    ROLLING_STUDENT_PPO_CRITIC_OBSERVATION_SIZE_3D
                ),
            }

        @property
        def action_size(self):
            return ROLLING_STUDENT_PPO_ACTION_SIZE_3D

        @property
        def backend(self):
            return "mjx"

        def _actor_observation(
            self,
            state,
            history,
            previous_controller_action,
            encoder_bias,
            noise_key,
        ):
            data = state.pipeline_state
            rotation = jp.reshape(
                data.xmat[self.torso_body_id], (3, 3)
            )
            angular_world = data.cvel[self.torso_body_id, :3]
            angular_body = rotation.T @ angular_world
            projected_gravity = rotation.T @ jp.asarray((0.0, 0.0, -1.0))
            joint_offset = (
                data.qpos[controller_qpos_indices]
                - compact_position
                + encoder_bias
            )
            frame = rolling_deploy_frame_3d(
                jp,
                angular_velocity_body=angular_body,
                projected_gravity=projected_gravity,
                joint_position_offset=joint_offset,
                last_action=previous_controller_action,
            )
            if observation_noise_scale > 0.0:
                frame = frame + (
                    observation_noise_scale
                    * frame_sigma
                    * jax.random.normal(noise_key, frame.shape)
                )
            return push_rolling_deploy_frame_3d(
                jp, history, jp.nan_to_num(frame)
            )

        def reset(self, rng):
            (
                base_key,
                latency_key,
                motor_key,
                encoder_key,
                observation_key,
                next_rng,
            ) = jax.random.split(rng, 6)
            base_state = self.base_env.reset(base_key)
            motor_zero_bias = jax.random.uniform(
                motor_key,
                (ROLLING_CONTROLLER_ACTION_SIZE_3D,),
                minval=-deploy_settings.motor_zero_bias_rad,
                maxval=deploy_settings.motor_zero_bias_rad,
            )
            encoder_bias = jax.random.uniform(
                encoder_key,
                (ROLLING_CONTROLLER_ACTION_SIZE_3D,),
                minval=-deploy_settings.encoder_fixed_bias_rad,
                maxval=deploy_settings.encoder_fixed_bias_rad,
            )
            previous_controller_action = jp.zeros(
                (ROLLING_CONTROLLER_ACTION_SIZE_3D,)
            )
            actor_history = self._actor_observation(
                base_state,
                initial_rolling_deploy_history_3d(jp),
                previous_controller_action,
                encoder_bias,
                observation_key,
            )
            info = {
                **base_state.info,
                "deploy_rng": next_rng,
                "deploy_action_queue": jp.zeros(
                    (3, ROLLING_STUDENT_PPO_ACTION_SIZE_3D)
                ),
                "deploy_applied_action": jp.zeros(
                    (ROLLING_STUDENT_PPO_ACTION_SIZE_3D,)
                ),
                "deploy_latency_steps": jax.random.choice(
                    latency_key, 3, p=latency_probabilities
                ),
                "motor_zero_bias_ctrl": motor_zero_bias,
                "encoder_bias": encoder_bias,
                "actor_history": actor_history,
                "previous_controller_action": previous_controller_action,
                "time_out": jp.zeros((), dtype=jp.float32),
            }
            metrics = {
                **base_state.metrics,
                "reward_student_anchor": jp.zeros((), dtype=jp.float32),
                "student_anchor_action_rmse": jp.zeros((), dtype=jp.float32),
                "deadline_missed": jp.zeros((), dtype=jp.float32),
                "latency_steps": info["deploy_latency_steps"].astype(jp.float32),
                "movement_success": jp.zeros((), dtype=jp.float32),
            }
            return base_state.replace(
                obs={
                    "state": actor_history,
                    "privileged_state": base_state.obs,
                },
                metrics=metrics,
                info=info,
            )

        def step(self, state, action):
            deadline_key, observation_key, next_rng = jax.random.split(
                state.info["deploy_rng"], 3
            )
            raw_action = jp.nan_to_num(
                jp.clip(action, -1.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            action_queue = jp.concatenate(
                (raw_action[None, :], state.info["deploy_action_queue"][:-1]),
                axis=0,
            )
            delayed_action = action_queue[state.info["deploy_latency_steps"]]
            deadline_missed = (
                jax.random.uniform(deadline_key)
                < deploy_settings.control_deadline_miss_probability
            )
            applied_action = jp.where(
                deadline_missed,
                state.info["deploy_applied_action"],
                delayed_action,
            )
            anchor_action = student_anchor_policy(state.obs["state"])
            anchor_mse = jp.mean(jp.square(raw_action - anchor_action))
            anchor_reward = -student_anchor_weight * anchor_mse
            base_input = state.replace(
                info={
                    **state.info,
                    "deploy_rng": next_rng,
                    "deploy_action_queue": action_queue,
                    "deploy_applied_action": applied_action,
                }
            )
            base_state = self.base_env.step(base_input, applied_action)
            controller_action = effective_action_to_controller_action_3d(
                jp, raw_action
            )
            actor_history = self._actor_observation(
                base_state,
                state.info["actor_history"],
                controller_action,
                state.info["encoder_bias"],
                observation_key,
            )
            reward = base_state.reward + anchor_reward
            movement_success = (
                (base_state.done > 0.0)
                & (base_state.metrics["failed"] <= 0.0)
                & (
                    base_state.info["previous_roll_potential"]
                    >= minimum_success_turns * (2.0 * jp.pi)
                )
            )
            metrics = {
                **base_state.metrics,
                "reward": reward,
                "reward_total": base_state.metrics["reward_total"] + anchor_reward,
                "reward_student_anchor": anchor_reward,
                "student_anchor_action_rmse": jp.sqrt(anchor_mse),
                "deadline_missed": deadline_missed.astype(jp.float32),
                "latency_steps": state.info["deploy_latency_steps"].astype(
                    jp.float32
                ),
                "movement_success": movement_success.astype(jp.float32),
            }
            info = {
                **base_state.info,
                "actor_history": actor_history,
                "previous_controller_action": controller_action,
                "time_out": base_state.metrics["timeout"],
            }
            return base_state.replace(
                obs={
                    "state": actor_history,
                    "privileged_state": base_state.obs,
                },
                reward=reward,
                metrics=metrics,
                info=info,
            )

    return RollingStudentDREnv()
