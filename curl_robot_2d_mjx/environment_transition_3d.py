"""MJX environment for one learned BRAKE + DEPLOY + STABILIZE policy.

The module intentionally imports JAX/Brax only inside the environment factory,
so configuration, deployment supervision, CLI parsing, and CPU model contract
tests remain usable on machines without the GPU training stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from curl_robot_2d.model_3d import FOOT_SITE_NAMES_3D
from curl_robot_2d_mjx.config_transition_3d import (
    TRANSITION_ACTION_SIZE_3D,
    TRANSITION_ACTOR_OBSERVATION_SIZE_3D,
    TRANSITION_CRITIC_OBSERVATION_SIZE_3D,
    Transition3DConfig,
    TransitionMode3D,
    transition_curriculum_config_3d,
    validate_transition_config_3d,
    stabilize_failure_update_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d,
    geometry_parameters_3d,
    model_path_3d,
)
from curl_robot_2d_mjx.environment_walking_3d import (
    FOOT_GEOM_NAMES_3D,
    WALKING_JOINT_NAMES_3D,
    validate_walking_morphology_3d,
)
from curl_robot_2d_mjx.reward_transition_3d import (
    TRANSITION_REWARD_TERM_NAMES_3D,
    Transition3DRewardConfig,
    reward_terms_transition_3d,
)
from curl_robot_2d_mjx.transition_initialization_3d import (
    walking_start_state_3d,
    load_roll_snapshots_3d,
    transition_target_ctrl_3d,
)
from curl_robot_2d_mjx.deployment_transition_3d import (
    transition_controller_frame_3d,
    initial_transition_history_3d,
    push_transition_frame_3d,
)
from curl_robot_2d_mjx.failure_transition_3d import (
    TRANSITION_FAILURE_CAUSE_NAMES_3D,
    TRANSITION_FAILURE_MODE_NAMES_3D,
    TRANSITION_SOURCE_OUTCOME_NAMES_3D,
    transition_failure_causes_3d,
    transition_failure_mode_metrics_3d,
    transition_source_metrics_3d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSITION_MODEL_PATH_3D = model_path_3d("rollingquad_2")
TRANSITION_KEYFRAME_NAMES_3D = ("compact", "stand")
TRANSITION_PER_STEP_METRICS_3D = (
    "linear_speed_m_s", "angular_speed_rad_s", "upright_tilt_rad", "root_z_m",
    "stand_pose_error_rms_rad", "foot_contact_count", "nonfoot_contact_count", "action_rms",
)


def add_transition_per_step_metrics_3d(metrics):
    # Brax 0.14 Evaluator divides names ending in 'per_step' by each episode's
    # own length before averaging across episodes. Keep legacy sums too.
    return {**metrics, **{name + "_per_step": metrics[name]
                         for name in TRANSITION_PER_STEP_METRICS_3D}}


def transition_reference_ctrl_3d(
    xp,
    stand_ctrl,
):
    """Walking startup is the sole action center, independent of mode/time."""
    return xp.asarray(stand_ctrl)


def _load_transition_dependencies_3d():
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "3-D transition MJX dependencies are unavailable. Install "
            "requirements-mjx.txt on the Linux GPU instance."
        ) from exc
    return jax, jp, mujoco, mjx, Env, State


def _duration_steps(duration_s: float, dt: float) -> int:
    return max(1, int(round(duration_s / dt)))


def make_brax_transition_env_3d(
    config: Transition3DConfig | None = None,
    *,
    reward_config: Transition3DRewardConfig | None = None,
    seed: int = 0,
):
    """Create the 12-DoF transition task.

    Reverse curriculum starts at the exact Walking startup state, expands its
    neighborhood, then resets from complete frozen-ROLL snapshots. Braking is
    performed solely by this policy's actuator outputs, never by editing qvel.
    """

    task = config or transition_curriculum_config_3d("walking_start")
    validate_transition_config_3d(task)
    use_roll_snapshots = task.curriculum_stage.startswith("brake_")
    if use_roll_snapshots and not task.roll_snapshots_path:
        raise ValueError("BRAKE training requires --roll-snapshots from a frozen "
                         "rollingquad_2 ROLL policy (qpos, qvel and ctrl)")
    rewards = reward_config or Transition3DRewardConfig()
    jax, jp, mujoco, mjx, Env, State = _load_transition_dependencies_3d()

    class CurlRobot3DTransitionMJXEnv(Env):
        def __init__(self):
            self.config = task
            self.reward_config = rewards
            self.seed = seed
            self.model_path = model_path_3d(task.geometry)
            self.geometry_parameters = geometry_parameters_3d(task.geometry)
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
            # Preserve rollingquad_2's CAD mesh collisions in every mode.
            validate_walking_morphology_3d(
                self.mj_model, self.geometry_parameters, geometry_name=task.geometry
            )
            apply_physics_options_3d(self.mj_model, task)
            self.cpu_data = mujoco.MjData(self.mj_model)
            self.sys = mjx.put_model(self.mj_model)
            self.base_data = mjx.put_data(self.mj_model, self.cpu_data)

            def object_id(kind, name):
                value = mujoco.mj_name2id(self.mj_model, kind, name)
                if value < 0:
                    raise ValueError(f"missing MuJoCo object: {name}")
                return int(value)

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
                [self.mj_model.jnt_qposadr[index] for index in joint_ids],
                dtype=jp.int32,
            )
            self.joint_dof_indices = jp.asarray(
                [self.mj_model.jnt_dofadr[index] for index in joint_ids],
                dtype=jp.int32,
            )
            self.joint_low = jp.asarray(
                [self.mj_model.jnt_range[index, 0] for index in joint_ids]
            )
            self.joint_high = jp.asarray(
                [self.mj_model.jnt_range[index, 1] for index in joint_ids]
            )
            walking_start = walking_start_state_3d(self.mj_model, task)
            self.stand_qpos = jp.asarray(walking_start["qpos"])
            self.stand_ctrl = jp.asarray(walking_start["ctrl"])
            self.stand_root_height = (
                self.stand_qpos[2] - task.walking_start_height_offset_m
            )
            compact_id = object_id(mujoco.mjtObj.mjOBJ_KEY, "compact")
            self.compact_qpos = jp.asarray(self.mj_model.key_qpos[compact_id])
            self.compact_ctrl = jp.asarray(self.mj_model.key_ctrl[compact_id])
            self.roll_snapshots = None
            self.snapshot_selection_report = None
            self.source_phase_bins = 0
            self.source_cycles = ()
            if use_roll_snapshots:
                bank, self.snapshot_selection_report = load_roll_snapshots_3d(
                    task.roll_snapshots_path, self.mj_model, task, return_report=True)
                self.source_phase_bins = task.snapshot_phase_bins
                self.source_cycles = tuple(
                    int(value) for value in np.unique(bank["source_cycle"])
                )
                self.roll_snapshots = {
                    key: jp.asarray(value)
                    for key, value in bank.items()
                }

            self.deploy_gate_steps = _duration_steps(
                task.deploy_gate_hold_s, task.control_timestep
            )
            self.stabilize_min_steps = _duration_steps(
                task.stabilize_min_s, task.control_timestep
            )
            self.ready_hold_steps = _duration_steps(
                task.ready_hold_s, task.control_timestep
            )
            self.brake_timeout_steps = _duration_steps(
                task.brake_timeout_s, task.control_timestep
            )

        @property
        def observation_size(self):
            return {
                "state": TRANSITION_ACTOR_OBSERVATION_SIZE_3D,
                "privileged_state": TRANSITION_CRITIC_OBSERVATION_SIZE_3D,
            }

        @property
        def action_size(self):
            return TRANSITION_ACTION_SIZE_3D

        @property
        def backend(self):
            return "mjx"

        def _zero_metrics(self):
            zero = jp.zeros((), dtype=jp.float32)
            return add_transition_per_step_metrics_3d({
                "reward": zero,
                "reward_total": zero,
                **{
                    f"reward_{name}": zero
                    for name in TRANSITION_REWARD_TERM_NAMES_3D
                },
                "mode": zero,
                "mode_brake": zero,
                "mode_deploy": zero,
                "mode_stabilize": zero,
                "transition_success": zero,
                "ready_gate": zero,
                "ready_hold_fraction": zero,
                "deploy_gate": zero,
                "deploy_progress": zero,
                "linear_speed_m_s": zero,
                "angular_speed_rad_s": zero,
                "combined_speed": zero,
                "upright_tilt_rad": zero,
                "root_z_m": zero,
                "reference_pose_error_rms_rad": zero,
                "stand_pose_error_rms_rad": zero,
                "foot_contact_count": zero,
                "nonfoot_contact_count": zero,
                "foot_slip_rms_m_s": zero,
                "action_rms": zero,
                "action_rate_rms": zero,
                "failed": zero,
                "failed_stabilize": zero,
                **{f"failure_{name}": zero
                   for name in TRANSITION_FAILURE_CAUSE_NAMES_3D},
                **{
                    f"failure_{cause}_mode_{mode}": zero
                    for cause in TRANSITION_FAILURE_CAUSE_NAMES_3D
                    for mode in TRANSITION_FAILURE_MODE_NAMES_3D
                },
                **{
                    f"source_phase_bin_{phase_bin}_{outcome}": zero
                    for phase_bin in range(self.source_phase_bins)
                    for outcome in TRANSITION_SOURCE_OUTCOME_NAMES_3D
                },
                **{
                    f"source_cycle_{cycle}_{outcome}": zero
                    for cycle in self.source_cycles
                    for outcome in TRANSITION_SOURCE_OUTCOME_NAMES_3D
                },
                "timeout": zero,
            })

        def _contact_arrays(self, data):
            contact = data.contact
            if hasattr(contact, "geom1"):
                return contact.geom1, contact.geom2, contact.dist
            return contact.geom[:, 0], contact.geom[:, 1], contact.dist

        def _geom_in_ids(self, geom, ids):
            return jp.any(geom[:, None] == ids[None, :], axis=1)

        def _contacts(self, data):
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
            nonfoot = ground & (~geom1_foot) & (~geom2_foot)
            return {
                "foot_ground": foot_ground,
                "foot_count": jp.sum(foot_ground),
                "nonfoot_count": jp.sum(nonfoot).astype(jp.float32),
            }

        def _kinematics(self, data):
            rotation = jp.reshape(data.xmat[self.torso_body_id], (3, 3))
            linear_velocity = rotation.T @ data.qvel[:3]
            angular_velocity = data.qvel[3:6]
            gravity = rotation.T @ jp.asarray((0.0, 0.0, -1.0))
            upright_tilt = jp.arccos(jp.clip(rotation[2, 2], -1.0, 1.0))
            return {
                "rotation": rotation,
                "linear_velocity": linear_velocity,
                "angular_velocity": angular_velocity,
                "gravity": gravity,
                "linear_speed": jp.linalg.norm(data.qvel[:3]),
                "angular_speed": jp.linalg.norm(data.qvel[3:6]),
                "upright_tilt": upright_tilt,
            }

        def _mode_progress(self, mode, mode_steps):
            deploy_progress = jp.clip(
                mode_steps * task.control_timestep / task.deploy_timeout_s,
                0.0,
                1.0,
            )
            brake_progress = jp.clip(
                mode_steps / self.brake_timeout_steps, 0.0, 1.0
            )
            stabilize_progress = jp.clip(
                mode_steps / self.ready_hold_steps, 0.0, 1.0
            )
            return jp.where(
                mode == int(TransitionMode3D.BRAKE),
                brake_progress,
                jp.where(
                    mode == int(TransitionMode3D.DEPLOY),
                    deploy_progress,
                    stabilize_progress,
                ),
            )

        def _reference(self, mode, mode_steps):
            del mode, mode_steps
            return transition_reference_ctrl_3d(jp, self.stand_ctrl)

        def _gates(self, data, contacts, kinematics, reference):
            joint_position = data.qpos[self.joint_qpos_indices]
            stand_error = jp.sqrt(
                jp.mean(jp.square(joint_position - self.stand_ctrl))
            )
            reference_error = jp.sqrt(
                jp.mean(jp.square(joint_position - reference))
            )
            deploy_gate = (
                (kinematics["linear_speed"] <= task.deploy_gate_linear_speed_m_s)
                & (kinematics["angular_speed"] <= task.deploy_gate_angular_speed_rad_s)
                & (kinematics["upright_tilt"] <= task.deploy_gate_tilt_rad)
            )
            ready_gate = (
                (kinematics["linear_speed"] <= task.ready_linear_speed_m_s)
                & (kinematics["angular_speed"] <= task.ready_angular_speed_rad_s)
                & (kinematics["upright_tilt"] <= task.ready_upright_tilt_rad)
                & (stand_error <= task.ready_joint_error_rad)
                & (data.qpos[2] >= task.ready_root_height_min_m)
                & (data.qpos[2] <= task.ready_root_height_max_m)
                & (contacts["foot_count"] >= task.ready_min_foot_contacts)
                & jp.all(jp.isfinite(data.qpos))
                & jp.all(jp.isfinite(data.qvel))
            )
            deploy_fraction = jp.minimum(
                task.deploy_gate_linear_speed_m_s
                / jp.maximum(kinematics["linear_speed"], 1.0e-6),
                task.deploy_gate_angular_speed_rad_s
                / jp.maximum(kinematics["angular_speed"], 1.0e-6),
            )
            return {
                "deploy_gate": deploy_gate,
                "ready_gate": ready_gate,
                "deploy_gate_fraction": jp.clip(deploy_fraction, 0.0, 1.0),
                "stand_error": stand_error,
                "reference_error": reference_error,
            }

        def _observation(
            self,
            data,
            contacts,
            kinematics,
            gates,
            *,
            mode,
            mode_steps,
            ready_steps,
            policy_action,
            reference,
            noise_key,
            actor_history,
        ):
            joint_position = data.qpos[self.joint_qpos_indices]
            joint_velocity = data.qvel[self.joint_dof_indices]
            foot_position = data.site_xpos[self.foot_site_ids]
            foot_height = jp.maximum(foot_position[:, 2], 0.0)
            roll_phase = 2.0 * jp.arctan2(data.qpos[5], data.qpos[3])
            mode_one_hot = jp.eye(4, dtype=jp.float32)[mode]
            progress = self._mode_progress(mode, mode_steps)
            ready_fraction = jp.clip(
                ready_steps / self.ready_hold_steps, 0.0, 1.0
            )
            combined_speed = jp.sqrt(
                jp.square(kinematics["linear_speed"])
                + 0.04 * jp.square(kinematics["angular_speed"])
            )
            critic_core = jp.concatenate(
                (
                    kinematics["linear_velocity"],
                    kinematics["angular_velocity"],
                    kinematics["gravity"],
                    joint_position - reference,
                    joint_velocity,
                    policy_action,
                    contacts["foot_ground"],
                    foot_height,
                    mode_one_hot,
                    jp.stack((jp.sin(roll_phase), jp.cos(roll_phase))),
                    jp.stack(
                        (
                            progress,
                            ready_fraction,
                            data.qpos[2],
                            gates["reference_error"],
                            combined_speed,
                            1.0,
                            gates["deploy_gate_fraction"],
                        )
                    ),
                )
            )
            privileged_extra = jp.concatenate(
                (
                    data.qpos[:3],
                    data.qpos[3:7],
                    foot_position.reshape((-1,)),
                    jp.asarray((contacts["nonfoot_count"],)),
                )
            )
            critic = jp.nan_to_num(jp.concatenate((critic_core, privileged_extra)))
            # Actor sees ONLY the real-controller sensor/command ABI. The
            # gyro is body angular velocity, not measured joint qvel.
            frame = transition_controller_frame_3d(
                jp, angular_velocity_body=kinematics["angular_velocity"],
                projected_gravity=kinematics["gravity"],
                joint_position_offset=joint_position - self.stand_ctrl,
                last_action=policy_action,
            )
            if task.observation_noise_enabled:
                scales = task.observation_noise_level * jp.concatenate((
                    jp.full((3,), task.observation_noise_velocity),
                    jp.full((3,), task.observation_noise_gravity),
                    jp.zeros((6,)),
                    jp.full((12,), task.observation_noise_joint_position),
                    jp.zeros((12,)),
                ))
                frame = frame + scales * jax.random.normal(noise_key, frame.shape)
            # Noise is sampled once per acquired frame; stored history is not
            # re-noised on every inference. Last action remains raw output.
            actor = push_transition_frame_3d(
                jp, actor_history, jp.nan_to_num(frame), task.observation_limit)
            return {"state": actor, "privileged_state": critic}

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            fraction_key, joint_key, velocity_key, phase_key, tilt_key = (
                jax.random.split(rng, 5)
            )
            low, high = task.reset_compact_fraction_range
            compact_fraction = jax.random.uniform(
                fraction_key, shape=(), minval=low, maxval=high
            )
            joint_noise = jax.random.uniform(
                joint_key,
                shape=(TRANSITION_ACTION_SIZE_3D,),
                minval=-task.reset_joint_noise_rad,
                maxval=task.reset_joint_noise_rad,
            )
            qpos = self.stand_qpos + compact_fraction * (
                self.compact_qpos - self.stand_qpos
            )
            joints = jp.clip(
                qpos[self.joint_qpos_indices] + joint_noise,
                self.joint_low,
                self.joint_high,
            )
            qpos = qpos.at[self.joint_qpos_indices].set(joints)
            phase_low, phase_high = task.reset_roll_phase_range_rad
            phase = jax.random.uniform(
                phase_key, shape=(), minval=phase_low, maxval=phase_high
            )
            tilt = jax.random.uniform(
                tilt_key,
                shape=(),
                minval=-task.reset_tilt_rad,
                maxval=task.reset_tilt_rad,
            )
            half_phase = 0.5 * phase
            half_tilt = 0.5 * tilt
            # qx(tilt) * qy(phase), MuJoCo quaternion order w,x,y,z.
            quaternion = jp.stack(
                (
                    jp.cos(half_tilt) * jp.cos(half_phase),
                    jp.sin(half_tilt) * jp.cos(half_phase),
                    jp.cos(half_tilt) * jp.sin(half_phase),
                    jp.sin(half_tilt) * jp.sin(half_phase),
                )
            )
            qpos = qpos.at[3:7].set(quaternion)
            velocity = jax.random.uniform(
                velocity_key,
                shape=(self.mj_model.nv,),
                minval=-1.0,
                maxval=1.0,
            )
            velocity = velocity.at[:3].multiply(task.reset_linear_speed_m_s)
            velocity = velocity.at[3:6].multiply(
                task.reset_angular_speed_rad_s
            )
            velocity = velocity.at[6:].multiply(task.reset_joint_velocity_rad_s)
            mode = jp.asarray(task.reset_start_mode, dtype=jp.int32)
            ctrl = joints
            source_phase_bin = jp.asarray(-1, dtype=jp.int32)
            source_cycle = jp.asarray(-1, dtype=jp.int32)
            if self.roll_snapshots is not None:
                index = jp.minimum(jp.searchsorted(
                    self.roll_snapshots["sampling_cdf"], jax.random.uniform(fraction_key),
                    side="right"), self.roll_snapshots["qpos"].shape[0] - 1)
                qpos = self.roll_snapshots["qpos"][index]
                velocity = self.roll_snapshots["qvel"][index]
                ctrl = self.roll_snapshots["ctrl"][index]
                source_phase_bin = self.roll_snapshots["source_phase_bin"][index]
                source_cycle = self.roll_snapshots["source_cycle"][index]
            data = self.base_data.replace(qpos=qpos, qvel=velocity, ctrl=ctrl)
            data = mjx.forward(self.sys, data)
            # Neighborhood perturbations may rotate a CAD mesh into the floor.
            # Only synthetic states are lifted. Never alter real takeover data
            # or the exact Walking-start anchor.
            if not use_roll_snapshots and task.curriculum_stage != "walking_start":
                g1, g2, dist = self._contact_arrays(data)
                ground = (g1 == self.floor_geom_id) | (g2 == self.floor_geom_id)
                lift = jp.max(jp.where(ground, jp.maximum(-dist, 0.0), 0.0),
                              initial=0.0)
                data = mjx.forward(self.sys, data.replace(
                    qpos=data.qpos.at[2].add(lift)))
            return self._initial_state(
                data, rng, mode, source_phase_bin=source_phase_bin,
                source_cycle=source_cycle,
            )

        def reset_from_roll_state(self, data, rng, actor_history=None, last_action=None):
            """Live ROLL -> Transition handoff on the SAME MJX model.

            Preserve full simulator state and last servo command; no velocity
            reset, warm-up, hidden rollout or external braking controller.
            actor_history is the PREVIOUS inference input (newest first), not
            the C++ post-inference rotated scratch buffer. last_action is the
            previous raw output in this policy's action convention. Omission
            intentionally cold-initializes only the observation buffer, never
            the simulator or motors; training snapshot resets use this path.
            """
            return self._initial_state(
                data, rng, jp.asarray(int(TransitionMode3D.BRAKE), dtype=jp.int32),
                actor_history=actor_history, last_action=last_action,
            )

        def _initial_state(
            self, data, rng, mode, actor_history=None, last_action=None,
            source_phase_bin=None, source_cycle=None,
        ):
            mode_steps = jp.asarray(0, dtype=jp.int32)
            reference = self._reference(mode, mode_steps)
            contacts = self._contacts(data)
            kinematics = self._kinematics(data)
            gates = self._gates(data, contacts, kinematics, reference)
            # Default cold history matches on_activate without moving motors.
            # Hot carry-over is explicit and requires identical obs/action ABI.
            previous_action = (jp.zeros((12,), dtype=data.qpos.dtype)
                               if last_action is None else last_action)
            actor_history = (initial_transition_history_3d(jp, dtype=data.qpos.dtype)
                             if actor_history is None else actor_history)
            if actor_history.shape != (TRANSITION_ACTOR_OBSERVATION_SIZE_3D,):
                raise ValueError("Transition actor_history must contain 720 values")
            if previous_action.shape != (12,):
                raise ValueError("Transition last_action must contain 12 values")
            info = {
                "rng": rng,
                "step_count": jp.asarray(0, dtype=jp.int32),
                "mode": mode,
                "mode_steps": mode_steps,
                "deploy_gate_steps": jp.asarray(0, dtype=jp.int32),
                "ready_steps": jp.asarray(0, dtype=jp.int32),
                "stabilize_bad_steps": jp.asarray(0, dtype=jp.int32),
                "last_action": previous_action,
                "last_foot_position": data.site_xpos[self.foot_site_ids],
                "previous_combined_speed": jp.sqrt(
                    jp.square(kinematics["linear_speed"])
                    + 0.04 * jp.square(kinematics["angular_speed"])
                ),
                "previous_reference_error": gates["reference_error"],
                "time_out": jp.asarray(0.0),
                "actor_history": actor_history,
                "source_phase_bin": (
                    jp.asarray(-1, dtype=jp.int32)
                    if source_phase_bin is None else jp.asarray(source_phase_bin, dtype=jp.int32)
                ),
                "source_cycle": (
                    jp.asarray(-1, dtype=jp.int32)
                    if source_cycle is None else jp.asarray(source_cycle, dtype=jp.int32)
                ),
            }
            obs = self._observation(
                data,
                contacts,
                kinematics,
                gates,
                mode=mode,
                mode_steps=mode_steps,
                ready_steps=info["ready_steps"],
                policy_action=previous_action,
                reference=reference,
                noise_key=jax.random.fold_in(rng, 99),
                actor_history=actor_history,
            )
            info["actor_history"] = obs["state"]
            return State(
                data,
                obs,
                jp.zeros((), dtype=jp.float32),
                jp.zeros((), dtype=jp.float32),
                metrics=self._zero_metrics(),
                info=info,
            )

        def step(self, state, action):
            mode = state.info["mode"]
            mode_steps = state.info["mode_steps"]
            reference = self._reference(mode, mode_steps)
            action_finite = jp.all(jp.isfinite(action))
            policy_action = jp.nan_to_num(
                action, nan=0.0, posinf=1.0, neginf=-1.0
            )
            target = transition_target_ctrl_3d(
                jp, policy_action, reference, self.joint_low, self.joint_high,
                task.action_range_fraction,
            )
            data = state.pipeline_state.replace(ctrl=target)

            def physics_step(carry, unused):
                del unused
                return mjx.step(self.sys, carry), None

            data, _ = jax.lax.scan(
                physics_step, data, None, length=task.action_repeat
            )
            contacts = self._contacts(data)
            kinematics = self._kinematics(data)
            gates = self._gates(data, contacts, kinematics, reference)

            next_deploy_gate_steps = jp.where(
                (mode == int(TransitionMode3D.BRAKE)) & gates["deploy_gate"],
                state.info["deploy_gate_steps"] + 1,
                0,
            )
            enter_deploy = (
                (mode == int(TransitionMode3D.BRAKE))
                & (next_deploy_gate_steps >= self.deploy_gate_steps)
            )
            deploy_complete = (
                (mode == int(TransitionMode3D.DEPLOY))
                & (gates["stand_error"] <= 0.45)
                & (contacts["foot_count"] >= 2)
                & (kinematics["upright_tilt"] <= task.deploy_gate_tilt_rad)
            )
            next_mode = jp.where(
                enter_deploy,
                int(TransitionMode3D.DEPLOY),
                jp.where(
                    deploy_complete,
                    int(TransitionMode3D.STABILIZE),
                    mode,
                ),
            ).astype(jp.int32)
            changed_mode = next_mode != mode
            next_mode_steps = jp.where(changed_mode, 0, mode_steps + 1)
            next_bad_steps, failed_stabilize = stabilize_failure_update_3d(
                jp, task, mode=next_mode, mode_steps=next_mode_steps,
                previous_bad_steps=state.info["stabilize_bad_steps"],
                root_height=data.qpos[2], joint_error=gates["stand_error"],
                tilt=kinematics["upright_tilt"], foot_contacts=contacts["foot_count"],
                nonfoot_contacts=contacts["nonfoot_count"],
            )

            ready_gate_active = (
                (next_mode == int(TransitionMode3D.STABILIZE))
                & (next_mode_steps >= self.stabilize_min_steps)
                & gates["ready_gate"]
            )
            next_ready_steps = jp.where(
                ready_gate_active, state.info["ready_steps"] + 1, 0
            )
            newly_ready = (next_ready_steps >= self.ready_hold_steps) & action_finite

            physics_finite = (
                jp.all(jp.isfinite(data.qpos))
                & jp.all(jp.isfinite(data.qvel))
            )
            failed_root_height_low = data.qpos[2] < task.failure_root_height_min_m
            failed_root_height_high = data.qpos[2] > task.failure_root_height_max_m
            failed_brake_timeout = (
                (mode == int(TransitionMode3D.BRAKE))
                & (mode_steps >= self.brake_timeout_steps)
                & (~enter_deploy)
            )
            failed_deploy_timeout = (
                (mode == int(TransitionMode3D.DEPLOY))
                & (mode_steps * task.control_timestep >= task.deploy_timeout_s)
                & (~deploy_complete)
            )
            failed = (
                (~action_finite)
                | (~physics_finite)
                | failed_stabilize
                | failed_root_height_low
                | failed_root_height_high
                | failed_brake_timeout
                | failed_deploy_timeout
            )
            failure_causes = transition_failure_causes_3d(
                jp, failed=failed, action_finite=action_finite,
                physics_finite=physics_finite,
                root_height_low=failed_root_height_low,
                root_height_high=failed_root_height_high,
                brake_timeout=failed_brake_timeout,
                deploy_timeout=failed_deploy_timeout,
                stabilize_guard=failed_stabilize,
            )
            failure_modes = transition_failure_mode_metrics_3d(
                jp, failure_causes, mode
            )
            # Terminal outcomes are mutually exclusive: invalid physics or
            # lost support cannot be counted as a simultaneous READY success.
            newly_ready = newly_ready & (~failed)
            next_mode = jp.where(
                newly_ready, int(TransitionMode3D.READY), next_mode
            ).astype(jp.int32)
            next_step_count = state.info["step_count"] + 1
            timeout = (next_step_count >= task.episode_length) & (~failed) & (~newly_ready)
            done = failed | newly_ready | timeout
            source_metrics = transition_source_metrics_3d(
                jp, done=done, success=newly_ready, failed=failed,
                timeout=timeout,
                root_height_low=failure_causes["root_height_low"],
                source_phase_bin=state.info["source_phase_bin"],
                source_cycle=state.info["source_cycle"],
                phase_bins=self.source_phase_bins, cycles=self.source_cycles,
            )

            foot_position = data.site_xpos[self.foot_site_ids]
            foot_velocity = (
                foot_position - state.info["last_foot_position"]
            ) / task.control_timestep
            foot_slip_squared = jp.mean(
                contacts["foot_ground"]
                * jp.sum(jp.square(foot_velocity[:, :2]), axis=1)
            )
            combined_speed = jp.sqrt(
                jp.square(kinematics["linear_speed"])
                + 0.04 * jp.square(kinematics["angular_speed"])
            )
            next_reference = self._reference(next_mode, next_mode_steps)
            next_reference_error = jp.sqrt(
                jp.mean(
                    jp.square(
                        data.qpos[self.joint_qpos_indices] - next_reference
                    )
                )
            )
            mode_brake = (mode == int(TransitionMode3D.BRAKE)).astype(jp.float32)
            mode_deploy = (mode == int(TransitionMode3D.DEPLOY)).astype(jp.float32)
            mode_stabilize = (
                mode == int(TransitionMode3D.STABILIZE)
            ).astype(jp.float32)
            reward_inputs = {
                "mode_brake": mode_brake,
                "mode_deploy": mode_deploy,
                "mode_stabilize": mode_stabilize,
                "combined_speed": combined_speed,
                "previous_combined_speed": state.info["previous_combined_speed"],
                "reference_pose_error_rms": next_reference_error,
                "previous_reference_pose_error_rms": state.info[
                    "previous_reference_error"
                ],
                "upright_tilt": kinematics["upright_tilt"],
                "root_height_error": data.qpos[2] - self.stand_root_height,
                "support_fraction": contacts["foot_count"] / 4.0,
                "newly_ready": newly_ready.astype(jp.float32),
                "action_rate_squared": jp.mean(
                    jp.square(policy_action - state.info["last_action"])
                ),
                "action_squared": jp.mean(jp.square(policy_action)),
                "joint_velocity_squared": jp.mean(
                    jp.square(data.qvel[self.joint_dof_indices])
                ),
                "foot_slip_velocity_squared": foot_slip_squared,
                # Precise contact force is intentionally deferred to the cloud
                # MJX validation pass because its storage differs by MuJoCo API.
                "contact_force_peak_n": jp.asarray(0.0),
                "nonfoot_contact_count": (
                    mode_stabilize * contacts["nonfoot_count"]
                ),
                "failed": failed.astype(jp.float32),
            }
            terms = reward_terms_transition_3d(jp, rewards, reward_inputs)
            reward = jp.sum(jp.stack(tuple(terms.values())))
            reward = jp.nan_to_num(reward, nan=-rewards.termination)

            next_info = {
                **state.info,
                "step_count": next_step_count,
                "mode": next_mode,
                "mode_steps": next_mode_steps,
                "deploy_gate_steps": jp.where(
                    changed_mode, 0, next_deploy_gate_steps
                ),
                "ready_steps": next_ready_steps,
                "stabilize_bad_steps": next_bad_steps,
                "last_action": policy_action,
                "last_foot_position": foot_position,
                "previous_combined_speed": combined_speed,
                "previous_reference_error": next_reference_error,
                "time_out": timeout.astype(jp.float32),
            }
            next_gates = self._gates(
                data, contacts, kinematics, next_reference
            )
            obs = self._observation(
                data,
                contacts,
                kinematics,
                next_gates,
                mode=next_mode,
                mode_steps=next_mode_steps,
                ready_steps=next_ready_steps,
                policy_action=policy_action,
                reference=next_reference,
                noise_key=jax.random.fold_in(
                    state.info["rng"], next_step_count + 99
                ),
                actor_history=state.info["actor_history"],
            )
            next_info["actor_history"] = obs["state"]
            metrics = {
                "reward": reward,
                "reward_total": reward,
                **{f"reward_{name}": value for name, value in terms.items()},
                "mode": next_mode.astype(jp.float32),
                "mode_brake": mode_brake,
                "mode_deploy": mode_deploy,
                "mode_stabilize": mode_stabilize,
                "transition_success": newly_ready.astype(jp.float32),
                "ready_gate": next_gates["ready_gate"].astype(jp.float32),
                "ready_hold_fraction": jp.clip(
                    next_ready_steps / self.ready_hold_steps, 0.0, 1.0
                ),
                "deploy_gate": next_gates["deploy_gate"].astype(jp.float32),
                "deploy_progress": jp.where(
                    next_mode == int(TransitionMode3D.DEPLOY),
                    self._mode_progress(next_mode, next_mode_steps),
                    0.0,
                ),
                "linear_speed_m_s": kinematics["linear_speed"],
                "angular_speed_rad_s": kinematics["angular_speed"],
                "combined_speed": combined_speed,
                "upright_tilt_rad": kinematics["upright_tilt"],
                "root_z_m": data.qpos[2],
                "reference_pose_error_rms_rad": next_reference_error,
                "stand_pose_error_rms_rad": next_gates["stand_error"],
                "foot_contact_count": contacts["foot_count"],
                "nonfoot_contact_count": contacts["nonfoot_count"],
                "foot_slip_rms_m_s": jp.sqrt(foot_slip_squared),
                "action_rms": jp.sqrt(jp.mean(jp.square(policy_action))),
                "action_rate_rms": jp.sqrt(
                    reward_inputs["action_rate_squared"]
                ),
                "failed": failed.astype(jp.float32),
                "failed_stabilize": failed_stabilize.astype(jp.float32),
                **{f"failure_{name}": value.astype(jp.float32)
                   for name, value in failure_causes.items()},
                **{name: value.astype(jp.float32)
                   for name, value in failure_modes.items()},
                **{name: value.astype(jp.float32)
                   for name, value in source_metrics.items()},
                "timeout": timeout.astype(jp.float32),
            }
            return State(
                data,
                obs,
                reward,
                done.astype(jp.float32),
                metrics=add_transition_per_step_metrics_3d(metrics),
                info=next_info,
            )

    return CurlRobot3DTransitionMJXEnv()
