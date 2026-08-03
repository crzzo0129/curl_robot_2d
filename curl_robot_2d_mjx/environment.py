"""Brax-compatible MJX environment for nominal-COM planar rolling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceConfig,
    advance_oscillator,
    effective_residual_action,
    reference_action,
)
from curl_robot_2d_mjx.config import (
    NominalRLConfig,
    smoothstep_ramp,
    validate_nominal_rl_config,
)
from curl_robot_2d_mjx.reward import (
    REWARD_TERM_NAMES,
    conservative_rolling_potential,
    reward_terms,
    stuck_termination_state,
)
from curl_robot_2d_mjx.reward_config import RollingRewardConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
JOINT_NAMES = (
    "front_hip",
    "front_knee",
    "rear_hip",
    "rear_knee",
)


def resolve_model_path(task: NominalRLConfig) -> Path:
    """Resolve the MuJoCo XML used by this task."""

    if task.model_xml is None:
        return MODEL_PATH
    path = Path(task.model_xml).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def apply_physics_options(model, task: NominalRLConfig) -> None:
    """Apply an MJX/CPU-comparable runtime profile to one MuJoCo model."""

    import mujoco

    solver_values = {
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "pgs": mujoco.mjtSolver.mjSOL_PGS,
    }
    integrator_values = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    cone_values = {
        "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
        "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    }
    jacobian_values = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
        "auto": mujoco.mjtJacobian.mjJAC_AUTO,
    }
    model.opt.solver = solver_values[task.solver_name]
    model.opt.integrator = integrator_values[task.integrator_name]
    model.opt.cone = cone_values[task.cone_name]
    model.opt.jacobian = jacobian_values[task.jacobian_name]
    model.opt.timestep = task.physics_timestep
    model.opt.iterations = task.solver_iterations
    model.opt.ls_iterations = task.solver_ls_iterations
    if task.disable_root_damping:
        for joint_name in ("root_x", "root_z", "root_pitch"):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"missing MuJoCo joint: {joint_name}")
            dof_id = int(model.jnt_dofadr[joint_id])
            model.dof_damping[dof_id] = 0.0


def _load_dependencies():
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "MJX training dependencies are unavailable. Install "
            "requirements-mjx.txt on the Linux GPU instance."
        ) from exc
    return jax, jp, mujoco, mjx, Env, State


def make_brax_env(
    config: NominalRLConfig | None = None,
    *,
    reward_config: RollingRewardConfig | None = None,
    cem_reference: CEMReferenceConfig | None = None,
    seed: int = 0,
):
    """Create the fixed-nominal-COM MJX environment.

    The environment loads the same generated XML used by the CPU CEM baseline.
    Runtime physics options can use a measured profile; no mass, COM, inertia,
    friction, gain or torque parameter is randomized.
    """

    task = config or NominalRLConfig()
    validate_nominal_rl_config(task)
    disturbance_enabled = (
        task.disturbance_probability > 0.0
        and (
            task.disturbance_root_x_velocity_m_s > 0.0
            or task.disturbance_root_pitch_velocity_rad_s > 0.0
        )
    )
    jax, jp, mujoco, mjx, Env, State = _load_dependencies()
    reward_settings = reward_config or RollingRewardConfig()
    reference_settings = cem_reference

    class CurlRobot2DMJXEnv(Env):
        def __init__(self):
            self.config = task
            self.reward_config = reward_settings
            self.cem_reference = reference_settings
            self.seed = seed
            self.model_path = resolve_model_path(task)
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
            apply_physics_options(self.mj_model, task)
            self.cpu_data = mujoco.MjData(self.mj_model)
            self.mjx_model = mjx.put_model(self.mj_model)
            self.base_data = mjx.put_data(self.mj_model, self.cpu_data)

            def object_id(object_type, name):
                value = mujoco.mj_name2id(
                    self.mj_model, object_type, name
                )
                if value < 0:
                    raise ValueError(f"missing MuJoCo object: {name}")
                return int(value)

            compact_key_id = object_id(
                mujoco.mjtObj.mjOBJ_KEY, "compact"
            )
            self.compact_qpos = jp.asarray(
                self.mj_model.key_qpos[compact_key_id]
            )
            self.compact_ctrl = jp.asarray(
                self.mj_model.key_ctrl[compact_key_id]
            )

            self.root_x_qpos = int(
                self.mj_model.jnt_qposadr[
                    object_id(mujoco.mjtObj.mjOBJ_JOINT, "root_x")
                ]
            )
            self.root_z_qpos = int(
                self.mj_model.jnt_qposadr[
                    object_id(mujoco.mjtObj.mjOBJ_JOINT, "root_z")
                ]
            )
            self.root_pitch_qpos = int(
                self.mj_model.jnt_qposadr[
                    object_id(mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")
                ]
            )
            self.root_x_dof = int(
                self.mj_model.jnt_dofadr[
                    object_id(mujoco.mjtObj.mjOBJ_JOINT, "root_x")
                ]
            )
            self.root_pitch_dof = int(
                self.mj_model.jnt_dofadr[
                    object_id(mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")
                ]
            )
            self.joint_qpos_indices = jp.asarray(
                [
                    int(
                        self.mj_model.jnt_qposadr[
                            object_id(
                                mujoco.mjtObj.mjOBJ_JOINT, joint_name
                            )
                        ]
                    )
                    for joint_name in JOINT_NAMES
                ]
            )
            self.joint_dof_indices = jp.asarray(
                [
                    int(
                        self.mj_model.jnt_dofadr[
                            object_id(
                                mujoco.mjtObj.mjOBJ_JOINT, joint_name
                            )
                        ]
                    )
                    for joint_name in JOINT_NAMES
                ]
            )
            self.joint_low = jp.asarray(
                [
                    self.mj_model.jnt_range[
                        object_id(mujoco.mjtObj.mjOBJ_JOINT, joint_name),
                        0,
                    ]
                    for joint_name in JOINT_NAMES
                ]
            )
            self.joint_high = jp.asarray(
                [
                    self.mj_model.jnt_range[
                        object_id(mujoco.mjtObj.mjOBJ_JOINT, joint_name),
                        1,
                    ]
                    for joint_name in JOINT_NAMES
                ]
            )
            self.action_scales = jp.asarray(task.action_scales)
            self.planar_indices = jp.asarray([0, 2])
            self.force_limits = jp.asarray(
                np.abs(self.mj_model.actuator_forcerange[:, 1])
            )
            if reference_settings is not None:
                bias_reference = replace(
                    reference_settings, coefficients=(0.0,) * 8
                )
                bias_action = reference_action(
                    np,
                    0.0,
                    bias_reference,
                    compact_ctrl=np.asarray(self.mj_model.key_ctrl[compact_key_id]),
                    action_scales=np.asarray(task.action_scales),
                    joint_low=np.asarray(self.joint_low),
                    joint_high=np.asarray(self.joint_high),
                )
                self.reference_bias_action = jp.asarray(bias_action)
                weighted_bias = reference_settings.reference_weight * bias_action
                start_ctrl = (
                    np.asarray(self.mj_model.key_ctrl[compact_key_id])
                    + weighted_bias * np.asarray(task.action_scales)
                )
                self.reference_start_ctrl = jp.asarray(start_ctrl)
                self.reference_start_root_z = float(
                    FIXED_PARAMETERS.leg_extension_height(
                        float(start_ctrl[0]), float(start_ctrl[1])
                    )
                    + FIXED_PARAMETERS.foot_radius
                )
            else:
                self.reference_bias_action = jp.zeros(4, dtype=jp.float32)
                self.reference_start_ctrl = self.compact_ctrl
                self.reference_start_root_z = float(
                    self.mj_model.key_qpos[compact_key_id, self.root_z_qpos]
                )

            self.floor_geom_id = object_id(
                mujoco.mjtObj.mjOBJ_GEOM, "floor"
            )
            self.front_foot_geom_id = object_id(
                mujoco.mjtObj.mjOBJ_GEOM, "front_foot_proxy"
            )
            self.rear_foot_geom_id = object_id(
                mujoco.mjtObj.mjOBJ_GEOM, "rear_foot_proxy"
            )
            self.front_foot_site_id = object_id(
                mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
            )
            self.rear_foot_site_id = object_id(
                mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
            )
            self.body_ids = {
                name: object_id(mujoco.mjtObj.mjOBJ_BODY, name)
                for name in (
                    "front_thigh",
                    "front_shank",
                    "rear_thigh",
                    "rear_shank",
                )
            }
            self.rolling_radius = FIXED_PARAMETERS.shell_contact_radius
            if task.terminate_root_z_min is None:
                self.root_low_termination_steps = 1
            else:
                self.root_low_termination_steps = max(
                    1,
                    int(
                        np.ceil(
                            task.terminate_root_z_low_duration_s
                            / task.control_timestep
                        )
                    ),
                )
            self.stuck_progress_window_steps = max(
                1,
                int(
                    np.ceil(
                        task.terminate_stuck_progress_window_s
                        / task.control_timestep
                    )
                ),
            )
            self.stuck_termination_steps = max(
                1,
                int(
                    np.ceil(
                        task.terminate_stuck_duration_s
                        / task.control_timestep
                    )
                ),
            )
            self.stuck_grace_steps = max(
                self.stuck_progress_window_steps,
                int(
                    np.ceil(
                        task.terminate_stuck_grace_s
                        / task.control_timestep
                    )
                ),
            )
            self.tail_progress_window_steps = max(
                1,
                min(
                    task.episode_length,
                    int(
                        np.ceil(
                            task.tail_progress_window_s
                            / task.control_timestep
                        )
                    ),
                ),
            )

        @property
        def observation_size(self):
            return 30 if reference_settings is not None else 23

        @property
        def action_size(self):
            return 4

        @property
        def backend(self):
            return "mjx"

        def _zero_metrics(self):
            # Keep reset and step PyTree signatures identical.  A scalar made
            # with jp.asarray(0.0) is weakly typed, while computed step metrics
            # are strong float32; that difference makes jax.jit compile step
            # again on its second invocation.
            zero = jp.zeros((), dtype=jp.float32)
            return {
                "reward": zero,
                "reward_total": zero,
                **{
                    f"reward_{name}": zero
                    for name in REWARD_TERM_NAMES
                },
                "roll_progress_rad": zero,
                "phase_progress_rad": zero,
                "translation_progress_rad": zero,
                "tail_roll_progress_rad": zero,
                "forbidden_contact_count": zero,
                "forbidden_penetration_m": zero,
                "allowed_foot_penetration_m": zero,
                "foot_contact_active": zero,
                "foot_contact_start": zero,
                "ground_contact_count": zero,
                "leg_crossing": zero,
                "root_height_m": zero,
                "root_low_active": zero,
                "root_low_step_count": zero,
                "stuck_active": zero,
                "stuck_deficit": zero,
                "stuck_step_count": zero,
                "rolling_window_progress_rad": zero,
                "foot_center_distance_m": zero,
                "action_rms": zero,
                "action_rate_rms": zero,
                "startup_action_ramp": zero,
                "normalized_torque_rms": zero,
                "reference_action_rms": zero,
                "residual_action_rms": zero,
                "reference_weight": zero,
                "residual_gain": zero,
                "failed": zero,
                "timeout": zero,
                "failure_nonfinite": zero,
                "failure_nonfinite_action": zero,
                "failure_nonfinite_physics": zero,
                "failure_root_low": zero,
                "failure_stuck": zero,
                "failure_root_high": zero,
                "failure_foot_gap": zero,
                "failure_leg_crossing": zero,
                "disturbance_applied": zero,
            }

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            (
                joint_key,
                velocity_key,
                disturbance_step_key,
                disturbance_value_key,
                disturbance_schedule_key,
                disturbance_level_key,
                disturbance_direction_key,
            ) = jax.random.split(rng, 7)
            joint_noise = jax.random.uniform(
                joint_key,
                shape=(4,),
                minval=-task.reset_joint_noise_rad,
                maxval=task.reset_joint_noise_rad,
            )
            velocity_noise = jax.random.uniform(
                velocity_key,
                shape=(self.mj_model.nv,),
                minval=-task.reset_velocity_noise,
                maxval=task.reset_velocity_noise,
            )
            disturbance_step = jax.random.randint(
                disturbance_step_key,
                shape=(),
                minval=task.disturbance_min_step,
                maxval=task.disturbance_max_step + 1,
            )
            disturbance_values = jax.random.uniform(
                disturbance_value_key,
                shape=(2,),
                minval=0.0,
                maxval=1.0,
            )
            disturbance_scheduled = jax.random.bernoulli(
                disturbance_schedule_key,
                p=task.disturbance_probability,
            )
            disturbance_level_index = jax.random.categorical(
                disturbance_level_key,
                jp.log(jp.asarray(task.disturbance_level_probabilities)),
            )
            sampled_disturbance_scale = jp.asarray(
                task.disturbance_level_scales
            )[disturbance_level_index]
            disturbance_scale = jp.where(
                disturbance_scheduled, sampled_disturbance_scale, 0.0
            )
            disturbance_backward = jax.random.bernoulli(
                disturbance_direction_key,
                p=task.disturbance_backward_probability,
            )
            disturbance_x_sign = jp.where(
                disturbance_backward, -1.0, 1.0
            )
            qpos = self.compact_qpos.at[self.joint_qpos_indices].set(
                self.reference_start_ctrl + joint_noise
            )
            qpos = qpos.at[self.root_x_qpos].set(0.0)
            qpos = qpos.at[self.root_z_qpos].set(
                self.reference_start_root_z
            )
            qvel = velocity_noise
            target = jp.clip(
                qpos[self.joint_qpos_indices],
                self.joint_low,
                self.joint_high,
            )
            data = self.base_data.replace(
                qpos=qpos,
                qvel=qvel,
                ctrl=target,
            )
            data = mjx.forward(self.mjx_model, data)
            contacts = self._contact_metrics(data)
            last_action = (
                reference_settings.reference_weight
                * self.reference_bias_action
                if reference_settings is not None
                else jp.zeros(4)
            )
            oscillator_phase = jp.zeros((), dtype=jp.float32)
            if reference_settings is not None:
                cem_action = reference_action(
                    jp,
                    oscillator_phase,
                    reference_settings,
                    compact_ctrl=self.compact_ctrl,
                    action_scales=self.action_scales,
                    joint_low=self.joint_low,
                    joint_high=self.joint_high,
                )
            else:
                cem_action = jp.zeros(4, dtype=jp.float32)
            info = {
                "initial_phase": data.qpos[self.root_pitch_qpos],
                "initial_root_x": data.qpos[self.root_x_qpos],
                "previous_phase": data.qpos[self.root_pitch_qpos],
                "previous_root_x": data.qpos[self.root_x_qpos],
                "previous_roll_potential": jp.zeros(
                    (), dtype=jp.float32
                ),
                "previous_mismatch_potential": jp.zeros(
                    (), dtype=jp.float32
                ),
                "last_action": last_action,
                "last_policy_action": last_action,
                "last_reference_action": cem_action,
                "oscillator_phase": oscillator_phase,
                "maximum_forbidden_penetration": jp.zeros(
                    (), dtype=jp.float32
                ),
                "maximum_allowed_excess": jp.zeros(
                    (), dtype=jp.float32
                ),
                "previous_foot_contact": (
                    contacts["allowed_count"] > 0
                ),
                "root_low_step_count": jp.asarray(0, dtype=jp.int32),
                "stuck_step_count": jp.asarray(0, dtype=jp.int32),
                "roll_potential_history": jp.zeros(
                    (self.stuck_progress_window_steps,), dtype=jp.float32
                ),
                "step_count": jp.asarray(0, dtype=jp.int32),
                "disturbance_step": disturbance_step,
                "disturbance_scheduled": disturbance_scheduled,
                "disturbance_scale": disturbance_scale,
                "disturbance_root_x_velocity": (
                    disturbance_x_sign
                    * disturbance_values[0]
                    * disturbance_scale
                    * task.disturbance_root_x_velocity_m_s
                ),
                "disturbance_root_pitch_velocity": (
                    (2.0 * disturbance_values[1] - 1.0)
                    * disturbance_scale
                    * task.disturbance_root_pitch_velocity_rad_s
                ),
            }
            observation = self._observation(
                data,
                last_action,
                contacts,
                reference_action_value=cem_action,
                oscillator_phase=oscillator_phase,
                action_ramp=jp.zeros((), dtype=jp.float32),
            )
            observation = jp.nan_to_num(
                observation, nan=0.0, posinf=0.0, neginf=0.0
            )
            return State(
                data,
                observation,
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
            control_dt = (
                float(self.mj_model.opt.timestep) * task.action_repeat
            )
            elapsed_s = (
                state.info["step_count"].astype(jp.float32) * control_dt
            )
            action_ramp = smoothstep_ramp(
                jp, elapsed_s, task.startup_action_ramp_s
            )
            disturbance_applied = jp.asarray(disturbance_enabled) & (
                state.info["disturbance_scheduled"]
                & (
                    state.info["step_count"]
                    == state.info["disturbance_step"]
                )
            )
            disturbed_qvel = state.pipeline_state.qvel
            disturbed_qvel = disturbed_qvel.at[self.root_x_dof].add(
                jp.where(
                    disturbance_applied,
                    state.info["disturbance_root_x_velocity"],
                    0.0,
                )
            )
            disturbed_qvel = disturbed_qvel.at[self.root_pitch_dof].add(
                jp.where(
                    disturbance_applied,
                    state.info["disturbance_root_pitch_velocity"],
                    0.0,
                )
            )
            pipeline_state = state.pipeline_state.replace(
                qvel=disturbed_qvel
            )
            if reference_settings is None:
                action = action_ramp * policy_action
                cem_action = jp.zeros(4, dtype=jp.float32)
                oscillator_phase = state.info["oscillator_phase"]
                target = jp.clip(
                    self.compact_ctrl + action * self.action_scales,
                    self.joint_low,
                    self.joint_high,
                )
                data = pipeline_state.replace(ctrl=target)

                def physics_step(carry, _):
                    return mjx.step(self.mjx_model, carry), None

                candidate_data = jax.lax.scan(
                    physics_step,
                    data,
                    (),
                    length=task.action_repeat,
                )[0]
            else:
                physics_dt = float(self.mj_model.opt.timestep)

                def residual_physics_step(carry, _):
                    current_data, current_oscillator_phase = carry
                    next_oscillator_phase = advance_oscillator(
                        jp,
                        current_data.qpos[self.root_pitch_qpos],
                        current_oscillator_phase,
                        physics_dt,
                        reference_settings,
                    )
                    current_reference_action = reference_action(
                        jp,
                        next_oscillator_phase,
                        reference_settings,
                        compact_ctrl=self.compact_ctrl,
                        action_scales=self.action_scales,
                        joint_low=self.joint_low,
                        joint_high=self.joint_high,
                    )
                    current_ramp = smoothstep_ramp(
                        jp,
                        current_data.time,
                        task.startup_action_ramp_s,
                    )
                    effective_action = effective_residual_action(
                        jp,
                        policy_action,
                        current_reference_action,
                        reference_settings,
                    )
                    retained_bias_action = (
                        reference_settings.reference_weight
                        * self.reference_bias_action
                    )
                    current_action = retained_bias_action + current_ramp * (
                        effective_action - retained_bias_action
                    )
                    current_target = jp.clip(
                        self.compact_ctrl
                        + current_action * self.action_scales,
                        self.joint_low,
                        self.joint_high,
                    )
                    next_data = mjx.step(
                        self.mjx_model,
                        current_data.replace(ctrl=current_target),
                    )
                    return (
                        next_data,
                        next_oscillator_phase,
                    ), (
                        current_action,
                        current_reference_action,
                        current_ramp,
                    )

                (
                    candidate_data,
                    candidate_oscillator_phase,
                ), residual_trace = jax.lax.scan(
                    residual_physics_step,
                    (
                        pipeline_state,
                        state.info["oscillator_phase"],
                    ),
                    (),
                    length=task.action_repeat,
                )
                action = residual_trace[0][-1]
                cem_action = residual_trace[1][-1]
                action_ramp = residual_trace[2][-1]
                oscillator_phase = candidate_oscillator_phase
            _, _, candidate_contact_distance = self._contact_arrays(
                candidate_data
            )
            physics_finite = (
                jp.all(jp.isfinite(candidate_data.qpos))
                & jp.all(jp.isfinite(candidate_data.qvel))
                & jp.all(jp.isfinite(candidate_data.qacc))
                & jp.all(jp.isfinite(candidate_data.actuator_force))
                & jp.all(jp.isfinite(candidate_data.xpos))
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
            action = jp.where(
                transition_finite, action, state.info["last_action"]
            )
            policy_action = jp.where(
                transition_finite,
                policy_action,
                state.info["last_policy_action"],
            )
            cem_action = jp.where(
                transition_finite,
                cem_action,
                state.info["last_reference_action"],
            )
            oscillator_phase = jp.where(
                transition_finite,
                oscillator_phase,
                state.info["oscillator_phase"],
            )
            contacts = self._contact_metrics(data)
            leg_crossing = self._leg_crossing(data)

            phase = data.qpos[self.root_pitch_qpos]
            root_x = data.qpos[self.root_x_qpos]
            phase_progress = phase - state.info["previous_phase"]
            translation_progress = (
                root_x - state.info["previous_root_x"]
            ) / self.rolling_radius
            cumulative_phase = phase - state.info["initial_phase"]
            cumulative_translation = (
                root_x - state.info["initial_root_x"]
            ) / self.rolling_radius
            roll_potential = conservative_rolling_potential(
                jp, cumulative_phase, cumulative_translation
            )
            conservative_progress = (
                roll_potential - state.info["previous_roll_potential"]
            )
            mismatch_potential = jp.abs(
                cumulative_phase - cumulative_translation
            )
            mismatch_progress = (
                mismatch_potential
                - state.info["previous_mismatch_potential"]
            )
            backward = jp.maximum(-phase_progress, 0.0) + jp.maximum(
                -translation_progress, 0.0
            )

            step_count = state.info["step_count"] + 1
            history_index = (
                state.info["step_count"] % self.stuck_progress_window_steps
            )
            rolling_window_progress = (
                roll_potential
                - state.info["roll_potential_history"][history_index]
            )
            roll_potential_history = state.info[
                "roll_potential_history"
            ].at[history_index].set(roll_potential)

            action_rate = jp.mean(
                jp.square(action - state.info["last_action"])
            )
            normalized_torque = data.actuator_force / jp.maximum(
                self.force_limits, 1e-6
            )
            torque_cost = jp.mean(jp.square(normalized_torque))
            airborne = (contacts["ground_count"] == 0).astype(jp.float32)
            foot_distance = jp.linalg.norm(
                data.site_xpos[
                    self.front_foot_site_id, self.planar_indices
                ]
                - data.site_xpos[
                    self.rear_foot_site_id, self.planar_indices
                ]
            )

            forbidden_depth = contacts["forbidden_depth"]
            allowed_excess = jp.maximum(
                contacts["allowed_depth"]
                - reward_settings.allowed_foot_penetration_m,
                0.0,
            )
            allowed_contact_active = contacts["allowed_count"] > 0
            allowed_contact_start = allowed_contact_active & (
                ~state.info["previous_foot_contact"]
            )
            new_forbidden_max = jp.maximum(
                state.info["maximum_forbidden_penetration"],
                forbidden_depth,
            )
            new_allowed_max = jp.maximum(
                state.info["maximum_allowed_excess"], allowed_excess
            )
            forbidden_max_increment = (
                new_forbidden_max
                - state.info["maximum_forbidden_penetration"]
            )
            allowed_max_increment = (
                new_allowed_max - state.info["maximum_allowed_excess"]
            )
            root_z = data.qpos[self.root_z_qpos]
            failure_nonfinite_action = ~action_finite
            failure_nonfinite_physics = action_finite & (~physics_finite)
            failure_nonfinite = ~transition_finite
            if task.terminate_root_z_min is None:
                root_low_active = jp.asarray(False)
                root_low_step_count = jp.asarray(0, dtype=jp.int32)
                failure_root_low = jp.asarray(False)
            else:
                root_low_active = root_z < task.terminate_root_z_min
                root_low_step_count = jp.where(
                    root_low_active,
                    state.info["root_low_step_count"] + 1,
                    jp.asarray(0, dtype=jp.int32),
                )
                failure_root_low = (
                    root_low_step_count >= self.root_low_termination_steps
                )
            if task.terminate_stuck_root_z_max is None:
                stuck_active = jp.asarray(False)
                stuck_deficit = jp.zeros((), dtype=jp.float32)
                stuck_step_count = jp.asarray(0, dtype=jp.int32)
                failure_stuck = jp.asarray(False)
            else:
                (
                    stuck_active,
                    stuck_deficit,
                    stuck_step_count,
                    failure_stuck,
                ) = (
                    stuck_termination_state(
                        jp,
                        root_z=root_z,
                        rolling_window_progress=rolling_window_progress,
                        previous_count=state.info["stuck_step_count"],
                        step_count=step_count,
                        root_z_max=task.terminate_stuck_root_z_max,
                        minimum_progress_rad=(
                            task.terminate_stuck_min_progress_rad
                        ),
                        grace_steps=self.stuck_grace_steps,
                        termination_steps=self.stuck_termination_steps,
                    )
                )
            failure_root_high = (
                jp.asarray(False)
                if task.terminate_root_z_max is None
                else root_z > task.terminate_root_z_max
            )
            failure_foot_gap = (
                jp.asarray(False)
                if task.maximum_foot_center_distance_m is None
                else foot_distance > task.maximum_foot_center_distance_m
            )
            failure_leg_crossing = (
                leg_crossing
                if task.terminate_leg_crossing
                else jp.asarray(False)
            )
            failed_bool = (
                failure_nonfinite
                | failure_root_low
                | failure_stuck
                | failure_root_high
                | failure_foot_gap
                | failure_leg_crossing
            )
            timeout_bool = step_count >= task.episode_length
            done = (failed_bool | timeout_bool).astype(jp.float32)
            remaining_fraction = jp.maximum(
                task.episode_length - step_count, 0
            ).astype(jp.float32) / max(task.episode_length - 1, 1)

            raw_reward_terms = reward_terms(
                jp,
                reward_settings,
                {
                    "conservative_progress": conservative_progress,
                    "phase_progress": phase_progress,
                    "translation_progress": translation_progress,
                    "mismatch_progress": mismatch_progress,
                    "backward": backward,
                    "action_rate": action_rate,
                    "residual_action_cost": jp.mean(
                        jp.square(policy_action)
                    ),
                    "torque_cost": torque_cost,
                    "airborne": airborne,
                    "stuck_deficit": stuck_deficit,
                    "foot_distance": foot_distance,
                    "control_dt": control_dt,
                    "forbidden_count": contacts["forbidden_count"],
                    "forbidden_depth": forbidden_depth,
                    "forbidden_max_increment": (
                        forbidden_max_increment
                    ),
                    "allowed_excess": allowed_excess,
                    "allowed_max_increment": allowed_max_increment,
                    "allowed_contact_active": (
                        allowed_contact_active.astype(jp.float32)
                    ),
                    "allowed_contact_start": (
                        allowed_contact_start.astype(jp.float32)
                    ),
                    "leg_crossing": leg_crossing.astype(jp.float32),
                    "failed": failed_bool.astype(jp.float32),
                    "failure_root_low": failure_root_low.astype(
                        jp.float32
                    ),
                    "failure_stuck": failure_stuck.astype(jp.float32),
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
            reward = sum(raw_reward_terms.values())
            nonfinite_terminal_reward = -reward_settings.termination * (
                1.0
                + reward_settings.early_termination_scale
                * remaining_fraction
            )
            reward = jp.nan_to_num(
                reward,
                nan=nonfinite_terminal_reward,
                posinf=nonfinite_terminal_reward,
                neginf=nonfinite_terminal_reward,
            )
            rewards = {
                f"reward_{name}": jp.nan_to_num(
                    value,
                    nan=(
                        -reward_settings.termination
                        if name == "termination"
                        else (
                            -reward_settings.termination
                            * reward_settings.early_termination_scale
                            * remaining_fraction
                            if name == "early_termination"
                            else 0.0
                        )
                    ),
                    posinf=0.0,
                    neginf=0.0,
                )
                for name, value in raw_reward_terms.items()
            }
            info = {
                **state.info,
                "previous_phase": phase,
                "previous_root_x": root_x,
                "previous_roll_potential": roll_potential,
                "previous_mismatch_potential": mismatch_potential,
                "last_action": action,
                "last_policy_action": policy_action,
                "last_reference_action": cem_action,
                "oscillator_phase": oscillator_phase,
                "maximum_forbidden_penetration": new_forbidden_max,
                "maximum_allowed_excess": new_allowed_max,
                "previous_foot_contact": allowed_contact_active,
                "root_low_step_count": root_low_step_count,
                "stuck_step_count": stuck_step_count,
                "roll_potential_history": roll_potential_history,
                "step_count": step_count,
            }
            metrics = {
                "reward": reward,
                "reward_total": reward,
                **rewards,
                "roll_progress_rad": conservative_progress,
                "phase_progress_rad": phase_progress,
                "translation_progress_rad": translation_progress,
                "tail_roll_progress_rad": jp.where(
                    step_count
                    > task.episode_length - self.tail_progress_window_steps,
                    conservative_progress,
                    0.0,
                ),
                "forbidden_contact_count": contacts[
                    "forbidden_count"
                ],
                "forbidden_penetration_m": forbidden_depth,
                "allowed_foot_penetration_m": contacts["allowed_depth"],
                "foot_contact_active": (
                    allowed_contact_active.astype(jp.float32)
                ),
                "foot_contact_start": (
                    allowed_contact_start.astype(jp.float32)
                ),
                "ground_contact_count": contacts["ground_count"],
                "leg_crossing": leg_crossing.astype(jp.float32),
                "root_height_m": root_z,
                "root_low_active": root_low_active.astype(jp.float32),
                "root_low_step_count": (
                    root_low_step_count.astype(jp.float32)
                ),
                "stuck_active": stuck_active.astype(jp.float32),
                "stuck_deficit": stuck_deficit,
                "stuck_step_count": stuck_step_count.astype(jp.float32),
                "rolling_window_progress_rad": rolling_window_progress,
                "foot_center_distance_m": foot_distance,
                "action_rms": jp.sqrt(jp.mean(jp.square(action))),
                "action_rate_rms": jp.sqrt(action_rate),
                "startup_action_ramp": action_ramp,
                "normalized_torque_rms": jp.sqrt(torque_cost),
                "reference_action_rms": jp.sqrt(
                    jp.mean(jp.square(cem_action))
                ),
                "residual_action_rms": jp.sqrt(
                    jp.mean(jp.square(policy_action))
                ),
                "reference_weight": jp.asarray(
                    (
                        reference_settings.reference_weight
                        if reference_settings is not None
                        else 0.0
                    ),
                    dtype=jp.float32,
                ),
                "residual_gain": jp.asarray(
                    (
                        reference_settings.residual_gain
                        if reference_settings is not None
                        else 0.0
                    ),
                    dtype=jp.float32,
                ),
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
                "failure_stuck": failure_stuck.astype(jp.float32),
                "failure_root_high": failure_root_high.astype(jp.float32),
                "failure_foot_gap": failure_foot_gap.astype(jp.float32),
                "failure_leg_crossing": (
                    failure_leg_crossing.astype(jp.float32)
                ),
                "disturbance_applied": (
                    disturbance_applied.astype(jp.float32)
                ),
            }
            observation = self._observation(
                data,
                action,
                contacts,
                reference_action_value=cem_action,
                oscillator_phase=oscillator_phase,
                action_ramp=action_ramp,
            )
            observation = jp.nan_to_num(
                observation, nan=0.0, posinf=0.0, neginf=0.0
            )
            metrics = {
                name: jp.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                )
                for name, value in metrics.items()
            }
            return State(
                data,
                observation,
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

        def _contact_metrics(self, data):
            geom1, geom2, distance = self._contact_arrays(data)
            valid = (geom1 >= 0) & (geom2 >= 0) & (distance <= 0.0)
            ground = valid & (
                (geom1 == self.floor_geom_id)
                | (geom2 == self.floor_geom_id)
            )
            allowed = valid & (
                (
                    (geom1 == self.front_foot_geom_id)
                    & (geom2 == self.rear_foot_geom_id)
                )
                | (
                    (geom2 == self.front_foot_geom_id)
                    & (geom1 == self.rear_foot_geom_id)
                )
            )
            forbidden = valid & (~ground) & (~allowed)
            forbidden_depth = jp.max(
                jp.where(forbidden, -distance, 0.0)
            )
            allowed_depth = jp.max(jp.where(allowed, -distance, 0.0))
            return {
                "ground_count": jp.sum(ground).astype(jp.float32),
                "allowed_count": jp.sum(allowed).astype(jp.float32),
                "forbidden_count": jp.sum(forbidden).astype(jp.float32),
                "forbidden_depth": forbidden_depth,
                "allowed_depth": allowed_depth,
            }

        @staticmethod
        def _proper_intersection(a, b, c, d):
            def cross(first, second):
                return first[0] * second[1] - first[1] * second[0]

            side_c = cross(b - a, c - a)
            side_d = cross(b - a, d - a)
            side_a = cross(d - c, a - c)
            side_b = cross(d - c, b - c)
            tolerance_squared = 1e-20
            return (side_c * side_d < -tolerance_squared) & (
                side_a * side_b < -tolerance_squared
            )

        def _leg_crossing(self, data):
            front_hip = data.xpos[
                self.body_ids["front_thigh"], self.planar_indices
            ]
            front_knee = data.xpos[
                self.body_ids["front_shank"], self.planar_indices
            ]
            front_foot = data.site_xpos[
                self.front_foot_site_id, self.planar_indices
            ]
            rear_hip = data.xpos[
                self.body_ids["rear_thigh"], self.planar_indices
            ]
            rear_knee = data.xpos[
                self.body_ids["rear_shank"], self.planar_indices
            ]
            rear_foot = data.site_xpos[
                self.rear_foot_site_id, self.planar_indices
            ]
            return (
                self._proper_intersection(
                    front_hip, front_knee, rear_hip, rear_knee
                )
                | self._proper_intersection(
                    front_hip, front_knee, rear_knee, rear_foot
                )
                | self._proper_intersection(
                    front_knee, front_foot, rear_hip, rear_knee
                )
                | self._proper_intersection(
                    front_knee, front_foot, rear_knee, rear_foot
                )
            )

        def _observation(
            self,
            data,
            last_action,
            contacts,
            *,
            reference_action_value,
            oscillator_phase,
            action_ramp,
        ):
            phase = data.qpos[self.root_pitch_qpos]
            root_velocity = data.qvel[:3]
            joint_position = data.qpos[self.joint_qpos_indices]
            joint_velocity = data.qvel[self.joint_dof_indices]
            contact_features = jp.stack(
                [
                    (contacts["ground_count"] > 0).astype(jp.float32),
                    (contacts["allowed_count"] > 0).astype(jp.float32),
                    (contacts["forbidden_count"] > 0).astype(jp.float32),
                    1000.0 * contacts["forbidden_depth"],
                    1000.0 * contacts["allowed_depth"],
                ]
            )
            observation = jp.concatenate(
                [
                    jp.asarray([jp.sin(phase), jp.cos(phase)]),
                    jp.asarray([data.qpos[self.root_z_qpos]]),
                    root_velocity,
                    joint_position,
                    joint_velocity,
                    last_action,
                    contact_features,
                ]
            )
            if reference_settings is None:
                return observation
            weighted_reference = (
                reference_settings.reference_weight
                * reference_action_value
            )
            reference_features = jp.concatenate(
                [
                    weighted_reference,
                    jp.asarray(
                        [
                            reference_settings.reference_weight
                            * jp.sin(oscillator_phase),
                            reference_settings.reference_weight
                            * jp.cos(oscillator_phase),
                            action_ramp,
                        ]
                    ),
                ]
            )
            return jp.concatenate([observation, reference_features])

    return CurlRobot2DMJXEnv()
