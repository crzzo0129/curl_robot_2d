"""Brax-compatible MJX environment for 3-D curl rolling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceConfig,
    advance_oscillator,
    load_cem_reference,
    reference_action,
    wrapped_phase_error,
)
from curl_robot_2d_mjx.config_3d import (
    Rolling3DConfig,
    smoothstep_ramp,
    validate_3d_config,
)
from curl_robot_2d_mjx.reward_3d import (
    REWARD_3D_TERM_NAMES,
    Rolling3DRewardConfig,
    conservative_rolling_potential,
    reward_terms_3d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH_3D = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"
DEFAULT_3D_CEM_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "collision_constrained_cem_foot_gap_2mm_short_contact"
    / "best_phase_controller.json"
)
ACTION_SIZE_3D = 8
OBSERVATION_SIZE_3D = 59
PLANAR_ACTION_SCALES = np.asarray((0.8, 1.2, 0.8, 1.2), dtype=np.float64)
PLANAR_COMPACT = np.asarray(
    (
        FIXED_PARAMETERS.compact_hip_angle,
        FIXED_PARAMETERS.compact_knee_angle,
        FIXED_PARAMETERS.compact_hip_angle,
        FIXED_PARAMETERS.compact_knee_angle,
    ),
    dtype=np.float64,
)
PLANAR_JOINT_LOW = np.asarray(
    (
        FIXED_PARAMETERS.hip.shell_compatible_range[0],
        FIXED_PARAMETERS.knee.shell_compatible_range[0],
        FIXED_PARAMETERS.hip.shell_compatible_range[0],
        FIXED_PARAMETERS.knee.shell_compatible_range[0],
    ),
    dtype=np.float64,
)
PLANAR_JOINT_HIGH = np.asarray(
    (
        FIXED_PARAMETERS.hip.shell_compatible_range[1],
        FIXED_PARAMETERS.knee.shell_compatible_range[1],
        FIXED_PARAMETERS.hip.shell_compatible_range[1],
        FIXED_PARAMETERS.knee.shell_compatible_range[1],
    ),
    dtype=np.float64,
)
FOOT_GEOM_NAMES_3D = (
    "front_left_foot_proxy",
    "front_right_foot_proxy",
    "rear_left_foot_proxy",
    "rear_right_foot_proxy",
)


def duplicate_planar_action_3d(xp, planar_action):
    """Map front/rear 2-D normalized actions to left/right 3-D rails."""

    front_hip, front_knee, rear_hip, rear_knee = planar_action
    return xp.stack(
        (
            front_hip,
            front_knee,
            front_hip,
            front_knee,
            rear_hip,
            rear_knee,
            rear_hip,
            rear_knee,
        )
    )


def advance_rolling_phase_3d(
    xp,
    rolling_phase,
    local_y_angular_velocity,
    timestep,
):
    """Integrate signed spin around the torso's local rolling axis."""

    return rolling_phase + timestep * local_y_angular_velocity


def apply_physics_options_3d(model, task: Rolling3DConfig) -> None:
    """Apply a runtime solver profile to the generated 3-D MuJoCo model."""

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
        root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        if root_id < 0:
            raise ValueError("missing MuJoCo freejoint: root")
        dof_id = int(model.jnt_dofadr[root_id])
        model.dof_damping[dof_id : dof_id + 6] = 0.0


def _load_dependencies_3d():
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "3-D MJX dependencies are unavailable. Install "
            "requirements-mjx.txt on the Linux GPU instance."
        ) from exc
    return jax, jp, mujoco, mjx, Env, State


def make_brax_env_3d(
    config: Rolling3DConfig | None = None,
    *,
    reward_config: Rolling3DRewardConfig | None = None,
    cem_reference: CEMReferenceConfig | None = None,
    seed: int = 0,
):
    """Create the first 3-D curl MJX env.

    The default is a CEM-reference residual task: zero residual exactly tracks
    the duplicated left/right 2-D CEM reference.
    """

    task = config or Rolling3DConfig()
    validate_3d_config(task)
    jax, jp, mujoco, mjx, Env, State = _load_dependencies_3d()
    reward_settings = reward_config or Rolling3DRewardConfig()
    reference_settings = cem_reference or load_cem_reference(
        DEFAULT_3D_CEM_CONTROLLER
    )

    class CurlRobot3DMJXEnv(Env):
        def __init__(self):
            self.config = task
            self.reward_config = reward_settings
            self.cem_reference = reference_settings
            self.seed = seed
            self.mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH_3D))
            apply_physics_options_3d(self.mj_model, task)
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
            self.action_scales = jp.asarray(task.action_scales)
            self.planar_compact = jp.asarray(PLANAR_COMPACT)
            self.planar_action_scales = jp.asarray(PLANAR_ACTION_SCALES)
            self.planar_joint_low = jp.asarray(PLANAR_JOINT_LOW)
            self.planar_joint_high = jp.asarray(PLANAR_JOINT_HIGH)

            self.torso_body_id = object_id(
                mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
            self.floor_geom_id = object_id(
                mujoco.mjtObj.mjOBJ_GEOM, "floor"
            )
            (
                self.front_left_foot_geom_id,
                self.front_right_foot_geom_id,
                self.rear_left_foot_geom_id,
                self.rear_right_foot_geom_id,
            ) = (
                object_id(mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in FOOT_GEOM_NAMES_3D
            )
            shell_geom_ids = []
            for geom_id in range(self.mj_model.ngeom):
                name = (
                    mujoco.mj_id2name(
                        self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    or ""
                )
                if "_shell_" in name:
                    shell_geom_ids.append(int(geom_id))
            self.shell_geom_ids = jp.asarray(shell_geom_ids, dtype=jp.int32)
            self.foot_geom_ids = jp.asarray(
                (
                    self.front_left_foot_geom_id,
                    self.front_right_foot_geom_id,
                    self.rear_left_foot_geom_id,
                    self.rear_right_foot_geom_id,
                ),
                dtype=jp.int32,
            )

            joint_ids = [
                object_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES_3D
            ]
            self.joint_qpos_indices = jp.asarray(
                [int(self.mj_model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
            )
            self.joint_dof_indices = jp.asarray(
                [int(self.mj_model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
            )
            self.joint_low = jp.asarray(
                [self.mj_model.jnt_range[joint_id, 0] for joint_id in joint_ids]
            )
            self.joint_high = jp.asarray(
                [self.mj_model.jnt_range[joint_id, 1] for joint_id in joint_ids]
            )
            actuator_ids = [
                object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_servo")
                for name in JOINT_NAMES_3D
            ]
            self.actuator_ids = jp.asarray(actuator_ids, dtype=jp.int32)
            self.force_limits = jp.asarray(
                np.abs(self.mj_model.actuator_forcerange[actuator_ids, 1])
            )
            self.rolling_radius = FIXED_PARAMETERS.shell_contact_radius
            self.root_low_termination_steps = _duration_to_steps(
                task.terminate_root_z_low_duration_s,
                task.control_timestep,
            )
            self.axis_tilt_termination_steps = _duration_to_steps(
                task.terminate_axis_tilt_duration_s,
                task.control_timestep,
            )
            self.forbidden_contact_termination_steps = _duration_to_steps(
                task.terminate_forbidden_contact_duration_s,
                task.control_timestep,
            )

        @property
        def observation_size(self):
            return OBSERVATION_SIZE_3D

        @property
        def action_size(self):
            return ACTION_SIZE_3D

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
                    for name in REWARD_3D_TERM_NAMES
                },
                "roll_progress_rad": zero,
                "rotation_progress_rad": zero,
                "translation_progress_rad": zero,
                "mismatch_progress_rad": zero,
                "root_x_m": zero,
                "root_y_m": zero,
                "root_z_m": zero,
                "lateral_drift_m": zero,
                "lateral_velocity_m_s": zero,
                "axis_tilt_rad": zero,
                "axis_tilt_step_count": zero,
                "root_low_active": zero,
                "root_low_step_count": zero,
                "shell_floor_contact_count": zero,
                "foot_floor_contact_count": zero,
                "same_side_foot_contact_count": zero,
                "same_side_foot_penetration_m": zero,
                "same_side_foot_contact_active": zero,
                "same_side_foot_contact_start": zero,
                "forbidden_contact_count": zero,
                "forbidden_penetration_m": zero,
                "forbidden_contact_step_count": zero,
                "cross_side_foot_contact_count": zero,
                "action_rms": zero,
                "action_rate_rms": zero,
                "startup_action_ramp": zero,
                "normalized_torque_rms": zero,
                "reference_action_rms": zero,
                "residual_action_rms": zero,
                "reference_weight": zero,
                "residual_gain": zero,
                "rolling_phase_rad": zero,
                "oscillator_phase_rad": zero,
                "phase_error_rad": zero,
                "oscillator_rate_rad_s": zero,
                "failed": zero,
                "timeout": zero,
                "failure_nonfinite": zero,
                "failure_nonfinite_action": zero,
                "failure_nonfinite_physics": zero,
                "failure_root_low": zero,
                "failure_root_high": zero,
                "failure_lateral_drift": zero,
                "failure_axis_tilt": zero,
                "failure_forbidden_depth": zero,
                "failure_forbidden_contact": zero,
            }

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            joint_key, velocity_key = jax.random.split(rng, 2)
            joint_noise = jax.random.uniform(
                joint_key,
                shape=(ACTION_SIZE_3D,),
                minval=-task.reset_joint_noise_rad,
                maxval=task.reset_joint_noise_rad,
            )
            velocity_noise = jax.random.uniform(
                velocity_key,
                shape=(self.mj_model.nv,),
                minval=-task.reset_velocity_noise,
                maxval=task.reset_velocity_noise,
            )
            oscillator_phase = jp.zeros((), dtype=jp.float32)
            cem_action = self._reference_action_8d(oscillator_phase)
            start_ctrl = jp.clip(
                self.compact_ctrl + cem_action * self.action_scales,
                self.joint_low,
                self.joint_high,
            )
            noisy_start_ctrl = jp.clip(
                start_ctrl + joint_noise,
                self.joint_low,
                self.joint_high,
            )
            qpos = self.compact_qpos.at[self.joint_qpos_indices].set(
                noisy_start_ctrl
            )
            qpos = qpos.at[0].set(0.0)
            qpos = qpos.at[1].set(0.0)
            qvel = velocity_noise
            data = self.base_data.replace(
                qpos=qpos,
                qvel=qvel,
                ctrl=start_ctrl,
            )
            data = mjx.forward(self.mjx_model, data)
            contacts = self._contact_metrics(data)
            axis_tilt = self._rolling_axis_tilt(data)
            info = {
                "initial_root_x": data.qpos[0],
                "initial_root_y": data.qpos[1],
                "previous_root_x": data.qpos[0],
                "cumulative_rotation": jp.zeros((), dtype=jp.float32),
                "previous_roll_potential": jp.zeros(
                    (), dtype=jp.float32
                ),
                "previous_mismatch_potential": jp.zeros(
                    (), dtype=jp.float32
                ),
                "last_action": cem_action,
                "last_policy_action": jp.zeros(
                    (ACTION_SIZE_3D,), dtype=jp.float32
                ),
                "last_reference_action": cem_action,
                "oscillator_phase": oscillator_phase,
                "rolling_phase": jp.zeros((), dtype=jp.float32),
                "maximum_forbidden_penetration": jp.zeros(
                    (), dtype=jp.float32
                ),
                "maximum_same_side_foot_excess": jp.zeros(
                    (), dtype=jp.float32
                ),
                "previous_same_side_foot_contact": (
                    contacts["same_side_foot_count"] > 0
                ),
                "root_low_step_count": jp.asarray(0, dtype=jp.int32),
                "axis_tilt_step_count": jp.asarray(0, dtype=jp.int32),
                "forbidden_contact_step_count": jp.asarray(
                    0, dtype=jp.int32
                ),
                "step_count": jp.asarray(0, dtype=jp.int32),
            }
            observation = self._observation(
                data,
                cem_action,
                contacts,
                axis_tilt=axis_tilt,
                reference_action_value=cem_action,
                oscillator_phase=oscillator_phase,
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
            control_dt = (
                float(self.mj_model.opt.timestep) * task.action_repeat
            )
            residual_gain_value = jp.asarray(
                reference_settings.residual_gain, dtype=jp.float32
            )
            reference_weight_value = jp.asarray(
                reference_settings.reference_weight, dtype=jp.float32
            )
            physics_dt = float(self.mj_model.opt.timestep)

            def reference_physics_step(carry, _):
                current_data, current_phase, current_rolling_phase = carry
                next_phase = advance_oscillator(
                    jp,
                    current_rolling_phase,
                    current_phase,
                    physics_dt,
                    reference_settings,
                    rate_scale=task.reference_phase_rate_scale,
                )
                current_reference_action = self._reference_action_8d(
                    next_phase
                )
                current_ramp = smoothstep_ramp(
                    jp,
                    current_data.time,
                    task.startup_action_ramp_s,
                )
                current_action = jp.clip(
                    reference_weight_value * current_reference_action
                    + current_ramp * residual_gain_value * policy_action,
                    -1.0,
                    1.0,
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
                next_rolling_phase = advance_rolling_phase_3d(
                    jp,
                    current_rolling_phase,
                    next_data.qvel[4],
                    physics_dt,
                )
                phase_rate = (next_phase - current_phase) / physics_dt
                return (
                    next_data,
                    next_phase,
                    next_rolling_phase,
                ), (
                    current_action,
                    current_reference_action,
                    current_ramp,
                    phase_rate,
                )

            (
                candidate_data,
                candidate_oscillator_phase,
                candidate_rolling_phase,
            ), reference_trace = jax.lax.scan(
                reference_physics_step,
                (
                    state.pipeline_state,
                    state.info["oscillator_phase"],
                    state.info["rolling_phase"],
                ),
                (),
                length=task.action_repeat,
            )
            effective_action = reference_trace[0][-1]
            cem_action = reference_trace[1][-1]
            action_ramp = reference_trace[2][-1]
            oscillator_rate = reference_trace[3][-1]
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
            effective_action = jp.where(
                transition_finite,
                effective_action,
                state.info["last_action"],
            )
            oscillator_phase = jp.where(
                transition_finite,
                candidate_oscillator_phase,
                state.info["oscillator_phase"],
            )
            rolling_phase = jp.where(
                transition_finite,
                candidate_rolling_phase,
                state.info["rolling_phase"],
            )
            oscillator_rate = jp.where(
                transition_finite,
                oscillator_rate,
                jp.zeros_like(oscillator_rate),
            )
            cem_action = jp.where(
                transition_finite,
                cem_action,
                state.info["last_reference_action"],
            )
            next_reference_action = cem_action
            phase_error = wrapped_phase_error(
                jp, rolling_phase, oscillator_phase
            )
            contacts = self._contact_metrics(data)
            axis_tilt = self._rolling_axis_tilt(data)

            root_x = data.qpos[0]
            root_y = data.qpos[1]
            root_z = data.qpos[2]
            translation_progress = (
                root_x - state.info["previous_root_x"]
            ) / self.rolling_radius
            angular_velocity_y = jp.abs(data.qvel[4])
            rotation_progress = angular_velocity_y * control_dt
            cumulative_rotation = (
                state.info["cumulative_rotation"] + rotation_progress
            )
            cumulative_translation = (
                root_x - state.info["initial_root_x"]
            ) / self.rolling_radius
            roll_potential = conservative_rolling_potential(
                jp, cumulative_rotation, cumulative_translation
            )
            conservative_progress = (
                roll_potential - state.info["previous_roll_potential"]
            )
            mismatch_potential = jp.abs(
                cumulative_rotation - cumulative_translation
            )
            mismatch_progress = (
                mismatch_potential
                - state.info["previous_mismatch_potential"]
            )
            backward_progress = jp.maximum(-translation_progress, 0.0)
            lateral_drift = root_y - state.info["initial_root_y"]
            lateral_drift_abs = jp.abs(lateral_drift)
            lateral_velocity = data.qvel[1]

            action_rate = jp.mean(
                jp.square(effective_action - state.info["last_action"])
            )
            normalized_torque = data.actuator_force[self.actuator_ids] / jp.maximum(
                self.force_limits, 1e-6
            )
            torque_cost = jp.mean(jp.square(normalized_torque))
            same_side_foot_excess = jp.maximum(
                contacts["same_side_foot_depth"]
                - reward_settings.allowed_foot_penetration_m,
                0.0,
            )
            new_forbidden_max = jp.maximum(
                state.info["maximum_forbidden_penetration"],
                contacts["forbidden_depth"],
            )
            new_same_side_foot_max = jp.maximum(
                state.info["maximum_same_side_foot_excess"],
                same_side_foot_excess,
            )
            forbidden_max_increment = (
                new_forbidden_max
                - state.info["maximum_forbidden_penetration"]
            )
            same_side_foot_max_increment = (
                new_same_side_foot_max
                - state.info["maximum_same_side_foot_excess"]
            )
            same_side_foot_active = contacts["same_side_foot_count"] > 0
            same_side_foot_start = same_side_foot_active & (
                ~state.info["previous_same_side_foot_contact"]
            )
            forbidden_active = contacts["forbidden_count"] > 0
            cross_side_foot_active = contacts["cross_side_foot_count"] > 0

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
            axis_tilt_active = axis_tilt > task.terminate_axis_tilt_rad
            axis_tilt_step_count = jp.where(
                axis_tilt_active,
                state.info["axis_tilt_step_count"] + 1,
                jp.asarray(0, dtype=jp.int32),
            )
            forbidden_contact_step_count = jp.where(
                forbidden_active,
                state.info["forbidden_contact_step_count"] + 1,
                jp.asarray(0, dtype=jp.int32),
            )
            failure_axis_tilt = (
                axis_tilt_step_count >= self.axis_tilt_termination_steps
            )
            failure_forbidden_contact = (
                forbidden_contact_step_count
                >= self.forbidden_contact_termination_steps
            )
            failure_root_high = root_z > task.terminate_root_z_max
            failure_lateral_drift = (
                lateral_drift_abs > task.terminate_lateral_drift_m
            )
            failure_forbidden_depth = (
                contacts["forbidden_depth"]
                > task.terminate_forbidden_depth_m
            )
            failed_bool = (
                failure_nonfinite
                | failure_root_low
                | failure_root_high
                | failure_lateral_drift
                | failure_axis_tilt
                | failure_forbidden_depth
                | failure_forbidden_contact
            )
            failure_severe = (
                failure_root_low
                | failure_lateral_drift
                | failure_axis_tilt
                | failure_forbidden_depth
                | failure_forbidden_contact
            )
            step_count = state.info["step_count"] + 1
            timeout_bool = step_count >= task.episode_length
            done = (failed_bool | timeout_bool).astype(jp.float32)
            remaining_fraction = jp.maximum(
                task.episode_length - step_count, 0
            ).astype(jp.float32) / max(task.episode_length - 1, 1)

            raw_reward_terms = reward_terms_3d(
                jp,
                reward_settings,
                {
                    "conservative_progress": conservative_progress,
                    "mismatch_progress": mismatch_progress,
                    "backward_progress": backward_progress,
                    "lateral_velocity_squared": jp.square(lateral_velocity),
                    "lateral_drift_abs": lateral_drift_abs,
                    "axis_tilt_squared": jp.square(axis_tilt),
                    "action_rate": action_rate,
                    "residual_action_cost": jp.mean(jp.square(policy_action)),
                    "torque_cost": torque_cost,
                    "control_dt": control_dt,
                    "forbidden_active": forbidden_active.astype(jp.float32),
                    "forbidden_depth": contacts["forbidden_depth"],
                    "forbidden_max_increment": forbidden_max_increment,
                    "same_side_foot_contact_start": (
                        same_side_foot_start.astype(jp.float32)
                    ),
                    "same_side_foot_contact_active": (
                        same_side_foot_active.astype(jp.float32)
                    ),
                    "same_side_foot_excess": same_side_foot_excess,
                    "same_side_foot_max_increment": (
                        same_side_foot_max_increment
                    ),
                    "cross_side_foot_contact": (
                        cross_side_foot_active.astype(jp.float32)
                    ),
                    "roll_potential_positive": jp.maximum(
                        roll_potential, 0.0
                    ),
                    "failed": failed_bool.astype(jp.float32),
                    "failure_severe": failure_severe.astype(jp.float32),
                    "failure_nonfinite": failure_nonfinite.astype(
                        jp.float32
                    ),
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
            reward = jp.nan_to_num(
                reward,
                nan=-reward_settings.nonfinite_termination,
                posinf=-reward_settings.nonfinite_termination,
                neginf=-reward_settings.nonfinite_termination,
            )
            rewards = {
                f"reward_{name}": jp.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                for name, value in raw_reward_terms.items()
            }
            info = {
                **state.info,
                "previous_root_x": root_x,
                "cumulative_rotation": cumulative_rotation,
                "previous_roll_potential": roll_potential,
                "previous_mismatch_potential": mismatch_potential,
                "last_action": effective_action,
                "last_policy_action": policy_action,
                "last_reference_action": next_reference_action,
                "oscillator_phase": oscillator_phase,
                "rolling_phase": rolling_phase,
                "maximum_forbidden_penetration": new_forbidden_max,
                "maximum_same_side_foot_excess": new_same_side_foot_max,
                "previous_same_side_foot_contact": same_side_foot_active,
                "root_low_step_count": root_low_step_count,
                "axis_tilt_step_count": axis_tilt_step_count,
                "forbidden_contact_step_count": (
                    forbidden_contact_step_count
                ),
                "step_count": step_count,
            }
            metrics = {
                "reward": reward,
                "reward_total": reward,
                **rewards,
                "roll_progress_rad": conservative_progress,
                "rotation_progress_rad": rotation_progress,
                "translation_progress_rad": translation_progress,
                "mismatch_progress_rad": mismatch_progress,
                "root_x_m": root_x,
                "root_y_m": root_y,
                "root_z_m": root_z,
                "lateral_drift_m": lateral_drift,
                "lateral_velocity_m_s": lateral_velocity,
                "axis_tilt_rad": axis_tilt,
                "axis_tilt_step_count": (
                    axis_tilt_step_count.astype(jp.float32)
                ),
                "root_low_active": root_low_active.astype(jp.float32),
                "root_low_step_count": (
                    root_low_step_count.astype(jp.float32)
                ),
                "shell_floor_contact_count": contacts["shell_floor_count"],
                "foot_floor_contact_count": contacts["foot_floor_count"],
                "same_side_foot_contact_count": (
                    contacts["same_side_foot_count"]
                ),
                "same_side_foot_penetration_m": (
                    contacts["same_side_foot_depth"]
                ),
                "same_side_foot_contact_active": (
                    same_side_foot_active.astype(jp.float32)
                ),
                "same_side_foot_contact_start": (
                    same_side_foot_start.astype(jp.float32)
                ),
                "forbidden_contact_count": contacts["forbidden_count"],
                "forbidden_penetration_m": contacts["forbidden_depth"],
                "forbidden_contact_step_count": (
                    forbidden_contact_step_count.astype(jp.float32)
                ),
                "cross_side_foot_contact_count": (
                    contacts["cross_side_foot_count"]
                ),
                "action_rms": jp.sqrt(jp.mean(jp.square(effective_action))),
                "action_rate_rms": jp.sqrt(action_rate),
                "startup_action_ramp": action_ramp,
                "normalized_torque_rms": jp.sqrt(torque_cost),
                "reference_action_rms": jp.sqrt(
                    jp.mean(jp.square(cem_action))
                ),
                "residual_action_rms": jp.sqrt(
                    jp.mean(jp.square(policy_action))
                ),
                "reference_weight": reference_weight_value,
                "residual_gain": residual_gain_value,
                "rolling_phase_rad": rolling_phase,
                "oscillator_phase_rad": oscillator_phase,
                "phase_error_rad": phase_error,
                "oscillator_rate_rad_s": oscillator_rate,
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
                "failure_lateral_drift": (
                    failure_lateral_drift.astype(jp.float32)
                ),
                "failure_axis_tilt": failure_axis_tilt.astype(jp.float32),
                "failure_forbidden_depth": (
                    failure_forbidden_depth.astype(jp.float32)
                ),
                "failure_forbidden_contact": (
                    failure_forbidden_contact.astype(jp.float32)
                ),
            }
            observation = self._observation(
                data,
                effective_action,
                contacts,
                axis_tilt=axis_tilt,
                reference_action_value=next_reference_action,
                oscillator_phase=oscillator_phase,
                action_ramp=action_ramp,
            )
            metrics = {
                name: jp.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                )
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

        def _reference_action_8d(self, oscillator_phase):
            planar_action = reference_action(
                jp,
                oscillator_phase,
                reference_settings,
                compact_ctrl=self.planar_compact,
                action_scales=self.planar_action_scales,
                joint_low=self.planar_joint_low,
                joint_high=self.planar_joint_high,
            )
            return duplicate_planar_action_3d(jp, planar_action)

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
            geom1_shell = self._geom_in_ids(geom1, self.shell_geom_ids)
            geom2_shell = self._geom_in_ids(geom2, self.shell_geom_ids)
            foot_foot = valid & geom1_foot & geom2_foot
            same_side_foot = valid & (
                _pair_matches(
                    geom1,
                    geom2,
                    self.front_left_foot_geom_id,
                    self.rear_left_foot_geom_id,
                )
                | _pair_matches(
                    geom1,
                    geom2,
                    self.front_right_foot_geom_id,
                    self.rear_right_foot_geom_id,
                )
            )
            cross_side_foot = foot_foot & (~same_side_foot)
            shell_floor = ground & (geom1_shell | geom2_shell)
            foot_floor = ground & (geom1_foot | geom2_foot)
            forbidden = valid & (~ground) & (~same_side_foot)
            return {
                "shell_floor_count": jp.sum(shell_floor).astype(jp.float32),
                "foot_floor_count": jp.sum(foot_floor).astype(jp.float32),
                "same_side_foot_count": jp.sum(same_side_foot).astype(
                    jp.float32
                ),
                "same_side_foot_depth": jp.max(
                    jp.where(same_side_foot, -distance, 0.0)
                ),
                "forbidden_count": jp.sum(forbidden).astype(jp.float32),
                "forbidden_depth": jp.max(
                    jp.where(forbidden, -distance, 0.0)
                ),
                "cross_side_foot_count": jp.sum(cross_side_foot).astype(
                    jp.float32
                ),
            }

        def _body_axes(self, data):
            rotation = jp.reshape(data.xmat[self.torso_body_id], (3, 3))
            return rotation[:, 1], rotation[:, 2]

        def _rolling_axis_tilt(self, data):
            body_y_axis, _ = self._body_axes(data)
            alignment = jp.clip(jp.abs(body_y_axis[1]), 0.0, 1.0)
            return jp.arccos(alignment)

        def _observation(
            self,
            data,
            last_action,
            contacts,
            *,
            axis_tilt,
            reference_action_value,
            oscillator_phase,
            action_ramp,
        ):
            body_y_axis, body_z_axis = self._body_axes(data)
            root_position_features = jp.asarray([data.qpos[2], data.qpos[1]])
            root_linear_velocity = data.qvel[:3]
            root_angular_velocity = data.qvel[3:6]
            joint_position = data.qpos[self.joint_qpos_indices]
            joint_velocity = data.qvel[self.joint_dof_indices]
            contact_features = jp.stack(
                (
                    (contacts["shell_floor_count"] > 0).astype(jp.float32),
                    (contacts["foot_floor_count"] > 0).astype(jp.float32),
                    (contacts["same_side_foot_count"] > 0).astype(
                        jp.float32
                    ),
                    (contacts["forbidden_count"] > 0).astype(jp.float32),
                    (contacts["cross_side_foot_count"] > 0).astype(
                        jp.float32
                    ),
                    1000.0 * contacts["same_side_foot_depth"],
                    1000.0 * contacts["forbidden_depth"],
                    axis_tilt,
                )
            )
            return jp.concatenate(
                (
                    root_position_features,
                    body_y_axis,
                    body_z_axis,
                    root_linear_velocity,
                    root_angular_velocity,
                    joint_position,
                    joint_velocity,
                    last_action,
                    contact_features,
                    reference_settings.reference_weight
                    * reference_action_value,
                    jp.asarray(
                        (
                            reference_settings.reference_weight
                            * jp.sin(oscillator_phase),
                            reference_settings.reference_weight
                            * jp.cos(oscillator_phase),
                            action_ramp,
                            reference_settings.reference_weight,
                            reference_settings.residual_gain,
                        )
                    ),
                )
            )

    return CurlRobot3DMJXEnv()


def _duration_to_steps(duration_s: float, control_timestep: float) -> int:
    return max(1, int(np.ceil(duration_s / control_timestep)))


def _pair_matches(geom1, geom2, first_id: int, second_id: int):
    return (
        ((geom1 == first_id) & (geom2 == second_id))
        | ((geom1 == second_id) & (geom2 == first_id))
    )
