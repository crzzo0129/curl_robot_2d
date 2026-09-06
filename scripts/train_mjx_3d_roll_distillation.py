#!/usr/bin/env python3
"""Distill the privileged 3-D rolling teacher into the real Pupper ABI.

The teacher keeps its 65-value simulator observation and eight residual
channels.  Teacher-controlled rollouts supervise a student that receives the
real controller's newest-first 36 x 20 observation history and predicts the
complete 12-motor normalized command.  The target is the *effective* CEM plus
residual action, never the residual alone.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.deployment_rolling_3d import (
    CONTROLLER_JOINT_NAMES_3D,
    HARDWARE_IMU_PUBLISH_FREQUENCY_HZ_3D,
    HARDWARE_POLICY_FREQUENCY_HZ_3D,
    ROLLING_CONTROLLER_ACTION_MASK_3D,
    ROLLING_CONTROLLER_ACTION_SIZE_3D,
    ROLLING_DEPLOY_OBSERVATION_HISTORY_3D,
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
    controller_action_to_effective_action_3d,
    effective_action_to_controller_action_3d,
    initial_rolling_deploy_history_3d,
    push_rolling_deploy_frame_3d,
    rolling_deploy_frame_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    ROLLINGQUAD_GEOMETRIES_3D,
    cem_controller_path_3d,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from curl_robot_2d_mjx.startup_rolling_3d import add_stand_startup_arguments, with_stand_startup
from curl_robot_2d_mjx.rolling_diagnostics_3d import (
    lateral_state_features_3d,
    save_lateral_trace,
)
from curl_robot_2d_mjx.randomization_3d import (
    RollingStudentDeployDomainRandomization,
)


PRESETS = {
    "smoke": {
        "envs": 32,
        "stats_steps": 8,
        "train_steps": 32,
        "dagger_steps": 16,
        "eval_envs": 16,
    },
    "h200": {
        "envs": 2048,
        "stats_steps": 500,
        "train_steps": 20_000,
        "dagger_steps": 10_000,
        "eval_envs": 256,
    },
}

TEACHER_HIDDEN_LAYERS = (256, 256, 128)
TEACHER_INITIAL_STD = 0.10
STUDENT_HIDDEN_LAYERS = (512, 256, 128)
# The accepted rolling teacher controls hip/knee only.  Keeping the four
# abduction scales at exactly zero makes the exported controller hold the
# compact pose even if the student's nominally-zero outputs have fit error.
CONTROLLER_ACTION_SCALES = np.asarray((0.0, 0.8, 1.2) * 4)

EVALUATION_FAILURE_METRICS = (
    "failure_nonfinite",
    "failure_nonfinite_action",
    "failure_nonfinite_physics",
    "failure_root_low",
    "failure_root_high",
    "failure_lateral_drift",
    "failure_axis_tilt",
    "failure_forbidden_depth",
    "failure_forbidden_contact",
)


def dagger_teacher_probability(step, total_steps, start, end):
    """Linear expert-intervention schedule, including both endpoints."""

    fraction = step / max(total_steps - 1, 1)
    return start + fraction * (end - start)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Distill the accepted rolling teacher to real observations"
    )
    parser.add_argument("teacher", type=Path, help="teacher params_best")
    parser.add_argument(
        "--geometry",
        choices=ROLLINGQUAD_GEOMETRIES_3D,
        default="rollingquad_2",
        help="collision geometry used by both the teacher and Student rollout",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="load --restore-student, skip all training, save lateral diagnostics only",
    )
    parser.add_argument(
        "--record-diagnostics", action="store_true",
        help="record signed lateral state and same-state teacher action errors during evaluation",
    )
    parser.add_argument(
        "--eval-seed", type=int,
        help="independent evaluation reset seed; defaults to seed+100000 in eval-only mode",
    )
    parser.add_argument(
        "--restore-student",
        type=Path,
        help=(
            "existing student_params checkpoint; reuse its normalizer and "
            "skip statistics plus behavior cloning, then run DAgger"
        ),
    )
    parser.add_argument(
        "--preset", choices=tuple(PRESETS), default="smoke"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument(
        "--controller",
        type=Path,
        help="CEM reference; defaults to the reference for --geometry",
    )
    parser.add_argument(
        "--lateral-drift-diagnostic-only",
        action="store_true",
        help=(
            "measure the 0.20 m lateral envelope without terminating or "
            "counting it as a physical failure"
        ),
    )
    parser.add_argument("--envs", type=int)
    parser.add_argument("--stats-steps", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--dagger-steps", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    add_stand_startup_arguments(parser)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dagger-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--dagger-teacher-start-probability", type=float, default=0.25
    )
    parser.add_argument(
        "--dagger-teacher-end-probability", type=float, default=0.0
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=list(STUDENT_HIDDEN_LAYERS),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--deploy-dr",
        action="store_true",
        help=(
            "continue an existing Student with deploy-style physics, "
            "calibration, latency and deadline randomization"
        ),
    )
    parser.add_argument(
        "--deploy-dr-strength",
        type=float,
        default=1.0,
        help=(
            "fraction of the train_ppo_deploy.py DR envelope; use "
            "0.25, 0.50 and 1.0 as a continuation curriculum"
        ),
    )
    parser.add_argument(
        "--minimum-closed-loop-turns",
        type=float,
        default=5.0,
        help="minimum net turns required for a closed-loop success",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa", "disable"),
        default="disable",
    )
    args = parser.parse_args(argv)
    if args.controller is None:
        args.controller = cem_controller_path_3d(args.geometry)
    if args.eval_only:
        if args.restore_student is None:
            parser.error("--eval-only requires --restore-student")
        args.record_diagnostics = True
        if args.eval_seed is None:
            args.eval_seed = args.seed + 100_000
    for name in (
        "envs",
        "stats_steps",
        "train_steps",
        "dagger_steps",
        "eval_envs",
    ):
        if getattr(args, name) is None:
            setattr(args, name, PRESETS[args.preset][name])
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.episode_length < 1 or args.log_every < 1:
        parser.error("--episode-length and --log-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if (
        not math.isfinite(args.dagger_learning_rate)
        or args.dagger_learning_rate <= 0.0
    ):
        parser.error("--dagger-learning-rate must be finite and positive")
    for name in (
        "dagger_teacher_start_probability",
        "dagger_teacher_end_probability",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error(
                f"--{name.replace('_', '-')} must be between zero and one"
            )
    if (
        not math.isfinite(args.observation_noise_scale)
        or args.observation_noise_scale < 0.0
    ):
        parser.error("--observation-noise-scale must be finite and nonnegative")
    if (
        not math.isfinite(args.deploy_dr_strength)
        or not 0.0 <= args.deploy_dr_strength <= 1.0
    ):
        parser.error("--deploy-dr-strength must be between zero and one")
    if args.deploy_dr and args.restore_student is None:
        parser.error("--deploy-dr requires --restore-student")
    if (
        not math.isfinite(args.minimum_closed_loop_turns)
        or args.minimum_closed_loop_turns < 0.0
    ):
        parser.error(
            "--minimum-closed-loop-turns must be finite and nonnegative"
        )
    if not args.teacher.is_file():
        parser.error(f"teacher checkpoint does not exist: {args.teacher}")
    if args.restore_student is not None and not args.restore_student.is_file():
        parser.error(
            f"student checkpoint does not exist: {args.restore_student}"
        )
    if not args.controller.is_file():
        parser.error(f"CEM controller does not exist: {args.controller}")
    if (
        args.out.exists()
        and any(args.out.iterdir())
        and not args.allow_existing_output
    ):
        parser.error(
            f"output directory is not empty: {args.out}; choose a new path"
        )
    return args


def student_controller_config(model, *, action_scales=CONTROLLER_ACTION_SCALES):
    """Build metadata consumed by export_rtneural and neural_controller."""

    import mujoco

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
    if key_id < 0:
        raise ValueError("missing compact keyframe")
    actuator_ids = []
    for joint_name in CONTROLLER_JOINT_NAMES_3D:
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{joint_name}_servo",
        )
        if actuator_id < 0:
            raise ValueError(f"missing actuator: {joint_name}_servo")
        actuator_ids.append(int(actuator_id))
    actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
    compact = np.asarray(model.key_ctrl[key_id, actuator_ids])
    limits = np.asarray(model.actuator_ctrlrange[actuator_ids])
    kp = np.asarray(model.actuator_gainprm[actuator_ids, 0])
    kd = -np.asarray(model.actuator_biasprm[actuator_ids, 2])
    return {
        "use_imu": True,
        "control_orientation": False,
        "observation_history": ROLLING_DEPLOY_OBSERVATION_HISTORY_3D,
        "kp": float(np.median(kp)),
        "kd": float(np.median(kd)),
        "action_scale": [float(value) for value in action_scales],
        "default_joint_pos": [float(value) for value in compact],
        "joint_lower_limits": [float(value) for value in limits[:, 0]],
        "joint_upper_limits": [float(value) for value in limits[:, 1]],
    }


def _task(
    *,
    episode_length,
    direct_effective_action=False,
    geometry="rollingquad_2",
    lateral_drift_diagnostic_only=False,
):
    return physics_profile_3d(
        "cg20",
        Rolling3DConfig(
            geometry=geometry,
            episode_length=episode_length,
            reset_joint_noise_rad=0.005,
            reset_velocity_noise=0.005,
            reset_root_velocity_noise=0.0,
            reset_pair_differential_scale=None,
            reset_axis_tilt_noise_rad=0.0,
            reference_phase_rate_scale=1.0,
            reference_action_scale=1.0,
            reference_ramp_start_scale=0.0,
            reference_ramp_duration_s=0.25,
            residual_pair_differential_scale=(
                None if direct_effective_action else 0.25
            ),
            # DAgger queries the privileged teacher on states visited by the
            # direct-action student.  The deployable student never reads
            # state.obs, but both environments must retain the teacher's 65-D
            # observation ABI so a direct state can be labelled by teacher_env.
            explicit_phase_observation=True,
            direct_effective_action=direct_effective_action,
            lateral_drift_termination=(
                not lateral_drift_diagnostic_only
            ),
        ),
    )


def main(argv=None):
    args = parse_args(argv)
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=False,
        xla_triton=False,
        mujoco_gl=args.mujoco_gl,
        verbose=False,
    )

    import jax
    import jax.numpy as jp
    from brax.io import model as model_io
    from brax.envs.wrappers import training as brax_training_wrappers
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    import flax.linen as linen
    import optax

    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d
    from curl_robot_2d_mjx.randomization_3d import (
        make_student_deploy_domain_randomization_fn_3d,
    )
    from scripts.export_rtneural import convert as convert_rtneural
    from scripts.train_mjx_3d_residual_ppo import (
        _zero_centered_residual_network_factory,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    reference = load_cem_reference(
        args.controller,
        reference_weight=1.0,
        minimum_residual_gain=0.15,
    )
    teacher_task = with_stand_startup(
        _task(
            episode_length=args.episode_length,
            geometry=args.geometry,
            lateral_drift_diagnostic_only=(
                args.lateral_drift_diagnostic_only
            ),
        ),
        args,
    )
    print(f"[startup] reset={teacher_task.reset_pose} "
          f"rolling_start={teacher_task.rolling_start_time_s:g}s; "
          "student must produce ALL startup actions (no reference assistance)", flush=True)
    teacher_env = make_brax_env_3d(
        teacher_task,
        cem_reference=reference,
        seed=args.seed,
    )
    if teacher_env.observation_size != 65 or teacher_env.action_size != 8:
        raise RuntimeError(
            "teacher contract mismatch: expected obs=65 action=8, got "
            f"obs={teacher_env.observation_size} action={teacher_env.action_size}"
        )

    teacher_factory = _zero_centered_residual_network_factory(
        TEACHER_HIDDEN_LAYERS,
        "elu",
        TEACHER_INITIAL_STD,
        reflection_equivariant=False,
    )
    teacher_networks = teacher_factory(
        teacher_env.observation_size,
        teacher_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    teacher_params = model_io.load_params(args.teacher)
    teacher_policy = ppo_networks.make_inference_fn(teacher_networks)(
        teacher_params, deterministic=True
    )
    teacher_policy_batch = jax.jit(
        jax.vmap(lambda observation, key: teacher_policy(observation, key)[0])
    )
    reset_batch = jax.jit(jax.vmap(teacher_env.reset))
    step_batch = jax.jit(jax.vmap(teacher_env.step))

    controller_qpos_indices = []
    import mujoco

    for name in CONTROLLER_JOINT_NAMES_3D:
        joint_id = mujoco.mj_name2id(
            teacher_env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise RuntimeError(f"missing deployment joint: {name}")
        controller_qpos_indices.append(
            int(teacher_env.mj_model.jnt_qposadr[joint_id])
        )
    controller_qpos_indices = jp.asarray(controller_qpos_indices)
    config = student_controller_config(teacher_env.mj_model)
    config["training_reset_pose"] = teacher_task.reset_pose
    config["startup_actions_provided_by"] = "student_network"
    compact_position = jp.asarray(config["default_joint_pos"])
    frame_sigma = jp.concatenate(
        (
            jp.full((3,), 0.20),
            jp.full((3,), 0.05),
            jp.zeros((6,)),
            jp.full((12,), 0.01),
            jp.zeros((12,)),
        )
    )
    initial_history = initial_rolling_deploy_history_3d(jp)
    zero_encoder_bias = jp.zeros(
        (args.envs, ROLLING_CONTROLLER_ACTION_SIZE_3D)
    )

    def deployment_observation(
        state,
        history,
        previous_controller_action,
        encoder_bias,
        noise_key,
        noise_scale,
    ):
        data = state.pipeline_state
        rotation = data.xmat[:, teacher_env.torso_body_id].reshape((-1, 3, 3))
        rotation_t = jp.swapaxes(rotation, -1, -2)
        angular_world = data.cvel[:, teacher_env.torso_body_id, :3]
        angular_body = jp.einsum("bij,bj->bi", rotation_t, angular_world)
        gravity_world = jp.broadcast_to(
            jp.asarray((0.0, 0.0, -1.0)), angular_body.shape
        )
        projected_gravity = jp.einsum(
            "bij,bj->bi", rotation_t, gravity_world
        )
        joint_offset = (
            data.qpos[:, controller_qpos_indices] - compact_position
            + encoder_bias
        )
        frame = rolling_deploy_frame_3d(
            jp,
            angular_velocity_body=angular_body,
            projected_gravity=projected_gravity,
            joint_position_offset=joint_offset,
            last_action=previous_controller_action,
        )
        if noise_scale > 0.0:
            frame = frame + noise_scale * frame_sigma * jax.random.normal(
                noise_key, frame.shape
            )
        return push_rolling_deploy_frame_3d(jp, history, frame)

    deployment_observation = jax.jit(
        deployment_observation, static_argnums=(5,)
    )

    class StudentPolicy(linen.Module):
        hidden_layers: tuple[int, ...]

        @linen.compact
        def __call__(self, observation):
            value = observation
            for index, width in enumerate(self.hidden_layers):
                value = linen.Dense(width, name=f"hidden_{index}")(value)
                value = linen.elu(value)
            value = linen.Dense(
                ROLLING_CONTROLLER_ACTION_SIZE_3D,
                name="location",
            )(value)
            return jp.tanh(value) * jp.asarray(
                ROLLING_CONTROLLER_ACTION_MASK_3D
            )

    student = StudentPolicy(tuple(args.hidden_layers))
    rng = jax.random.PRNGKey(args.seed)
    restored_student_checkpoint = (
        model_io.load_params(args.restore_student)
        if args.restore_student is not None
        else None
    )

    def reset_rollout(rng_key, batch_size):
        reset_keys = jax.random.split(rng_key, batch_size)
        state = reset_batch(reset_keys)
        history = jp.broadcast_to(
            initial_history, (batch_size, ROLLING_DEPLOY_OBSERVATION_SIZE_3D)
        )
        previous_controller_action = jp.zeros(
            (batch_size, ROLLING_CONTROLLER_ACTION_SIZE_3D)
        )
        return state, history, previous_controller_action

    rng, reset_key = jax.random.split(rng)
    if restored_student_checkpoint is None:
        state, history, previous_controller_action = reset_rollout(
            reset_key, args.envs
        )
    observation_sum = jp.zeros((ROLLING_DEPLOY_OBSERVATION_SIZE_3D,))
    observation_square_sum = jp.zeros_like(observation_sum)
    observation_count = 0

    print(
        "[distillation contract]\n"
        f"  teacher={args.teacher.resolve()} obs=65 residual_action=8\n"
        f"  student_obs=36x20={ROLLING_DEPLOY_OBSERVATION_SIZE_3D} "
        "student_action=12 effective_motor_command\n"
        f"  simulation=50Hz hardware={HARDWARE_POLICY_FREQUENCY_HZ_3D:g}Hz\n"
        f"  stats_steps={args.stats_steps} bc_steps={args.train_steps} "
        f"dagger_steps={args.dagger_steps} "
        f"envs={args.envs}",
        flush=True,
    )

    stats_steps_to_run = (
        0 if restored_student_checkpoint is not None else args.stats_steps
    )
    for step in range(stats_steps_to_run):
        rng, policy_key, noise_key = jax.random.split(rng, 3)
        policy_keys = jax.random.split(policy_key, args.envs)
        history = deployment_observation(
            state,
            history,
            previous_controller_action,
            zero_encoder_bias,
            noise_key,
            args.observation_noise_scale,
        )
        observation_sum += jp.sum(history, axis=0)
        observation_square_sum += jp.sum(jp.square(history), axis=0)
        observation_count += args.envs
        teacher_action = teacher_policy_batch(state.obs, policy_keys)
        state = step_batch(state, teacher_action)
        previous_controller_action = effective_action_to_controller_action_3d(
            jp, state.info["last_action"]
        )
        if (step + 1) % args.episode_length == 0:
            rng, reset_key = jax.random.split(rng)
            state, history, previous_controller_action = reset_rollout(
                reset_key, args.envs
            )

    if restored_student_checkpoint is None:
        observation_mean = observation_sum / observation_count
        observation_variance = jp.maximum(
            observation_square_sum / observation_count
            - jp.square(observation_mean),
            1e-6,
        )
        observation_std = jp.sqrt(observation_variance)
    else:
        restored_normalizer = restored_student_checkpoint[0]
        observation_mean = jp.asarray(restored_normalizer["mean"])
        observation_std = jp.asarray(restored_normalizer["std"])
        if (
            observation_mean.shape
            != (ROLLING_DEPLOY_OBSERVATION_SIZE_3D,)
            or observation_std.shape
            != (ROLLING_DEPLOY_OBSERVATION_SIZE_3D,)
        ):
            raise RuntimeError(
                "restored student normalizer must contain 720-value mean/std"
            )
        print(
            "[restore student]\n"
            f"  checkpoint={args.restore_student.resolve()}\n"
            "  skipping observation statistics and behavior cloning",
            flush=True,
        )
    rng, init_key, reset_key = jax.random.split(rng, 3)
    student_params = (
        student.init(
            init_key,
            jp.zeros((1, ROLLING_DEPLOY_OBSERVATION_SIZE_3D)),
        )
        if restored_student_checkpoint is None
        else restored_student_checkpoint[1]
    )
    # model_io restores arrays as NumPy values, whereas freshly initialized
    # Flax parameters are JAX arrays.  Normalize both paths before using JAX's
    # indexed-update API and before initializing the DAgger optimizer.
    student_params = jax.tree_util.tree_map(jp.asarray, student_params)

    # The C++ controller writes raw network output back into the last-action
    # observation even when action_scale is zero.  Projecting the final layer
    # therefore matters: it keeps the exported RTNeural model's four locked
    # abduction outputs exactly zero, not merely close to zero on teacher data.
    from flax.core import FrozenDict, freeze, unfreeze

    params_were_frozen = isinstance(student_params, FrozenDict)
    mutable_student_params = unfreeze(student_params)
    location_params = mutable_student_params["params"]["location"]
    locked_indices = jp.asarray((0, 3, 6, 9))
    location_params["kernel"] = location_params["kernel"].at[
        :, locked_indices
    ].set(0.0)
    location_params["bias"] = location_params["bias"].at[
        locked_indices
    ].set(0.0)
    student_params = (
        freeze(mutable_student_params)
        if params_were_frozen
        else mutable_student_params
    )
    def make_train_step(current_optimizer):
        @jax.jit
        def train_step(params, opt_state, observation, target):
            normalized = (observation - observation_mean) / observation_std

            def loss_fn(current_params):
                prediction = student.apply(current_params, normalized)
                # The four abduction outputs are structurally locked at zero.
                # Report and optimize only the eight controlled hip/knee
                # channels so the BC/DAgger diagnostics are not diluted.
                error = controller_action_to_effective_action_3d(
                    jp, prediction - target
                )
                mse = jp.mean(jp.square(error))
                return mse, (
                    jp.sqrt(mse),
                    jp.max(jp.abs(error)),
                )

            (loss, diagnostics), gradients = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            updates, next_opt_state = current_optimizer.update(
                gradients, opt_state, params
            )
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, loss, diagnostics

        return train_step

    optimizer = optax.adam(args.learning_rate)
    optimizer_state = optimizer.init(student_params)
    train_step = make_train_step(optimizer)

    if restored_student_checkpoint is None:
        state, history, previous_controller_action = reset_rollout(
            reset_key, args.envs
        )
    loss_history = []
    bc_steps_to_run = (
        0 if restored_student_checkpoint is not None else args.train_steps
    )
    for step in range(bc_steps_to_run):
        rng, policy_key, noise_key = jax.random.split(rng, 3)
        policy_keys = jax.random.split(policy_key, args.envs)
        history = deployment_observation(
            state,
            history,
            previous_controller_action,
            zero_encoder_bias,
            noise_key,
            args.observation_noise_scale,
        )
        teacher_action = teacher_policy_batch(state.obs, policy_keys)
        next_state = step_batch(state, teacher_action)
        target = effective_action_to_controller_action_3d(
            jp, next_state.info["last_action"]
        )
        (
            student_params,
            optimizer_state,
            loss,
            diagnostics,
        ) = train_step(
            student_params,
            optimizer_state,
            history,
            target,
        )
        state = next_state
        previous_controller_action = target
        if (step + 1) % args.episode_length == 0:
            rng, reset_key = jax.random.split(rng)
            state, history, previous_controller_action = reset_rollout(
                reset_key, args.envs
            )
        if step == 0 or (step + 1) % args.log_every == 0:
            record = {
                "stage": "behavior_cloning",
                "step": step + 1,
                "loss": float(loss),
                "action_rmse": float(diagnostics[0]),
                "action_max_abs": float(diagnostics[1]),
            }
            loss_history.append(record)
            print(
                f"[student {step + 1:>6}/{args.train_steps}] "
                f"loss={record['loss']:.6g} "
                f"rmse={record['action_rmse']:.5f} "
                f"max={record['action_max_abs']:.5f}",
                flush=True,
            )

    # Run the distilled policy without the CEM teacher.  The environment's
    # direct mode interprets the eight hip/knee channels as the complete
    # normalized motor command, exactly matching the supervised target.
    direct_task = _task(
        episode_length=args.episode_length,
        direct_effective_action=True,
        geometry=args.geometry,
        lateral_drift_diagnostic_only=(
            args.lateral_drift_diagnostic_only
        ),
    )
    direct_task = with_stand_startup(direct_task, args)
    direct_env = make_brax_env_3d(
        direct_task,
        cem_reference=reference,
        seed=args.seed + 50_000,
    )
    if direct_env.observation_size != teacher_env.observation_size:
        raise RuntimeError(
            "DAgger contract mismatch: direct states must retain the "
            f"teacher's {teacher_env.observation_size}-D observation, got "
            f"{direct_env.observation_size}"
        )
    deploy_dr_settings = (
        RollingStudentDeployDomainRandomization().scaled(
            args.deploy_dr_strength
        )
        if args.deploy_dr
        else None
    )
    if deploy_dr_settings is not None:
        print(
            "[deploy DR]\n"
            f"  strength={args.deploy_dr_strength:g} "
            f"friction={deploy_dr_settings.sliding_friction} "
            f"torso_mass={deploy_dr_settings.torso_mass_scale} "
            f"leg_mass={deploy_dr_settings.leg_mass_scale}\n"
            f"  inertia={deploy_dr_settings.inertia_scale} "
            f"kp={deploy_dr_settings.motor_kp_scale} "
            f"kd={deploy_dr_settings.motor_kd_scale} "
            f"torque={deploy_dr_settings.motor_torque_scale}\n"
            f"  latency={deploy_dr_settings.action_latency_probabilities} "
            "for 0/20/40ms; "
            f"deadline_miss="
            f"{deploy_dr_settings.control_deadline_miss_probability:.1%} "
            f"motor_zero=±{deploy_dr_settings.motor_zero_bias_rad:.4f}rad "
            f"encoder=±{deploy_dr_settings.encoder_fixed_bias_rad:.4f}rad",
            flush=True,
        )

    def batched_direct_environment(env, batch_size, model_seed):
        """Create nominal vmap or per-environment randomized MJX calls."""

        if deploy_dr_settings is None:
            return (
                None,
                jax.jit(jax.vmap(env.reset)),
                jax.jit(jax.vmap(env.step)),
            )
        randomization_fn = make_student_deploy_domain_randomization_fn_3d(
            deploy_dr_settings,
            torso_body_id=env.torso_body_id,
        )
        model_keys = jax.random.split(
            jax.random.PRNGKey(model_seed), batch_size
        )
        wrapper = brax_training_wrappers.DomainRandomizationVmapWrapper(
            env,
            lambda model: randomization_fn(model, model_keys),
        )
        return wrapper, jax.jit(wrapper.reset), jax.jit(wrapper.step)

    (
        direct_train_wrapper,
        direct_reset_batch,
        direct_step_raw_batch,
    ) = batched_direct_environment(
        direct_env, args.envs, args.seed + 60_000
    )

    def make_active_batch_step(raw_step):
        @jax.jit
        def step_if_active(state, action, active):
            candidate = raw_step(state, action)

            def choose(current, next_value):
                mask = jp.reshape(
                    active,
                    active.shape + (1,) * (next_value.ndim - active.ndim),
                )
                return jp.where(mask, next_value, current)

            return jax.tree_util.tree_map(choose, state, candidate)

        return step_if_active

    student_policy_batch = jax.jit(
        lambda params, observation: student.apply(
            params,
            (observation - observation_mean) / observation_std,
        )
    )

    def make_episode_randomization(batch_size):
        if deploy_dr_settings is None:
            @jax.jit
            def attach_nominal(state, key):
                del key
                return state

            @jax.jit
            def transport_nominal(state, action, key):
                del key
                return state, action, jp.zeros((), dtype=jp.float32)

            return attach_nominal, transport_nominal

        probabilities = jp.asarray(
            deploy_dr_settings.action_latency_probabilities
        )

        @jax.jit
        def attach_randomization(state, key):
            latency_key, motor_key, encoder_key = jax.random.split(key, 3)
            info = {
                **state.info,
                "deploy_action_queue": jp.zeros(
                    (batch_size, 3, ROLLING_CONTROLLER_ACTION_SIZE_3D)
                ),
                "deploy_applied_action": jp.zeros(
                    (batch_size, ROLLING_CONTROLLER_ACTION_SIZE_3D)
                ),
                "deploy_latency_steps": jax.random.choice(
                    latency_key,
                    3,
                    shape=(batch_size,),
                    p=probabilities,
                ),
                "motor_zero_bias_ctrl": jax.random.uniform(
                    motor_key,
                    (batch_size, ROLLING_CONTROLLER_ACTION_SIZE_3D),
                    minval=-deploy_dr_settings.motor_zero_bias_rad,
                    maxval=deploy_dr_settings.motor_zero_bias_rad,
                ),
                "encoder_bias": jax.random.uniform(
                    encoder_key,
                    (batch_size, ROLLING_CONTROLLER_ACTION_SIZE_3D),
                    minval=-deploy_dr_settings.encoder_fixed_bias_rad,
                    maxval=deploy_dr_settings.encoder_fixed_bias_rad,
                ),
            }
            return state.replace(info=info)

        @jax.jit
        def transport_randomized(state, action, key):
            queue = jp.concatenate(
                (action[:, None, :], state.info["deploy_action_queue"][:, :-1]),
                axis=1,
            )
            delayed = jp.take_along_axis(
                queue,
                state.info["deploy_latency_steps"][:, None, None],
                axis=1,
            )[:, 0, :]
            deadline_missed = jax.random.uniform(
                key, (batch_size,)
            ) < deploy_dr_settings.control_deadline_miss_probability
            applied = jp.where(
                deadline_missed[:, None],
                state.info["deploy_applied_action"],
                delayed,
            )
            info = {
                **state.info,
                "deploy_action_queue": queue,
                "deploy_applied_action": applied,
            }
            return (
                state.replace(info=info),
                applied,
                jp.mean(deadline_missed.astype(jp.float32)),
            )

        return attach_randomization, transport_randomized

    attach_train_episode_randomization, transport_train_action = (
        make_episode_randomization(args.envs)
    )

    @jax.jit
    def reset_finished_rollouts(
        current_state,
        reset_state,
        current_history,
        current_previous_action,
    ):
        finished = current_state.done > 0.5

        def choose_reset(reset_value, current_value):
            mask_shape = finished.shape + (1,) * (
                current_value.ndim - finished.ndim
            )
            return jp.where(
                jp.reshape(finished, mask_shape),
                reset_value,
                current_value,
            )

        next_state = jax.tree_util.tree_map(
            choose_reset, reset_state, current_state
        )
        next_history = jp.where(
            finished[:, None],
            jp.broadcast_to(initial_history, current_history.shape),
            current_history,
        )
        next_previous_action = jp.where(
            finished[:, None],
            jp.zeros_like(current_previous_action),
            current_previous_action,
        )
        return (
            next_state,
            next_history,
            next_previous_action,
            jp.mean(finished.astype(jp.float32)),
        )

    # Online DAgger: visit states under the current student (with a decaying
    # amount of expert intervention), ask the privileged teacher for the
    # complete effective command at those exact states, and update the student
    # on that label before moving on.  This attacks the covariate shift that
    # ordinary teacher-forced behavior cloning cannot see.
    if not args.eval_only:
        dagger_optimizer = optax.adam(args.dagger_learning_rate)
        dagger_optimizer_state = dagger_optimizer.init(student_params)
        dagger_train_step = make_train_step(dagger_optimizer)
        rng, dagger_reset_key, dagger_episode_key = jax.random.split(rng, 3)
        dagger_reset_keys = jax.random.split(dagger_reset_key, args.envs)
        dagger_state = attach_train_episode_randomization(
            direct_reset_batch(dagger_reset_keys), dagger_episode_key
        )
        dagger_history = jp.broadcast_to(
            initial_history,
            (args.envs, ROLLING_DEPLOY_OBSERVATION_SIZE_3D),
        )
        dagger_previous_controller_action = jp.zeros(
            (args.envs, ROLLING_CONTROLLER_ACTION_SIZE_3D)
        )
    dagger_loss_history = []
    if args.eval_only:
        print("[evaluation only] BC=0 DAgger=0; no checkpoint updates", flush=True)
    else:
        print(
            "[DAgger]\n"
            f"  steps={args.dagger_steps} lr={args.dagger_learning_rate:g} "
            "teacher_intervention="
            f"{args.dagger_teacher_start_probability:.1%}->"
            f"{args.dagger_teacher_end_probability:.1%}\n"
            f"  deploy_dr={args.deploy_dr} "
            f"strength={args.deploy_dr_strength:g}", flush=True,
        )

    for step in range(0 if args.eval_only else args.dagger_steps):
        (
            rng,
            policy_key,
            noise_key,
            mixture_key,
            transport_key,
            reset_key,
            reset_episode_key,
        ) = (
            jax.random.split(rng, 7)
        )
        policy_keys = jax.random.split(policy_key, args.envs)
        dagger_history = deployment_observation(
            dagger_state,
            dagger_history,
            dagger_previous_controller_action,
            dagger_state.info.get(
                "encoder_bias", zero_encoder_bias
            ),
            noise_key,
            args.observation_noise_scale,
        )
        student_controller_action = student_policy_batch(
            student_params, dagger_history
        )

        # teacher_env and direct_env share the same MJX/state/info structure.
        # teacher_env.step is evaluated on a copy solely to recover the full
        # CEM+residual action; its next physics state is never used by DAgger.
        teacher_residual_action = teacher_policy_batch(
            dagger_state.obs, policy_keys
        )
        teacher_label_state = step_batch(
            dagger_state, teacher_residual_action
        )
        teacher_controller_action = (
            effective_action_to_controller_action_3d(
                jp, teacher_label_state.info["last_action"]
            )
        )
        (
            student_params,
            dagger_optimizer_state,
            loss,
            diagnostics,
        ) = dagger_train_step(
            student_params,
            dagger_optimizer_state,
            dagger_history,
            teacher_controller_action,
        )

        teacher_probability = dagger_teacher_probability(
            step,
            args.dagger_steps,
            args.dagger_teacher_start_probability,
            args.dagger_teacher_end_probability,
        )
        use_teacher = jax.random.bernoulli(
            mixture_key,
            teacher_probability,
            (args.envs,),
        )
        behavior_controller_action = jp.where(
            use_teacher[:, None],
            teacher_controller_action,
            student_controller_action,
        )
        (
            dagger_state,
            applied_controller_action,
            deadline_miss_rate,
        ) = transport_train_action(
            dagger_state, behavior_controller_action, transport_key
        )
        behavior_effective_action = controller_action_to_effective_action_3d(
            jp, applied_controller_action
        )
        dagger_state = direct_step_raw_batch(
            dagger_state, behavior_effective_action
        )
        dagger_previous_controller_action = behavior_controller_action

        reset_keys = jax.random.split(reset_key, args.envs)
        reset_state = attach_train_episode_randomization(
            direct_reset_batch(reset_keys), reset_episode_key
        )
        (
            dagger_state,
            dagger_history,
            dagger_previous_controller_action,
            reset_rate,
        ) = reset_finished_rollouts(
            dagger_state,
            reset_state,
            dagger_history,
            dagger_previous_controller_action,
        )

        if step == 0 or (step + 1) % args.log_every == 0:
            record = {
                "stage": "dagger",
                "step": step + 1,
                "loss": float(loss),
                "action_rmse": float(diagnostics[0]),
                "action_max_abs": float(diagnostics[1]),
                "teacher_probability": float(teacher_probability),
                "teacher_fraction": float(jp.mean(use_teacher)),
                "deadline_miss_rate": float(deadline_miss_rate),
                "reset_rate": float(reset_rate),
            }
            dagger_loss_history.append(record)
            loss_history.append(record)
            print(
                f"[dagger {step + 1:>6}/{args.dagger_steps}] "
                f"loss={record['loss']:.6g} "
                f"rmse={record['action_rmse']:.5f} "
                f"max={record['action_max_abs']:.5f} "
                f"expert={record['teacher_fraction']:.1%} "
                f"miss={record['deadline_miss_rate']:.1%} "
                f"reset={record['reset_rate']:.1%}",
                flush=True,
            )

    direct_eval_env = make_brax_env_3d(
        direct_task,
        cem_reference=reference,
        seed=args.seed + 70_000,
    )
    (
        direct_eval_wrapper,
        direct_eval_reset_batch,
        direct_eval_step_raw_batch,
    ) = batched_direct_environment(
        direct_eval_env,
        args.eval_envs,
        args.eval_seed if args.eval_seed is not None else args.seed + 80_000,
    )
    direct_eval_step_batch = make_active_batch_step(
        direct_eval_step_raw_batch
    )
    attach_eval_episode_randomization, transport_eval_action = (
        make_episode_randomization(args.eval_envs)
    )
    rng, eval_reset_key = jax.random.split(rng)
    if args.eval_seed is not None:
        eval_reset_key = jax.random.PRNGKey(args.eval_seed)
    eval_reset_key, eval_episode_key = jax.random.split(eval_reset_key)
    eval_reset_keys = jax.random.split(eval_reset_key, args.eval_envs)
    eval_state = attach_eval_episode_randomization(
        direct_eval_reset_batch(eval_reset_keys), eval_episode_key
    )
    eval_history = jp.broadcast_to(
        initial_history,
        (args.eval_envs, ROLLING_DEPLOY_OBSERVATION_SIZE_3D),
    )
    eval_previous_controller_action = jp.zeros(
        (args.eval_envs, ROLLING_CONTROLLER_ACTION_SIZE_3D)
    )
    eval_active = jp.ones((args.eval_envs,), dtype=jp.bool_)
    eval_failed = jp.zeros_like(eval_active)
    eval_non_lateral_failed = jp.zeros_like(eval_active)
    eval_steps = jp.zeros((args.eval_envs,), dtype=jp.int32)
    eval_roll_progress = jp.zeros((args.eval_envs,))
    eval_failure_flags = {
        name: jp.zeros_like(eval_active)
        for name in EVALUATION_FAILURE_METRICS
    }
    abduction_square_sum = jp.asarray(0.0)
    abduction_sample_count = jp.asarray(0, dtype=jp.int32)
    abduction_max_abs = jp.asarray(0.0)
    abduction_indices = jp.asarray((0, 3, 6, 9))
    eval_deadline_miss_sum = jp.asarray(0.0)
    diagnostic_frames = []
    diagnostic_rng = jax.random.PRNGKey(args.seed + 200_000)

    @jax.jit
    def diagnostic_frame(before, after, active, student_action, teacher_action, label_valid, turns):
        def state_values(state):
            data = state.pipeline_state
            rotation = data.xmat[:, direct_env.torso_body_id].reshape((-1, 3, 3))
            return lateral_state_features_3d(
                jp, data.qpos, data.qvel, rotation, state.info["initial_root_y"]
            )

        return {
            "time_s": before.pipeline_state.time,
            "next_time_s": after.pipeline_state.time,
            "active": active,
            **state_values(before),
            **{f"next_{name}": value for name, value in state_values(after).items()},
            "student_action": student_action,
            "teacher_action": teacher_action,
            "teacher_label_valid": label_valid,
            "turns": turns,
            "failed": after.metrics["failed"] > 0.5,
            "lateral_failed": after.metrics["failure_lateral_drift"] > 0.5,
        }

    if args.record_diagnostics:
        print(
            "[lateral diagnostics] teacher queries only; no interventions. "
            "Recording active transitions including termination.", flush=True,
        )

    for eval_step in range(args.episode_length):
        # Nominal acceptance remains noise-free.  A deploy-DR evaluation uses
        # the same observation corruption that the continuation stage sees.
        rng, eval_noise_key, eval_transport_key = jax.random.split(rng, 3)
        eval_encoder_bias = eval_state.info.get(
            "encoder_bias",
            jp.zeros(
                (args.eval_envs, ROLLING_CONTROLLER_ACTION_SIZE_3D)
            ),
        )
        eval_history = deployment_observation(
            eval_state,
            eval_history,
            eval_previous_controller_action,
            eval_encoder_bias,
            eval_noise_key,
            args.observation_noise_scale if args.deploy_dr else 0.0,
        )
        controller_action = student_policy_batch(
            student_params, eval_history
        )
        abduction_action = jp.take(
            controller_action, abduction_indices, axis=-1
        )
        active_abduction = jp.where(
            eval_active[:, None], abduction_action, 0.0
        )
        abduction_square_sum += jp.sum(jp.square(active_abduction))
        abduction_sample_count += 4 * jp.sum(eval_active.astype(jp.int32))
        abduction_max_abs = jp.maximum(
            abduction_max_abs,
            jp.max(jp.abs(active_abduction)),
        )
        raw_effective_action = controller_action_to_effective_action_3d(
            jp, controller_action
        )
        if args.record_diagnostics:
            diagnostic_rng, label_key = jax.random.split(diagnostic_rng)
            label_keys = jax.random.split(label_key, args.eval_envs)
            label_residual = teacher_policy_batch(eval_state.obs, label_keys)
            # Query a copy of the same PRE-step student state, exactly as in
            # DAgger. Never feed the resulting physics state into the rollout.
            label_state = step_batch(eval_state, label_residual)
            label_action = label_state.info["last_action"]
            label_valid = label_state.metrics["failure_nonfinite"] < 0.5
        was_active = eval_active
        (
            eval_state,
            eval_applied_controller_action,
            eval_deadline_miss_rate,
        ) = transport_eval_action(
            eval_state, controller_action, eval_transport_key
        )
        eval_deadline_miss_sum += eval_deadline_miss_rate
        applied_effective_action = controller_action_to_effective_action_3d(
            jp, eval_applied_controller_action
        )
        next_eval_state = direct_eval_step_batch(
            eval_state, applied_effective_action, was_active
        )
        eval_roll_progress += jp.where(
            was_active,
            next_eval_state.metrics["roll_progress_rad"],
            0.0,
        )
        if args.record_diagnostics:
            diagnostic_frames.append(jax.device_get(diagnostic_frame(
                eval_state, next_eval_state, was_active, raw_effective_action,
                label_action, label_valid, eval_roll_progress / (2.0 * math.pi),
            )))
        eval_steps += was_active.astype(jp.int32)
        failed_now = (
            next_eval_state.metrics["failed"] > 0.5
        ) & was_active
        non_lateral_failed_now = (
            next_eval_state.metrics["failed_non_lateral"] > 0.5
        ) & was_active
        eval_failed = eval_failed | failed_now
        eval_non_lateral_failed = (
            eval_non_lateral_failed | non_lateral_failed_now
        )
        for name in EVALUATION_FAILURE_METRICS:
            eval_failure_flags[name] = eval_failure_flags[name] | (
                (next_eval_state.metrics[name] > 0.5) & was_active
            )
        eval_active = was_active & (next_eval_state.done < 0.5)
        eval_state = next_eval_state
        eval_previous_controller_action = jp.where(
            was_active[:, None],
            controller_action,
            eval_previous_controller_action,
        )
        if args.record_diagnostics and (
            (eval_step + 1) % args.log_every == 0
            or eval_step + 1 == args.episode_length
        ):
            print(
                f"[diagnostic {eval_step + 1}/{args.episode_length}] "
                f"active_before_step={int(jp.sum(was_active))}/{args.eval_envs}",
                flush=True,
            )

    eval_turns = np.asarray(
        jax.device_get(eval_roll_progress / (2.0 * math.pi))
    )
    eval_failed_np = np.asarray(jax.device_get(eval_failed))
    eval_non_lateral_failed_np = np.asarray(
        jax.device_get(eval_non_lateral_failed)
    )
    eval_lateral_failed_np = np.asarray(
        jax.device_get(eval_failure_flags["failure_lateral_drift"])
    )
    eval_strict_failed_np = (
        eval_non_lateral_failed_np | eval_lateral_failed_np
    )
    failure_rates = {
        name: float(
            np.mean(np.asarray(jax.device_get(flags), dtype=np.float64))
        )
        for name, flags in eval_failure_flags.items()
    }
    movement_success = (
        (~eval_failed_np)
        & (eval_turns >= args.minimum_closed_loop_turns)
    )
    non_lateral_movement_success = (
        (~eval_non_lateral_failed_np)
        & (eval_turns >= args.minimum_closed_loop_turns)
    )
    strict_movement_success = (
        (~eval_strict_failed_np)
        & (eval_turns >= args.minimum_closed_loop_turns)
    )
    closed_loop_evaluation = {
        "episodes": args.eval_envs,
        "episode_length": args.episode_length,
        "duration_s": args.episode_length * direct_task.control_timestep,
        "deploy_dr": args.deploy_dr,
        "deploy_dr_strength": args.deploy_dr_strength if args.deploy_dr else 0.0,
        "observation_noise_scale": (
            args.observation_noise_scale if args.deploy_dr else 0.0
        ),
        "mean_deadline_miss_rate": float(
            eval_deadline_miss_sum / args.episode_length
        ),
        "minimum_success_turns": args.minimum_closed_loop_turns,
        "failure_free_rate": float(1.0 - np.mean(eval_failed_np)),
        "success_rate": float(np.mean(movement_success)),
        "strict_failure_free_rate": float(
            1.0 - np.mean(eval_strict_failed_np)
        ),
        "strict_success_rate": float(np.mean(strict_movement_success)),
        "non_lateral_failure_free_rate": float(
            1.0 - np.mean(eval_non_lateral_failed_np)
        ),
        "non_lateral_success_rate": float(
            np.mean(non_lateral_movement_success)
        ),
        "mean_turns": float(np.mean(eval_turns)),
        "median_turns": float(np.median(eval_turns)),
        "minimum_turns": float(np.min(eval_turns)),
        "maximum_turns": float(np.max(eval_turns)),
        "mean_episode_steps": float(
            np.mean(np.asarray(jax.device_get(eval_steps)))
        ),
        "abduction_output_rms": float(
            jp.sqrt(
                abduction_square_sum
                / jp.maximum(abduction_sample_count, 1)
            )
        ),
        "abduction_output_max_abs": float(abduction_max_abs),
        "failure_rates": failure_rates,
    }
    print(
        "[student closed loop]\n"
        f"  success={closed_loop_evaluation['success_rate']:.1%} "
        "strict_success="
        f"{closed_loop_evaluation['strict_success_rate']:.1%} "
        "non_lateral_success="
        f"{closed_loop_evaluation['non_lateral_success_rate']:.1%} "
        f"failure_free={closed_loop_evaluation['failure_free_rate']:.1%} "
        f"turns_mean={closed_loop_evaluation['mean_turns']:.3f} "
        f"turns_min={closed_loop_evaluation['minimum_turns']:.3f}\n"
        f"  abduction_rms="
        f"{closed_loop_evaluation['abduction_output_rms']:.6f} "
        f"abduction_max="
        f"{closed_loop_evaluation['abduction_output_max_abs']:.6f}",
        flush=True,
    )

    lateral_diagnostics = None
    if args.record_diagnostics:
        trace = {
            name: np.stack([frame[name] for frame in diagnostic_frames])
            for name in diagnostic_frames[0]
        }
        lateral_diagnostics = save_lateral_trace(args.out, trace)
        print(
            "[lateral diagnostics]\n"
            f"  failed_y_positive={lateral_diagnostics['lateral_failure_positive_count']} "
            f"failed_y_negative={lateral_diagnostics['lateral_failure_negative_count']}\n"
            f"  common_error_rmse={lateral_diagnostics['common_error']['rmse']} "
            f"differential_error_rmse={lateral_diagnostics['differential_error']['rmse']}\n"
            f"  reports={args.out / 'lateral_diagnostics.json'}", flush=True,
        )

    if args.eval_only:
        report = {
            "mode": "evaluation_only_no_training_or_checkpoint_export",
            "teacher": str(args.teacher.resolve()),
            "student": str(args.restore_student.resolve()),
            "controller": str(args.controller.resolve()),
            "eval_seed": args.eval_seed,
            "eval_reset_keys": np.asarray(jax.device_get(eval_reset_keys)).tolist(),
            "closed_loop_task": asdict(direct_task),
            "teacher_task": asdict(teacher_task),
            "deploy_domain_randomization": (
                asdict(deploy_dr_settings)
                if deploy_dr_settings is not None
                else None
            ),
            "observation_noise_scale": (
                args.observation_noise_scale if args.deploy_dr else 0.0
            ),
            "closed_loop_evaluation": closed_loop_evaluation,
            "lateral_diagnostics": lateral_diagnostics,
        }
        with (args.out / "evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"[saved evaluation only] {args.out / 'evaluation.json'}", flush=True)
        return

    normalizer = {
        "mean": np.asarray(observation_mean),
        "std": np.asarray(observation_std),
    }
    checkpoint = (
        normalizer,
        jax.tree_util.tree_map(np.asarray, student_params),
        {},
    )
    checkpoint_path = args.out / "student_params"
    model_io.save_params(checkpoint_path, checkpoint)
    with (args.out / "controller_config.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    rtneural = convert_rtneural(
        checkpoint,
        config,
        activation="elu",
        observation_history=ROLLING_DEPLOY_OBSERVATION_HISTORY_3D,
    )
    with (args.out / "student_rtneural.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(rtneural, output_file, separators=(",", ":"))
        output_file.write("\n")

    run_config = {
        "teacher": str(args.teacher.resolve()),
        "controller": str(args.controller.resolve()),
        "runtime": describe_runtime(),
        "teacher_task": asdict(teacher_task),
        "closed_loop_task": asdict(direct_task),
        "deploy_domain_randomization": (
            asdict(deploy_dr_settings)
            if deploy_dr_settings is not None
            else None
        ),
        "student_observation_size": ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
        "student_action_size": ROLLING_CONTROLLER_ACTION_SIZE_3D,
        "hardware_policy_frequency_hz": HARDWARE_POLICY_FREQUENCY_HZ_3D,
        "hardware_imu_publish_frequency_hz": (
            HARDWARE_IMU_PUBLISH_FREQUENCY_HZ_3D
        ),
        "simulation_policy_frequency_hz": 1.0 / teacher_task.control_timestep,
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "loss_history": loss_history,
        "dagger_loss_history": dagger_loss_history,
        "closed_loop_evaluation": closed_loop_evaluation,
        "eval_reset_keys": np.asarray(jax.device_get(eval_reset_keys)).tolist(),
        "lateral_diagnostics": lateral_diagnostics,
    }
    with (args.out / "distillation.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(run_config, output_file, indent=2)
        output_file.write("\n")
    print(
        "[saved]\n"
        f"  checkpoint={checkpoint_path}\n"
        f"  rtneural={args.out / 'student_rtneural.json'}\n"
        f"  config={args.out / 'controller_config.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
