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
from curl_robot_2d_mjx.environment_3d import cem_controller_path_3d
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime


PRESETS = {
    "smoke": {
        "envs": 32,
        "stats_steps": 8,
        "train_steps": 32,
        "eval_envs": 16,
    },
    "h200": {
        "envs": 2048,
        "stats_steps": 500,
        "train_steps": 20_000,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Distill the accepted rolling teacher to real observations"
    )
    parser.add_argument("teacher", type=Path, help="teacher params_best")
    parser.add_argument(
        "--preset", choices=tuple(PRESETS), default="smoke"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument(
        "--controller",
        type=Path,
        default=cem_controller_path_3d("rollingquad_2"),
    )
    parser.add_argument("--envs", type=int)
    parser.add_argument("--stats-steps", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=list(STUDENT_HIDDEN_LAYERS),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-noise-scale", type=float, default=1.0)
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
    for name in ("envs", "stats_steps", "train_steps", "eval_envs"):
        if getattr(args, name) is None:
            setattr(args, name, PRESETS[args.preset][name])
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.episode_length < 1 or args.log_every < 1:
        parser.error("--episode-length and --log-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if (
        not math.isfinite(args.observation_noise_scale)
        or args.observation_noise_scale < 0.0
    ):
        parser.error("--observation-noise-scale must be finite and nonnegative")
    if (
        not math.isfinite(args.minimum_closed_loop_turns)
        or args.minimum_closed_loop_turns < 0.0
    ):
        parser.error(
            "--minimum-closed-loop-turns must be finite and nonnegative"
        )
    if not args.teacher.is_file():
        parser.error(f"teacher checkpoint does not exist: {args.teacher}")
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


def _task(*, episode_length, direct_effective_action=False):
    return physics_profile_3d(
        "cg20",
        Rolling3DConfig(
            geometry="rollingquad_2",
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
            explicit_phase_observation=not direct_effective_action,
            direct_effective_action=direct_effective_action,
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
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    import flax.linen as linen
    import optax

    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d
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
    teacher_task = _task(episode_length=args.episode_length)
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

    def deployment_observation(
        state,
        history,
        previous_controller_action,
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
        deployment_observation, static_argnums=(4,)
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
        f"  stats_steps={args.stats_steps} train_steps={args.train_steps} "
        f"envs={args.envs}",
        flush=True,
    )

    for step in range(args.stats_steps):
        rng, policy_key, noise_key = jax.random.split(rng, 3)
        policy_keys = jax.random.split(policy_key, args.envs)
        history = deployment_observation(
            state,
            history,
            previous_controller_action,
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

    observation_mean = observation_sum / observation_count
    observation_variance = jp.maximum(
        observation_square_sum / observation_count
        - jp.square(observation_mean),
        1e-6,
    )
    observation_std = jp.sqrt(observation_variance)
    rng, init_key, reset_key = jax.random.split(rng, 3)
    student_params = student.init(
        init_key,
        jp.zeros((1, ROLLING_DEPLOY_OBSERVATION_SIZE_3D)),
    )

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
    optimizer = optax.adam(args.learning_rate)
    optimizer_state = optimizer.init(student_params)

    @jax.jit
    def train_step(params, opt_state, observation, target):
        normalized = (observation - observation_mean) / observation_std

        def loss_fn(current_params):
            prediction = student.apply(current_params, normalized)
            error = prediction - target
            return jp.mean(jp.square(error)), (
                jp.sqrt(jp.mean(jp.square(error))),
                jp.max(jp.abs(error)),
            )

        (loss, diagnostics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, next_opt_state = optimizer.update(
            gradients, opt_state, params
        )
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss, diagnostics

    state, history, previous_controller_action = reset_rollout(
        reset_key, args.envs
    )
    loss_history = []
    for step in range(args.train_steps):
        rng, policy_key, noise_key = jax.random.split(rng, 3)
        policy_keys = jax.random.split(policy_key, args.envs)
        history = deployment_observation(
            state,
            history,
            previous_controller_action,
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
    )
    direct_env = make_brax_env_3d(
        direct_task,
        cem_reference=reference,
        seed=args.seed + 50_000,
    )
    direct_reset_batch = jax.jit(jax.vmap(direct_env.reset))

    def direct_step_if_active(single_state, single_action, single_active):
        return jax.lax.cond(
            single_active,
            lambda operands: direct_env.step(*operands),
            lambda operands: operands[0],
            (single_state, single_action),
        )

    direct_step_batch = jax.jit(jax.vmap(direct_step_if_active))
    student_policy_batch = jax.jit(
        lambda observation: student.apply(
            student_params,
            (observation - observation_mean) / observation_std,
        )
    )
    rng, eval_reset_key = jax.random.split(rng)
    eval_reset_keys = jax.random.split(eval_reset_key, args.eval_envs)
    eval_state = direct_reset_batch(eval_reset_keys)
    eval_history = jp.broadcast_to(
        initial_history,
        (args.eval_envs, ROLLING_DEPLOY_OBSERVATION_SIZE_3D),
    )
    eval_previous_controller_action = jp.zeros(
        (args.eval_envs, ROLLING_CONTROLLER_ACTION_SIZE_3D)
    )
    eval_active = jp.ones((args.eval_envs,), dtype=jp.bool_)
    eval_failed = jp.zeros_like(eval_active)
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

    for _ in range(args.episode_length):
        # No synthetic sensor noise in acceptance: this measures whether the
        # student itself can close the loop through the deployable ABI.
        rng, eval_noise_key = jax.random.split(rng)
        eval_history = deployment_observation(
            eval_state,
            eval_history,
            eval_previous_controller_action,
            eval_noise_key,
            0.0,
        )
        controller_action = student_policy_batch(eval_history)
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
        effective_action = controller_action_to_effective_action_3d(
            jp, controller_action
        )
        was_active = eval_active
        next_eval_state = direct_step_batch(
            eval_state, effective_action, was_active
        )
        eval_roll_progress += jp.where(
            was_active,
            next_eval_state.metrics["roll_progress_rad"],
            0.0,
        )
        eval_steps += was_active.astype(jp.int32)
        failed_now = (
            next_eval_state.metrics["failed"] > 0.5
        ) & was_active
        eval_failed = eval_failed | failed_now
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

    eval_turns = np.asarray(
        jax.device_get(eval_roll_progress / (2.0 * math.pi))
    )
    eval_failed_np = np.asarray(jax.device_get(eval_failed))
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
    closed_loop_evaluation = {
        "episodes": args.eval_envs,
        "episode_length": args.episode_length,
        "duration_s": args.episode_length * direct_task.control_timestep,
        "minimum_success_turns": args.minimum_closed_loop_turns,
        "failure_free_rate": float(1.0 - np.mean(eval_failed_np)),
        "success_rate": float(np.mean(movement_success)),
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
        f"failure_free={closed_loop_evaluation['failure_free_rate']:.1%} "
        f"turns_mean={closed_loop_evaluation['mean_turns']:.3f} "
        f"turns_min={closed_loop_evaluation['minimum_turns']:.3f}\n"
        f"  abduction_rms="
        f"{closed_loop_evaluation['abduction_output_rms']:.6f} "
        f"abduction_max="
        f"{closed_loop_evaluation['abduction_output_max_abs']:.6f}",
        flush=True,
    )

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
        "closed_loop_evaluation": closed_loop_evaluation,
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
