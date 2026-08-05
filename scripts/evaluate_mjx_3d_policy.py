"""Evaluate a saved 3-D residual PPO policy with a large deterministic batch."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.environment_3d import DEFAULT_3D_CEM_CONTROLLER
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from scripts.train_mjx_ppo import _network_factory
from scripts.train_mjx_3d_residual_ppo import (
    TANH_NORMAL_MIN_STD,
    _zero_centered_residual_network_factory,
)


FAILURE_METRICS = (
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


def _distribution(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p25": float(np.percentile(array, 25.0)),
        "p75": float(np.percentile(array, 75.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controller", type=Path, default=DEFAULT_3D_CEM_CONTROLLER)
    parser.add_argument("--physics-profile", default="cg20")
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-joint-noise-rad", type=float, default=0.005)
    parser.add_argument("--reset-velocity-noise", type=float, default=0.005)
    parser.add_argument("--reference-weight", type=float, default=1.0)
    parser.add_argument("--minimum-residual-gain", type=float, default=0.15)
    parser.add_argument("--phase-rate-scale", type=float, default=1.0)
    parser.add_argument("--reference-action-scale", type=float, default=1.0)
    parser.add_argument("--reference-ramp-start-scale", type=float, default=0.50)
    parser.add_argument("--reference-ramp-duration-s", type=float, default=0.10)
    parser.add_argument("--reference-startup-boost", type=float, default=0.0)
    parser.add_argument(
        "--reference-startup-boost-duration-s",
        type=float,
        default=0.25,
    )
    parser.add_argument("--residual-pair-differential-scale", type=float, default=0.25)
    parser.add_argument(
        "--explicit-phase-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hidden-layers", type=int, nargs="+", default=[256, 256, 128]
    )
    parser.add_argument(
        "--activation",
        choices=("elu", "relu", "swish", "tanh"),
        default="elu",
    )
    parser.add_argument(
        "--zero-residual-policy-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the zero-centered residual policy module used by "
            "phase_locked_coupled_v6 checkpoints."
        ),
    )
    parser.add_argument(
        "--initial-policy-std",
        type=float,
        default=0.20,
        help="Initial pre-tanh policy std used to build the saved policy tree.",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--save-rollout", action="store_true")
    parser.add_argument("--rollout-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.episode_length < 1 or args.batch_size < 1 or args.chunk_size < 1:
        parser.error("--episode-length, --batch-size and --chunk-size must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    if not 0 <= args.rollout_index < args.batch_size:
        parser.error("--rollout-index must be in [0, batch-size)")
    if args.reset_joint_noise_rad < 0.0 or args.reset_velocity_noise < 0.0:
        parser.error("--reset-* noise values must be nonnegative")
    if args.initial_policy_std <= TANH_NORMAL_MIN_STD:
        parser.error(
            "--initial-policy-std must be greater than "
            f"{TANH_NORMAL_MIN_STD:g}"
        )
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=False,
        xla_triton=False,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )

    import jax
    import jax.numpy as jp
    from brax.io import model as model_io
    from brax.training.agents.ppo import networks as ppo_networks
    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d

    task = physics_profile_3d(
        args.physics_profile,
        Rolling3DConfig(
            episode_length=args.episode_length,
            reset_joint_noise_rad=args.reset_joint_noise_rad,
            reset_velocity_noise=args.reset_velocity_noise,
            reference_phase_rate_scale=args.phase_rate_scale,
            reference_action_scale=args.reference_action_scale,
            reference_ramp_start_scale=args.reference_ramp_start_scale,
            reference_ramp_duration_s=args.reference_ramp_duration_s,
            reference_startup_boost=args.reference_startup_boost,
            reference_startup_boost_duration_s=(
                args.reference_startup_boost_duration_s
            ),
            residual_pair_differential_scale=(
                args.residual_pair_differential_scale
            ),
            explicit_phase_observation=args.explicit_phase_observation,
        ),
    )
    reference = load_cem_reference(
        args.controller,
        reference_weight=args.reference_weight,
        minimum_residual_gain=args.minimum_residual_gain,
    )
    env = make_brax_env_3d(task, cem_reference=reference, seed=args.seed)
    network_factory = (
        _zero_centered_residual_network_factory(
            args.hidden_layers,
            args.activation,
            args.initial_policy_std,
        )
        if args.zero_residual_policy_init
        else _network_factory(args.hidden_layers, args.activation)
    )
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    params = model_io.load_params(args.checkpoint)
    try:
        policy = make_policy(params, deterministic=True)
    except TypeError:
        policy = make_policy(params)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.batch_size)
    reset_batch = jax.jit(jax.vmap(env.reset))
    policy_batch = jax.jit(
        jax.vmap(lambda obs, key: policy(obs, key)[0])
    )

    def step_one(state, action, active):
        return jax.lax.cond(
            active,
            lambda _: env.step(state, action),
            lambda _: state,
            operand=None,
        )

    step_batch = jax.jit(jax.vmap(step_one))
    print(
        "[3-D policy deterministic evaluation]\n"
        f"  checkpoint={args.checkpoint}\n"
        f"  batch={args.batch_size} chunk={args.chunk_size} "
        f"episode={args.episode_length} "
        f"physics={task.physics_profile} seed={args.seed}\n"
        f"  reset_noise q={task.reset_joint_noise_rad:g} "
        f"v={task.reset_velocity_noise:g}\n"
        f"  reference_ramp_start={task.reference_ramp_start_scale} "
        f"ramp_s={task.reference_ramp_duration_s:g}\n"
        f"  zero_residual_policy_init={args.zero_residual_policy_init}",
        flush=True,
    )

    start = time.perf_counter()
    array_chunks = {
        "steps": [],
        "reward": [],
        "conservative_turns": [],
        "rotation_turns": [],
        "translation_turns": [],
        "lateral_drift_m": [],
        "axis_tilt_rad": [],
        "failed": [],
        "timeout": [],
    }
    failure_chunks = {metric: [] for metric in FAILURE_METRICS}
    rollout_qpos = []
    rollout_action = []
    rollout_reward = []
    scale = 1.0 / (2.0 * math.pi)
    chunk_count = math.ceil(args.batch_size / args.chunk_size)

    for chunk_id, chunk_start in enumerate(
        range(0, args.batch_size, args.chunk_size),
        start=1,
    ):
        chunk_end = min(chunk_start + args.chunk_size, args.batch_size)
        current_batch = chunk_end - chunk_start
        save_this_rollout = (
            args.save_rollout
            and chunk_start <= args.rollout_index < chunk_end
        )
        rollout_local_index = args.rollout_index - chunk_start
        print(
            f"  chunk {chunk_id}/{chunk_count} "
            f"envs={current_batch} seed_index=[{chunk_start}, {chunk_end}) "
            "compiling/running",
            flush=True,
        )
        chunk_t0 = time.perf_counter()
        state = reset_batch(keys[chunk_start:chunk_end])
        jax.block_until_ready(state.obs)
        active = jp.ones((current_batch,), dtype=bool)
        steps = jp.zeros((current_batch,), dtype=jp.int32)
        reward_total = jp.zeros((current_batch,), dtype=jp.float32)
        metric_totals = {
            name: jp.zeros((current_batch,), dtype=jp.float32)
            for name in env._zero_metrics()
            if name not in ("reward", "reward_total")
        }

        for step_index in range(task.episode_length):
            rng = jax.random.PRNGKey(
                args.seed + 1 + chunk_start * 1_000_003 + step_index
            )
            action_keys = jax.random.split(rng, current_batch)
            actions = policy_batch(state.obs, action_keys)
            was_active = active
            state = step_batch(state, actions, active)
            weight = was_active.astype(jp.float32)
            steps = steps + was_active.astype(jp.int32)
            reward_total = reward_total + weight * state.reward
            for name in metric_totals:
                metric_totals[name] = (
                    metric_totals[name] + weight * state.metrics[name]
                )
            active = active & (state.done < 0.5)
            if save_this_rollout:
                selected_active = bool(
                    np.asarray(
                        jax.device_get(was_active[rollout_local_index])
                    )
                )
                if selected_active:
                    rollout_qpos.append(
                        np.asarray(
                            jax.device_get(
                                state.pipeline_state.qpos[
                                    rollout_local_index
                                ]
                            )
                        )
                    )
                    rollout_action.append(
                        np.asarray(
                            jax.device_get(actions[rollout_local_index])
                        )
                    )
                    rollout_reward.append(
                        float(
                            np.asarray(
                                jax.device_get(
                                    state.reward[rollout_local_index]
                                )
                            )
                        )
                    )
            if args.progress_every and (
                (step_index + 1) % args.progress_every == 0
                or step_index + 1 == task.episode_length
            ):
                jax.block_until_ready(state.obs)
                active_count = int(
                    np.sum(np.asarray(jax.device_get(active)))
                )
                print(
                    f"    step {step_index + 1}/{task.episode_length} "
                    f"active={active_count}/{current_batch}",
                    flush=True,
                )

        jax.block_until_ready(state.obs)
        chunk_wall = time.perf_counter() - chunk_t0
        chunk_arrays = {
            "steps": np.asarray(jax.device_get(steps)),
            "reward": np.asarray(jax.device_get(reward_total)),
            "conservative_turns": np.asarray(
                jax.device_get(metric_totals["roll_progress_rad"] * scale)
            ),
            "rotation_turns": np.asarray(
                jax.device_get(
                    metric_totals["rotation_progress_rad"] * scale
                )
            ),
            "translation_turns": np.asarray(
                jax.device_get(
                    metric_totals["translation_progress_rad"] * scale
                )
            ),
            "lateral_drift_m": np.asarray(
                jax.device_get(
                    metric_totals["lateral_drift_m"]
                    / jp.maximum(steps, 1)
                )
            ),
            "axis_tilt_rad": np.asarray(
                jax.device_get(
                    metric_totals["axis_tilt_rad"] / jp.maximum(steps, 1)
                )
            ),
            "failed": np.asarray(jax.device_get(state.metrics["failed"])),
            "timeout": np.asarray(jax.device_get(state.metrics["timeout"])),
        }
        for name, values in chunk_arrays.items():
            array_chunks[name].append(values)
        for metric in FAILURE_METRICS:
            if metric in state.metrics:
                failure_chunks[metric].append(
                    np.asarray(jax.device_get(state.metrics[metric]))
                )
        print(
            f"    chunk_done wall={chunk_wall:.1f}s "
            f"turns_median={np.median(chunk_arrays['conservative_turns']):.3f} "
            f"failed={np.mean(chunk_arrays['failed']):.2%}",
            flush=True,
        )

    wall_time = time.perf_counter() - start
    arrays = {
        name: np.concatenate(chunks)
        for name, chunks in array_chunks.items()
    }
    failure_rates = {
        metric.removeprefix("failure_"): float(np.mean(np.concatenate(chunks)))
        for metric, chunks in failure_chunks.items()
        if chunks
    }
    summary = {
        "runtime": describe_runtime(),
        "checkpoint": str(args.checkpoint),
        "wall_time_s": wall_time,
        "task": asdict(task),
        "controller": str(reference.source),
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "progress_every": args.progress_every,
        "episode_length": args.episode_length,
        "zero_residual_policy_init": args.zero_residual_policy_init,
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "initial_policy_std": args.initial_policy_std,
        "average_steps": float(np.mean(arrays["steps"])),
        "failure_rate": float(np.mean(arrays["failed"])),
        "timeout_rate": float(np.mean(arrays["timeout"])),
        "failure_rates": failure_rates,
        "reward": _distribution(arrays["reward"]),
        "conservative_turns": _distribution(arrays["conservative_turns"]),
        "rotation_turns": _distribution(arrays["rotation_turns"]),
        "translation_turns": _distribution(arrays["translation_turns"]),
        "lateral_drift_m": _distribution(arrays["lateral_drift_m"]),
        "axis_tilt_rad": _distribution(arrays["axis_tilt_rad"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "deterministic_eval.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.save_rollout:
        np.savez_compressed(
            args.out / "evaluation_rollout.npz",
            qpos=np.asarray(rollout_qpos),
            action=np.asarray(rollout_action),
            reward=np.asarray(rollout_reward),
        )
    print(
        f"  turns median={summary['conservative_turns']['median']:.3f} "
        f"range=[{summary['conservative_turns']['min']:.3f}, "
        f"{summary['conservative_turns']['max']:.3f}] "
        f"failed={summary['failure_rate']:.2%} "
        f"timeout={summary['timeout_rate']:.2%}\n"
        f"  output={args.out.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
