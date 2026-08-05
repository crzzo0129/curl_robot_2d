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
    if args.episode_length < 1 or args.batch_size < 1:
        parser.error("--episode-length and --batch-size must be positive")
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
        f"  batch={args.batch_size} episode={args.episode_length} "
        f"physics={task.physics_profile} seed={args.seed}\n"
        f"  reset_noise q={task.reset_joint_noise_rad:g} "
        f"v={task.reset_velocity_noise:g}\n"
        f"  reference_ramp_start={task.reference_ramp_start_scale} "
        f"ramp_s={task.reference_ramp_duration_s:g}\n"
        f"  zero_residual_policy_init={args.zero_residual_policy_init}",
        flush=True,
    )

    start = time.perf_counter()
    state = reset_batch(keys)
    jax.block_until_ready(state.obs)
    active = jp.ones((args.batch_size,), dtype=bool)
    steps = jp.zeros((args.batch_size,), dtype=jp.int32)
    reward_total = jp.zeros((args.batch_size,), dtype=jp.float32)
    metric_totals = {
        name: jp.zeros((args.batch_size,), dtype=jp.float32)
        for name in env._zero_metrics()
        if name not in ("reward", "reward_total")
    }
    rollout_qpos = []
    rollout_action = []
    rollout_reward = []

    for step_index in range(task.episode_length):
        rng = jax.random.PRNGKey(args.seed + step_index + 1)
        action_keys = jax.random.split(rng, args.batch_size)
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
        if args.save_rollout:
            first_active = bool(
                np.asarray(
                    jax.device_get(was_active[args.rollout_index])
                )
            )
            if first_active:
                rollout_qpos.append(
                    np.asarray(
                        jax.device_get(
                            state.pipeline_state.qpos[args.rollout_index]
                        )
                    )
                )
                rollout_action.append(
                    np.asarray(jax.device_get(actions[args.rollout_index]))
                )
                rollout_reward.append(
                    float(
                        np.asarray(
                            jax.device_get(state.reward[args.rollout_index])
                        )
                    )
                )
        active = active & (state.done < 0.5)

    jax.block_until_ready(state.obs)
    wall_time = time.perf_counter() - start
    scale = 1.0 / (2.0 * math.pi)
    arrays = {
        "steps": np.asarray(jax.device_get(steps)),
        "reward": np.asarray(jax.device_get(reward_total)),
        "conservative_turns": np.asarray(
            jax.device_get(metric_totals["roll_progress_rad"] * scale)
        ),
        "rotation_turns": np.asarray(
            jax.device_get(metric_totals["rotation_progress_rad"] * scale)
        ),
        "translation_turns": np.asarray(
            jax.device_get(metric_totals["translation_progress_rad"] * scale)
        ),
        "lateral_drift_m": np.asarray(
            jax.device_get(
                metric_totals["lateral_drift_m"] / jp.maximum(steps, 1)
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
    failure_rates = {
        metric.removeprefix("failure_"): float(
            np.mean(np.asarray(jax.device_get(state.metrics[metric])))
        )
        for metric in FAILURE_METRICS
        if metric in state.metrics
    }
    summary = {
        "runtime": describe_runtime(),
        "checkpoint": str(args.checkpoint),
        "wall_time_s": wall_time,
        "task": asdict(task),
        "controller": str(reference.source),
        "batch_size": args.batch_size,
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
