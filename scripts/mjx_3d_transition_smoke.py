"""Compile and step the 3-D transition environment on a JAX/MJX machine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from curl_robot_2d_mjx.config_transition_3d import (
    TRANSITION_CURRICULUM_STAGE_NAMES_3D,
    TRANSITION_PHYSICS_PROFILE_NAMES_3D,
    Transition3DConfig,
    transition_curriculum_config_3d,
    transition_physics_profile_3d,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=TRANSITION_CURRICULUM_STAGE_NAMES_3D,
        default="walking_start",
    )
    parser.add_argument("--roll-snapshots", type=Path)
    parser.add_argument("--snapshot-tail-fraction", type=float, default=1.0)
    parser.add_argument(
        "--physics-profile",
        choices=TRANSITION_PHYSICS_PROFILE_NAMES_3D,
        default="cg12",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--mujoco-gl", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1 or args.steps < 1:
        raise SystemExit("--batch-size and --steps must be positive")
    if args.stage.startswith("brake_") and not args.roll_snapshots:
        raise SystemExit("BRAKE smoke requires --roll-snapshots")
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )
    import jax
    import jax.numpy as jp

    from curl_robot_2d_mjx.environment_transition_3d import (
        make_brax_transition_env_3d,
    )

    task = transition_curriculum_config_3d(
        args.stage, Transition3DConfig(
            curriculum_stage=args.stage,
            roll_snapshots_path=str(args.roll_snapshots.resolve())
            if args.roll_snapshots else None,
            snapshot_tail_fraction=args.snapshot_tail_fraction,
        )
    )
    task = transition_physics_profile_3d(args.physics_profile, task)
    env = make_brax_transition_env_3d(task, seed=args.seed)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.batch_size)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    started = time.perf_counter()
    state = reset(keys)
    jax.block_until_ready(state.obs["state"])
    reset_compile_s = time.perf_counter() - started
    expected_actor = (args.batch_size, env.observation_size["state"])
    expected_critic = (
        args.batch_size,
        env.observation_size["privileged_state"],
    )
    if state.obs["state"].shape != expected_actor:
        raise RuntimeError(f"unexpected actor observation: {state.obs['state'].shape}")
    if state.obs["privileged_state"].shape != expected_critic:
        raise RuntimeError(
            f"unexpected critic observation: {state.obs['privileged_state'].shape}"
        )
    actions = jp.zeros((args.batch_size, env.action_size))
    started = time.perf_counter()
    state = step(state, actions)
    jax.block_until_ready(state.reward)
    step_compile_s = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(args.steps):
        state = step(state, actions)
    jax.block_until_ready(state.reward)
    rollout_s = time.perf_counter() - started
    if not bool(jp.all(jp.isfinite(state.obs["state"]))):
        raise RuntimeError("non-finite transition actor observation")
    if not bool(jp.all(jp.isfinite(state.reward))):
        raise RuntimeError("non-finite transition reward")
    result = {
        "status": "ok",
        "runtime": describe_runtime(),
        "stage": args.stage,
        "geometry": task.geometry,
        "model_path": str(env.model_path),
        "physics_profile": args.physics_profile,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "reset_compile_s": reset_compile_s,
        "step_compile_s": step_compile_s,
        "cached_rollout_s": rollout_s,
        "mean_reward": float(jp.mean(state.reward)),
        "mean_mode": float(jp.mean(state.metrics["mode"])),
        "success_fraction": float(
            jp.mean(state.metrics["transition_success"])
        ),
        "failure_fraction": float(jp.mean(state.metrics["failed"])),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
