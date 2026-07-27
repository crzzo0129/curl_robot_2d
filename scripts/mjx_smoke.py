"""Compile and step the nominal-COM MJX environment on a cloud GPU."""

from __future__ import annotations

import argparse
import json
import time

from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    describe_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    args = parser.parse_args()

    configure_cloud_runtime(memory_fraction=args.memory_fraction)
    import jax
    import jax.numpy as jp

    from curl_robot_2d_mjx.environment import make_brax_env

    print(json.dumps(describe_runtime(), indent=2), flush=True)
    env = make_brax_env(seed=args.seed)
    keys = jax.random.split(
        jax.random.PRNGKey(args.seed), args.batch_size
    )
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))

    start = time.perf_counter()
    state = reset_batch(keys)
    jax.block_until_ready(state.obs)
    reset_compile_s = time.perf_counter() - start
    if state.obs.shape != (args.batch_size, env.observation_size):
        raise RuntimeError(f"unexpected observation shape: {state.obs.shape}")

    actions = jp.zeros((args.batch_size, env.action_size))
    start = time.perf_counter()
    for _ in range(args.steps):
        state = step_batch(state, actions)
    jax.block_until_ready(state.reward)
    step_compile_and_run_s = time.perf_counter() - start
    if not bool(jp.all(jp.isfinite(state.obs))):
        raise RuntimeError("non-finite MJX observation")
    if not bool(jp.all(jp.isfinite(state.reward))):
        raise RuntimeError("non-finite MJX reward")

    result = {
        "status": "ok",
        "batch_size": args.batch_size,
        "steps": args.steps,
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "reset_compile_s": reset_compile_s,
        "step_compile_and_run_s": step_compile_and_run_s,
        "mean_reward": float(jp.mean(state.reward)),
        "done_fraction": float(jp.mean(state.done)),
        "mean_roll_progress_rad": float(
            jp.mean(state.metrics["roll_progress_rad"])
        ),
        "mean_forbidden_contacts": float(
            jp.mean(state.metrics["forbidden_contact_count"])
        ),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
