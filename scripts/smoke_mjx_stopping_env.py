"""Run reset and a few zero-residual steps in the MJX stopping environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.config import NominalRLConfig, physics_profile
from curl_robot_2d_mjx.environment_stopping_2d import (
    make_stopping_brax_env,
    scaled_reference_frequency,
)
from scripts.search_braking_schedule import (
    DEFAULT_CONTROLLER,
    DEFAULT_MODEL,
    DEFAULT_SNAPSHOTS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    import jax
    import jax.numpy as jp

    reference = scaled_reference_frequency(
        load_cem_reference(
            args.controller, reference_weight=1.0, minimum_residual_gain=0.10
        ),
        0.40,
    )
    task = physics_profile(
        "newton4",
        NominalRLConfig(
            model_xml=str(args.model.resolve()),
            episode_length=250,
            disturbance_probability=0.0,
            terminate_stuck_root_z_max=None,
            mjx_compatible_collision_proxies=True,
        ),
    )
    env = make_stopping_brax_env(
        task,
        cem_reference=reference,
        snapshots=args.snapshots,
        maximum_initial_angular_speed_rad_s=3.0,
        seed=123,
    )
    reset = jax.jit(env.reset) if args.jit else env.reset
    step = jax.jit(env.step) if args.jit else env.step
    state = reset(jax.random.PRNGKey(0))
    initial_observation_size = int(state.obs.shape[0])
    for _ in range(args.steps):
        state = step(state, jp.zeros(env.action_size, dtype=jp.float32))
    leaves = jax.tree_util.tree_leaves(state)
    finite = all(bool(jp.all(jp.isfinite(value))) for value in leaves)
    report = {
        "backend": env.backend,
        "observation_size": env.observation_size,
        "actual_observation_size": initial_observation_size,
        "action_size": env.action_size,
        "steps": args.steps,
        "jit": args.jit,
        "finite": finite,
        "reward": float(state.reward),
        "done": float(state.done),
        "target_remaining_rad": float(state.metrics["target_remaining_rad"]),
        "linear_speed_m_s": float(state.metrics["linear_speed_m_s"]),
        "angular_speed_rad_s": float(state.metrics["angular_speed_rad_s"]),
    }
    if initial_observation_size != env.observation_size:
        raise RuntimeError(f"observation width mismatch: {report}")
    if not finite:
        raise RuntimeError(f"nonfinite stopping state: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
