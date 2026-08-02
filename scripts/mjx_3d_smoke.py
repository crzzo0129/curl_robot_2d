"""Compile and step the 3-D curl MJX CEM-reference residual env."""

from __future__ import annotations

import argparse
import json
import time

from curl_robot_2d_mjx.config_3d import (
    PHYSICS_PROFILE_NAMES_3D,
    Rolling3DConfig,
    physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import DEFAULT_3D_CEM_CONTROLLER
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    describe_runtime,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILE_NAMES_3D,
        default="cg12",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller", default=DEFAULT_3D_CEM_CONTROLLER)
    parser.add_argument("--reference-weight", type=float, default=1.0)
    parser.add_argument("--minimum-residual-gain", type=float, default=0.05)
    parser.add_argument("--phase-rate-scale", type=float, default=1.0)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument(
        "--mujoco-gl",
        default="auto",
        help="Use 'disable' on a headless node without EGL.",
    )
    parser.add_argument("--xla-triton", action="store_true", default=True)
    parser.add_argument(
        "--no-xla-triton", dest="xla_triton", action="store_false"
    )
    parser.add_argument(
        "--preallocate", action="store_true", default=True
    )
    parser.add_argument(
        "--no-preallocate", dest="preallocate", action="store_false"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")

    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=args.preallocate,
        xla_triton=args.xla_triton,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )
    import jax
    import jax.numpy as jp

    from curl_robot_2d_mjx.cem_reference import load_cem_reference
    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d

    print(json.dumps(describe_runtime(), indent=2), flush=True)
    task = physics_profile_3d(
        args.physics_profile,
        Rolling3DConfig(
            episode_length=args.episode_length,
            reference_phase_rate_scale=args.phase_rate_scale,
        ),
    )
    reference = load_cem_reference(
        args.controller,
        reference_weight=args.reference_weight,
        minimum_residual_gain=args.minimum_residual_gain,
    )
    print("stage=environment_create_start", flush=True)
    env = make_brax_env_3d(task, cem_reference=reference, seed=args.seed)
    print(
        "stage=environment_create_done "
        f"nq={env.mj_model.nq} nv={env.mj_model.nv} "
        f"nu={env.mj_model.nu} ngeom={env.mj_model.ngeom} "
        f"physics_profile={env.config.physics_profile} "
        f"timestep={env.mj_model.opt.timestep} "
        f"solver={env.config.solver_name} "
        f"integrator={env.config.integrator_name} "
        f"cone={env.config.cone_name} "
        f"iterations={env.mj_model.opt.iterations} "
        f"ls_iterations={env.mj_model.opt.ls_iterations} "
        f"action_repeat={env.config.action_repeat} "
        f"observation_size={env.observation_size} "
        f"action_size={env.action_size}",
        flush=True,
    )

    keys = jax.random.split(
        jax.random.PRNGKey(args.seed), args.batch_size
    )
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))

    print(
        f"stage=reset_compile_start batch_size={args.batch_size}",
        flush=True,
    )
    start = time.perf_counter()
    state = reset_batch(keys)
    jax.block_until_ready(state.obs)
    reset_compile_s = time.perf_counter() - start
    print(
        f"stage=reset_compile_done seconds={reset_compile_s:.3f}",
        flush=True,
    )
    if state.obs.shape != (args.batch_size, env.observation_size):
        raise RuntimeError(f"unexpected observation shape: {state.obs.shape}")

    actions = jp.zeros((args.batch_size, env.action_size))
    print(
        f"stage=step_compile_start batch_size={args.batch_size}",
        flush=True,
    )
    start = time.perf_counter()
    state = step_batch(state, actions)
    jax.block_until_ready(state.reward)
    step_compile_s = time.perf_counter() - start
    print(
        f"stage=step_compile_done seconds={step_compile_s:.3f}",
        flush=True,
    )

    print("stage=step_signature_check_start", flush=True)
    start = time.perf_counter()
    state = step_batch(state, actions)
    jax.block_until_ready(state.reward)
    step_signature_check_s = time.perf_counter() - start
    print(
        "stage=step_signature_check_done "
        f"seconds={step_signature_check_s:.3f}",
        flush=True,
    )

    print(f"stage=cached_rollout_start steps={args.steps}", flush=True)
    start = time.perf_counter()
    for _ in range(args.steps):
        state = step_batch(state, actions)
    jax.block_until_ready(state.reward)
    cached_rollout_s = time.perf_counter() - start
    print(
        f"stage=cached_rollout_done seconds={cached_rollout_s:.3f}",
        flush=True,
    )
    if not bool(jp.all(jp.isfinite(state.obs))):
        raise RuntimeError("non-finite 3-D MJX observation")
    if not bool(jp.all(jp.isfinite(state.reward))):
        raise RuntimeError("non-finite 3-D MJX reward")

    result = {
        "status": "ok",
        "batch_size": args.batch_size,
        "steps": args.steps,
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "physics_profile": env.config.physics_profile,
        "reset_compile_s": reset_compile_s,
        "step_compile_s": step_compile_s,
        "step_signature_check_s": step_signature_check_s,
        "cached_rollout_s": cached_rollout_s,
        "cached_steps_per_second": (
            args.batch_size * args.steps / max(cached_rollout_s, 1e-9)
        ),
        "mean_reward": float(jp.mean(state.reward)),
        "done_fraction": float(jp.mean(state.done)),
        "mean_root_x_m": float(jp.mean(state.metrics["root_x_m"])),
        "mean_lateral_drift_m": float(
            jp.mean(state.metrics["lateral_drift_m"])
        ),
        "mean_axis_tilt_rad": float(jp.mean(state.metrics["axis_tilt_rad"])),
        "mean_roll_progress_rad": float(
            jp.mean(state.metrics["roll_progress_rad"])
        ),
        "mean_forbidden_contacts": float(
            jp.mean(state.metrics["forbidden_contact_count"])
        ),
        "mean_same_side_foot_contacts": float(
            jp.mean(state.metrics["same_side_foot_contact_count"])
        ),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
