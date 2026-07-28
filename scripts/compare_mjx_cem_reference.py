"""Compare pure CEM rollouts across MJX reset and root-damping settings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.cem_reference import (
    DEFAULT_CEM_CONTROLLER,
    load_cem_reference,
)
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    describe_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "mjx_cem_reference_ablation"
    / "summary.json"
)


def parse_args(argv=None):
    from curl_robot_2d_mjx.config import (
        PHYSICS_PROFILE_NAMES,
        NominalRLConfig,
        validate_nominal_rl_config,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run a zero-policy CEM reference in MJX with controlled reset "
            "noise and root damping."
        )
    )
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILE_NAMES,
        default="cg12",
    )
    parser.add_argument(
        "--controller",
        type=Path,
        default=DEFAULT_CEM_CONTROLLER,
    )
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument(
        "--disturbance-root-x-velocity", type=float, default=0.0
    )
    parser.add_argument(
        "--disturbance-root-pitch-velocity", type=float, default=0.0
    )
    parser.add_argument("--disturbance-min-step", type=int, default=100)
    parser.add_argument("--disturbance-max-step", type=int, default=400)
    parser.add_argument(
        "--noise-seeds",
        type=int,
        default=32,
        help="Number of parallel reset-noise samples in case A.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("A", "B", "C", "D"),
        default=("A", "B", "C", "D"),
        help="Ablation cases to run; use '--cases D' for the training default.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-fraction", type=float, default=0.50)
    parser.add_argument(
        "--mujoco-gl",
        default="disable",
        help="Rendering is unused; 'disable' is suitable locally.",
    )
    args = parser.parse_args(argv)
    try:
        validate_nominal_rl_config(
            NominalRLConfig(
                episode_length=args.episode_length,
                disturbance_root_x_velocity_m_s=(
                    args.disturbance_root_x_velocity
                ),
                disturbance_root_pitch_velocity_rad_s=(
                    args.disturbance_root_pitch_velocity
                ),
                disturbance_min_step=args.disturbance_min_step,
                disturbance_max_step=args.disturbance_max_step,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _distribution(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _run_cpu_reference(task, controller_path) -> dict[str, object]:
    import mujoco

    from curl_robot_2d_mjx.environment import MODEL_PATH, apply_physics_options
    from scripts.optimize_phase_controller import (
        _load_controller_parameters,
        rollout_controller,
    )

    parameters = _load_controller_parameters(controller_path)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    apply_physics_options(model, task)
    start = time.perf_counter()
    rollout = rollout_controller(
        model,
        parameters[:8],
        duration=task.episode_length * task.control_timestep,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        objective="sustained",
        detailed=False,
    )
    elapsed_s = time.perf_counter() - start
    result = {
        "elapsed_s": elapsed_s,
        "summary": rollout.summary,
        "note": (
            "The established CPU CEM rollout starts from exact compact, "
            "zero velocity, and disables root damping."
        ),
    }
    print(
        "[CPU_reference] "
        f"turns={rollout.summary['conservative_rolling_turns']:+.3f} "
        f"x={rollout.summary['root_x_displacement_m']:+.3f}m "
        f"wall={elapsed_s:.1f}s",
        flush=True,
    )
    return result


def _run_case(
    *,
    name,
    task,
    reference,
    batch_size,
    seed,
):
    import jax
    import jax.numpy as jp

    from curl_robot_2d_mjx.environment import make_brax_env

    env = make_brax_env(task, cem_reference=reference, seed=seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    reset_batch = jax.jit(jax.vmap(env.reset))

    def step_one(state, action, active):
        return jax.lax.cond(
            active,
            lambda _: env.step(state, action),
            lambda _: state,
            operand=None,
        )

    step_batch = jax.jit(jax.vmap(step_one))
    start = time.perf_counter()
    state = reset_batch(keys)
    jax.block_until_ready(state.obs)
    initial_phase = state.pipeline_state.qpos[:, env.root_pitch_qpos]
    initial_x = state.pipeline_state.qpos[:, env.root_x_qpos]
    active = jp.ones((batch_size,), dtype=bool)
    steps = jp.zeros((batch_size,), dtype=jp.int32)
    roll_progress = jp.zeros((batch_size,), dtype=jp.float32)
    root_height_sum = jp.zeros((batch_size,), dtype=jp.float32)
    foot_gap_sum = jp.zeros((batch_size,), dtype=jp.float32)
    maximum_foot_gap = jp.zeros((batch_size,), dtype=jp.float32)
    maximum_forbidden_depth = jp.zeros((batch_size,), dtype=jp.float32)
    reference_rms_sum = jp.zeros((batch_size,), dtype=jp.float32)
    disturbance_count = jp.zeros((batch_size,), dtype=jp.float32)
    actions = jp.zeros((batch_size, env.action_size), dtype=jp.float32)

    for _ in range(task.episode_length):
        was_active = active
        state = step_batch(state, actions, active)
        active_float = was_active.astype(jp.float32)
        steps = steps + was_active.astype(jp.int32)
        roll_progress = (
            roll_progress + active_float * state.metrics["roll_progress_rad"]
        )
        root_height_sum = (
            root_height_sum + active_float * state.metrics["root_height_m"]
        )
        foot_gap_sum = (
            foot_gap_sum
            + active_float * state.metrics["foot_center_distance_m"]
        )
        maximum_foot_gap = jp.maximum(
            maximum_foot_gap,
            active_float * state.metrics["foot_center_distance_m"],
        )
        maximum_forbidden_depth = jp.maximum(
            maximum_forbidden_depth,
            active_float * state.metrics["forbidden_penetration_m"],
        )
        reference_rms_sum = (
            reference_rms_sum
            + active_float * state.metrics["reference_action_rms"]
        )
        disturbance_count = (
            disturbance_count
            + active_float * state.metrics["disturbance_applied"]
        )
        active = active & (state.done < 0.5)

    jax.block_until_ready(state.obs)
    elapsed_s = time.perf_counter() - start
    final_phase = state.pipeline_state.qpos[:, env.root_pitch_qpos]
    final_x = state.pipeline_state.qpos[:, env.root_x_qpos]
    divisor = jp.maximum(steps, 1).astype(jp.float32)
    arrays = {
        "steps": np.asarray(jax.device_get(steps)),
        "phase_turns": np.asarray(
            jax.device_get((final_phase - initial_phase) / (2.0 * math.pi))
        ),
        "conservative_turns": np.asarray(
            jax.device_get(roll_progress / (2.0 * math.pi))
        ),
        "root_x_displacement_m": np.asarray(
            jax.device_get(final_x - initial_x)
        ),
        "average_root_height_m": np.asarray(
            jax.device_get(root_height_sum / divisor)
        ),
        "average_foot_gap_m": np.asarray(
            jax.device_get(foot_gap_sum / divisor)
        ),
        "maximum_foot_gap_m": np.asarray(
            jax.device_get(maximum_foot_gap)
        ),
        "maximum_forbidden_depth_m": np.asarray(
            jax.device_get(maximum_forbidden_depth)
        ),
        "average_reference_action_rms": np.asarray(
            jax.device_get(reference_rms_sum / divisor)
        ),
        "disturbance_count": np.asarray(
            jax.device_get(disturbance_count)
        ),
        "disturbance_root_x_velocity_m_s": np.asarray(
            jax.device_get(state.info["disturbance_root_x_velocity"])
        ),
        "disturbance_root_pitch_velocity_rad_s": np.asarray(
            jax.device_get(state.info["disturbance_root_pitch_velocity"])
        ),
        "failed": np.asarray(jax.device_get(state.metrics["failed"])),
        "failure_root_high": np.asarray(
            jax.device_get(state.metrics["failure_root_high"])
        ),
        "failure_foot_gap": np.asarray(
            jax.device_get(state.metrics["failure_foot_gap"])
        ),
        "failure_leg_crossing": np.asarray(
            jax.device_get(state.metrics["failure_leg_crossing"])
        ),
        "failure_nonfinite": np.asarray(
            jax.device_get(state.metrics["failure_nonfinite"])
        ),
    }
    result = {
        "name": name,
        "batch_size": batch_size,
        "elapsed_s": elapsed_s,
        "task": asdict(task),
        "phase_turns": _distribution(arrays["phase_turns"]),
        "conservative_turns": _distribution(
            arrays["conservative_turns"]
        ),
        "root_x_displacement_m": _distribution(
            arrays["root_x_displacement_m"]
        ),
        "average_root_height_m": _distribution(
            arrays["average_root_height_m"]
        ),
        "average_foot_gap_m": _distribution(
            arrays["average_foot_gap_m"]
        ),
        "maximum_foot_gap_m": _distribution(
            arrays["maximum_foot_gap_m"]
        ),
        "maximum_forbidden_depth_m": _distribution(
            arrays["maximum_forbidden_depth_m"]
        ),
        "average_reference_action_rms": _distribution(
            arrays["average_reference_action_rms"]
        ),
        "disturbance_count": _distribution(
            arrays["disturbance_count"]
        ),
        "disturbance_root_x_velocity_m_s": _distribution(
            arrays["disturbance_root_x_velocity_m_s"]
        ),
        "disturbance_root_pitch_velocity_rad_s": _distribution(
            arrays["disturbance_root_pitch_velocity_rad_s"]
        ),
        "average_steps": _mean(arrays["steps"]),
        "failure_rate": _mean(arrays["failed"]),
        "failure_rates": {
            key.removeprefix("failure_"): _mean(value)
            for key, value in arrays.items()
            if key.startswith("failure_")
        },
        "samples": [
            {
                key: (
                    int(value[index])
                    if key == "steps"
                    else float(value[index])
                )
                for key, value in arrays.items()
            }
            for index in range(batch_size)
        ],
    }
    turns = result["conservative_turns"]
    displacement = result["root_x_displacement_m"]
    print(
        f"[{name}] batch={batch_size} "
        f"turns={turns['mean']:+.3f} "
        f"[{turns['min']:+.3f}, {turns['max']:+.3f}] "
        f"x={displacement['mean']:+.3f}m "
        f"failed={result['failure_rate']:.1%} "
        f"pushes={result['disturbance_count']['mean']:.2f}/episode "
        f"wall={elapsed_s:.1f}s",
        flush=True,
    )
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.episode_length < 1:
        raise SystemExit("--episode-length must be at least 1")
    if args.noise_seeds < 1:
        raise SystemExit("--noise-seeds must be at least 1")

    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=False,
        xla_triton=False,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )
    from curl_robot_2d_mjx.config import NominalRLConfig, physics_profile

    base_task = physics_profile(
        args.physics_profile,
        NominalRLConfig(
            episode_length=args.episode_length,
            disable_root_damping=False,
            disturbance_root_x_velocity_m_s=(
                args.disturbance_root_x_velocity
            ),
            disturbance_root_pitch_velocity_rad_s=(
                args.disturbance_root_pitch_velocity
            ),
            disturbance_min_step=args.disturbance_min_step,
            disturbance_max_step=args.disturbance_max_step,
        ),
    )
    reference = load_cem_reference(
        args.controller,
        reference_weight=1.0,
        minimum_residual_gain=0.0,
    )
    case_options = {
        "A": (
            "A_noise_root_damping",
            base_task,
            args.noise_seeds,
        ),
        "B": (
            "B_compact_root_damping",
            replace(
                base_task,
                reset_joint_noise_rad=0.0,
                reset_velocity_noise=0.0,
            ),
            1,
        ),
        "C": (
            "C_compact_no_root_damping",
            replace(
                base_task,
                reset_joint_noise_rad=0.0,
                reset_velocity_noise=0.0,
                disable_root_damping=True,
            ),
            1,
        ),
        "D": (
            "D_noise_no_root_damping",
            replace(base_task, disable_root_damping=True),
            args.noise_seeds,
        ),
    }
    cases = [case_options[name] for name in args.cases]
    print(
        "[pure CEM MJX ablation]\n"
        f"  profile={args.physics_profile} "
        f"episode={args.episode_length} "
        f"noise_seeds={args.noise_seeds} "
        f"cases={list(args.cases)}\n"
        f"  disturbance root_x<=+/-"
        f"{args.disturbance_root_x_velocity:g}m/s "
        f"root_pitch<=+/-"
        f"{args.disturbance_root_pitch_velocity:g}rad/s "
        f"step=[{args.disturbance_min_step},"
        f"{args.disturbance_max_step}]\n"
        f"  policy_action=0 residual_gain=0 "
        f"controller={reference.source}",
        flush=True,
    )
    cpu_reference = _run_cpu_reference(base_task, args.controller)
    results = [
        _run_case(
            name=name,
            task=task,
            reference=reference,
            batch_size=batch_size,
            seed=args.seed,
        )
        for name, task, batch_size in cases
    ]
    payload = {
        "runtime": describe_runtime(),
        "controller": asdict(reference),
        "physics_profile": args.physics_profile,
        "episode_length": args.episode_length,
        "cpu_reference": cpu_reference,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
