"""Compare the 3-D CEM reference across CPU MuJoCo and MJX solvers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.environment_3d import DEFAULT_3D_CEM_CONTROLLER
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "mjx_3d_reference_parity" / "summary.json"
)
CASE_NAMES = (
    "cpu_newton_exact",
    "cpu_newton8_exact",
    "cpu_cg12_exact",
    "cpu_cg20_exact",
    "mjx_newton8_exact",
    "mjx_newton8_noisy",
    "mjx_cg12_exact",
    "mjx_cg12_noisy",
    "mjx_cg20_exact",
    "mjx_cg20_noisy",
)
DEFAULT_CASE_NAMES = (
    "cpu_newton_exact",
    "cpu_cg12_exact",
    "mjx_cg12_exact",
    "mjx_cg12_noisy",
)
FAILURE_METRICS = (
    "failure_nonfinite",
    "failure_root_low",
    "failure_root_high",
    "failure_lateral_drift",
    "failure_axis_tilt",
    "failure_forbidden_depth",
    "failure_forbidden_contact",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller", type=Path, default=DEFAULT_3D_CEM_CONTROLLER
    )
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--noise-seeds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_NAMES, default=DEFAULT_CASE_NAMES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-fraction", type=float, default=0.50)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--reference-action-scale", type=float, default=1.0)
    parser.add_argument("--reference-ramp-start-scale", type=float, default=0.0)
    parser.add_argument("--reference-ramp-duration-s", type=float, default=0.25)
    parser.add_argument("--reference-startup-boost", type=float, default=0.0)
    parser.add_argument(
        "--reference-startup-boost-duration-s",
        type=float,
        default=0.25,
    )
    args = parser.parse_args(argv)
    if args.episode_length < 1:
        parser.error("--episode-length must be at least 1")
    if args.noise_seeds < 1:
        parser.error("--noise-seeds must be at least 1")
    if (
        not math.isfinite(args.reference_action_scale)
        or args.reference_action_scale <= 0.0
    ):
        parser.error("--reference-action-scale must be positive")
    if args.reference_ramp_start_scale is not None:
        if (
            not math.isfinite(args.reference_ramp_start_scale)
            or args.reference_ramp_start_scale < 0.0
        ):
            parser.error("--reference-ramp-start-scale must be nonnegative")
    if (
        not math.isfinite(args.reference_ramp_duration_s)
        or args.reference_ramp_duration_s <= 0.0
    ):
        parser.error("--reference-ramp-duration-s must be positive")
    if (
        not math.isfinite(args.reference_startup_boost)
        or args.reference_startup_boost < 0.0
    ):
        parser.error("--reference-startup-boost must be nonnegative")
    if (
        not math.isfinite(args.reference_startup_boost_duration_s)
        or args.reference_startup_boost_duration_s <= 0.0
    ):
        parser.error("--reference-startup-boost-duration-s must be positive")
    return args


def _distribution(values) -> dict[str, float]:
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


def _cpu_result(summary, *, name=None) -> dict[str, object]:
    net_rotation_turns = float(summary["rolling_phase_turns"])
    absolute_rotation_turns = float(
        summary.get("absolute_rotation_turns", abs(net_rotation_turns))
    )
    translation_turns = float(summary["distance_as_shell_turns"])
    conservative_turns = min(absolute_rotation_turns, translation_turns)
    profile = str(summary.get("physics_profile", "reference"))
    solver = str(summary.get("solver", "newton"))
    return {
        "name": name or f"cpu_{profile}_exact",
        "backend": "cpu_mujoco",
        "solver": solver,
        "physics_profile": profile,
        "reference_action_scale": float(summary.get("target_scale", 1.0)),
        "reference_ramp_start_scale": summary.get("startup_target_scale"),
        "reference_ramp_duration_s": float(
            summary.get("target_ramp_duration_s", 0.25)
        ),
        "reference_startup_boost": float(
            summary.get("startup_target_boost", 0.0)
        ),
        "reference_startup_boost_duration_s": float(
            summary.get("startup_target_boost_duration_s", 0.25)
        ),
        "reset": "exact",
        "batch_size": 1,
        "net_rotation_turns": _distribution([net_rotation_turns]),
        "absolute_rotation_turns": _distribution([absolute_rotation_turns]),
        "translation_turns": _distribution([translation_turns]),
        "conservative_turns": _distribution([conservative_turns]),
        "rotation_translation_mismatch_turns": _distribution(
            [absolute_rotation_turns - translation_turns]
        ),
        "failure_rate": float(bool(summary.get("nonfinite", False))),
        "source_summary": summary,
    }


def _run_cpu_case(args, *, name, physics_profile):
    from scripts.evaluate_3d_symmetric_cem_reference import (
        parse_args as parse_cpu_args,
        run_smoke,
    )

    duration = 0.02 * args.episode_length
    cpu_args = parse_cpu_args(
        [
            "--controller",
            str(args.controller),
            "--duration",
            str(duration),
            "--control-dt",
            "0.02",
            "--physics-profile",
            physics_profile,
            "--target-scale",
            str(args.reference_action_scale),
            "--target-ramp-duration-s",
            str(args.reference_ramp_duration_s),
            *(
                []
                if args.reference_ramp_start_scale is None
                else [
                    "--startup-target-scale",
                    str(args.reference_ramp_start_scale),
                ]
            ),
            "--startup-target-boost",
            str(args.reference_startup_boost),
            "--startup-target-boost-duration-s",
            str(args.reference_startup_boost_duration_s),
        ]
    )
    start = time.perf_counter()
    result = _cpu_result(run_smoke(cpu_args), name=name)
    result["wall_time_s"] = time.perf_counter() - start
    _print_case(result)
    return result


def _mjx_case_specs(
    episode_length: int,
    *,
    reference_action_scale: float = 1.0,
    reference_ramp_start_scale: float | None = None,
    reference_ramp_duration_s: float = 0.25,
    reference_startup_boost: float = 0.0,
    reference_startup_boost_duration_s: float = 0.25,
):
    from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d

    base = Rolling3DConfig(
        episode_length=episode_length,
        reference_action_scale=reference_action_scale,
        reference_ramp_start_scale=reference_ramp_start_scale,
        reference_ramp_duration_s=reference_ramp_duration_s,
        reference_startup_boost=reference_startup_boost,
        reference_startup_boost_duration_s=reference_startup_boost_duration_s,
    )
    exact = {"reset_joint_noise_rad": 0.0, "reset_velocity_noise": 0.0}
    return {
        "mjx_newton8_exact": (
            physics_profile_3d("newton8", replace(base, **exact)),
            1,
            "exact",
        ),
        "mjx_newton8_noisy": (
            physics_profile_3d("newton8", base),
            None,
            "noise",
        ),
        "mjx_cg12_exact": (
            physics_profile_3d("cg12", replace(base, **exact)),
            1,
            "exact",
        ),
        "mjx_cg12_noisy": (
            physics_profile_3d("cg12", base),
            None,
            "noise",
        ),
        "mjx_cg20_exact": (
            physics_profile_3d("cg20", replace(base, **exact)),
            1,
            "exact",
        ),
        "mjx_cg20_noisy": (
            physics_profile_3d("cg20", base),
            None,
            "noise",
        ),
    }


def _run_mjx_case(
    *, name, task, reset_name, batch_size, reference, environment_seed, rollout_seed
):
    import jax
    import jax.numpy as jp

    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d

    env = make_brax_env_3d(
        task, cem_reference=reference, seed=environment_seed
    )
    keys = jax.random.split(jax.random.PRNGKey(rollout_seed), batch_size)
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
    print(
        f"[{name}] compiling/running MJX; low GPU utilization is expected "
        "during the first XLA compile",
        flush=True,
    )
    state = reset_batch(keys)
    jax.block_until_ready(state.obs)
    initial_x = state.pipeline_state.qpos[:, 0]
    initial_y = state.pipeline_state.qpos[:, 1]
    actions = jp.zeros((batch_size, env.action_size), dtype=jp.float32)
    active = jp.ones((batch_size,), dtype=bool)
    steps = jp.zeros((batch_size,), dtype=jp.int32)
    absolute_rotation_progress = jp.zeros((batch_size,), dtype=jp.float32)
    translation_progress = jp.zeros((batch_size,), dtype=jp.float32)
    conservative_progress = jp.zeros((batch_size,), dtype=jp.float32)

    for _ in range(task.episode_length):
        was_active = active
        state = step_batch(state, actions, active)
        weight = was_active.astype(jp.float32)
        steps = steps + was_active.astype(jp.int32)
        absolute_rotation_progress += (
            weight * state.metrics["rotation_progress_rad"]
        )
        translation_progress += weight * state.metrics[
            "translation_progress_rad"
        ]
        conservative_progress += weight * state.metrics["roll_progress_rad"]
        active = active & (state.done < 0.5)

    jax.block_until_ready(state.obs)
    wall_time = time.perf_counter() - start
    scale = 1.0 / (2.0 * math.pi)
    arrays = {
        "steps": np.asarray(jax.device_get(steps)),
        "net_rotation_turns": np.asarray(
            jax.device_get(state.info["rolling_phase"] * scale)
        ),
        "absolute_rotation_turns": np.asarray(
            jax.device_get(absolute_rotation_progress * scale)
        ),
        "translation_turns": np.asarray(
            jax.device_get(translation_progress * scale)
        ),
        "conservative_turns": np.asarray(
            jax.device_get(conservative_progress * scale)
        ),
        "root_x_displacement_m": np.asarray(
            jax.device_get(state.pipeline_state.qpos[:, 0] - initial_x)
        ),
        "final_lateral_drift_m": np.asarray(
            jax.device_get(state.pipeline_state.qpos[:, 1] - initial_y)
        ),
        "failed": np.asarray(jax.device_get(state.metrics["failed"])),
    }
    for metric in FAILURE_METRICS:
        arrays[metric] = np.asarray(jax.device_get(state.metrics[metric]))
    arrays["rotation_translation_mismatch_turns"] = (
        arrays["absolute_rotation_turns"] - arrays["translation_turns"]
    )
    result = {
        "name": name,
        "backend": "mjx",
        "solver": task.solver_name,
        "physics_profile": task.physics_profile,
        "solver_iterations": task.solver_iterations,
        "solver_ls_iterations": task.solver_ls_iterations,
        "reference_action_scale": task.reference_action_scale,
        "reference_ramp_start_scale": task.reference_ramp_start_scale,
        "reference_ramp_duration_s": task.reference_ramp_duration_s,
        "reference_startup_boost": task.reference_startup_boost,
        "reference_startup_boost_duration_s": (
            task.reference_startup_boost_duration_s
        ),
        "reset": reset_name,
        "reset_joint_noise_rad": task.reset_joint_noise_rad,
        "reset_velocity_noise": task.reset_velocity_noise,
        "batch_size": batch_size,
        "environment_seed": environment_seed,
        "rollout_seed": rollout_seed,
        "wall_time_s": wall_time,
        "task": asdict(task),
        "average_steps": float(np.mean(arrays["steps"])),
        "net_rotation_turns": _distribution(arrays["net_rotation_turns"]),
        "absolute_rotation_turns": _distribution(
            arrays["absolute_rotation_turns"]
        ),
        "translation_turns": _distribution(arrays["translation_turns"]),
        "conservative_turns": _distribution(arrays["conservative_turns"]),
        "rotation_translation_mismatch_turns": _distribution(
            arrays["rotation_translation_mismatch_turns"]
        ),
        "root_x_displacement_m": _distribution(
            arrays["root_x_displacement_m"]
        ),
        "final_lateral_drift_m": _distribution(
            arrays["final_lateral_drift_m"]
        ),
        "failure_rate": float(np.mean(arrays["failed"])),
        "failure_rates": {
            metric.removeprefix("failure_"): float(np.mean(arrays[metric]))
            for metric in FAILURE_METRICS
        },
    }
    _print_case(result)
    return result


def _print_case(result) -> None:
    turns = result["conservative_turns"]
    profile = result.get("physics_profile", "")
    profile_text = f" profile={profile}" if profile else ""
    solver_detail = ""
    if "solver_iterations" in result:
        solver_detail = (
            f" iter={result['solver_iterations']}"
            f" ls={result['solver_ls_iterations']}"
        )
    reference_detail = ""
    if "reference_action_scale" in result:
        ramp_start = result.get("reference_ramp_start_scale")
        ramp_start_text = (
            "auto" if ramp_start is None else f"{float(ramp_start):.4g}"
        )
        reference_detail = (
            f" ref_scale={result['reference_action_scale']:.4g}"
            f" ramp_start={ramp_start_text}"
            f" ramp_s={result['reference_ramp_duration_s']:.4g}"
            f" ref_boost={result['reference_startup_boost']:.4g}"
            f" boost_s={result['reference_startup_boost_duration_s']:.4g}"
        )
    noise_detail = ""
    if "reset_joint_noise_rad" in result:
        noise_detail = (
            f" q_noise={result['reset_joint_noise_rad']:.4g}"
            f" v_noise={result['reset_velocity_noise']:.4g}"
        )
    print(
        f"[{result['name']}] backend={result['backend']} "
        f"solver={result['solver']}{profile_text}{solver_detail} "
        f"reset={result['reset']}{noise_detail}{reference_detail} "
        f"batch={result['batch_size']}\n"
        f"  conservative={turns['mean']:+.3f} "
        f"median={turns['median']:+.3f} "
        f"p95={turns['p95']:+.3f} "
        f"range=[{turns['min']:+.3f}, {turns['max']:+.3f}]\n"
        f"  net_rotation={result['net_rotation_turns']['mean']:+.3f} "
        f"abs_rotation={result['absolute_rotation_turns']['mean']:+.3f} "
        f"translation={result['translation_turns']['mean']:+.3f} "
        f"mismatch="
        f"{result['rotation_translation_mismatch_turns']['mean']:+.3f} "
        f"failed={result['failure_rate']:.1%} "
        f"wall={result['wall_time_s']:.1f}s",
        flush=True,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=False,
        xla_triton=False,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )
    reference = load_cem_reference(
        args.controller, reference_weight=1.0, minimum_residual_gain=0.0
    )
    print(
        "[3-D reference backend parity]\n"
        f"  episode={args.episode_length} "
        f"({0.02 * args.episode_length:.2f}s) "
        f"noise_seeds={args.noise_seeds} seed={args.seed}\n"
        f"  policy_action=0 residual_gain=0 "
        f"controller={reference.source}",
        flush=True,
    )
    results = []
    cpu_cases = {
        "cpu_newton_exact": "reference",
        "cpu_newton8_exact": "newton8",
        "cpu_cg12_exact": "cg12",
        "cpu_cg20_exact": "cg20",
    }
    for case_name in args.cases:
        if case_name in cpu_cases:
            results.append(
                _run_cpu_case(
                    args,
                    name=case_name,
                    physics_profile=cpu_cases[case_name],
                )
            )
    specs = _mjx_case_specs(
        args.episode_length,
        reference_action_scale=args.reference_action_scale,
        reference_ramp_start_scale=args.reference_ramp_start_scale,
        reference_ramp_duration_s=args.reference_ramp_duration_s,
        reference_startup_boost=args.reference_startup_boost,
        reference_startup_boost_duration_s=(
            args.reference_startup_boost_duration_s
        ),
    )
    for case_name in args.cases:
        if case_name in cpu_cases:
            continue
        task, fixed_batch_size, reset_name = specs[case_name]
        results.append(
            _run_mjx_case(
                name=case_name,
                task=task,
                reset_name=reset_name,
                batch_size=fixed_batch_size or args.noise_seeds,
                reference=reference,
                environment_seed=args.seed,
                rollout_seed=args.seed,
            )
        )
    payload = {
        "runtime": describe_runtime(),
        "controller": asdict(reference),
        "episode_length": args.episode_length,
        "rolling_radius_m": FIXED_PARAMETERS.shell_contact_radius,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
