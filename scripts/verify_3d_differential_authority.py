"""Verify whether the 3-D rolling differential channels can correct lateral drift.

Runs the CEM reference controller with an injected left/right differential
feedback and measures the lateral-drift failure rate, so we can decide whether a
residual policy has the authority to fix the ~15% step-0 baseline drift instead
of only learning a spurious unidirectional bias.

The action space is 8-D: raw[:4] = common (front_hip, front_knee, rear_hip,
rear_knee), raw[4:] = differential for the same four channels. The differential
maps left/right hip or knee in opposite directions (see
``pair_coupled_residual_action_3d``). With ``residual_pair_differential_scale``
set to the training value (0.25) and ``residual_gain`` 0.15, the maximum
actuator authority of the differential channel is 0.25 * 0.15 = 0.0375 rad.

Each case injects a linear feedback on the raw differential slots:

    d_channel = clip(bias + gain_y * y + gain_vy * vy, -1, 1)

applied to a configurable subset of the four differential channels, where y and
vy are the world-frame lateral position and velocity of the root.
"""

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
from curl_robot_2d_mjx.environment_3d import (
    DEFAULT_3D_CEM_CONTROLLER,
    make_brax_env_3d,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "differential_authority" / "summary.json"
)

CHANNEL_NAMES = ("front_hip", "front_knee", "rear_hip", "rear_knee")


def _distribution(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
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


def _parse_channel_mask(text: str) -> tuple[bool, ...]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if names == ["all"]:
        return (True, True, True, True)
    if names == ["hip"]:
        return (True, False, True, False)
    if names == ["front_hip"]:
        return (True, False, False, False)
    mask = [False, False, False, False]
    for name in names:
        if name not in CHANNEL_NAMES:
            raise ValueError(f"unknown channel {name!r}")
        mask[CHANNEL_NAMES.index(name)] = True
    return tuple(mask)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller", type=Path, default=DEFAULT_3D_CEM_CONTROLLER
    )
    parser.add_argument("--geometry", default="pupper_open60")
    parser.add_argument("--physics-profile", default="cg20")
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=10000)
    parser.add_argument("--minimum-residual-gain", type=float, default=0.15)
    parser.add_argument(
        "--differential-scale", type=float, default=0.25,
        help="residual_pair_differential_scale; 0.25 matches training",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.50)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "zero",
            "damp_y_p_k0p5",
            "damp_y_p_k1",
            "damp_y_p_k2",
            "damp_y_p_k5",
            "damp_vy_p_k1",
            "damp_vy_p_k2",
            "damp_vy_p_k5",
            "pd_p_y2_vy1",
            "pd_p_y2_vy2",
        ],
        help="case names to run (see CASE_SPECS)",
    )
    args = parser.parse_args(argv)
    if args.episode_length < 1:
        parser.error("--episode-length must be at least 1")
    if args.envs < 1:
        parser.error("--envs must be at least 1")
    if args.differential_scale is None or not 0.0 <= args.differential_scale <= 1.0:
        parser.error("--differential-scale must be in [0, 1]")
    return args


def _case_specs():
    hip = (True, False, True, False)
    all_channels = (True, True, True, True)
    return {
        "zero": dict(channels=(False, False, False, False), bias=0.0, gain_y=0.0, gain_vy=0.0),
        "bias_p1_hip": dict(channels=hip, bias=1.0, gain_y=0.0, gain_vy=0.0),
        "bias_n1_hip": dict(channels=hip, bias=-1.0, gain_y=0.0, gain_vy=0.0),
        "damp_y_hip_k2": dict(channels=hip, bias=0.0, gain_y=-2.0, gain_vy=0.0),
        "damp_y_hip_k5": dict(channels=hip, bias=0.0, gain_y=-5.0, gain_vy=0.0),
        "damp_y_hip_k10": dict(channels=hip, bias=0.0, gain_y=-10.0, gain_vy=0.0),
        "damp_vy_hip_k5": dict(channels=hip, bias=0.0, gain_y=0.0, gain_vy=-5.0),
        "damp_vy_hip_k10": dict(channels=hip, bias=0.0, gain_y=0.0, gain_vy=-10.0),
        "damp_y_all_k5": dict(channels=all_channels, bias=0.0, gain_y=-5.0, gain_vy=0.0),
        "pd_hip_y5_vy5": dict(channels=hip, bias=0.0, gain_y=-5.0, gain_vy=-5.0),
        # Corrected sign: d = +k*y and d = +k*vy oppose the induced drift.
        "damp_y_p_k0p5": dict(channels=hip, bias=0.0, gain_y=0.5, gain_vy=0.0),
        "damp_y_p_k1": dict(channels=hip, bias=0.0, gain_y=1.0, gain_vy=0.0),
        "damp_y_p_k2": dict(channels=hip, bias=0.0, gain_y=2.0, gain_vy=0.0),
        "damp_y_p_k5": dict(channels=hip, bias=0.0, gain_y=5.0, gain_vy=0.0),
        "damp_vy_p_k1": dict(channels=hip, bias=0.0, gain_y=0.0, gain_vy=1.0),
        "damp_vy_p_k2": dict(channels=hip, bias=0.0, gain_y=0.0, gain_vy=2.0),
        "damp_vy_p_k5": dict(channels=hip, bias=0.0, gain_y=0.0, gain_vy=5.0),
        "pd_p_y2_vy1": dict(channels=hip, bias=0.0, gain_y=2.0, gain_vy=1.0),
        "pd_p_y2_vy2": dict(channels=hip, bias=0.0, gain_y=2.0, gain_vy=2.0),
    }


def _run_case(
    *,
    name,
    task,
    reference,
    spec,
    batch_size,
    environment_seed,
    rollout_seed,
):
    import jax
    import jax.numpy as jp

    env = make_brax_env_3d(task, cem_reference=reference, seed=environment_seed)
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
    channels = jp.asarray(spec["channels"], dtype=jp.float32)
    gain_y = jp.asarray(spec["gain_y"], dtype=jp.float32)
    gain_vy = jp.asarray(spec["gain_vy"], dtype=jp.float32)
    bias = jp.asarray(spec["bias"], dtype=jp.float32)

    print(f"[{name}] compiling/running MJX...", flush=True)
    start = time.perf_counter()
    state = reset_batch(keys)
    jax.block_until_ready(state.obs)
    initial_y = state.pipeline_state.qpos[:, 1]
    active = jp.ones((batch_size,), dtype=bool)
    steps = jp.zeros((batch_size,), dtype=jp.int32)
    mean_abs_y = jp.zeros((batch_size,), dtype=jp.float32)
    max_abs_y = jp.zeros((batch_size,), dtype=jp.float32)
    conservative_progress = jp.zeros((batch_size,), dtype=jp.float32)

    for _ in range(task.episode_length):
        was_active = active
        drift_y = state.pipeline_state.qpos[:, 1] - initial_y
        vy = state.pipeline_state.qvel[:, 1]
        d = jp.clip(bias + gain_y * drift_y + gain_vy * vy, -1.0, 1.0)
        raw = jp.zeros((batch_size, env.action_size), dtype=jp.float32)
        raw = raw.at[:, 4:8].set(channels * d[:, None])
        state = step_batch(state, raw, active)
        weight = was_active.astype(jp.float32)
        abs_y = jp.abs(state.pipeline_state.qpos[:, 1] - initial_y)
        mean_abs_y += weight * abs_y
        max_abs_y = jp.maximum(max_abs_y, weight * abs_y)
        steps += was_active.astype(jp.int32)
        conservative_progress += (
            weight * state.metrics["roll_progress_rad"]
        )
        active = active & (state.done < 0.5)

    jax.block_until_ready(state.obs)
    wall_time = time.perf_counter() - start
    scale = 1.0 / (2.0 * math.pi)
    arrays = {
        "steps": np.asarray(jax.device_get(steps)),
        "final_lateral_drift_m": np.asarray(
            jax.device_get(state.pipeline_state.qpos[:, 1] - initial_y)
        ),
        "mean_abs_lateral_drift_m": np.asarray(
            jax.device_get(mean_abs_y / jp.maximum(steps, 1).astype(jp.float32))
        ),
        "max_abs_lateral_drift_m": np.asarray(jax.device_get(max_abs_y)),
        "conservative_turns": np.asarray(
            jax.device_get(conservative_progress * scale)
        ),
        "failed": np.asarray(jax.device_get(state.metrics["failed"])),
    }
    arrays["failure_lateral_drift"] = np.asarray(
        jax.device_get(state.metrics["failure_lateral_drift"])
    )
    result = {
        "name": name,
        "differential_scale": task.residual_pair_differential_scale,
        "channels": [CHANNEL_NAMES[i] for i, on in enumerate(spec["channels"]) if on],
        "bias": spec["bias"],
        "gain_y": spec["gain_y"],
        "gain_vy": spec["gain_vy"],
        "wall_time_s": wall_time,
        "failure_rate": float(np.mean(arrays["failed"])),
        "failure_lateral_drift": float(
            np.mean(arrays["failure_lateral_drift"])
        ),
        "final_lateral_drift_m": _distribution(
            arrays["final_lateral_drift_m"]
        ),
        "mean_abs_lateral_drift_m": _distribution(
            arrays["mean_abs_lateral_drift_m"]
        ),
        "max_abs_lateral_drift_m": _distribution(
            arrays["max_abs_lateral_drift_m"]
        ),
        "conservative_turns": _distribution(arrays["conservative_turns"]),
    }
    _print_case(result)
    return result


def _print_case(result) -> None:
    final = result["final_lateral_drift_m"]
    mean_abs = result["mean_abs_lateral_drift_m"]
    turns = result["conservative_turns"]
    channels = ",".join(result["channels"]) or "-"
    print(
        f"[{result['name']}] channels={channels} scale={result['differential_scale']} "
        f"bias={result['bias']:+.2g} ky={result['gain_y']:+.2g} kvy={result['gain_vy']:+.2g}\n"
        f"  failed={result['failure_rate']:.1%} "
        f"lat_drift={result['failure_lateral_drift']:.1%} "
        f"turns={turns['median']:+.3f} "
        f"final_y mean={final['mean']:+.4f} med|max={final['median']:+.4f}/{final['max']:+.4f} "
        f"mean_abs_y={mean_abs['median']:.4f} "
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
        args.controller,
        reference_weight=1.0,
        minimum_residual_gain=args.minimum_residual_gain,
    )
    base = Rolling3DConfig(
        geometry=args.geometry,
        episode_length=args.episode_length,
        residual_pair_differential_scale=args.differential_scale,
        explicit_phase_observation=True,
    )
    task = physics_profile_3d(args.physics_profile, base)
    specs = _case_specs()
    unknown = [name for name in args.cases if name not in specs]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}")
    print(
        "[3-D differential authority]\n"
        f"  episode={args.episode_length} envs={args.envs} "
        f"geometry={args.geometry} physics={args.physics_profile}\n"
        f"  differential_scale={args.differential_scale} "
        f"residual_gain={reference.residual_gain}\n"
        f"  reset q=0.005 v=0.005 pair_differential=None "
        f"seed={args.seed} env_seed={args.environment_seed}\n"
        f"  controller={reference.source}",
        flush=True,
    )
    results = []
    for name in args.cases:
        results.append(
            _run_case(
                name=name,
                task=task,
                reference=reference,
                spec=specs[name],
                batch_size=args.envs,
                environment_seed=args.environment_seed,
                rollout_seed=args.seed,
            )
        )
    payload = {
        "runtime": describe_runtime(),
        "controller": asdict(reference),
        "episode_length": args.episode_length,
        "differential_scale": args.differential_scale,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
