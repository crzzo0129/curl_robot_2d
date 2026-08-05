"""Evaluate and diagnose a saved 3-D residual PPO policy."""

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
FAILURE_NAMES = tuple(name.removeprefix("failure_") for name in FAILURE_METRICS)
CHUNK_FORMAT_VERSION = 1


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


def _failure_code(arrays: dict[str, np.ndarray]) -> np.ndarray:
    codes = np.zeros(np.asarray(arrays["failed"]).shape, dtype=np.int8)
    for code, metric in enumerate(FAILURE_METRICS, start=1):
        active = np.asarray(arrays[metric]) > 0.5
        codes[(codes == 0) & active] = code
    return codes


def _failure_name(code: int) -> str | None:
    if code <= 0:
        return None
    return FAILURE_NAMES[code - 1]


def _summarize_arrays(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    failure_rates = {
        name.removeprefix("failure_"): float(np.mean(arrays[name]))
        for name in FAILURE_METRICS
    }
    return {
        "average_steps": float(np.mean(arrays["steps"])),
        "failure_rate": float(np.mean(arrays["failed"])),
        "timeout_rate": float(np.mean(arrays["timeout"])),
        "failure_rates": failure_rates,
        "reward": _distribution(arrays["reward"]),
        "conservative_turns": _distribution(arrays["conservative_turns"]),
        "rotation_turns": _distribution(arrays["rotation_turns"]),
        "translation_turns": _distribution(arrays["translation_turns"]),
        "average_lateral_drift_m": _distribution(
            arrays["average_lateral_drift_m"]
        ),
        # Kept for compatibility with summaries written before per-rollout output.
        "lateral_drift_m": _distribution(arrays["average_lateral_drift_m"]),
        "final_lateral_drift_m": _distribution(
            arrays["final_lateral_drift_m"]
        ),
        "max_abs_lateral_drift_m": _distribution(
            arrays["max_abs_lateral_drift_m"]
        ),
        "lateral_path_m": _distribution(arrays["lateral_path_m"]),
        "average_axis_tilt_rad": _distribution(
            arrays["average_axis_tilt_rad"]
        ),
        "axis_tilt_rad": _distribution(arrays["average_axis_tilt_rad"]),
        "max_axis_tilt_rad": _distribution(arrays["max_axis_tilt_rad"]),
        "residual_action_rms": _distribution(
            arrays["residual_action_rms"]
        ),
        "differential_residual_rms": _distribution(
            arrays["differential_residual_rms"]
        ),
    }


def _select_diagnostic_rollouts(
    arrays: dict[str, np.ndarray], limit: int
) -> list[dict[str, object]]:
    """Select diverse failures and boundary cases without saving every trace."""

    if limit <= 0:
        return []
    failed = np.flatnonzero(np.asarray(arrays["failed"]) > 0.5)
    succeeded = np.flatnonzero(np.asarray(arrays["failed"]) <= 0.5)
    selected: list[int] = []
    reasons: dict[int, list[str]] = {}

    def take(reason: str, candidates, count: int) -> None:
        added = 0
        for value in np.asarray(candidates, dtype=np.int64):
            index = int(value)
            if index in reasons:
                if reason not in reasons[index]:
                    reasons[index].append(reason)
                continue
            if len(selected) >= limit or added >= count:
                break
            selected.append(index)
            reasons[index] = [reason]
            added += 1

    if failed.size:
        take(
            "earliest_failure",
            failed[np.argsort(arrays["steps"][failed])],
            2,
        )
        take(
            "latest_failure",
            failed[np.argsort(-arrays["steps"][failed])],
            2,
        )
        positive = failed[arrays["final_lateral_drift_m"][failed] > 0.0]
        negative = failed[arrays["final_lateral_drift_m"][failed] < 0.0]
        take(
            "positive_lateral_failure",
            positive[
                np.argsort(-arrays["max_abs_lateral_drift_m"][positive])
            ],
            1,
        )
        take(
            "negative_lateral_failure",
            negative[
                np.argsort(-arrays["max_abs_lateral_drift_m"][negative])
            ],
            1,
        )
        take(
            "largest_lateral_failure",
            failed[
                np.argsort(-arrays["max_abs_lateral_drift_m"][failed])
            ],
            1,
        )

    if succeeded.size:
        take(
            "near_lateral_limit_success",
            succeeded[
                np.argsort(-arrays["max_abs_lateral_drift_m"][succeeded])
            ],
            2,
        )
        median_turns = float(np.median(arrays["conservative_turns"][succeeded]))
        take(
            "median_success",
            succeeded[
                np.argsort(
                    np.abs(
                        arrays["conservative_turns"][succeeded] - median_turns
                    )
                )
            ],
            1,
        )

    if len(selected) < limit:
        all_indices = np.arange(arrays["steps"].size)
        take(
            "largest_remaining_lateral_drift",
            all_indices[
                np.argsort(-arrays["max_abs_lateral_drift_m"])
            ],
            limit - len(selected),
        )

    return [
        {
            "array_index": index,
            "seed_index": int(arrays["seed_index"][index]),
            "reasons": reasons[index],
        }
        for index in selected
    ]


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


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
    parser.add_argument("--reset-pair-differential-scale", type=float)
    parser.add_argument(
        "--reset-axis-tilt-noise-rad", type=float, default=0.0
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed chunk files from an interrupted identical run.",
    )
    parser.add_argument(
        "--diagnostic-rollouts",
        type=int,
        default=0,
        help=(
            "Automatically replay this many representative policy rollouts "
            "and the same reset keys with zero residual action."
        ),
    )
    parser.add_argument(
        "--diagnostic-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay selected diagnostic reset keys with reference-only action.",
    )
    parser.add_argument("--save-rollout", action="store_true")
    parser.add_argument("--rollout-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.episode_length < 1 or args.batch_size < 1 or args.chunk_size < 1:
        parser.error("--episode-length, --batch-size and --chunk-size must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    if args.diagnostic_rollouts < 0:
        parser.error("--diagnostic-rollouts must be nonnegative")
    if not 0 <= args.rollout_index < args.batch_size:
        parser.error("--rollout-index must be in [0, batch-size)")
    if (
        args.reset_joint_noise_rad < 0.0
        or args.reset_velocity_noise < 0.0
        or args.reset_axis_tilt_noise_rad < 0.0
    ):
        parser.error("--reset-* noise values must be nonnegative")
    if (
        args.reset_pair_differential_scale is not None
        and not 0.0 <= args.reset_pair_differential_scale <= 1.0
    ):
        parser.error("--reset-pair-differential-scale must be in [0, 1]")
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
            reset_pair_differential_scale=(
                args.reset_pair_differential_scale
            ),
            reset_axis_tilt_noise_rad=args.reset_axis_tilt_noise_rad,
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
    ppo_network = network_factory(env.observation_size, env.action_size)
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    params = model_io.load_params(args.checkpoint)
    try:
        policy = make_policy(params, deterministic=True)
    except TypeError:
        policy = make_policy(params)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.batch_size)
    reset_batch = jax.jit(jax.vmap(env.reset))
    policy_batch = jax.jit(jax.vmap(lambda obs, key: policy(obs, key)[0]))

    def step_one(state, action, active):
        return jax.lax.cond(
            active,
            lambda _: env.step(state, action),
            lambda _: state,
            operand=None,
        )

    step_batch = jax.jit(jax.vmap(step_one))

    def run_rollouts(
        reset_keys,
        seed_indices,
        *,
        mode: str,
        capture: bool = False,
        label: str,
    ):
        batch = int(reset_keys.shape[0])
        state = reset_batch(reset_keys)
        jax.block_until_ready(state.obs)
        active = jp.ones((batch,), dtype=bool)
        steps = jp.zeros((batch,), dtype=jp.int32)
        reward_total = jp.zeros((batch,), dtype=jp.float32)
        conservative = jp.zeros((batch,), dtype=jp.float32)
        rotation = jp.zeros((batch,), dtype=jp.float32)
        translation = jp.zeros((batch,), dtype=jp.float32)
        lateral_sum = jp.zeros((batch,), dtype=jp.float32)
        lateral_path = jp.zeros((batch,), dtype=jp.float32)
        previous_lateral = jp.zeros((batch,), dtype=jp.float32)
        max_abs_lateral = jp.zeros((batch,), dtype=jp.float32)
        axis_tilt_sum = jp.zeros((batch,), dtype=jp.float32)
        max_axis_tilt = jp.zeros((batch,), dtype=jp.float32)
        residual_square_sum = jp.zeros((batch,), dtype=jp.float32)
        differential_square_sum = jp.zeros((batch,), dtype=jp.float32)
        traces = {
            name: []
            for name in (
                "qpos",
                "action",
                "mapped_residual_action",
                "reward",
                "lateral_drift_m",
                "axis_tilt_rad",
                "residual_action_rms",
                "differential_residual_rms",
                "failed",
                *FAILURE_METRICS,
            )
        }

        rollout_start = time.perf_counter()
        for step_index in range(task.episode_length):
            if mode == "reference":
                actions = jp.zeros((batch, env.action_size), dtype=jp.float32)
            else:
                action_keys = jax.random.split(
                    jax.random.PRNGKey(args.seed + 1 + step_index), batch
                )
                actions = policy_batch(state.obs, action_keys)
            was_active = active
            state = step_batch(state, actions, active)
            weight = was_active.astype(jp.float32)
            steps = steps + was_active.astype(jp.int32)
            reward_total = reward_total + weight * state.reward
            conservative += weight * state.metrics["roll_progress_rad"]
            rotation += weight * state.metrics["rotation_progress_rad"]
            translation += weight * state.metrics["translation_progress_rad"]
            lateral = state.metrics["lateral_drift_m"]
            lateral_sum += weight * lateral
            lateral_path += weight * jp.abs(lateral - previous_lateral)
            previous_lateral = jp.where(was_active, lateral, previous_lateral)
            max_abs_lateral = jp.maximum(
                max_abs_lateral, weight * jp.abs(lateral)
            )
            axis_tilt = state.metrics["axis_tilt_rad"]
            axis_tilt_sum += weight * axis_tilt
            max_axis_tilt = jp.maximum(max_axis_tilt, weight * axis_tilt)
            mapped_residual = state.info["last_policy_action"]
            residual_square = jp.mean(jp.square(mapped_residual), axis=1)
            pair_differential = 0.5 * jp.stack(
                (
                    mapped_residual[:, 0] - mapped_residual[:, 2],
                    mapped_residual[:, 1] - mapped_residual[:, 3],
                    mapped_residual[:, 4] - mapped_residual[:, 6],
                    mapped_residual[:, 5] - mapped_residual[:, 7],
                ),
                axis=1,
            )
            differential_square = jp.mean(jp.square(pair_differential), axis=1)
            residual_square_sum += weight * residual_square
            differential_square_sum += weight * differential_square
            active = active & (state.done < 0.5)

            if capture:
                trace_values = {
                    "qpos": state.pipeline_state.qpos,
                    "action": actions,
                    "mapped_residual_action": mapped_residual,
                    "reward": state.reward,
                    "lateral_drift_m": lateral,
                    "axis_tilt_rad": axis_tilt,
                    "residual_action_rms": jp.sqrt(residual_square),
                    "differential_residual_rms": jp.sqrt(differential_square),
                    "failed": state.metrics["failed"],
                    **{
                        name: state.metrics[name]
                        for name in FAILURE_METRICS
                    },
                }
                for name, value in trace_values.items():
                    traces[name].append(value)

            if args.progress_every and (
                (step_index + 1) % args.progress_every == 0
                or step_index + 1 == task.episode_length
            ):
                jax.block_until_ready(state.obs)
                active_count = int(
                    np.sum(np.asarray(jax.device_get(active)))
                )
                print(
                    f"    {label} step {step_index + 1}/{task.episode_length} "
                    f"active={active_count}/{batch}",
                    flush=True,
                )

        jax.block_until_ready(state.obs)
        wall_time = time.perf_counter() - rollout_start
        scale = 1.0 / (2.0 * math.pi)
        denominator = jp.maximum(steps, 1)
        arrays = {
            "seed_index": np.asarray(seed_indices, dtype=np.int32),
            "reset_key": np.asarray(jax.device_get(reset_keys)),
            "steps": np.asarray(jax.device_get(steps)),
            "reward": np.asarray(jax.device_get(reward_total)),
            "conservative_turns": np.asarray(
                jax.device_get(conservative * scale)
            ),
            "rotation_turns": np.asarray(jax.device_get(rotation * scale)),
            "translation_turns": np.asarray(
                jax.device_get(translation * scale)
            ),
            "average_lateral_drift_m": np.asarray(
                jax.device_get(lateral_sum / denominator)
            ),
            "final_lateral_drift_m": np.asarray(
                jax.device_get(state.metrics["lateral_drift_m"])
            ),
            "max_abs_lateral_drift_m": np.asarray(
                jax.device_get(max_abs_lateral)
            ),
            "lateral_path_m": np.asarray(jax.device_get(lateral_path)),
            "average_axis_tilt_rad": np.asarray(
                jax.device_get(axis_tilt_sum / denominator)
            ),
            "max_axis_tilt_rad": np.asarray(jax.device_get(max_axis_tilt)),
            "residual_action_rms": np.asarray(
                jax.device_get(jp.sqrt(residual_square_sum / denominator))
            ),
            "differential_residual_rms": np.asarray(
                jax.device_get(
                    jp.sqrt(differential_square_sum / denominator)
                )
            ),
            "failed": np.asarray(jax.device_get(state.metrics["failed"])),
            "timeout": np.asarray(jax.device_get(state.metrics["timeout"])),
            **{
                name: np.asarray(jax.device_get(state.metrics[name]))
                for name in FAILURE_METRICS
            },
        }
        arrays["failure_code"] = _failure_code(arrays)
        captured = None
        if capture:
            captured = {
                name: np.asarray(jax.device_get(jp.stack(values, axis=0)))
                for name, values in traces.items()
            }
        return arrays, captured, wall_time

    print(
        "[3-D policy deterministic evaluation]\n"
        f"  checkpoint={args.checkpoint}\n"
        f"  batch={args.batch_size} chunk={args.chunk_size} "
        f"episode={args.episode_length} "
        f"physics={task.physics_profile} seed={args.seed}\n"
        f"  reset_noise q={task.reset_joint_noise_rad:g} "
        f"v={task.reset_velocity_noise:g} "
        f"differential={task.reset_pair_differential_scale} "
        f"tilt={task.reset_axis_tilt_noise_rad:g}rad\n"
        f"  reference_ramp_start={task.reference_ramp_start_scale} "
        f"ramp_s={task.reference_ramp_duration_s:g}\n"
        f"  zero_residual_policy_init={args.zero_residual_policy_init} "
        f"resume={args.resume} diagnostics={args.diagnostic_rollouts}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.out / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "format_version": CHUNK_FORMAT_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "controller": str(Path(reference.source).resolve()),
        "task": asdict(task),
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "episode_length": args.episode_length,
        "seed": args.seed,
        "zero_residual_policy_init": args.zero_residual_policy_init,
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "initial_policy_std": args.initial_policy_std,
    }
    config_path = args.out / "eval_config.json"
    config_text = json.dumps(config_payload, indent=2, sort_keys=True) + "\n"
    if args.resume and config_path.exists():
        existing = json.dumps(
            json.loads(config_path.read_text(encoding="utf-8")),
            indent=2,
            sort_keys=True,
        ) + "\n"
        if existing != config_text:
            raise SystemExit(
                f"Cannot resume because evaluation config changed: {config_path}"
            )
    elif args.resume and any(chunks_dir.glob("policy_*.npz")):
        raise SystemExit(
            f"Cannot resume without the matching config file: {config_path}"
        )
    else:
        config_path.write_text(config_text, encoding="utf-8")

    start = time.perf_counter()
    chunk_results: list[dict[str, np.ndarray]] = []
    chunk_wall_total = 0.0
    chunk_count = math.ceil(args.batch_size / args.chunk_size)
    for chunk_id, chunk_start in enumerate(
        range(0, args.batch_size, args.chunk_size), start=1
    ):
        chunk_end = min(chunk_start + args.chunk_size, args.batch_size)
        chunk_path = chunks_dir / (
            f"policy_{chunk_start:06d}_{chunk_end:06d}.npz"
        )
        print(
            f"  chunk {chunk_id}/{chunk_count} "
            f"seed_index=[{chunk_start}, {chunk_end})",
            flush=True,
        )
        if args.resume and chunk_path.exists():
            archive = _load_npz(chunk_path)
            chunk_wall = float(archive.pop("_wall_time_s"))
            chunk_arrays = archive
            print("    loaded completed chunk", flush=True)
        else:
            chunk_arrays, _, chunk_wall = run_rollouts(
                keys[chunk_start:chunk_end],
                np.arange(chunk_start, chunk_end, dtype=np.int32),
                mode="policy",
                capture=False,
                label="policy",
            )
            _atomic_savez(
                chunk_path,
                **chunk_arrays,
                _wall_time_s=np.asarray(chunk_wall),
            )
        chunk_results.append(chunk_arrays)
        chunk_wall_total += chunk_wall
        print(
            f"    chunk_done wall={chunk_wall:.1f}s "
            f"turns_median={np.median(chunk_arrays['conservative_turns']):.3f} "
            f"failed={np.mean(chunk_arrays['failed']):.2%}",
            flush=True,
        )

    arrays = {
        name: np.concatenate([chunk[name] for chunk in chunk_results], axis=0)
        for name in chunk_results[0]
    }
    _atomic_savez(args.out / "eval_arrays.npz", **arrays)
    policy_summary = _summarize_arrays(arrays)
    summary = {
        "runtime": describe_runtime(),
        "checkpoint": str(args.checkpoint),
        "wall_time_s": time.perf_counter() - start,
        "chunk_wall_time_s": chunk_wall_total,
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
        **policy_summary,
    }
    (args.out / "deterministic_eval.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    selections = _select_diagnostic_rollouts(
        arrays, args.diagnostic_rollouts
    )
    if args.save_rollout:
        legacy = {
            "array_index": args.rollout_index,
            "seed_index": int(arrays["seed_index"][args.rollout_index]),
            "reasons": ["requested_rollout"],
        }
        if all(
            item["array_index"] != args.rollout_index for item in selections
        ):
            selections.append(legacy)

    if selections:
        selected_indices = np.asarray(
            [item["array_index"] for item in selections], dtype=np.int32
        )
        selected_keys = jp.asarray(arrays["reset_key"][selected_indices])
        selected_seed_indices = arrays["seed_index"][selected_indices]
        print(
            f"  replaying {len(selections)} selected policy rollouts",
            flush=True,
        )
        diagnostic_policy, policy_traces, _ = run_rollouts(
            selected_keys,
            selected_seed_indices,
            mode="policy",
            capture=True,
            label="diagnostic_policy",
        )
        diagnostic_reference = None
        reference_traces = None
        if args.diagnostic_reference and args.diagnostic_rollouts > 0:
            print(
                f"  replaying the same {len(selections)} reference-only rollouts",
                flush=True,
            )
            diagnostic_reference, reference_traces, _ = run_rollouts(
                selected_keys,
                selected_seed_indices,
                mode="reference",
                capture=True,
                label="diagnostic_reference",
            )

        rollout_dir = args.out / "diagnostic_rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        def save_traces(mode, diagnostic, traces) -> None:
            if diagnostic is None or traces is None:
                return
            for local_index, item in enumerate(selections):
                sample_count = int(diagnostic["steps"][local_index])
                seed_index = int(item["seed_index"])
                path = rollout_dir / f"seed_{seed_index:06d}_{mode}.npz"
                payload = {
                    name: values[:sample_count, local_index]
                    for name, values in traces.items()
                }
                payload.update(
                    {
                        "seed_index": np.asarray(seed_index),
                        "mode": np.asarray(mode),
                        "selection_reasons": np.asarray(
                            ",".join(item["reasons"])
                        ),
                        "steps": np.asarray(sample_count),
                        "failure_code": np.asarray(
                            diagnostic["failure_code"][local_index]
                        ),
                    }
                )
                _atomic_savez(path, **payload)

        save_traces("policy", diagnostic_policy, policy_traces)
        save_traces("reference", diagnostic_reference, reference_traces)

        for local_index, item in enumerate(selections):
            item["policy"] = {
                "steps": int(diagnostic_policy["steps"][local_index]),
                "failed": bool(diagnostic_policy["failed"][local_index]),
                "failure": _failure_name(
                    int(diagnostic_policy["failure_code"][local_index])
                ),
                "conservative_turns": float(
                    diagnostic_policy["conservative_turns"][local_index]
                ),
                "final_lateral_drift_m": float(
                    diagnostic_policy["final_lateral_drift_m"][local_index]
                ),
                "max_abs_lateral_drift_m": float(
                    diagnostic_policy["max_abs_lateral_drift_m"][local_index]
                ),
            }
            if diagnostic_reference is not None:
                item["reference"] = {
                    "steps": int(diagnostic_reference["steps"][local_index]),
                    "failed": bool(
                        diagnostic_reference["failed"][local_index]
                    ),
                    "failure": _failure_name(
                        int(
                            diagnostic_reference["failure_code"][local_index]
                        )
                    ),
                    "conservative_turns": float(
                        diagnostic_reference["conservative_turns"][local_index]
                    ),
                    "final_lateral_drift_m": float(
                        diagnostic_reference[
                            "final_lateral_drift_m"
                        ][local_index]
                    ),
                    "max_abs_lateral_drift_m": float(
                        diagnostic_reference[
                            "max_abs_lateral_drift_m"
                        ][local_index]
                    ),
                }
                policy_failed = item["policy"]["failed"]
                reference_failed = item["reference"]["failed"]
                if policy_failed and not reference_failed:
                    comparison = "policy_only_failure"
                elif reference_failed and not policy_failed:
                    comparison = "reference_only_failure"
                elif policy_failed:
                    comparison = "both_fail"
                else:
                    comparison = "both_success"
                item["comparison"] = comparison

        (args.out / "diagnostic_selection.json").write_text(
            json.dumps(selections, indent=2) + "\n", encoding="utf-8"
        )

        if args.save_rollout:
            requested = next(
                index
                for index, item in enumerate(selections)
                if item["array_index"] == args.rollout_index
            )
            sample_count = int(diagnostic_policy["steps"][requested])
            _atomic_savez(
                args.out / "evaluation_rollout.npz",
                **{
                    name: values[:sample_count, requested]
                    for name, values in policy_traces.items()
                },
            )

    print(
        f"  turns median={summary['conservative_turns']['median']:.3f} "
        f"range=[{summary['conservative_turns']['min']:.3f}, "
        f"{summary['conservative_turns']['max']:.3f}] "
        f"failed={summary['failure_rate']:.2%} "
        f"timeout={summary['timeout_rate']:.2%}\n"
        f"  per_rollout={args.out.resolve() / 'eval_arrays.npz'}\n"
        f"  output={args.out.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
