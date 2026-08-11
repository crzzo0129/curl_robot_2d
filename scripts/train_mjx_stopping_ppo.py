"""Train a snapshot-reset feedback braking residual with Brax PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.config import NominalRLConfig, physics_profile
from curl_robot_2d_mjx.environment_stopping_2d import (
    _load_snapshot_arrays,
    make_stopping_brax_env,
    scaled_reference_frequency,
)
from curl_robot_2d_mjx.reward_stopping import (
    StoppingRewardConfig,
    StoppingTaskConfig,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from scripts.search_braking_schedule import (
    DEFAULT_CONTROLLER,
    DEFAULT_MODEL,
    DEFAULT_SNAPSHOTS,
)
from scripts.train_mjx_ppo import _float, _network_factory


PRESETS = {
    "smoke": {
        "steps": 65_536,
        "envs": 64,
        "eval_envs": 16,
        "num_evals": 4,
        "batch_size": 64,
        "num_minibatches": 4,
    },
    "4090": {
        "steps": 5_000_000,
        "envs": 512,
        "eval_envs": 128,
        "num_evals": 10,
        "batch_size": 512,
        "num_minibatches": 16,
    },
    "h200": {
        "steps": 10_000_000,
        "envs": 2048,
        "eval_envs": 256,
        "num_evals": 10,
        "batch_size": 1024,
        "num_minibatches": 32,
    },
}


def _dataset_summary(
    path: Path,
    maximum_initial_angular_speed_rad_s: float | None,
) -> dict[str, object]:
    arrays = _load_snapshot_arrays(
        path,
        maximum_initial_angular_speed_rad_s=(
            maximum_initial_angular_speed_rad_s
        ),
    )
    angular = np.abs(arrays["qvel"][:, 2])
    return {
        "path": str(path.resolve()),
        "safe_snapshots": int(len(arrays["qpos"])),
        "qpos_shape": list(arrays["qpos"].shape),
        "qvel_shape": list(arrays["qvel"].shape),
        "median_initial_angular_speed_rad_s": float(np.median(angular)),
        "maximum_initial_angular_speed_rad_s": float(np.max(angular)),
        "filter_maximum_initial_angular_speed_rad_s": (
            maximum_initial_angular_speed_rad_s
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--stage", choices=("low", "full"), default="low")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--out", type=Path, default=Path("results/mjx_stopping_ppo"))
    parser.add_argument("--physics-profile", default="newton4")
    parser.add_argument("--frequency-hz", type=float, default=0.40)
    parser.add_argument("--residual-gain", type=float, default=0.35)
    parser.add_argument("--maximum-duration", type=float, default=5.0)
    parser.add_argument("--maximum-initial-angular-speed", type=float, default=None)
    parser.add_argument("--restore-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-cost", type=float, default=1.0e-3)
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--unroll-length", type=int, default=16)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=(256, 256))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.residual_gain <= 1.0:
        parser.error("--residual-gain must lie in (0, 1]")
    if args.maximum_duration <= 0.0:
        parser.error("--maximum-duration must be positive")
    maximum_initial_angular_speed = args.maximum_initial_angular_speed
    if maximum_initial_angular_speed is None and args.stage == "low":
        maximum_initial_angular_speed = 3.0

    stopping = StoppingTaskConfig(maximum_duration_s=args.maximum_duration)
    reward = StoppingRewardConfig()
    reference = load_cem_reference(
        args.controller,
        reference_weight=1.0,
        minimum_residual_gain=args.residual_gain,
    )
    reference = scaled_reference_frequency(reference, args.frequency_hz)
    task = physics_profile(
        args.physics_profile,
        NominalRLConfig(
            model_xml=str(args.model.resolve()),
            episode_length=int(math.ceil(
                stopping.maximum_duration_s / (0.001 * 20)
            )),
            disturbance_probability=0.0,
            terminate_stuck_root_z_max=None,
            mjx_compatible_collision_proxies=True,
            tail_progress_window_s=min(2.0, stopping.maximum_duration_s),
        ),
    )
    payload = {
        "preset": args.preset,
        "stage": args.stage,
        "dataset": _dataset_summary(
            args.snapshots, maximum_initial_angular_speed
        ),
        "task": asdict(task),
        "stopping": asdict(stopping),
        "reward": asdict(reward),
        "reference": asdict(reference),
        "training": PRESETS[args.preset],
        "seed": args.seed,
        "restore_checkpoint": (
            str(args.restore_checkpoint.resolve())
            if args.restore_checkpoint is not None
            else None
        ),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    configure_cloud_runtime(verbose=False)
    import jax
    from brax.io import model as model_io
    from brax.training.agents.ppo import train as ppo

    args.out.mkdir(parents=True, exist_ok=True)
    payload["runtime"] = describe_runtime()
    (args.out / "training_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    train_env = make_stopping_brax_env(
        task,
        cem_reference=reference,
        snapshots=args.snapshots,
        stopping_config=stopping,
        reward_config=reward,
        maximum_initial_angular_speed_rad_s=maximum_initial_angular_speed,
        seed=args.seed,
    )
    eval_env = make_stopping_brax_env(
        task,
        cem_reference=reference,
        snapshots=args.snapshots,
        stopping_config=stopping,
        reward_config=reward,
        maximum_initial_angular_speed_rad_s=maximum_initial_angular_speed,
        seed=args.seed + 10_000,
    )
    values = PRESETS[args.preset]
    history: list[dict[str, float | int]] = []

    def progress(step, metrics):
        clean = {name: _float(value) for name, value in metrics.items()}
        record = {"step": int(step), **clean}
        history.append(record)
        success = clean.get("eval/episode_stop_success", 0.0)
        phase = clean.get("eval/episode_phase_error_rad", math.nan)
        angular = clean.get("eval/episode_angular_speed_rad_s", math.nan)
        print(
            f"[stop eval] step={int(step)} success={success:.1%} "
            f"phase_sum={phase:.4f} angular_sum={angular:.4f}",
            flush=True,
        )

    print(
        f"[stopping PPO] snapshots={payload['dataset']['safe_snapshots']} "
        f"frequency={args.frequency_hz:.3f}Hz residual_gain={args.residual_gain:.2f} "
        f"episode={task.episode_length}x{task.control_timestep:.3f}s",
        flush=True,
    )
    start = time.perf_counter()
    checkpoint_kwargs = {}
    train_parameters = inspect.signature(ppo.train).parameters
    if "save_checkpoint_path" in train_parameters:
        checkpoint_kwargs["save_checkpoint_path"] = str(
            (args.out / "ppo_checkpoint").resolve()
        )
    if args.restore_checkpoint is not None:
        if "restore_checkpoint_path" not in train_parameters:
            raise SystemExit("Installed Brax cannot restore PPO checkpoints")
        checkpoint_kwargs["restore_checkpoint_path"] = str(
            args.restore_checkpoint.resolve()
        )
    _, params, final_metrics = ppo.train(
        environment=train_env,
        eval_env=eval_env,
        num_timesteps=values["steps"],
        episode_length=task.episode_length,
        action_repeat=1,
        num_envs=values["envs"],
        num_evals=values["num_evals"],
        num_eval_envs=values["eval_envs"],
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=1.0,
        unroll_length=args.unroll_length,
        batch_size=values["batch_size"],
        num_minibatches=values["num_minibatches"],
        num_updates_per_batch=args.updates_per_batch,
        normalize_observations=True,
        network_factory=_network_factory(args.hidden_layers, "swish"),
        seed=args.seed,
        progress_fn=progress,
        **checkpoint_kwargs,
    )
    model_io.save_params(args.out / "params_final", params)
    (args.out / "metrics_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "elapsed_s": time.perf_counter() - start,
        "final_metrics": {
            name: _float(value) for name, value in (final_metrics or {}).items()
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
