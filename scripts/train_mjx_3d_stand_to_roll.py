#!/usr/bin/env python3
"""BC warm-start and staged PPO for one-policy stand-to-roll training.

Examples:
  python -m scripts.train_mjx_3d_stand_to_roll --stage bc --out results/stand_to_roll
  python -m scripts.train_mjx_3d_stand_to_roll --stage compact \
      --bc-params results/stand_to_roll/bc/bc_params --out results/stand_to_roll
  python -m scripts.train_mjx_3d_stand_to_roll --stage slightly_open \
      --bc-params results/stand_to_roll/bc/bc_params \
      --restore-checkpoint results/stand_to_roll/compact/ppo_checkpoint \
      --out results/stand_to_roll

Every PPO stage uses the same actor, the same fixed BC observation normalizer,
the same reward weights, and pure policy actions.  Only reset alpha changes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.config_stand_to_roll import (
    STAND_TO_ROLL_CURRICULUM_STAGES,
    StandToRollConfig,
    stand_to_roll_curriculum_config,
)
from curl_robot_2d_mjx.deployment_rolling_3d import (
    CONTROLLER_JOINT_NAMES_3D,
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
)
from curl_robot_2d_mjx.environment_3d import model_path_3d
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from curl_robot_2d_mjx.stand_to_roll_training import (
    STAND_TO_ROLL_ACTION_SIZE,
    STAND_TO_ROLL_HIDDEN_LAYERS,
    action_center_and_scale,
    build_cem_bc_dataset,
    initialize_ppo_actor_from_bc,
    observation_normalizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CEM_DATA = PROJECT_ROOT / "results" / "cem_cycle_data" / "cem_cycles.npz"
PRESETS = {
    "cpu_smoke": {
        "steps": 2048,
        "envs": 4,
        "eval_envs": 4,
        "num_evals": 2,
        "batch_size": 8,
        "num_minibatches": 2,
    },
    "smoke": {
        "steps": 131_072,
        "envs": 64,
        "eval_envs": 16,
        "num_evals": 4,
        "batch_size": 64,
        "num_minibatches": 4,
    },
    "4090": {
        "steps": 10_000_000,
        "envs": 512,
        "eval_envs": 64,
        "num_evals": 10,
        "batch_size": 256,
        "num_minibatches": 8,
    },
    "h200": {
        "steps": 20_000_000,
        "envs": 2048,
        "eval_envs": 256,
        "num_evals": 10,
        "batch_size": 512,
        "num_minibatches": 16,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("bc",) + STAND_TO_ROLL_CURRICULUM_STAGES,
                        required=True)
    parser.add_argument("--out", type=Path, default=Path("results/mjx_3d_stand_to_roll"))
    parser.add_argument("--cem-data", type=Path, default=DEFAULT_CEM_DATA)
    parser.add_argument("--bc-params", type=Path)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-layers", type=int, nargs="+",
                        default=STAND_TO_ROLL_HIDDEN_LAYERS)
    parser.add_argument("--initial-policy-std", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--entropy-cost", type=float, default=3.0e-4)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.cem_data.is_file() and not args.dry_run:
        parser.error(f"CEM data does not exist: {args.cem_data}")
    if args.stage != "bc" and args.bc_params is None and not args.dry_run:
        parser.error("PPO stages require --bc-params for the fixed normalizer")
    if args.stage != "bc" and args.stage != "compact" and args.restore_checkpoint is None:
        parser.error("post-compact stages must restore the preceding PPO checkpoint")
    if args.stage == "compact" and args.restore_checkpoint is not None:
        parser.error("compact starts from BC; do not pass --restore-checkpoint")
    if args.bc_steps < 1 or args.bc_batch_size < 1:
        parser.error("BC step and batch counts must be positive")
    for value, name in (
        (args.learning_rate, "learning rate"),
        (args.bc_learning_rate, "BC learning rate"),
        (args.initial_policy_std, "initial policy std"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    return args


def _controller_qpos_indices(model):
    import mujoco

    result = []
    for name in CONTROLLER_JOINT_NAMES_3D:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing controller joint: {name}")
        result.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(result)


def _train_bc(args, stage_out):
    import flax.linen as linen
    import jax
    import jax.numpy as jp
    import mujoco
    import optax
    from brax.io import model as model_io

    task = StandToRollConfig()
    center, scale = action_center_and_scale(task)
    model = mujoco.MjModel.from_xml_path(str(model_path_3d(task.geometry)))
    observations, targets = build_cem_bc_dataset(
        args.cem_data,
        controller_qpos_indices=_controller_qpos_indices(model),
        action_center=center,
        action_scale=scale,
    )
    normalizer = observation_normalizer(observations)
    normalized = (observations - normalizer["mean"]) / normalizer["std"]

    class BCPolicy(linen.Module):
        hidden_layers: tuple[int, ...]

        @linen.compact
        def __call__(self, observation):
            value = observation
            for index, width in enumerate(self.hidden_layers):
                value = linen.Dense(width, name=f"hidden_{index}")(value)
                value = linen.elu(value)
            return jp.tanh(linen.Dense(12, name="location")(value))

    policy = BCPolicy(tuple(args.hidden_layers))
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    params = policy.init(init_key, jp.zeros((1, 720), dtype=jp.float32))
    optimizer = optax.adam(args.bc_learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(current, opt_state, obs, target):
        def loss_fn(p):
            prediction = policy.apply(p, obs)
            error = prediction - target
            return jp.mean(jp.square(error)), (
                jp.sqrt(jp.mean(jp.square(error))), jp.max(jp.abs(error))
            )
        (loss, diagnostics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(current)
        updates, next_opt_state = optimizer.update(gradients, opt_state, current)
        return optax.apply_updates(current, updates), next_opt_state, loss, diagnostics

    history = []
    for step in range(args.bc_steps):
        rng, batch_key = jax.random.split(rng)
        indices = jax.random.randint(
            batch_key, (args.bc_batch_size,), 0, normalized.shape[0]
        )
        params, optimizer_state, loss, diagnostics = update(
            params,
            optimizer_state,
            jp.asarray(normalized)[indices],
            jp.asarray(targets)[indices],
        )
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.bc_steps:
            row = {
                "step": step + 1,
                "loss": float(loss),
                "rmse": float(diagnostics[0]),
                "max_abs_error": float(diagnostics[1]),
            }
            history.append(row)
            print(f"[BC] {row}", flush=True)

    checkpoint = (
        {name: np.asarray(value) for name, value in normalizer.items()},
        jax.tree_util.tree_map(np.asarray, params),
    )
    model_io.save_params(stage_out / "bc_params", checkpoint)
    report = {
        "samples": int(observations.shape[0]),
        "observation_size": int(observations.shape[1]),
        "action_size": int(targets.shape[1]),
        "cem_data": str(args.cem_data.resolve()),
        "action_center": center.tolist(),
        "action_scale": scale.tolist(),
        "history": history,
    }
    (stage_out / "bc_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _train_ppo(args, stage_out):
    import jax
    import jax.numpy as jp
    import jax.nn as jnn
    from brax.io import model as model_io
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment_stand_to_roll_3d import (
        make_stand_to_roll_env_3d,
    )

    bc_normalizer, bc_params = model_io.load_params(args.bc_params)
    mean = jp.asarray(bc_normalizer["mean"])
    std = jp.asarray(bc_normalizer["std"])
    if mean.shape != (720,) or std.shape != (720,) or bool(jp.any(std <= 0.0)):
        raise ValueError("BC normalizer must contain positive 720-value mean/std")
    bc_params = jax.tree_util.tree_map(jp.asarray, bc_params)
    task = stand_to_roll_curriculum_config(args.stage)
    train_env = make_stand_to_roll_env_3d(task, matcher_npz=args.cem_data, seed=args.seed)
    eval_env = make_stand_to_roll_env_3d(
        StandToRollConfig(**{**asdict(task), "observation_noise_enabled": False}),
        matcher_npz=args.cem_data,
        seed=args.seed + 10_000,
    )
    if train_env.observation_size != 720 or train_env.action_size != 12:
        raise RuntimeError("stand-to-roll actor contract must be obs=720, action=12")

    def fixed_preprocess(observation, unused_statistics):
        del unused_statistics
        return (observation - mean) / std

    def network_factory(observation_size, action_size, preprocess_observations_fn):
        del preprocess_observations_fn
        networks = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=fixed_preprocess,
            policy_hidden_layer_sizes=tuple(args.hidden_layers),
            value_hidden_layer_sizes=tuple(args.hidden_layers),
            activation=jnn.elu,
            distribution_type="tanh_normal",
        )
        if args.restore_checkpoint is None:
            original_init = networks.policy_network.init

            def init(key):
                return initialize_ppo_actor_from_bc(
                    jp,
                    original_init(key),
                    bc_params,
                    hidden_layers=tuple(args.hidden_layers),
                    initial_std=args.initial_policy_std,
                )

            networks = replace(
                networks,
                policy_network=replace(networks.policy_network, init=init),
            )
        return networks

    preset = PRESETS[args.preset]
    progress_history = []

    def progress(step, metrics):
        row = {"step": int(step)}
        for name, value in metrics.items():
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                row[name] = float(value.item())
        progress_history.append(row)
        print(
            f"[PPO {args.stage}] step={step} "
            f"capture={row.get('eval/episode_captured', 0.0):.3f} "
            f"roll={row.get('eval/episode_roll_progress', 0.0):.3f} "
            f"failed={row.get('eval/episode_failed', 0.0):.3f}",
            flush=True,
        )

    train_parameters = inspect.signature(ppo.train).parameters
    kwargs = {}
    if "save_checkpoint_path" in train_parameters:
        kwargs["save_checkpoint_path"] = str((stage_out / "ppo_checkpoint").resolve())
    if args.restore_checkpoint is not None:
        if "restore_checkpoint_path" not in train_parameters:
            raise RuntimeError("installed Brax PPO cannot restore curriculum checkpoints")
        kwargs["restore_checkpoint_path"] = str(args.restore_checkpoint.resolve())

    started = time.perf_counter()
    _, params, final_metrics = ppo.train(
        environment=train_env,
        eval_env=eval_env,
        num_timesteps=preset["steps"],
        episode_length=task.episode_length,
        action_repeat=1,
        num_envs=preset["envs"],
        num_eval_envs=preset["eval_envs"],
        num_evals=preset["num_evals"],
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=1.0,
        unroll_length=args.unroll_length,
        batch_size=preset["batch_size"],
        num_minibatches=preset["num_minibatches"],
        num_updates_per_batch=args.updates_per_batch,
        normalize_observations=False,
        deterministic_eval=True,
        network_factory=network_factory,
        seed=args.seed,
        progress_fn=progress,
        **kwargs,
    )
    model_io.save_params(stage_out / "params_final", params)
    clean_metrics = {name: float(value) for name, value in (final_metrics or {}).items()}
    capture_rate = clean_metrics.get("eval/episode_captured", 0.0)
    failure_rate = clean_metrics.get("eval/episode_failed", 1.0)
    summary = {
        "stage": args.stage,
        "task": asdict(task),
        "elapsed_s": time.perf_counter() - started,
        "final_metrics": clean_metrics,
        "capture_rate": capture_rate,
        "failure_rate": failure_rate,
        "stage_passed": capture_rate >= 0.80 and failure_rate <= 0.20,
        "next_stage": (
            STAND_TO_ROLL_CURRICULUM_STAGES[
                STAND_TO_ROLL_CURRICULUM_STAGES.index(args.stage) + 1
            ]
            if args.stage != STAND_TO_ROLL_CURRICULUM_STAGES[-1]
            else None
        ),
    }
    (stage_out / "metrics_history.json").write_text(
        json.dumps(progress_history, indent=2) + "\n", encoding="utf-8"
    )
    (stage_out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    task = None if args.stage == "bc" else stand_to_roll_curriculum_config(args.stage)
    payload = {
        "pipeline": "one_policy_bc_then_reset_curriculum_v1",
        "stage": args.stage,
        "actor_observation": "train_ppo_deploy 36x20 newest-first",
        "actor_observation_size": ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
        "action_size": STAND_TO_ROLL_ACTION_SIZE,
        "cem_online_control": False,
        "teacher_shaping_annealed": False,
        "task": asdict(task) if task else None,
        "preset": PRESETS[args.preset],
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return payload
    stage_out = args.out / args.stage
    if stage_out.exists() and any(stage_out.iterdir()):
        raise SystemExit(f"output directory is not empty: {stage_out}")
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=False,
        xla_triton=False,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )
    stage_out.mkdir(parents=True, exist_ok=True)
    payload["runtime"] = describe_runtime()
    (stage_out / "training_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    result = _train_bc(args, stage_out) if args.stage == "bc" else _train_ppo(args, stage_out)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
