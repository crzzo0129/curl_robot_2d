"""Train the 3-D BRAKE + DEPLOY + STABILIZE transition policy with PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.config_transition_3d import (
    TRANSITION_ACTOR_OBSERVATION_SIZE_3D,
    TRANSITION_CRITIC_OBSERVATION_SIZE_3D,
    TRANSITION_CURRICULUM_STAGE_NAMES_3D,
    TRANSITION_GEOMETRY_NAMES_3D,
    TRANSITION_PHYSICS_PROFILE_NAMES_3D,
    Transition3DConfig,
    transition_curriculum_config_3d,
    transition_physics_profile_3d,
)
from curl_robot_2d_mjx.reward_transition_3d import Transition3DRewardConfig
from curl_robot_2d_mjx.failure_transition_3d import (
    TRANSITION_FAILURE_CAUSE_NAMES_3D,
    transition_failure_breakdown_3d,
    transition_failure_mode_breakdown_3d,
    transition_source_breakdown_3d,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from curl_robot_2d_mjx.training_transition_3d import (
    TRANSITION_INITIAL_POLICY_STD, TRANSITION_TRAINING_REVISION,
    initialize_transition_actor, transition_scale_logit, transition_curriculum_acceptance,
    resolve_transition_checkpoint,
)
from curl_robot_2d_mjx.transition_snapshot_cli_3d import (
    add_cycle_selection_arguments, apply_cycle_selection_arguments,
)


PRESETS_TRANSITION_3D = {
    "smoke": {
        "steps": 131_072,
        "envs": 64,
        "eval_envs": 8,
        "num_evals": 4,
        "batch_size": 64,
        "num_minibatches": 4,
    },
    "4090": {
        "steps": 8_000_000,
        "envs": 512,
        "eval_envs": 64,
        "num_evals": 10,
        "batch_size": 512,
        "num_minibatches": 16,
    },
    "h200": {
        "steps": 16_000_000,
        "envs": 2048,
        "eval_envs": 256,
        "num_evals": 10,
        "batch_size": 1024,
        "num_minibatches": 32,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=TRANSITION_CURRICULUM_STAGE_NAMES_3D,
        default="walking_start",
    )
    parser.add_argument("--geometry", choices=TRANSITION_GEOMETRY_NAMES_3D,
                        default="rollingquad_2")
    parser.add_argument("--roll-snapshots", type=Path)
    parser.add_argument("--snapshot-tail-fraction", type=float, default=1.0)
    add_cycle_selection_arguments(parser)
    parser.add_argument("--preset", choices=tuple(PRESETS_TRANSITION_3D), default="smoke")
    parser.add_argument(
        "--physics-profile",
        choices=TRANSITION_PHYSICS_PROFILE_NAMES_3D,
        default="newton4",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/mjx_3d_transition_ppo")
    )
    parser.add_argument("--restore-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="load frozen Transition weights and run independent episodes without PPO updates",
    )
    parser.add_argument(
        "--eval-params", type=Path, default=None,
        help="params_final/params_best file for --eval-only (alternative to an Orbax checkpoint)",
    )
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--entropy-cost", type=float, default=3.0e-4)
    parser.add_argument("--initial-policy-std", type=float, default=TRANSITION_INITIAL_POLICY_STD,
                        help="fresh actor pre-tanh std; never changes restored weights")
    parser.add_argument("--discounting", type=float, default=0.985)
    parser.add_argument("--unroll-length", type=int, default=24)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=(256, 256, 128))
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--mujoco-gl", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_task(args) -> Transition3DConfig:
    task = transition_curriculum_config_3d(
        args.stage, Transition3DConfig(
            geometry=args.geometry, curriculum_stage=args.stage,
            roll_snapshots_path=str(args.roll_snapshots.resolve())
            if args.roll_snapshots else None,
            snapshot_tail_fraction=args.snapshot_tail_fraction,
        )
    )
    return transition_physics_profile_3d(
        args.physics_profile, apply_cycle_selection_arguments(task, args))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value.item())


def make_transition_networks(observation_size, action_size,
                             preprocess_observations_fn, *, hidden_layers=(256, 256, 128),
                             initial_std=TRANSITION_INITIAL_POLICY_STD):
    """One shared actor architecture for PPO and cloud export parity tests."""
    from brax.training.agents.ppo import networks as ppo_networks
    import jax.nn as jnn
    import jax.numpy as jp
    transition_scale_logit(initial_std)
    nets = ppo_networks.make_ppo_networks(
        observation_size, action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=tuple(hidden_layers),
        value_hidden_layer_sizes=tuple(hidden_layers),
        activation=jnn.elu,
        policy_obs_key="state", value_obs_key="privileged_state",
    )
    original_init = nets.policy_network.init

    def init(key):
        return initialize_transition_actor(
            jp, original_init(key), hidden_layers, action_size, initial_std)

    return replace(nets, policy_network=replace(nets.policy_network, init=init))


def evaluate_transition_policy(env, policy, *, count, seed):
    """Evaluate frozen weights in independent raw-environment episodes.

    This deliberately bypasses PPO training and its auto-reset wrapper.  Every
    terminal metric is accumulated once per lane, which makes phase/cycle
    conditional rates auditable from the saved arrays.
    """
    import jax

    keys = jax.random.split(jax.random.PRNGKey(seed), count)
    state = jax.jit(jax.vmap(env.reset))(keys)

    def one_step(single_state, key):
        action = policy(single_state.obs, key)[0]
        return env.step(single_state, action)

    batched_step = jax.jit(jax.vmap(one_step))
    totals = {name: np.zeros(count, dtype=np.float64)
              for name in env._zero_metrics()}
    active = np.ones(count, dtype=bool)
    terminal_step = np.zeros(count, dtype=np.int32)
    for step_index in range(env.config.episode_length):
        step_keys = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(seed + 1), step_index), count
        )
        next_state = batched_step(state, step_keys)
        metrics = jax.device_get(next_state.metrics)
        for name in totals:
            values = np.asarray(metrics[name], dtype=np.float64)
            totals[name] += values * active
        done = np.asarray(jax.device_get(next_state.done)) > 0
        newly_done = active & done
        terminal_step[newly_done] = step_index + 1
        active &= ~done
        state = next_state
        if step_index % 50 == 49:
            print(
                f"[transition eval-only] step={step_index + 1}/"
                f"{env.config.episode_length} active={int(active.sum())}/{count}",
                flush=True,
            )
        if not active.any():
            break

    means = {
        "eval/episode_" + name: float(values.mean())
        for name, values in totals.items()
    }
    success = totals["transition_success"] > 0
    failed = totals["failed"] > 0
    timeout = totals["timeout"] > 0
    report = {
        "episodes": count,
        "seed": seed,
        "success_rate": float(success.mean()),
        "failure_rate": float(failed.mean()),
        "timeout_rate": float(timeout.mean()),
        "outcome_consistency_error": float(
            success.mean() + failed.mean() + timeout.mean() - 1.0
        ),
        "mean_episode_length": float(terminal_step.mean()),
        "incomplete_episodes": int(active.sum()),
        "failure_breakdown": transition_failure_breakdown_3d(means),
        "failure_mode_breakdown": transition_failure_mode_breakdown_3d(means),
        "source_breakdown": transition_source_breakdown_3d(
            means, phase_bins=env.source_phase_bins, cycles=env.source_cycles,
            episode_count=count,
        ),
        "passes_nominal_curriculum_thresholds": bool(
            success.mean() >= 0.9 and failed.mean() <= 0.05
            and timeout.mean() <= 0.05 and not active.any()
        ),
        "policy_updates": 0,
    }
    arrays = {**totals, "terminal_step": terminal_step}
    return report, arrays


def main(argv=None) -> None:
    args = parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive")
    if not math.isfinite(args.entropy_cost) or args.entropy_cost < 0:
        raise SystemExit("--entropy-cost must be finite and nonnegative")
    transition_scale_logit(args.initial_policy_std)
    if args.unroll_length < 1 or args.updates_per_batch < 1:
        raise SystemExit("rollout and update lengths must be positive")
    if args.eval_envs < 1:
        raise SystemExit("--eval-envs must be positive")
    if args.eval_only:
        if (args.restore_checkpoint is None) == (args.eval_params is None):
            raise SystemExit(
                "--eval-only requires exactly one of --restore-checkpoint or --eval-params"
            )
    elif args.eval_params is not None:
        raise SystemExit("--eval-params is only valid with --eval-only")

    task = build_task(args)
    if args.stage.startswith("brake_") and not args.roll_snapshots and not args.dry_run:
        raise SystemExit("BRAKE stages require --roll-snapshots; no synthetic "
                         "or artificially slowed fallback is used")
    reward = Transition3DRewardConfig()
    preset = PRESETS_TRANSITION_3D[args.preset]
    stage_out = args.out / args.stage
    payload = {
        "training_revision": TRANSITION_TRAINING_REVISION,
        "contract_version": "transition_neural_controller_36x20_v3",
        "actor_observation_size": TRANSITION_ACTOR_OBSERVATION_SIZE_3D,
        "critic_observation_size": TRANSITION_CRITIC_OBSERVATION_SIZE_3D,
        "actor_activation": "elu",
        "actor_distribution": "default_tanh_normal",
        "control": "one policy; fixed Walking action center; no external brake",
        "reset_source": "roll_snapshots" if args.stage.startswith("brake_")
                        else "walking_start_neighborhood",
        "task": asdict(task),
        "reward": asdict(reward),
        "training": {
            **preset,
            "learning_rate": args.learning_rate,
            "entropy_cost": args.entropy_cost,
            "discounting": args.discounting,
            "unroll_length": args.unroll_length,
            "updates_per_batch": args.updates_per_batch,
            "hidden_layers": list(args.hidden_layers),
            "initial_policy_std": args.initial_policy_std,
            "actor_init": "zero_location_fixed_small_initial_scale",
            "episode_reset": "full_task_state_with_fresh_rng",
        },
        "run_mode": "evaluation_only" if args.eval_only else "training",
        "eval_only": {
            "episodes": args.eval_envs,
            "seed": args.eval_seed if args.eval_seed is not None else args.seed + 20_000,
            "params": str(args.eval_params.resolve()) if args.eval_params else None,
            "policy_updates": 0,
        } if args.eval_only else None,
        "restore_checkpoint": (
            str(args.restore_checkpoint.resolve())
            if args.restore_checkpoint is not None
            else None
        ),
        "curriculum_order": list(TRANSITION_CURRICULUM_STAGE_NAMES_3D),
        "curriculum_next_stage": (
            TRANSITION_CURRICULUM_STAGE_NAMES_3D[
                TRANSITION_CURRICULUM_STAGE_NAMES_3D.index(args.stage) + 1
            ]
            if args.stage != TRANSITION_CURRICULUM_STAGE_NAMES_3D[-1]
            else None
        ),
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.restore_checkpoint is not None:
        requested = args.restore_checkpoint
        args.restore_checkpoint = resolve_transition_checkpoint(requested)
        payload["restore_checkpoint_requested"] = str(requested.resolve())
        payload["restore_checkpoint"] = str(args.restore_checkpoint)
        print(f"[restore Transition] {args.restore_checkpoint}", flush=True)
    if args.eval_params is not None:
        args.eval_params = args.eval_params.resolve()
        if not args.eval_params.is_file():
            raise ValueError(f"evaluation params file does not exist: {args.eval_params}")
        payload["eval_only"]["params"] = str(args.eval_params)

    if stage_out.exists() and any(stage_out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {stage_out}. "
                         "Use a new --out; existing weights/logs will not be overwritten.")

    # Validate cycle/phase coverage before importing JAX or creating outputs.
    if args.stage.startswith("brake_"):
        from scripts.inspect_transition_roll_snapshots import inspect_bank
        payload["snapshot_selection"] = inspect_bank(args.roll_snapshots, task)
        print("[ROLL snapshot selection] " + json.dumps(payload["snapshot_selection"],
                                                       sort_keys=True), flush=True)

    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )
    from brax.io import model as model_io

    from curl_robot_2d_mjx.environment_transition_3d import (
        make_brax_transition_env_3d,
    )
    if not args.eval_only:
        from brax.training.agents.ppo import train as ppo
        from curl_robot_2d_mjx.wrappers_transition_3d import wrap_transition_3d
        if "wrap_env_fn" not in inspect.signature(ppo.train).parameters:
            raise SystemExit("Installed Brax must support wrap_env_fn for full Transition resets")

    payload["runtime"] = describe_runtime()
    stage_out.mkdir(parents=True, exist_ok=True)
    (stage_out / "training_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if "snapshot_selection" in payload:
        (stage_out / "snapshot_selection.json").write_text(
            json.dumps(payload["snapshot_selection"], indent=2) + "\n", encoding="utf-8")
    eval_env = make_brax_transition_env_3d(
        replace(task, observation_noise_enabled=False),
        reward_config=reward, seed=args.seed + 10_000
    )
    train_env = None if args.eval_only else make_brax_transition_env_3d(
        task, reward_config=reward, seed=args.seed
    )
    from curl_robot_2d_mjx.deployment_transition_3d import transition_controller_metadata_3d
    (stage_out / "deployment_config.json").write_text(
        json.dumps(transition_controller_metadata_3d(eval_env.mj_model, task),
                   indent=2) + "\n", encoding="utf-8")

    def network_factory(observation_size, action_size, preprocess_observations_fn):
        return make_transition_networks(
            observation_size, action_size, preprocess_observations_fn,
            hidden_layers=args.hidden_layers, initial_std=args.initial_policy_std)

    if args.eval_only:
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
        networks = network_factory(
            eval_env.observation_size, eval_env.action_size,
            running_statistics.normalize,
        )
        if args.eval_params is not None:
            params = model_io.load_params(args.eval_params)
            parameter_source = str(args.eval_params)
        else:
            from brax.training.agents.ppo import checkpoint as ppo_checkpoint
            params = ppo_checkpoint.load(str(args.restore_checkpoint))
            parameter_source = str(args.restore_checkpoint)
        policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
        eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 20_000
        started = time.perf_counter()
        report, arrays = evaluate_transition_policy(
            eval_env, policy, count=args.eval_envs, seed=eval_seed
        )
        report.update(
            stage=args.stage,
            training_revision=TRANSITION_TRAINING_REVISION,
            parameter_source=parameter_source,
            elapsed_s=time.perf_counter() - started,
            snapshot_selection=payload.get("snapshot_selection"),
        )
        (stage_out / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(stage_out / "evaluation_arrays.npz", **arrays)
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    history = []

    def progress(step, metrics):
        clean = {name: _float(value) for name, value in metrics.items()}
        history.append({"step": int(step), **clean})
        success = clean.get("eval/episode_transition_success", 0.0)
        failed = clean.get("eval/episode_failed", 0.0)
        timeout = clean.get("eval/episode_timeout", 0.0)
        causes = sorted(((name, clean.get(f"eval/episode_failure_{name}", 0.0))
                         for name in TRANSITION_FAILURE_CAUSE_NAMES_3D),
                        key=lambda item: item[1], reverse=True)
        cause_text = ",".join(f"{name}={value:.3f}" for name, value in causes if value > 0)
        mode_report = transition_failure_mode_breakdown_3d(clean)
        mode_text = ",".join(
            f"{mode}={row['rates']['root_height_low']:.3f}"
            for mode, row in mode_report["by_mode"].items()
            if row["rates"]["root_height_low"] > 0
        )
        source_report = transition_source_breakdown_3d(
            clean, phase_bins=eval_env.source_phase_bins,
            cycles=eval_env.source_cycles, episode_count=preset["eval_envs"],
        )
        phase_rows = [
            (label, row["failed_rate"])
            for label, row in source_report["by_phase_bin"].items()
            if row["failed_rate"] is not None
        ]
        worst_phase = (
            max(phase_rows, key=lambda item: item[1]) if phase_rows else None
        )
        worst_phase_text = (
            "none" if worst_phase is None
            else f"{worst_phase[0]}:{worst_phase[1]:.3f}"
        )
        print(
            f"[transition eval] stage={args.stage} step={int(step)} "
            f"success={success:.3f} failure={failed:.3f} timeout={timeout:.3f} "
            f"causes=[{cause_text}] root_low_modes=[{mode_text}] "
            f"worst_phase_failure={worst_phase_text}",
            flush=True,
        )

    checkpoint_kwargs = {}
    train_parameters = inspect.signature(ppo.train).parameters
    if "save_checkpoint_path" in train_parameters:
        checkpoint_kwargs["save_checkpoint_path"] = str(
            (stage_out / "ppo_checkpoint").resolve()
        )
    if args.restore_checkpoint is not None:
        if "restore_checkpoint_path" not in train_parameters:
            raise SystemExit("Installed Brax cannot restore PPO checkpoints")
        checkpoint_kwargs["restore_checkpoint_path"] = str(
            args.restore_checkpoint.resolve()
        )

    print(
        f"[transition PPO] stage={args.stage} preset={args.preset} "
        f"steps={preset['steps']:,} envs={preset['envs']}",
        flush=True,
    )
    started = time.perf_counter()
    _, params, final_metrics = ppo.train(
        environment=train_env,
        eval_env=eval_env,
        wrap_env_fn=wrap_transition_3d,
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
        normalize_observations=True,
        deterministic_eval=True,
        network_factory=network_factory,
        seed=args.seed,
        progress_fn=progress,
        **checkpoint_kwargs,
    )
    model_io.save_params(stage_out / "params_final", params)
    (stage_out / "metrics_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    acceptance = transition_curriculum_acceptance(history)
    clean_final_metrics = {
        name: _float(value) for name, value in (final_metrics or {}).items()
    }
    summary = {
        "stage": args.stage,
        "training_revision": TRANSITION_TRAINING_REVISION,
        "elapsed_s": time.perf_counter() - started,
        "params": str((stage_out / "params_final").resolve()),
        "final_metrics": clean_final_metrics,
        "failure_breakdown": transition_failure_breakdown_3d(clean_final_metrics),
        "failure_mode_breakdown": transition_failure_mode_breakdown_3d(
            clean_final_metrics
        ),
        "source_breakdown": transition_source_breakdown_3d(
            clean_final_metrics, phase_bins=eval_env.source_phase_bins,
            cycles=eval_env.source_cycles, episode_count=preset["eval_envs"],
        ),
        "curriculum_next_stage": payload["curriculum_next_stage"],
        "stage_passed": acceptance["passed"],
        "snapshot_selection": payload.get("snapshot_selection"),
        "acceptance": acceptance,
        "next_stage": payload["curriculum_next_stage"] if acceptance["passed"] else None,
    }
    (stage_out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
