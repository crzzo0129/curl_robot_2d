"""Train a from-scratch PPO rolling policy at the current nominal COM."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.config import (
    PHYSICS_PROFILE_NAMES,
    NominalRLConfig,
    physics_profile,
)
from curl_robot_2d_mjx.reward import REWARD_TERM_NAMES
from curl_robot_2d_mjx.reward_config import RollingRewardConfig
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    describe_runtime,
)


PRESETS = {
    "smoke": {
        "steps": 65_536,
        "envs": 64,
        "eval_envs": 8,
        "num_evals": 4,
        "batch_size": 64,
        "num_minibatches": 4,
    },
    "4090": {
        "steps": 20_000_000,
        "envs": 512,
        "eval_envs": 64,
        "num_evals": 10,
        "batch_size": 512,
        "num_minibatches": 16,
    },
    "h200": {
        "steps": 50_000_000,
        "envs": 2048,
        "eval_envs": 256,
        "num_evals": 10,
        "batch_size": 1024,
        "num_minibatches": 32,
    },
}


def _add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose reward dataclass fields without duplicating their defaults."""

    for field in fields(RollingRewardConfig):
        option = f"--reward-{field.name.replace('_', '-')}"
        parser.add_argument(
            option,
            dest=f"reward_{field.name}",
            type=float,
            default=None,
            help=(
                f"Override RollingRewardConfig.{field.name}; "
                "the default comes from reward_config.py."
            ),
        )


def _reward_config_from_args(args) -> RollingRewardConfig:
    overrides = {
        field.name: value
        for field in fields(RollingRewardConfig)
        if (
            value := getattr(args, f"reward_{field.name}", None)
        )
        is not None
    }
    return replace(RollingRewardConfig(), **overrides)


def _resolve_restore_checkpoint(path: Path) -> Path:
    """Resolve a Brax checkpoint root to its latest numbered child."""

    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        return path
    numbered = sorted(
        (
            child
            for child in path.iterdir()
            if child.is_dir() and child.name.isdigit()
        ),
        key=lambda child: int(child.name),
    )
    return numbered[-1] if numbered else path


def _float(value):
    try:
        return float(value)
    except TypeError:
        return float(value.item())


PER_STEP_EVAL_METRICS = (
    "root_height_m",
    "foot_center_distance_m",
    "action_rms",
    "action_rate_rms",
    "startup_action_ramp",
    "normalized_torque_rms",
    "forbidden_contact_count",
    "forbidden_penetration_m",
    "allowed_foot_penetration_m",
    "ground_contact_count",
    "roll_progress_rad",
    "phase_progress_rad",
    "translation_progress_rad",
)


def _add_per_step_eval_metrics(metrics):
    episode_length = metrics.get("eval/avg_episode_length")
    if episode_length is None or episode_length <= 0:
        return
    for name in PER_STEP_EVAL_METRICS:
        key = f"eval/episode_{name}"
        if key in metrics:
            metrics[f"eval/avg_{name}"] = metrics[key] / episode_length
    for name in REWARD_TERM_NAMES:
        key = f"eval/episode_reward_{name}"
        if key in metrics:
            metrics[f"eval/avg_reward_{name}"] = (
                metrics[key] / episode_length
            )
    if "eval/episode_reward" in metrics:
        metrics["eval/avg_reward"] = (
            metrics["eval/episode_reward"] / episode_length
        )


def _is_reward_metric(name: str) -> bool:
    return (
        name == "reward"
        or name == "reward_total"
        or name.startswith("reward_")
        or name.startswith("eval/episode_reward")
        or "/avg_reward" in name
    )


def _split_metrics(metrics):
    reward_metrics = {
        name: value
        for name, value in metrics.items()
        if _is_reward_metric(name)
    }
    ordinary_metrics = {
        name: value
        for name, value in metrics.items()
        if not _is_reward_metric(name)
    }
    return reward_metrics, ordinary_metrics


def _network_factory(hidden_layers, activation_name):
    import jax.nn as jnn
    from brax.training.agents.ppo import networks

    activation = {
        "elu": jnn.elu,
        "relu": jnn.relu,
        "swish": jnn.swish,
        "tanh": jnn.tanh,
    }[activation_name]

    def factory(*args, **kwargs):
        return networks.make_ppo_networks(
            *args,
            policy_hidden_layer_sizes=tuple(hidden_layers),
            activation=activation,
            **kwargs,
        )

    return factory


def _evaluate_policy(
    env,
    make_inference_fn,
    params,
    *,
    seed,
    episode_length,
    output_dir,
):
    import jax

    try:
        policy = make_inference_fn(params, deterministic=True)
    except TypeError:
        policy = make_inference_fn(params)
    policy_step = jax.jit(policy)
    env_reset = jax.jit(env.reset)
    env_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = env_reset(rng)
    initial_phase = _float(
        state.pipeline_state.qpos[env.root_pitch_qpos]
    )
    initial_x = _float(state.pipeline_state.qpos[env.root_x_qpos])
    qpos_rows = []
    action_rows = []
    reward_rows = []
    metric_totals = {}
    reward_term_totals = {name: 0.0 for name in REWARD_TERM_NAMES}

    for _ in range(episode_length):
        rng, action_key = jax.random.split(rng)
        action, _ = policy_step(state.obs, action_key)
        state = env_step(state, action)
        qpos_rows.append(
            np.asarray(jax.device_get(state.pipeline_state.qpos))
        )
        action_rows.append(np.asarray(jax.device_get(action)))
        reward_rows.append(_float(state.reward))
        for name, value in state.metrics.items():
            scalar = _float(value)
            if name.startswith("reward_") and name not in (
                "reward_total",
            ):
                term_name = name.removeprefix("reward_")
                reward_term_totals[term_name] = (
                    reward_term_totals.get(term_name, 0.0) + scalar
                )
            elif name not in ("reward", "reward_total"):
                metric_totals[name] = (
                    metric_totals.get(name, 0.0) + scalar
                )
        if _float(state.done) > 0.5:
            break

    final_phase = _float(
        state.pipeline_state.qpos[env.root_pitch_qpos]
    )
    final_x = _float(state.pipeline_state.qpos[env.root_x_qpos])
    steps = len(reward_rows)
    metric_averages = {
        name: value / max(steps, 1)
        for name, value in metric_totals.items()
    }
    failure_reasons = {
        name.removeprefix("failure_"): bool(metric_totals.get(name, 0.0))
        for name in (
            "failure_nonfinite",
            "failure_nonfinite_action",
            "failure_nonfinite_physics",
            "failure_root_low",
            "failure_root_high",
            "failure_foot_gap",
            "failure_leg_crossing",
        )
    }
    summary = {
        "episode_steps": steps,
        "episode_duration_s": (
            steps
            * float(env.mj_model.opt.timestep)
            * env.config.action_repeat
        ),
        "total_reward": float(sum(reward_rows)),
        "net_phase_rad": final_phase - initial_phase,
        "net_turns": (final_phase - initial_phase) / (2.0 * math.pi),
        "root_x_displacement_m": final_x - initial_x,
        "terminated": bool(_float(state.done) > 0.5),
        "reward_breakdown": {
            "total": float(sum(reward_rows)),
            "terms": reward_term_totals,
            "per_step": {
                name: value / max(steps, 1)
                for name, value in reward_term_totals.items()
            },
        },
        "metrics": {
            "totals": metric_totals,
            "per_step_averages": metric_averages,
        },
        "failure_reasons": failure_reasons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "evaluation_rollout.npz",
        qpos=np.asarray(qpos_rows),
        action=np.asarray(action_rows),
        reward=np.asarray(reward_rows),
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=tuple(PRESETS), default="smoke"
    )
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILE_NAMES,
        default="cg12",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument(
        "--hidden-layers", type=int, nargs="+", default=[256, 256, 128]
    )
    parser.add_argument(
        "--activation",
        choices=("elu", "relu", "swish", "tanh"),
        default="elu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument(
        "--mujoco-gl",
        default="disable",
        help="Use 'disable' for headless training or 'egl' when available.",
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
    parser.add_argument(
        "--runtime-diagnostics", action="store_true", default=True
    )
    parser.add_argument(
        "--no-runtime-diagnostics",
        dest="runtime_diagnostics",
        action="store_false",
    )
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "mjx_ppo_nominal",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Explicitly allow writing into a non-empty output directory.",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    _add_reward_arguments(parser)
    args = parser.parse_args()

    values = PRESETS[args.preset].copy()
    for name in (
        "steps",
        "envs",
        "eval_envs",
        "num_evals",
        "batch_size",
        "num_minibatches",
    ):
        override = getattr(args, name)
        if override is not None:
            values[name] = override
    if (
        args.out.exists()
        and any(args.out.iterdir())
        and not args.allow_existing_output
    ):
        raise SystemExit(
            f"Output directory is not empty: {args.out}. "
            "Use a new --out path so historical results are preserved."
        )

    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=args.preallocate,
        xla_triton=args.xla_triton,
        mujoco_gl=args.mujoco_gl,
        verbose=args.runtime_diagnostics,
    )
    import jax
    from brax.io import model as model_io
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment import make_brax_env

    args.out.mkdir(parents=True, exist_ok=True)
    runtime = describe_runtime()
    print(json.dumps(runtime, indent=2), flush=True)
    task = physics_profile(
        args.physics_profile,
        NominalRLConfig(episode_length=args.episode_length),
    )
    reward_config = _reward_config_from_args(args)
    train_env = make_brax_env(
        task, reward_config=reward_config, seed=args.seed
    )
    eval_env = make_brax_env(
        task, reward_config=reward_config, seed=args.seed + 10_000
    )

    metric_history = []
    reward_history = []
    best = {
        "reward": float("-inf"),
        "step": None,
        "params": None,
        "candidate_step": None,
        "candidate_params": None,
    }

    def policy_params_fn(step, make_policy, params):
        del make_policy
        best["candidate_step"] = int(step)
        best["candidate_params"] = params
        if best["step"] == int(step):
            best["params"] = params

    def progress_fn(step, metrics):
        clean = {name: _float(value) for name, value in metrics.items()}
        _add_per_step_eval_metrics(clean)
        reward_metrics, ordinary_metrics = _split_metrics(clean)
        reward_history.append({"step": int(step), **reward_metrics})
        metric_history.append({"step": int(step), **ordinary_metrics})
        reward = clean.get(
            "eval/episode_reward",
            clean.get("eval/episode_reward_mean"),
        )
        if reward is not None and reward > best["reward"]:
            best["reward"] = reward
            best["step"] = int(step)
            if best["candidate_step"] == int(step):
                best["params"] = best["candidate_params"]
        message = f"steps={int(step)}"
        if reward is not None:
            message += f" eval_reward={reward:.4f}"
        if "eval/episode_length" in clean:
            message += (
                f" eval_length={clean['eval/episode_length']:.1f}"
            )
        if "eval/avg_episode_length" in clean:
            message += (
                f" avg_length={clean['eval/avg_episode_length']:.1f}"
            )
        if "eval/episode_failed" in clean:
            message += f" failed={clean['eval/episode_failed']:.2f}"
        for short_name, metric_name in (
            ("low", "eval/episode_failure_root_low"),
            ("high", "eval/episode_failure_root_high"),
            ("gap", "eval/episode_failure_foot_gap"),
            ("cross", "eval/episode_failure_leg_crossing"),
            ("nan", "eval/episode_failure_nonfinite"),
            ("nan_action", "eval/episode_failure_nonfinite_action"),
            ("nan_physics", "eval/episode_failure_nonfinite_physics"),
        ):
            if clean.get(metric_name, 0.0) > 0.0:
                message += f" fail_{short_name}={clean[metric_name]:.2f}"
        for short_name, metric_name in (
            ("phase", "eval/avg_phase_progress_rad"),
            ("translation", "eval/avg_translation_progress_rad"),
            ("roll", "eval/avg_roll_progress_rad"),
        ):
            if metric_name in clean:
                message += (
                    f" avg_{short_name}_step={clean[metric_name]:.4f}"
                )
        print(message, flush=True)

    config_payload = {
        "preset": args.preset,
        **values,
        "episode_length": args.episode_length,
        "unroll_length": args.unroll_length,
        "updates_per_batch": args.updates_per_batch,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "reward_scaling": args.reward_scaling,
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "seed": args.seed,
        "restore_checkpoint": (
            str(args.restore_checkpoint)
            if args.restore_checkpoint is not None
            else None
        ),
        "task": asdict(task),
        "reward": asdict(reward_config),
        "runtime": runtime,
    }
    (args.out / "training_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "reward_config.json").write_text(
        json.dumps(asdict(reward_config), indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "stage=train_start "
        f"preset={args.preset} steps={values['steps']} "
        f"envs={values['envs']} episode_length={args.episode_length}",
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
            raise SystemExit(
                "Installed Brax does not support restore_checkpoint_path."
            )
        checkpoint_kwargs["restore_checkpoint_path"] = str(
            _resolve_restore_checkpoint(args.restore_checkpoint)
        )
        print(
            "stage=checkpoint_restore "
            f"path={checkpoint_kwargs['restore_checkpoint_path']}",
            flush=True,
        )

    make_inference_fn, final_params, final_metrics = ppo.train(
        environment=train_env,
        eval_env=eval_env,
        num_timesteps=values["steps"],
        episode_length=args.episode_length,
        action_repeat=1,
        num_envs=values["envs"],
        num_evals=values["num_evals"],
        num_eval_envs=values["eval_envs"],
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=args.reward_scaling,
        unroll_length=args.unroll_length,
        batch_size=values["batch_size"],
        num_minibatches=values["num_minibatches"],
        num_updates_per_batch=args.updates_per_batch,
        normalize_observations=True,
        network_factory=_network_factory(
            args.hidden_layers, args.activation
        ),
        seed=args.seed,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        **checkpoint_kwargs,
    )
    elapsed = time.perf_counter() - start
    best_params = best["params"] if best["params"] is not None else final_params
    model_io.save_params(args.out / "params_final", final_params)
    model_io.save_params(args.out / "params_best", best_params)
    (args.out / "metrics_history.json").write_text(
        json.dumps(metric_history, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "reward_history.json").write_text(
        json.dumps(reward_history, indent=2) + "\n", encoding="utf-8"
    )
    clean_final_metrics = {
        name: _float(value)
        for name, value in (final_metrics or {}).items()
    }
    _add_per_step_eval_metrics(clean_final_metrics)
    final_reward_metrics, final_ordinary_metrics = _split_metrics(
        clean_final_metrics
    )
    train_summary = {
        "elapsed_s": elapsed,
        "best_eval_reward": best["reward"],
        "best_step": best["step"],
        "final_metrics": final_ordinary_metrics,
        "final_reward_metrics": final_reward_metrics,
    }
    (args.out / "training_summary.json").write_text(
        json.dumps(train_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(train_summary, indent=2), flush=True)

    if not args.skip_evaluation:
        evaluation = _evaluate_policy(
            eval_env,
            make_inference_fn,
            best_params,
            seed=args.seed + 20_000,
            episode_length=args.episode_length,
            output_dir=args.out,
        )
        print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()
