"""Train a from-scratch PPO rolling policy at the current nominal COM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.config import NominalRLConfig
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    describe_runtime,
)


PRESETS = {
    "smoke": {
        "steps": 200_000,
        "envs": 256,
        "eval_envs": 32,
        "num_evals": 4,
        "batch_size": 256,
        "num_minibatches": 8,
    },
    "4090": {
        "steps": 20_000_000,
        "envs": 2048,
        "eval_envs": 128,
        "num_evals": 10,
        "batch_size": 1024,
        "num_minibatches": 32,
    },
    "h200": {
        "steps": 50_000_000,
        "envs": 8192,
        "eval_envs": 512,
        "num_evals": 10,
        "batch_size": 2048,
        "num_minibatches": 32,
    },
}


def _float(value):
    try:
        return float(value)
    except TypeError:
        return float(value.item())


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
            metric_totals[name] = metric_totals.get(name, 0.0) + _float(
                value
            )
        if _float(state.done) > 0.5:
            break

    final_phase = _float(
        state.pipeline_state.qpos[env.root_pitch_qpos]
    )
    final_x = _float(state.pipeline_state.qpos[env.root_x_qpos])
    steps = len(reward_rows)
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
        "metric_totals": metric_totals,
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
        "--out",
        type=Path,
        default=Path("results") / "mjx_ppo_nominal",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
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

    configure_cloud_runtime(memory_fraction=args.memory_fraction)
    import jax
    from brax.io import model as model_io
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment import make_brax_env

    args.out.mkdir(parents=True, exist_ok=True)
    runtime = describe_runtime()
    print(json.dumps(runtime, indent=2), flush=True)
    task = NominalRLConfig(episode_length=args.episode_length)
    train_env = make_brax_env(task, seed=args.seed)
    eval_env = make_brax_env(task, seed=args.seed + 10_000)

    history = []
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
        row = {"step": int(step), **clean}
        history.append(row)
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
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "seed": args.seed,
        "task": task.__dict__,
        "runtime": runtime,
    }
    (args.out / "training_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "stage=train_start "
        f"preset={args.preset} steps={values['steps']} "
        f"envs={values['envs']} episode_length={args.episode_length}",
        flush=True,
    )
    start = time.perf_counter()
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
        reward_scaling=1.0,
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
    )
    elapsed = time.perf_counter() - start
    best_params = best["params"] if best["params"] is not None else final_params
    model_io.save_params(args.out / "params_final", final_params)
    model_io.save_params(args.out / "params_best", best_params)
    (args.out / "metrics_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    train_summary = {
        "elapsed_s": elapsed,
        "best_eval_reward": best["reward"],
        "best_step": best["step"],
        "final_metrics": {
            name: _float(value)
            for name, value in (final_metrics or {}).items()
        },
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
