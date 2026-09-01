#!/usr/bin/env python3
"""Reward-driven deploy-DR PPO fine-tuning for an existing rolling Student."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.deployment_rolling_3d import (
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
    controller_action_to_effective_action_3d,
)
from curl_robot_2d_mjx.environment_3d import cem_controller_path_3d
from curl_robot_2d_mjx.randomization_3d import (
    RollingStudentDeployDomainRandomization,
)
from curl_robot_2d_mjx.rolling_student_dr_ppo_3d import (
    ROLLING_STUDENT_PPO_ACTION_SIZE_3D,
    ROLLING_STUDENT_PPO_CRITIC_OBSERVATION_SIZE_3D,
    expand_ppo_actor_to_controller_3d,
    initialize_ppo_actor_from_student_3d,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from curl_robot_2d_mjx.startup_rolling_3d import (
    add_stand_startup_arguments,
    with_stand_startup,
)
from scripts.train_mjx_3d_roll_distillation import (
    _task,
    student_controller_config,
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
        "batch_size": 256,
        "num_minibatches": 8,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student", type=Path, help="existing student_params")
    parser.add_argument(
        "--restore-ppo",
        type=Path,
        help="previous DR PPO params_final; restores actor, critic and normalizer",
    )
    parser.add_argument(
        "--controller",
        type=Path,
        default=cem_controller_path_3d("rollingquad_2"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--minimum-success-turns", type=float, default=5.0)
    add_stand_startup_arguments(parser)
    parser.add_argument("--dr-strength", type=float, default=0.25)
    parser.add_argument("--student-anchor-weight", type=float, default=0.02)
    parser.add_argument("--observation-noise-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--entropy-cost", type=float, default=1.0e-4)
    parser.add_argument("--initial-policy-std", type=float, default=0.05)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument(
        "--hidden-layers", type=int, nargs="+", default=(512, 256, 128)
    )
    parser.add_argument(
        "--critic-hidden-layers", type=int, nargs="+", default=(256, 256, 128)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-devices", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa", "disable"),
        default="disable",
    )
    args = parser.parse_args(argv)
    values = PRESETS[args.preset].copy()
    for name in values:
        override = getattr(args, name)
        if override is not None:
            values[name] = override
        if values[name] < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
        setattr(args, name, values[name])
    for path, name in (
        (args.student, "student"),
        (args.controller, "controller"),
    ):
        if not path.is_file():
            parser.error(f"{name} file does not exist: {path}")
    if args.restore_ppo is not None and not args.restore_ppo.is_file():
        parser.error(f"PPO params do not exist: {args.restore_ppo}")
    if args.out.exists() and any(args.out.iterdir()) and not args.allow_existing_output:
        parser.error(f"output directory is not empty: {args.out}")
    for value, name in (
        (args.dr_strength, "--dr-strength"),
        (args.student_anchor_weight, "--student-anchor-weight"),
        (args.observation_noise_scale, "--observation-noise-scale"),
        (args.entropy_cost, "--entropy-cost"),
        (args.minimum_success_turns, "--minimum-success-turns"),
    ):
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.dr_strength > 1.0:
        parser.error("--dr-strength must not exceed one")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.initial_policy_std) or args.initial_policy_std <= 0.001:
        parser.error("--initial-policy-std must be greater than 0.001")
    if not 0.0 < args.discounting <= 1.0:
        parser.error("--discounting must be in (0, 1]")
    if args.episode_length < 1 or args.unroll_length < 1 or args.updates_per_batch < 1:
        parser.error("episode and rollout lengths must be positive")
    if args.batch_size * args.num_minibatches % args.envs:
        parser.error("batch-size * num-minibatches must be divisible by envs")
    if args.max_devices is not None:
        if args.max_devices < 1:
            parser.error("--max-devices must be positive")
        if args.envs % args.max_devices or args.eval_envs % args.max_devices:
            parser.error("training and evaluation envs must divide across devices")
    return args


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value.item())


def main(argv=None):
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
    import jax.nn as jnn
    from brax.io import model as model_io
    from brax.training import networks as training_networks
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d
    from curl_robot_2d_mjx.environment_rolling_student_dr_3d import (
        make_rolling_student_dr_env_3d,
    )
    from curl_robot_2d_mjx.randomization_3d import (
        make_student_deploy_domain_randomization_fn_3d,
    )
    from curl_robot_2d_mjx.wrappers_rolling_student_dr_3d import (
        wrap_rolling_student_dr_3d,
    )
    from scripts.export_rtneural import convert as convert_rtneural

    signature = inspect.signature(ppo.train).parameters
    for required in ("restore_params", "randomization_fn", "wrap_env_fn"):
        if required not in signature:
            raise SystemExit(f"Installed Brax PPO lacks required {required}")

    student_checkpoint = model_io.load_params(args.student)
    student_normalizer = student_checkpoint[0]
    student_params = jax.tree_util.tree_map(jp.asarray, student_checkpoint[1])
    frozen_mean_np = np.asarray(student_normalizer["mean"])
    frozen_std_np = np.asarray(student_normalizer["std"])
    if frozen_mean_np.shape != (ROLLING_DEPLOY_OBSERVATION_SIZE_3D,):
        raise ValueError("Student normalizer must contain a 720-value mean")
    if frozen_std_np.shape != frozen_mean_np.shape or not np.all(
        np.isfinite(frozen_std_np) & (frozen_std_np > 0.0)
    ):
        raise ValueError("Student normalizer std must be 720 positive values")
    frozen_mean = jp.asarray(frozen_mean_np)
    frozen_std = jp.asarray(frozen_std_np)

    student_layers = student_params["params"]

    @jax.jit
    def student_anchor_policy(observation):
        value = (observation - frozen_mean) / frozen_std
        for index in range(len(args.hidden_layers)):
            layer = student_layers[f"hidden_{index}"]
            value = jnn.elu(value @ layer["kernel"] + layer["bias"])
        head = student_layers["location"]
        controller_action = jp.tanh(value @ head["kernel"] + head["bias"])
        return controller_action_to_effective_action_3d(
            jp, controller_action
        )

    reference = load_cem_reference(
        args.controller,
        reference_weight=1.0,
        minimum_residual_gain=0.15,
    )
    task = with_stand_startup(
        _task(
            episode_length=args.episode_length,
            direct_effective_action=True,
        ),
        args,
    )
    deploy_settings = RollingStudentDeployDomainRandomization().scaled(
        args.dr_strength
    )

    def make_env(seed, noise_scale):
        base = make_brax_env_3d(task, cem_reference=reference, seed=seed)
        return make_rolling_student_dr_env_3d(
            base,
            deploy_settings,
            student_anchor_policy=student_anchor_policy,
            student_anchor_weight=args.student_anchor_weight,
            observation_noise_scale=noise_scale,
            minimum_success_turns=args.minimum_success_turns,
        )

    train_env = make_env(args.seed, args.observation_noise_scale)
    eval_env = make_env(args.seed + 10_000, args.observation_noise_scale)
    if train_env.observation_size != {
        "state": ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
        "privileged_state": ROLLING_STUDENT_PPO_CRITIC_OBSERVATION_SIZE_3D,
    }:
        raise RuntimeError("asymmetric rolling observation contract mismatch")

    def hybrid_preprocess(observation, statistics):
        if observation.shape[-1] == ROLLING_DEPLOY_OBSERVATION_SIZE_3D:
            return (observation - frozen_mean) / frozen_std
        return running_statistics.normalize(observation, statistics)

    def network_factory(observation_size, action_size, preprocess_observations_fn):
        del preprocess_observations_fn
        networks = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=hybrid_preprocess,
            policy_hidden_layer_sizes=tuple(args.hidden_layers),
            value_hidden_layer_sizes=tuple(args.critic_hidden_layers),
            activation=jnn.elu,
            policy_obs_key="state",
            value_obs_key="privileged_state",
            distribution_type="tanh_normal",
        )
        original_init = networks.policy_network.init

        def initialize_policy(key):
            return initialize_ppo_actor_from_student_3d(
                jp,
                original_init(key),
                student_params,
                hidden_layers=tuple(args.hidden_layers),
                initial_std=args.initial_policy_std,
            )

        policy_network = training_networks.FeedForwardNetwork(
            init=initialize_policy,
            apply=networks.policy_network.apply,
        )
        return replace(networks, policy_network=policy_network)

    initialized_networks = network_factory(
        train_env.observation_size,
        train_env.action_size,
        running_statistics.normalize,
    )
    running_statistics_supports_mode = (
        "mode" in inspect.signature(running_statistics.init_state).parameters
    )
    if args.restore_ppo is None:
        policy_key, value_key = jax.random.split(
            jax.random.PRNGKey(args.seed + 20_000)
        )
        normalizer = running_statistics.init_state(
            {
                "state": jp.zeros((ROLLING_DEPLOY_OBSERVATION_SIZE_3D,)),
                "privileged_state": jp.zeros(
                    (ROLLING_STUDENT_PPO_CRITIC_OBSERVATION_SIZE_3D,)
                ),
            },
            **({"mode": "ema"} if running_statistics_supports_mode else {}),
        )
        restore_params = (
            normalizer,
            initialized_networks.policy_network.init(policy_key),
            initialized_networks.value_network.init(value_key),
        )
        restore_source = "existing_student_actor_plus_fresh_privileged_critic"
    else:
        restore_params = model_io.load_params(args.restore_ppo)
        restore_source = str(args.restore_ppo.resolve())

    randomization_fn = make_student_deploy_domain_randomization_fn_3d(
        deploy_settings,
        torso_body_id=train_env.torso_body_id,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    controller_config = student_controller_config(train_env.mj_model)
    history = []

    def progress(step, metrics):
        clean = {name: _float(value) for name, value in metrics.items()}
        record = {"step": int(step), **clean}
        history.append(record)
        turns = clean.get("eval/episode_roll_progress_rad", 0.0) / (
            2.0 * math.pi
        )
        failed = clean.get("eval/episode_failed", 0.0)
        success = clean.get("eval/episode_movement_success", 0.0)
        lateral = clean.get("eval/episode_failure_lateral_drift", 0.0)
        anchor = clean.get("eval/episode_student_anchor_action_rmse", 0.0)
        length = max(clean.get("eval/avg_episode_length", 1.0), 1.0)
        print(
            f"[Student DR PPO eval] step={int(step):,} "
            f"turns={turns:.3f} success={success:.1%} failed={failed:.1%} "
            f"lateral={lateral:.1%} anchor_rmse/step={anchor / length:.5f}",
            flush=True,
        )

    optional_train_kwargs = {}
    if "save_checkpoint_path" in signature:
        optional_train_kwargs["save_checkpoint_path"] = str(
            (args.out / "ppo_checkpoint").resolve()
        )
    if (
        running_statistics_supports_mode
        and "normalize_observations_mode" in signature
    ):
        optional_train_kwargs["normalize_observations_mode"] = "ema"
    if "bootstrap_on_timeout" in signature:
        optional_train_kwargs["bootstrap_on_timeout"] = True
    if args.max_devices is not None:
        if "max_devices_per_host" not in signature:
            raise SystemExit("Installed Brax cannot limit devices per host")
        optional_train_kwargs["max_devices_per_host"] = args.max_devices

    run_config = {
        "mode": "reward_dr_ppo_not_imitation_learning",
        "student": str(args.student.resolve()),
        "restore_source": restore_source,
        "controller": str(args.controller.resolve()),
        "task": asdict(task),
        "deploy_domain_randomization": asdict(deploy_settings),
        "actor_observation": "real_controller_36x20",
        "actor_observation_size": ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
        "critic_observation": "privileged_65d",
        "critic_observation_size": ROLLING_STUDENT_PPO_CRITIC_OBSERVATION_SIZE_3D,
        "policy_action_size": ROLLING_STUDENT_PPO_ACTION_SIZE_3D,
        "controller_action_size": 12,
        "student_anchor_weight": args.student_anchor_weight,
        "runtime": describe_runtime(),
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    with (args.out / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")
    with (args.out / "controller_config.json").open("w", encoding="utf-8") as handle:
        json.dump(controller_config, handle, indent=2)
        handle.write("\n")

    print(
        "[rolling Student DR PPO]\n"
        f"  student={args.student.resolve()}\n"
        f"  restore={restore_source}\n"
        f"  actor=720D-real -> 8D-effective critic=65D-privileged\n"
        f"  DR strength={args.dr_strength:g} anchor={args.student_anchor_weight:g} "
        f"noise={args.observation_noise_scale:g}\n"
        f"  steps={args.steps:,} envs={args.envs} eval_envs={args.eval_envs}",
        flush=True,
    )
    started = time.perf_counter()
    _, final_params, final_metrics = ppo.train(
        environment=train_env,
        eval_env=eval_env,
        wrap_env_fn=wrap_rolling_student_dr_3d,
        randomization_fn=randomization_fn,
        restore_params=restore_params,
        num_timesteps=args.steps,
        episode_length=args.episode_length,
        action_repeat=1,
        num_envs=args.envs,
        num_eval_envs=args.eval_envs,
        num_evals=args.num_evals,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=1.0,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.updates_per_batch,
        normalize_observations=True,
        deterministic_eval=True,
        network_factory=network_factory,
        seed=args.seed,
        progress_fn=progress,
        **optional_train_kwargs,
    )
    params_path = args.out / "params_final"
    model_io.save_params(params_path, final_params)

    controller_actor = expand_ppo_actor_to_controller_3d(
        np, jax.tree_util.tree_map(np.asarray, final_params[1])
    )
    export_checkpoint = (
        {
            "mean": np.asarray(frozen_mean),
            "std": np.asarray(frozen_std),
        },
        controller_actor,
        {},
    )
    rtneural = convert_rtneural(
        export_checkpoint,
        controller_config,
        activation="elu",
        observation_history=20,
    )
    with (args.out / "student_rtneural.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rtneural, handle, separators=(",", ":"))
        handle.write("\n")
    with (args.out / "metrics_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    summary = {
        **run_config,
        "elapsed_s": time.perf_counter() - started,
        "params_final": str(params_path.resolve()),
        "rtneural": str((args.out / "student_rtneural.json").resolve()),
        "final_metrics": {
            name: _float(value) for name, value in (final_metrics or {}).items()
        },
    }
    with (args.out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(
        "[saved]\n"
        f"  PPO={params_path}\n"
        f"  deploy={args.out / 'student_rtneural.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
