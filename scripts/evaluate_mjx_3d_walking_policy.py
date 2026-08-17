"""Evaluate a saved 3-D walking policy across speeds and gait phases."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
from pathlib import Path

from curl_robot_2d_mjx.config_walking_3d import Walking3DConfig
from curl_robot_2d_mjx.reward_walking_3d import Walking3DRewardConfig
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from scripts.train_mjx_3d_walking_ppo import (
    _evaluate_policy_grid_walking_3d,
    _format_evaluation_grid_walking_3d,
    _unique_finite_values,
    _walking_network_factory,
)


def _find_training_config(checkpoint: Path) -> Path:
    for directory in (checkpoint.parent, *checkpoint.parents):
        candidate = directory / "training_config.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find training_config.json above {checkpoint}"
    )


def _dataclass_kwargs(cls, values):
    names = {field.name for field in fields(cls)}
    return {name: value for name, value in values.items() if name in names}


def _default_grid(config, evaluation_task):
    grid = config.get("evaluation_grid", {})
    speeds = grid.get("forward_speeds_m_s")
    if speeds is None:
        training_task = config.get("task", {})
        forward_range = training_task.get(
            "command_forward_velocity_range_m_s",
            evaluation_task.command_forward_velocity_range_m_s,
        )
        desired_speed = training_task.get(
            "desired_speed_m_s", evaluation_task.desired_speed_m_s
        )
        speeds = (
            forward_range[0],
            desired_speed,
            forward_range[1],
        )
    phases = grid.get("initial_gait_phases")
    if phases is None:
        gait_phase_enabled = config.get("task", {}).get(
            "gait_phase_enabled", evaluation_task.gait_phase_enabled
        )
        phases = (0.0, 0.5) if gait_phase_enabled else (0.0,)
    return (
        _unique_finite_values(speeds),
        _unique_finite_values(phases, modulo=1.0),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--speeds", type=float, nargs="+")
    parser.add_argument("--gait-phases", type=float, nargs="+")
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa", "disable"),
        default="auto",
    )
    parser.add_argument("--xla-triton", action="store_true", default=True)
    parser.add_argument(
        "--no-xla-triton", dest="xla_triton", action="store_false"
    )
    parser.add_argument("--preallocate", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.episode_length is not None and args.episode_length < 1:
        parser.error("--episode-length must be positive")
    for values, name in (
        (args.speeds, "--speeds"),
        (args.gait_phases, "--gait-phases"),
    ):
        if values is not None and not all(math.isfinite(value) for value in values):
            parser.error(f"{name} values must be finite")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    training_config_path = (
        args.training_config.resolve()
        if args.training_config is not None
        else _find_training_config(checkpoint)
    )
    config = json.loads(training_config_path.read_text(encoding="utf-8"))
    task_values = config.get("evaluation_task", config.get("task"))
    if not isinstance(task_values, dict):
        raise SystemExit("training config does not contain a walking task")
    task = Walking3DConfig(
        **_dataclass_kwargs(Walking3DConfig, task_values)
    )
    task = replace(
        task,
        episode_length=args.episode_length or task.episode_length,
        command_lateral_velocity_range_m_s=(0.0, 0.0),
        command_yaw_rate_range_rad_s=(0.0, 0.0),
        command_deadband_probability=0.0,
        observation_noise_enabled=False,
        reset_joint_noise_rad=0.0,
        reset_velocity_noise=0.0,
        reset_root_xy_velocity_noise_m_s=0.0,
        reset_root_yaw_rate_noise_rad_s=0.0,
        terminate_low_progress_enabled=False,
    )
    reward = Walking3DRewardConfig(
        **_dataclass_kwargs(
            Walking3DRewardConfig,
            config.get("reward", {}),
        )
    )
    default_speeds, default_phases = _default_grid(config, task)
    speeds = _unique_finite_values(args.speeds or default_speeds)
    phases = _unique_finite_values(
        args.gait_phases or default_phases,
        modulo=1.0,
    )
    output_dir = (
        args.out.resolve()
        if args.out is not None
        else checkpoint.parent / f"evaluation_grid_{checkpoint.name}"
    )

    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=args.preallocate,
        xla_triton=args.xla_triton,
        mujoco_gl=args.mujoco_gl,
        verbose=False,
    )

    from brax.io import model as model_io
    from brax.training import types as training_types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from curl_robot_2d_mjx.environment_walking_3d import (
        make_brax_walking_env_3d,
    )

    env = make_brax_walking_env_3d(task, reward_config=reward, seed=args.seed)
    preprocess_observations_fn = (
        running_statistics.normalize
        if config.get("observation_normalization", False)
        else training_types.identity_observation_preprocessor
    )
    network_factory = _walking_network_factory(
        config.get("hidden_layers", (256, 256, 128)),
        config.get("critic_hidden_layers", (256, 256, 128)),
        config.get("activation", "elu"),
        config.get("init_noise_std", 0.30),
        asymmetric_observations=config.get(
            "asymmetric_observations", False
        ),
        small_actor_mean_init=(
            config.get("policy_mean_kernel_init") == "small_uniform"
        ),
    )
    networks = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    params = model_io.load_params(checkpoint)
    runtime = describe_runtime()
    print(
        "[runtime]\n"
        f"  python={runtime['python_version']} "
        f"jax={runtime['jax_version']} backend={runtime['backend']}\n"
        f"  devices={', '.join(runtime['devices'])}\n"
        f"[evaluation setup]\n"
        f"  checkpoint={checkpoint}\n"
        f"  training_config={training_config_path}\n"
        f"  speeds={','.join(f'{value:g}' for value in speeds)} "
        f"phases={','.join(f'{value:g}' for value in phases)}\n"
        f"  output={output_dir}",
        flush=True,
    )
    grid = _evaluate_policy_grid_walking_3d(
        env,
        make_inference_fn,
        params,
        seed=args.seed,
        episode_length=task.episode_length,
        output_dir=output_dir,
        forward_speeds=speeds,
        gait_phases=phases,
    )
    print(_format_evaluation_grid_walking_3d("policy", grid), flush=True)


if __name__ == "__main__":
    main()
