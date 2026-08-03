"""Train a residual PPO walking policy for the 3-D curl robot."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.config_walking_3d import (
    WALKING_PHYSICS_PROFILE_NAMES_3D,
    Walking3DConfig,
    WalkingReference3DConfig,
    walking_physics_profile_3d,
)
from curl_robot_2d_mjx.reward_walking_3d import (
    WALKING_REWARD_TERM_NAMES_3D,
    Walking3DRewardConfig,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from scripts.train_mjx_ppo import (
    _float,
    _network_factory,
    _resolve_restore_checkpoint,
    _split_metrics,
    _training_step_schedule,
)


PRESETS_WALKING_3D = {
    "smoke": {
        "steps": 131_072,
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
        "batch_size": 512,
        "num_minibatches": 16,
    },
    "h200": {
        "steps": 20_000_000,
        "envs": 2048,
        "eval_envs": 256,
        "num_evals": 10,
        "batch_size": 1024,
        "num_minibatches": 32,
    },
}


WALKING_RECIPES_3D = {
    "stability_v1": {
        "description": (
            "Long double-support curriculum around the morphology-aware "
            "paired-leg reference."
        ),
        "args": {
            "frequency_hz": 0.70,
            "step_length_m": 0.040,
            "foot_lift_m": 0.010,
            "duty_factor": 0.90,
            "reset_reference_weight": 1.0,
            "residual_gain": 0.65,
            "learning_rate": 1e-4,
            "entropy_cost": 3e-3,
        },
        "reward": {},
    },
}


PER_STEP_WALKING_METRICS_3D = (
    "forward_velocity_m_s",
    "forward_progress_m",
    "velocity_error_m_s",
    "root_x_m",
    "root_y_m",
    "root_z_m",
    "root_height_error_m",
    "lateral_drift_m",
    "lateral_velocity_m_s",
    "upright_tilt_rad",
    "heading_error_rad",
    "foot_contact_count",
    "stance_miss_fraction",
    "swing_contact_fraction",
    "swing_clearance_cost",
    "nonfoot_ground_contact_count",
    "nonfoot_ground_depth_m",
    "self_contact_count",
    "self_contact_depth_m",
    "airborne_active",
    "airborne_step_count",
    "root_low_step_count",
    "upright_tilt_step_count",
    "nonfoot_contact_step_count",
    "self_contact_step_count",
    "joint_tracking_rms",
    "residual_action_rms",
    "action_rate_rms",
    "normalized_torque_rms",
    "reference_blend",
    "startup_action_ramp",
    "oscillator_phase_rad",
    "desired_speed_m_s",
    "reset_reference_weight",
)


WALKING_FAILURE_METRICS_3D = (
    "failure_nonfinite",
    "failure_nonfinite_action",
    "failure_nonfinite_physics",
    "failure_root_low",
    "failure_root_high",
    "failure_upright_tilt",
    "failure_lateral_drift",
    "failure_airborne",
    "failure_nonfoot_depth",
    "failure_nonfoot_contact",
    "failure_self_contact_depth",
    "failure_self_contact",
)


def _add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    for field in fields(Walking3DRewardConfig):
        parser.add_argument(
            f"--reward-{field.name.replace('_', '-')}",
            dest=f"reward_{field.name}",
            type=float,
            default=None,
            help=f"Override Walking3DRewardConfig.{field.name}.",
        )


def _reward_config_from_args(args) -> Walking3DRewardConfig:
    overrides = dict(WALKING_RECIPES_3D[args.recipe]["reward"])
    overrides.update(
        {
            field.name: value
            for field in fields(Walking3DRewardConfig)
            if (value := getattr(args, f"reward_{field.name}", None))
            is not None
        }
    )
    return replace(Walking3DRewardConfig(), **overrides)


def _add_per_step_walking_metrics_3d(metrics) -> None:
    average_length = metrics.get("eval/avg_episode_length")
    if average_length is None or average_length <= 0:
        return
    for name in PER_STEP_WALKING_METRICS_3D:
        key = f"eval/episode_{name}"
        if key in metrics:
            metrics[f"eval/avg_{name}"] = metrics[key] / average_length
    for name in WALKING_REWARD_TERM_NAMES_3D:
        key = f"eval/episode_reward_{name}"
        if key in metrics:
            metrics[f"eval/avg_reward_{name}"] = (
                metrics[key] / average_length
            )
    if "eval/episode_reward" in metrics:
        metrics["eval/avg_reward"] = (
            metrics["eval/episode_reward"] / average_length
        )


def _metric(metrics, name, default=0.0):
    return float(metrics.get(name, default))


def _checkpoint_selection_walking_3d(
    metrics,
    episode_length,
    *,
    target_distance_m,
    desired_speed_m_s,
):
    """Select checkpoints using walking behavior rather than reward scale."""

    average_length = metrics.get("eval/avg_episode_length", 0.0)
    failed_rate = metrics.get("eval/episode_failed", 1.0)
    nonfinite_rate = metrics.get("eval/episode_failure_nonfinite", 0.0)
    distance = metrics.get("eval/episode_forward_progress_m", -math.inf)
    velocity = metrics.get("eval/avg_forward_velocity_m_s", math.inf)
    upright_tilt = metrics.get("eval/avg_upright_tilt_rad", math.inf)
    lateral_drift = abs(metrics.get("eval/avg_lateral_drift_m", math.inf))
    nonfoot = metrics.get(
        "eval/avg_nonfoot_ground_contact_count", math.inf
    )
    self_contact = metrics.get("eval/avg_self_contact_count", math.inf)
    survival = min(max(average_length / episode_length, 0.0), 1.0)
    progress_quality = min(
        max(distance / max(target_distance_m, 1.0e-6), -1.0), 1.0
    )
    velocity_quality = 1.0 - min(
        abs(velocity - desired_speed_m_s)
        / max(abs(desired_speed_m_s), 1.0e-4),
        1.0,
    )
    nonfailure_quality = 1.0 - min(max(failed_rate, 0.0), 1.0)
    upright_quality = 1.0 - min(max(upright_tilt / 0.30, 0.0), 1.0)
    lateral_quality = 1.0 - min(max(lateral_drift / 0.05, 0.0), 1.0)
    contact_quality = 1.0 - min(
        max(nonfoot / 0.02, 0.0) + max(self_contact / 0.02, 0.0),
        1.0,
    )
    score = (
        0.30 * survival
        + 0.30 * progress_quality
        + 0.15 * velocity_quality
        + 0.10 * nonfailure_quality
        + 0.05 * upright_quality
        + 0.05 * lateral_quality
        + 0.05 * contact_quality
    )
    rejected = (
        nonfinite_rate > 0.0
        or not math.isfinite(distance)
        or not math.isfinite(velocity)
        or not math.isfinite(upright_tilt)
        or not math.isfinite(lateral_drift)
        or not math.isfinite(nonfoot)
        or not math.isfinite(self_contact)
        or not math.isfinite(score)
    )
    return {
        "score": -1_000_000.0 if rejected else score,
        "rejected": rejected,
        "survival": survival,
        "distance_m": distance,
        "forward_velocity_m_s": velocity,
        "velocity_quality": velocity_quality,
        "upright_tilt_rad": upright_tilt,
        "lateral_drift_m": lateral_drift,
        "contact_quality": contact_quality,
    }


def _format_eval_report_walking_3d(
    eval_index,
    total_evals,
    step,
    metrics,
    *,
    episode_length,
    control_dt,
    target_distance_m,
    desired_speed_m_s,
    selection,
    selected,
):
    marker = " new_best" if selected else ""
    lines = [
        (
            f"[eval {eval_index}/{total_evals}] step={int(step)} "
            f"physical_score={selection['score']:.4f}{marker}"
        ),
        (
            f"  outcome reward={_metric(metrics, 'eval/episode_reward'):+.3f} "
            f"avg/step={_metric(metrics, 'eval/avg_reward'):+.4f} "
            f"length={_metric(metrics, 'eval/avg_episode_length'):.1f}/"
            f"{episode_length} "
            f"time={_metric(metrics, 'eval/avg_episode_length') * control_dt:.2f}s "
            f"failed={_metric(metrics, 'eval/episode_failed'):.1%}"
        ),
        (
            f"  motion  distance={selection['distance_m']:+.3f}/"
            f"{target_distance_m:.3f}m "
            f"velocity={selection['forward_velocity_m_s']:+.3f}/"
            f"{desired_speed_m_s:.3f}m/s "
            f"velocity_quality={selection['velocity_quality']:.2f}"
        ),
        (
            f"  pose    z={_metric(metrics, 'eval/avg_root_z_m'):.3f}m "
            f"tilt={selection['upright_tilt_rad']:.3f}rad "
            f"heading={_metric(metrics, 'eval/avg_heading_error_rad'):+.3f}rad "
            f"lateral={selection['lateral_drift_m']:.3f}m"
        ),
        (
            f"  gait    feet={_metric(metrics, 'eval/avg_foot_contact_count'):.2f} "
            f"stance_miss={_metric(metrics, 'eval/avg_stance_miss_fraction'):.2%} "
            f"swing_contact={_metric(metrics, 'eval/avg_swing_contact_fraction'):.2%} "
            f"clearance_cost={_metric(metrics, 'eval/avg_swing_clearance_cost'):.3f}"
        ),
        (
            f"  safety  nonfoot="
            f"{_metric(metrics, 'eval/avg_nonfoot_ground_contact_count'):.3f} "
            f"self={_metric(metrics, 'eval/avg_self_contact_count'):.3f} "
            f"air={_metric(metrics, 'eval/avg_airborne_active'):.2%}"
        ),
        (
            f"  control residual="
            f"{_metric(metrics, 'eval/avg_residual_action_rms'):.3f} "
            f"rate={_metric(metrics, 'eval/avg_action_rate_rms'):.3f} "
            f"tracking={_metric(metrics, 'eval/avg_joint_tracking_rms'):.3f} "
            f"torque={_metric(metrics, 'eval/avg_normalized_torque_rms'):.3f}"
        ),
    ]
    reward_text = " ".join(
        f"{name}={_metric(metrics, f'eval/avg_reward_{name}'):+.3f}"
        for name in WALKING_REWARD_TERM_NAMES_3D
    )
    lines.append(f"  reward/step {reward_text}")
    lines.append(
        "  failures "
        + " ".join(
            f"{name.removeprefix('failure_')}="
            f"{_metric(metrics, f'eval/episode_{name}'):.1%}"
            for name in WALKING_FAILURE_METRICS_3D
        )
    )
    if "training/sps" in metrics:
        lines.append(
            f"  ppo     sps={_metric(metrics, 'training/sps'):.0f} "
            f"kl={_metric(metrics, 'training/kl_mean'):.4f} "
            f"policy_loss={_metric(metrics, 'training/policy_loss'):+.4f} "
            f"value_loss={_metric(metrics, 'training/v_loss'):.4f}"
        )
    return "\n".join(lines)


def _evaluate_policy_walking_3d(
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
    initial_x = _float(state.pipeline_state.qpos[0])
    initial_y = _float(state.pipeline_state.qpos[1])
    qpos_rows = []
    action_rows = []
    reward_rows = []
    metric_totals = {}
    reward_totals = {
        name: 0.0 for name in WALKING_REWARD_TERM_NAMES_3D
    }
    minimum_root_z = math.inf
    maximum_tilt = 0.0

    for _ in range(episode_length):
        rng, action_key = jax.random.split(rng)
        action, _ = policy_step(state.obs, action_key)
        state = env_step(state, action)
        qpos_rows.append(np.asarray(jax.device_get(state.pipeline_state.qpos)))
        action_rows.append(np.asarray(jax.device_get(action)))
        reward_rows.append(_float(state.reward))
        minimum_root_z = min(
            minimum_root_z, _float(state.metrics["root_z_m"])
        )
        maximum_tilt = max(
            maximum_tilt, _float(state.metrics["upright_tilt_rad"])
        )
        for name, value in state.metrics.items():
            scalar = _float(value)
            if name.startswith("reward_") and name != "reward_total":
                term_name = name.removeprefix("reward_")
                reward_totals[term_name] = (
                    reward_totals.get(term_name, 0.0) + scalar
                )
            elif name not in ("reward", "reward_total"):
                metric_totals[name] = metric_totals.get(name, 0.0) + scalar
        if _float(state.done) > 0.5:
            break

    steps = len(reward_rows)
    final_x = _float(state.pipeline_state.qpos[0])
    final_y = _float(state.pipeline_state.qpos[1])
    averages = {
        name: value / max(steps, 1) for name, value in metric_totals.items()
    }
    failures = {
        name.removeprefix("failure_"): bool(metric_totals.get(name, 0.0))
        for name in WALKING_FAILURE_METRICS_3D
    }
    summary = {
        "episode_steps": steps,
        "episode_duration_s": steps * env.config.control_timestep,
        "total_reward": float(sum(reward_rows)),
        "root_x_displacement_m": final_x - initial_x,
        "final_lateral_drift_m": final_y - initial_y,
        "average_forward_velocity_m_s": averages.get(
            "forward_velocity_m_s", 0.0
        ),
        "minimum_root_z_m": minimum_root_z,
        "maximum_upright_tilt_rad": maximum_tilt,
        "terminated": bool(_float(state.done) > 0.5),
        "failure_reasons": failures,
        "reward_breakdown": {
            "terms": reward_totals,
            "per_step": {
                name: value / max(steps, 1)
                for name, value in reward_totals.items()
            },
        },
        "metrics": {"totals": metric_totals, "per_step": averages},
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=tuple(PRESETS_WALKING_3D), default="smoke"
    )
    parser.add_argument(
        "--recipe", choices=tuple(WALKING_RECIPES_3D), default="stability_v1"
    )
    parser.add_argument(
        "--physics-profile",
        choices=WALKING_PHYSICS_PROFILE_NAMES_3D,
        default="cg12",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--frequency-hz", type=float)
    parser.add_argument("--step-length-m", type=float)
    parser.add_argument("--foot-lift-m", type=float)
    parser.add_argument("--duty-factor", type=float)
    parser.add_argument("--reset-keyframe", default="walk")
    parser.add_argument("--reset-reference-weight", type=float)
    parser.add_argument("--residual-gain", type=float)
    parser.add_argument("--terminate-root-z-min", type=float, default=0.145)
    parser.add_argument(
        "--terminate-root-z-low-duration", type=float, default=0.08
    )
    parser.add_argument("--terminate-root-z-max", type=float, default=0.46)
    parser.add_argument("--terminate-upright-tilt", type=float, default=0.72)
    parser.add_argument(
        "--terminate-upright-tilt-duration", type=float, default=0.08
    )
    parser.add_argument(
        "--terminate-lateral-drift", type=float, default=0.25
    )
    parser.add_argument(
        "--terminate-airborne-duration", type=float, default=0.14
    )
    parser.add_argument(
        "--terminate-nonfoot-depth", type=float, default=0.004
    )
    parser.add_argument(
        "--terminate-nonfoot-contact-duration", type=float, default=0.06
    )
    parser.add_argument(
        "--terminate-self-contact-depth", type=float, default=0.004
    )
    parser.add_argument(
        "--terminate-self-contact-duration", type=float, default=0.08
    )
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-cost", type=float)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument(
        "--hidden-layers", type=int, nargs="+", default=[256, 256, 128]
    )
    parser.add_argument(
        "--activation", choices=("elu", "relu", "swish", "tanh"), default="elu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa", "disable"),
        default="auto",
    )
    parser.add_argument("--xla-triton", action="store_true", default=True)
    parser.add_argument(
        "--no-xla-triton", dest="xla_triton", action="store_false"
    )
    parser.add_argument("--preallocate", action="store_true", default=True)
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
        default=Path("results") / "mjx_3d_walking_stability_v1",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--save-ppo-checkpoints", action="store_true")
    parser.add_argument("--ppo-checkpoint-dir", type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--selection-target-distance", type=float)
    _add_reward_arguments(parser)
    return parser


def _apply_recipe_defaults(args) -> None:
    for name, value in WALKING_RECIPES_3D[args.recipe]["args"].items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_recipe_defaults(args)
    if not 0.0 <= args.reset_reference_weight <= 1.0:
        parser.error("--reset-reference-weight must be in [0, 1]")
    if not 0.0 <= args.residual_gain <= 1.0:
        parser.error("--residual-gain must be in [0, 1]")
    if not 0.5 < args.duty_factor < 1.0:
        parser.error("--duty-factor must be between 0.5 and 1")
    if args.ppo_checkpoint_dir is not None and not args.save_ppo_checkpoints:
        parser.error("--ppo-checkpoint-dir requires --save-ppo-checkpoints")
    if (
        args.selection_target_distance is not None
        and args.selection_target_distance <= 0.0
    ):
        parser.error("--selection-target-distance must be positive")
    for name in (
        "episode_length",
        "unroll_length",
        "updates_per_batch",
        "frequency_hz",
        "step_length_m",
        "foot_lift_m",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    values = PRESETS_WALKING_3D[args.preset].copy()
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
    schedule = _training_step_schedule(
        requested_steps=values["steps"],
        num_evals=values["num_evals"],
        batch_size=values["batch_size"],
        unroll_length=args.unroll_length,
        num_minibatches=values["num_minibatches"],
    )
    if (
        args.out.exists()
        and any(args.out.iterdir())
        and not args.allow_existing_output
    ):
        raise SystemExit(
            f"Output directory is not empty: {args.out}. Use a new --out path."
        )

    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=args.preallocate,
        xla_triton=args.xla_triton,
        mujoco_gl=args.mujoco_gl,
        verbose=False,
    )
    from brax.io import model as model_io
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment_walking_3d import (
        make_brax_walking_env_3d,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    runtime = describe_runtime()
    if args.runtime_diagnostics:
        print(
            "[runtime]\n"
            f"  python={runtime['python_version']} "
            f"jax={runtime['jax_version']} backend={runtime['backend']}\n"
            f"  devices={', '.join(runtime['devices'])}\n"
            f"  mujoco_gl={runtime['mujoco_gl']} "
            f"memory_fraction={runtime['memory_fraction']}",
            flush=True,
        )

    reference = replace(
        WalkingReference3DConfig(),
        frequency_hz=args.frequency_hz,
        step_length_m=args.step_length_m,
        foot_lift_m=args.foot_lift_m,
        duty_factor=args.duty_factor,
    )
    task = walking_physics_profile_3d(
        args.physics_profile,
        Walking3DConfig(
            episode_length=args.episode_length,
            reset_keyframe_name=args.reset_keyframe,
            reference=reference,
            residual_gain=args.residual_gain,
            reset_reference_weight=args.reset_reference_weight,
            terminate_root_z_min=args.terminate_root_z_min,
            terminate_root_z_low_duration_s=(
                args.terminate_root_z_low_duration
            ),
            terminate_root_z_max=args.terminate_root_z_max,
            terminate_upright_tilt_rad=args.terminate_upright_tilt,
            terminate_upright_tilt_duration_s=(
                args.terminate_upright_tilt_duration
            ),
            terminate_lateral_drift_m=args.terminate_lateral_drift,
            terminate_airborne_duration_s=(
                args.terminate_airborne_duration
            ),
            terminate_nonfoot_depth_m=args.terminate_nonfoot_depth,
            terminate_nonfoot_contact_duration_s=(
                args.terminate_nonfoot_contact_duration
            ),
            terminate_self_contact_depth_m=(
                args.terminate_self_contact_depth
            ),
            terminate_self_contact_duration_s=(
                args.terminate_self_contact_duration
            ),
        ),
    )
    reward_config = _reward_config_from_args(args)
    train_env = make_brax_walking_env_3d(
        task, reward_config=reward_config, seed=args.seed
    )
    eval_env = make_brax_walking_env_3d(
        task, reward_config=reward_config, seed=args.seed + 10_000
    )
    target_distance_m = args.selection_target_distance or (
        reference.desired_speed_m_s
        * args.episode_length
        * task.control_timestep
    )

    metric_history = []
    reward_history = []
    best = {
        "score": float("-inf"),
        "reward": float("-inf"),
        "step": None,
        "params": None,
        "candidate_step": None,
        "candidate_params": None,
    }
    reward_peak = {"reward": float("-inf"), "step": None}
    eval_counter = {"value": 0}

    def policy_params_fn(step, make_policy, params):
        del make_policy
        best["candidate_step"] = int(step)
        best["candidate_params"] = params
        if best["step"] == int(step):
            best["params"] = params

    def progress_fn(step, metrics):
        clean = {name: _float(value) for name, value in metrics.items()}
        _add_per_step_walking_metrics_3d(clean)
        reward_metrics, ordinary_metrics = _split_metrics(clean)
        reward_history.append({"step": int(step), **reward_metrics})
        metric_history.append({"step": int(step), **ordinary_metrics})
        reward = clean.get(
            "eval/episode_reward", clean.get("eval/episode_reward_mean")
        )
        if reward is not None and reward > reward_peak["reward"]:
            reward_peak["reward"] = reward
            reward_peak["step"] = int(step)
        selection = _checkpoint_selection_walking_3d(
            clean,
            args.episode_length,
            target_distance_m=target_distance_m,
            desired_speed_m_s=reference.desired_speed_m_s,
        )
        selected = (
            not selection["rejected"]
            and selection["score"] > best["score"]
        )
        if selected:
            best["score"] = selection["score"]
            best["reward"] = reward
            best["step"] = int(step)
            if best["candidate_step"] == int(step):
                best["params"] = best["candidate_params"]
        eval_counter["value"] += 1
        print(
            _format_eval_report_walking_3d(
                eval_counter["value"],
                values["num_evals"],
                step,
                clean,
                episode_length=args.episode_length,
                control_dt=task.control_timestep,
                target_distance_m=target_distance_m,
                desired_speed_m_s=reference.desired_speed_m_s,
                selection=selection,
                selected=selected,
            ),
            flush=True,
        )

    config_payload = {
        "preset": args.preset,
        "recipe": args.recipe,
        **values,
        "unroll_length": args.unroll_length,
        "updates_per_batch": args.updates_per_batch,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "reward_scaling": args.reward_scaling,
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "seed": args.seed,
        "task": asdict(task),
        "reward": asdict(reward_config),
        "runtime": runtime,
        "selection_target_distance_m": target_distance_m,
        "training_step_schedule": schedule,
        "restore_checkpoint": (
            str(args.restore_checkpoint)
            if args.restore_checkpoint is not None
            else None
        ),
        "save_ppo_checkpoints": args.save_ppo_checkpoints,
    }
    (args.out / "training_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "reward_config.json").write_text(
        json.dumps(asdict(reward_config), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[training]\n"
        f"  preset={args.preset} recipe={args.recipe} "
        f"physics={args.physics_profile}\n"
        f"  requested_steps={schedule['requested_steps']:,} "
        f"effective_steps={schedule['effective_steps']:,} "
        f"evals={values['num_evals']}\n"
        f"  envs={values['envs']} eval_envs={values['eval_envs']} "
        f"batch={values['batch_size']} "
        f"minibatches={values['num_minibatches']}\n"
        f"  episode={args.episode_length * task.control_timestep:.2f}s "
        f"speed={reference.desired_speed_m_s:.3f}m/s "
        f"target_distance={target_distance_m:.3f}m\n"
        f"  gait={reference.frequency_hz:.2f}Hz "
        f"stride={reference.step_length_m:.3f}m "
        f"lift={reference.foot_lift_m:.3f}m "
        f"duty={reference.duty_factor:.2f}\n"
        f"  reset_reference_weight={task.reset_reference_weight:.2f} "
        f"residual_gain={task.residual_gain:.2f}\n"
        f"  lr={args.learning_rate:g} entropy={args.entropy_cost:g} "
        f"discount={args.discounting:g} seed={args.seed}\n"
        f"  output={args.out.resolve()}",
        flush=True,
    )

    checkpoint_kwargs = {}
    train_parameters = inspect.signature(ppo.train).parameters
    if args.save_ppo_checkpoints and "save_checkpoint_path" in train_parameters:
        checkpoint_dir = args.ppo_checkpoint_dir or (
            args.out / "ppo_checkpoint"
        )
        checkpoint_kwargs["save_checkpoint_path"] = str(
            checkpoint_dir.resolve()
        )
    if args.restore_checkpoint is not None:
        if "restore_checkpoint_path" not in train_parameters:
            raise SystemExit(
                "Installed Brax does not support restore_checkpoint_path."
            )
        checkpoint_kwargs["restore_checkpoint_path"] = str(
            _resolve_restore_checkpoint(args.restore_checkpoint)
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
    clean_final = {
        name: _float(value) for name, value in (final_metrics or {}).items()
    }
    _add_per_step_walking_metrics_3d(clean_final)
    final_reward_metrics, final_metrics_clean = _split_metrics(clean_final)
    summary = {
        "elapsed_s": elapsed,
        "best_selection_score": (
            best["score"] if math.isfinite(best["score"]) else None
        ),
        "best_eval_reward": (
            best["reward"] if math.isfinite(best["reward"]) else None
        ),
        "best_step": best["step"],
        "reward_peak": (
            reward_peak["reward"]
            if math.isfinite(reward_peak["reward"])
            else None
        ),
        "reward_peak_step": reward_peak["step"],
        "final_metrics": final_metrics_clean,
        "final_reward_metrics": final_reward_metrics,
    }
    (args.out / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    throughput = schedule["effective_steps"] / max(elapsed, 1.0e-9)
    print(
        "[training complete]\n"
        f"  elapsed={elapsed / 60.0:.1f}min "
        f"throughput={throughput:,.0f} steps/s\n"
        f"  best_step={best['step']} score={best['score']:.4f}\n"
        f"  checkpoints best={args.out / 'params_best'} "
        f"final={args.out / 'params_final'}",
        flush=True,
    )

    if not args.skip_evaluation:
        best_eval = _evaluate_policy_walking_3d(
            eval_env,
            make_inference_fn,
            best_params,
            seed=args.seed + 20_000,
            episode_length=args.episode_length,
            output_dir=args.out / "evaluation_best",
        )
        final_eval = _evaluate_policy_walking_3d(
            eval_env,
            make_inference_fn,
            final_params,
            seed=args.seed + 20_000,
            episode_length=args.episode_length,
            output_dir=args.out / "evaluation_final",
        )
        comparison = {
            "selection": {
                "best_step": best["step"],
                "best_selection_score": summary["best_selection_score"],
            },
            "best": best_eval,
            "final": final_eval,
        }
        (args.out / "policy_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "[evaluation]\n"
            f"  best distance={best_eval['root_x_displacement_m']:+.3f}m "
            f"time={best_eval['episode_duration_s']:.2f}s "
            f"tilt={best_eval['maximum_upright_tilt_rad']:.3f}rad\n"
            f"  final distance={final_eval['root_x_displacement_m']:+.3f}m "
            f"time={final_eval['episode_duration_s']:.2f}s "
            f"tilt={final_eval['maximum_upright_tilt_rad']:.3f}rad",
            flush=True,
        )


if __name__ == "__main__":
    main()
