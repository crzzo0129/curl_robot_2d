"""Train a reference-free PPO walking policy for the 3-D curl robot."""

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
    WALKING_GEOMETRY_NAMES_3D,
    WALKING_PHYSICS_PROFILE_NAMES_3D,
    Walking3DConfig,
    walking_physics_profile_3d,
)
from curl_robot_2d_mjx.reward_walking_3d import (
    WALKING_REWARD_TERM_NAMES_3D,
    Walking3DRewardConfig,
)
from curl_robot_2d_mjx.randomization_3d import (
    Walking3DDomainRandomization,
    make_walking_domain_randomization_fn_3d,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from scripts.train_mjx_ppo import (
    _float,
    _resolve_restore_checkpoint,
    _split_metrics,
    _training_step_schedule,
)


WALKING_ACTOR_MEAN_INIT_SCALE = 1.0e-3
WALKING_ACTOR_MEAN_CLIP_SCALE = 1.0


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
    "laptop": {
        "steps": 5_000_000,
        "envs": 64,
        "eval_envs": 16,
        "num_evals": 10,
        "batch_size": 128,
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


WALKING_RECIPES_3D = {
    "anymal_v1": {
        "description": (
            "Command-conditioned 12-DoF locomotion with shaped task rewards "
            "and batched MJX domain randomization."
        ),
        "args": {
            "desired_speed_m_s": 0.20,
            "command_forward_min": -0.10,
            "command_forward_max": 0.35,
            "command_lateral_max": 0.15,
            "command_yaw_rate_max": 0.60,
            "command_resample_time": 4.0,
            "command_stop_probability": 0.10,
            "no_observation_noise": False,
            "no_domain_randomization": False,
            "action_scale_abduction": 0.10,
            "action_scale_hip": 0.40,
            "action_scale_knee": 0.55,
            "reset_joint_noise": 0.015,
            "reset_velocity_noise": 0.05,
            "reset_root_xy_velocity_noise": 0.15,
            "reset_root_yaw_rate_noise": 0.20,
            "terminate_airborne_duration": 0.25,
            "terminate_nonfoot_contact_duration": 0.12,
            "terminate_self_contact_duration": 0.10,
            "updates_per_batch": 4,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "discounting": 0.99,
            "reward_scaling": 1.0,
            "init_noise_std": 0.30,
            "clipping_epsilon": 0.20,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "learning_rate_schedule": "ADAPTIVE_KL",
        },
        "reward": {},
    },
    "direct_v1": {
        "description": (
            "Direct joint-position locomotion with no gait phase, contact "
            "schedule, or trajectory tracking."
        ),
        "args": {
            "desired_speed_m_s": 0.080,
            "command_forward_min": -0.10,
            "command_forward_max": 0.35,
            "command_lateral_max": 0.15,
            "command_yaw_rate_max": 0.60,
            "command_resample_time": 4.0,
            "command_stop_probability": 0.10,
            "no_observation_noise": False,
            "no_domain_randomization": False,
            "action_scale_abduction": 0.10,
            "action_scale_hip": 0.40,
            "action_scale_knee": 0.55,
            "reset_joint_noise": 0.015,
            "reset_velocity_noise": 0.05,
            "reset_root_xy_velocity_noise": 0.15,
            "reset_root_yaw_rate_noise": 0.20,
            "terminate_airborne_duration": 0.25,
            "terminate_nonfoot_contact_duration": 0.12,
            "terminate_self_contact_duration": 0.10,
            "updates_per_batch": 4,
            "learning_rate": 2e-4,
            "entropy_cost": 1e-2,
            "discounting": 0.99,
            "reward_scaling": 1.0,
            "init_noise_std": 0.30,
            "clipping_epsilon": 0.20,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "learning_rate_schedule": "ADAPTIVE_KL",
        },
        "reward": {},
    },
    "forward_stage1_v1": {
        "description": (
            "Stage-1 curriculum: learn stable 0.10 m/s straight walking "
            "from an exact stand reset before adding commands or randomization."
        ),
        "args": {
            "desired_speed_m_s": 0.10,
            "command_forward_min": 0.10,
            "command_forward_max": 0.10,
            "command_lateral_max": 0.0,
            "command_yaw_rate_max": 0.0,
            "command_resample_time": 4.0,
            "command_stop_probability": 0.0,
            "no_observation_noise": True,
            "no_domain_randomization": True,
            "action_scale_abduction": 0.06,
            "action_scale_hip": 0.25,
            "action_scale_knee": 0.35,
            "reset_joint_noise": 0.0,
            "reset_velocity_noise": 0.0,
            "reset_root_xy_velocity_noise": 0.0,
            "reset_root_yaw_rate_noise": 0.0,
            "terminate_airborne_duration": 0.25,
            "terminate_nonfoot_contact_duration": 0.12,
            "terminate_self_contact_duration": 0.10,
            "updates_per_batch": 1,
            "learning_rate": 2e-5,
            "entropy_cost": 0.0,
            "discounting": 0.99,
            "reward_scaling": 0.05,
            "init_noise_std": 0.08,
            "clipping_epsilon": 0.20,
            "max_grad_norm": 1.0,
            "desired_kl": 0.003,
            "learning_rate_schedule": "ADAPTIVE_KL",
        },
        "reward": {
            "velocity_tracking": 4.0,
            "velocity_tracking_sigma_m_s": 0.05,
            "yaw_rate_tracking": 0.25,
            "forward_progress": 0.0,
            "upright": 0.2,
            "upright_sigma_rad": 0.20,
            "angular_velocity": 0.15,
            "action_rate": 0.04,
            "termination": 20.0,
            "severe_extra_termination": 0.0,
            "early_termination_scale": 0.5,
        },
    },
}


PER_STEP_WALKING_METRICS_3D = (
    "forward_velocity_m_s",
    "forward_progress_m",
    "velocity_error_m_s",
    "vertical_velocity_m_s",
    "roll_pitch_angular_velocity_rms",
    "root_x_m",
    "root_y_m",
    "root_z_m",
    "root_height_error_m",
    "lateral_drift_m",
    "lateral_drift_exceeded",
    "lateral_velocity_m_s",
    "upright_tilt_rad",
    "heading_error_rad",
    "foot_contact_count",
    "foot_air_time_reward",
    "swing_clearance_cost",
    "foot_slip_rms_m_s",
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
    "action_rms",
    "action_rate_rms",
    "joint_velocity_rms_rad_s",
    "joint_limit_cost",
    "normalized_torque_rms",
    "desired_speed_m_s",
    "command_forward_velocity_m_s",
    "command_lateral_velocity_m_s",
    "command_yaw_rate_rad_s",
    "planar_velocity_error_m_s",
    "yaw_rate_error_rad_s",
)


WALKING_FAILURE_METRICS_3D = (
    "failure_nonfinite",
    "failure_nonfinite_action",
    "failure_nonfinite_physics",
    "failure_root_low",
    "failure_root_high",
    "failure_upright_tilt",
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


def _walking_network_factory(
    hidden_layers,
    activation_name,
    init_noise_std,
):
    """Build an ETH-style actor with state-independent exploration noise."""

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
            distribution_type="normal",
            noise_std_type="log",
            init_noise_std=init_noise_std,
            state_dependent_std=False,
            mean_kernel_init_fn=jnn.initializers.uniform,
            mean_kernel_init_kwargs={"scale": WALKING_ACTOR_MEAN_INIT_SCALE},
            mean_clip_scale=WALKING_ACTOR_MEAN_CLIP_SCALE,
            **kwargs,
        )

    return factory


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
    upright_failure_rate = metrics.get(
        "eval/episode_failure_upright_tilt", failed_rate
    )
    nonfinite_rate = metrics.get("eval/episode_failure_nonfinite", 0.0)
    distance = metrics.get("eval/episode_forward_progress_m", -math.inf)
    velocity = metrics.get("eval/avg_forward_velocity_m_s", math.inf)
    planar_tracking_error = abs(
        metrics.get(
            "eval/avg_planar_velocity_error_m_s",
            abs(velocity - desired_speed_m_s),
        )
    )
    yaw_tracking_error = abs(
        metrics.get("eval/avg_yaw_rate_error_rad_s", 0.0)
    )
    upright_tilt = metrics.get("eval/avg_upright_tilt_rad", math.inf)
    lateral_drift = abs(metrics.get("eval/avg_lateral_drift_m", math.inf))
    nonfoot = metrics.get(
        "eval/avg_nonfoot_ground_contact_count", math.inf
    )
    self_contact = metrics.get("eval/avg_self_contact_count", math.inf)
    survival = min(max(average_length / episode_length, 0.0), 1.0)
    raw_progress_quality = min(
        max(distance / max(target_distance_m, 1.0e-6), -1.0), 1.0
    )
    progress_quality = survival * raw_progress_quality
    velocity_quality = 1.0 - min(
        abs(velocity - desired_speed_m_s)
        / max(abs(desired_speed_m_s), 1.0e-4),
        1.0,
    )
    planar_tracking_quality = 1.0 - min(
        planar_tracking_error / max(abs(desired_speed_m_s), 0.10),
        1.0,
    )
    yaw_tracking_quality = 1.0 - min(
        yaw_tracking_error / 0.60,
        1.0,
    )
    tracking_quality = (
        0.75 * planar_tracking_quality + 0.25 * yaw_tracking_quality
    )
    nonfailure_quality = 1.0 - min(max(failed_rate, 0.0), 1.0)
    upright_quality = 1.0 - min(max(upright_tilt / 0.30, 0.0), 1.0)
    lateral_quality = 1.0 - min(max(lateral_drift / 0.05, 0.0), 1.0)
    contact_quality = 1.0 - min(
        max(nonfoot / 0.02, 0.0) + max(self_contact / 0.02, 0.0),
        1.0,
    )
    completed = float(survival >= 0.999 and failed_rate <= 0.001)
    rank = (
        completed,
        survival,
        1.0 - min(max(upright_failure_rate, 0.0), 1.0),
        nonfailure_quality,
        tracking_quality,
        progress_quality,
        contact_quality,
        upright_quality,
        lateral_quality,
    )
    # A readable scalar for logs.  Actual selection uses the lexicographic
    # rank above so contact quality can never outweigh a survival regression.
    score = (
        1000.0 * completed
        + 100.0 * survival
        + 20.0 * rank[2]
        + 10.0 * nonfailure_quality
        + 5.0 * tracking_quality
        + 3.0 * progress_quality
        + upright_quality
        + 0.5 * lateral_quality
        + 0.5 * contact_quality
    )
    rejected = (
        nonfinite_rate > 0.0
        or not math.isfinite(distance)
        or not math.isfinite(velocity)
        or not math.isfinite(planar_tracking_error)
        or not math.isfinite(yaw_tracking_error)
        or not math.isfinite(upright_tilt)
        or not math.isfinite(lateral_drift)
        or not math.isfinite(nonfoot)
        or not math.isfinite(self_contact)
        or not math.isfinite(upright_failure_rate)
        or not math.isfinite(score)
    )
    return {
        "score": -1_000_000.0 if rejected else score,
        "rank": ((float("-inf"),) * len(rank)) if rejected else rank,
        "rejected": rejected,
        "completed": completed,
        "survival": survival,
        "distance_m": distance,
        "raw_progress_quality": raw_progress_quality,
        "progress_quality": progress_quality,
        "forward_velocity_m_s": velocity,
        "velocity_quality": velocity_quality,
        "planar_tracking_error_m_s": planar_tracking_error,
        "yaw_tracking_error_rad_s": yaw_tracking_error,
        "tracking_quality": tracking_quality,
        "upright_failure_rate": upright_failure_rate,
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
            f"tracking_quality={selection['tracking_quality']:.2f}"
        ),
        (
            f"  tracking planar_error="
            f"{selection['planar_tracking_error_m_s']:.3f}m/s "
            f"yaw_rate_error="
            f"{selection['yaw_tracking_error_rad_s']:.3f}rad/s"
        ),
        (
            f"  pose    z={_metric(metrics, 'eval/avg_root_z_m'):.3f}m "
            f"tilt={selection['upright_tilt_rad']:.3f}rad "
            f"heading={_metric(metrics, 'eval/avg_heading_error_rad'):+.3f}rad "
            f"lateral={selection['lateral_drift_m']:.3f}m"
        ),
        (
            f"  feet    contacts="
            f"{_metric(metrics, 'eval/avg_foot_contact_count'):.2f} "
            f"air_time={_metric(metrics, 'eval/avg_foot_air_time_reward'):.3f} "
            f"clearance={_metric(metrics, 'eval/avg_swing_clearance_cost'):.3f} "
            f"slip={_metric(metrics, 'eval/avg_foot_slip_rms_m_s'):.3f}m/s"
        ),
        (
            f"  safety  nonfoot="
            f"{_metric(metrics, 'eval/avg_nonfoot_ground_contact_count'):.3f} "
            f"self={_metric(metrics, 'eval/avg_self_contact_count'):.3f} "
            f"air={_metric(metrics, 'eval/avg_airborne_active'):.2%}"
        ),
        (
            f"  control action="
            f"{_metric(metrics, 'eval/avg_action_rms'):.3f} "
            f"rate={_metric(metrics, 'eval/avg_action_rate_rms'):.3f} "
            f"joint_vel="
            f"{_metric(metrics, 'eval/avg_joint_velocity_rms_rad_s'):.3f} "
            f"limit={_metric(metrics, 'eval/avg_joint_limit_cost'):.3f} "
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
            f"std={_metric(metrics, 'training/policy_dist_mean_std'):.4f} "
            f"lr={_metric(metrics, 'training/learning_rate'):.2e} "
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
        "--recipe", choices=tuple(WALKING_RECIPES_3D), default="anymal_v1"
    )
    parser.add_argument(
        "--physics-profile",
        choices=WALKING_PHYSICS_PROFILE_NAMES_3D,
        default="cg12",
    )
    parser.add_argument(
        "--geometry",
        choices=WALKING_GEOMETRY_NAMES_3D,
        default="pupper_open60",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--desired-speed", dest="desired_speed_m_s", type=float)
    parser.add_argument("--command-forward-min", type=float)
    parser.add_argument("--command-forward-max", type=float)
    parser.add_argument("--command-lateral-max", type=float)
    parser.add_argument("--command-yaw-rate-max", type=float)
    parser.add_argument("--command-resample-time", type=float)
    parser.add_argument("--command-stop-probability", type=float)
    observation_noise_group = parser.add_mutually_exclusive_group()
    observation_noise_group.add_argument(
        "--no-observation-noise",
        dest="no_observation_noise",
        action="store_true",
    )
    observation_noise_group.add_argument(
        "--observation-noise",
        dest="no_observation_noise",
        action="store_false",
    )
    parser.set_defaults(no_observation_noise=None)
    parser.add_argument("--action-scale-abduction", type=float)
    parser.add_argument("--action-scale-hip", type=float)
    parser.add_argument("--action-scale-knee", type=float)
    parser.add_argument("--reset-keyframe", default="stand")
    parser.add_argument("--reset-joint-noise", type=float)
    parser.add_argument("--reset-velocity-noise", type=float)
    parser.add_argument("--reset-root-xy-velocity-noise", type=float)
    parser.add_argument("--reset-root-yaw-rate-noise", type=float)
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
        "--diagnostic-lateral-drift",
        "--terminate-lateral-drift",
        dest="diagnostic_lateral_drift",
        type=float,
        default=1.50,
        help=(
            "absolute world-y displacement threshold used only for logging; "
            "the legacy --terminate-lateral-drift name is retained as an alias"
        ),
    )
    parser.add_argument(
        "--terminate-airborne-duration", type=float
    )
    parser.add_argument(
        "--terminate-nonfoot-depth", type=float, default=0.004
    )
    parser.add_argument(
        "--terminate-nonfoot-contact-duration", type=float
    )
    parser.add_argument(
        "--terminate-self-contact-depth", type=float, default=0.004
    )
    parser.add_argument(
        "--terminate-self-contact-duration", type=float
    )
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-cost", type=float)
    parser.add_argument("--discounting", type=float)
    parser.add_argument("--reward-scaling", type=float)
    parser.add_argument("--init-noise-std", type=float)
    parser.add_argument("--clipping-epsilon", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--desired-kl", type=float)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("NONE", "ADAPTIVE_KL"),
    )
    parser.add_argument(
        "--deterministic-eval",
        dest="deterministic_eval",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--stochastic-eval",
        dest="deterministic_eval",
        action="store_false",
        help="sample policy actions during periodic eval instead of using the mean",
    )
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
        default=Path("results") / "mjx_3d_walking_direct_v1",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--save-ppo-checkpoints", action="store_true")
    parser.add_argument("--ppo-checkpoint-dir", type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--selection-target-distance", type=float)
    domain_randomization_group = parser.add_mutually_exclusive_group()
    domain_randomization_group.add_argument(
        "--no-domain-randomization",
        dest="no_domain_randomization",
        action="store_true",
    )
    domain_randomization_group.add_argument(
        "--domain-randomization",
        dest="no_domain_randomization",
        action="store_false",
    )
    parser.set_defaults(no_domain_randomization=None)
    parser.add_argument("--friction-range", type=float, nargs=2, default=(0.60, 1.40))
    parser.add_argument("--mass-range", type=float, nargs=2, default=(0.90, 1.10))
    parser.add_argument("--actuator-gain-range", type=float, nargs=2, default=(0.90, 1.10))
    parser.add_argument("--joint-damping-range", type=float, nargs=2, default=(0.80, 1.20))
    parser.add_argument("--joint-armature-range", type=float, nargs=2, default=(0.80, 1.20))
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
        "desired_speed_m_s",
        "action_scale_abduction",
        "action_scale_hip",
        "action_scale_knee",
        "learning_rate",
        "init_noise_std",
        "clipping_epsilon",
        "max_grad_norm",
        "desired_kl",
        "reward_scaling",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command_forward_max < args.command_forward_min:
        parser.error("command forward range must be ordered")
    for name in (
        "command_lateral_max",
        "command_yaw_rate_max",
        "reset_joint_noise",
        "reset_velocity_noise",
        "reset_root_xy_velocity_noise",
        "reset_root_yaw_rate_noise",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0.0 <= args.command_stop_probability <= 1.0:
        parser.error("--command-stop-probability must be in [0, 1]")
    if not 0.0 < args.discounting <= 1.0:
        parser.error("--discounting must be in (0, 1]")
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

    action_scales = (
        args.action_scale_abduction,
        args.action_scale_hip,
        args.action_scale_knee,
    ) * 4
    task = walking_physics_profile_3d(
        args.physics_profile,
        Walking3DConfig(
            geometry=args.geometry,
            episode_length=args.episode_length,
            reset_keyframe_name=args.reset_keyframe,
            desired_speed_m_s=args.desired_speed_m_s,
            command_forward_velocity_range_m_s=(
                args.command_forward_min, args.command_forward_max
            ),
            command_lateral_velocity_range_m_s=(
                -args.command_lateral_max, args.command_lateral_max
            ),
            command_yaw_rate_range_rad_s=(
                -args.command_yaw_rate_max, args.command_yaw_rate_max
            ),
            command_resample_time_s=args.command_resample_time,
            command_deadband_probability=args.command_stop_probability,
            observation_noise_enabled=not args.no_observation_noise,
            action_scales=action_scales,
            reset_joint_noise_rad=args.reset_joint_noise,
            reset_velocity_noise=args.reset_velocity_noise,
            reset_root_xy_velocity_noise_m_s=(
                args.reset_root_xy_velocity_noise
            ),
            reset_root_yaw_rate_noise_rad_s=(
                args.reset_root_yaw_rate_noise
            ),
            terminate_root_z_min=args.terminate_root_z_min,
            terminate_root_z_low_duration_s=(
                args.terminate_root_z_low_duration
            ),
            terminate_root_z_max=args.terminate_root_z_max,
            terminate_upright_tilt_rad=args.terminate_upright_tilt,
            terminate_upright_tilt_duration_s=(
                args.terminate_upright_tilt_duration
            ),
            diagnostic_lateral_drift_m=args.diagnostic_lateral_drift,
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
    domain_randomization = Walking3DDomainRandomization(
        geom_friction_scale=tuple(args.friction_range),
        body_mass_scale=tuple(args.mass_range),
        actuator_gain_scale=tuple(args.actuator_gain_range),
        joint_damping_scale=tuple(args.joint_damping_range),
        joint_armature_scale=tuple(args.joint_armature_range),
    )
    randomization_fn = None if args.no_domain_randomization else (
        make_walking_domain_randomization_fn_3d(domain_randomization)
    )
    train_env = make_brax_walking_env_3d(
        task, reward_config=reward_config, seed=args.seed
    )
    eval_task = replace(
        task,
        command_forward_velocity_range_m_s=(
            task.desired_speed_m_s, task.desired_speed_m_s
        ),
        command_lateral_velocity_range_m_s=(0.0, 0.0),
        command_yaw_rate_range_rad_s=(0.0, 0.0),
        command_deadband_probability=0.0,
        observation_noise_enabled=False,
        reset_joint_noise_rad=0.0,
        reset_velocity_noise=0.0,
        reset_root_xy_velocity_noise_m_s=0.0,
        reset_root_yaw_rate_noise_rad_s=0.0,
    )
    eval_env = make_brax_walking_env_3d(
        eval_task, reward_config=reward_config, seed=args.seed + 10_000
    )
    target_distance_m = args.selection_target_distance or (
        task.desired_speed_m_s
        * args.episode_length
        * task.control_timestep
    )

    metric_history = []
    reward_history = []
    best = {
        "score": float("-inf"),
        "rank": None,
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
            desired_speed_m_s=task.desired_speed_m_s,
        )
        selected = (
            not selection["rejected"]
            and (
                best["rank"] is None
                or selection["rank"] > best["rank"]
            )
        )
        if selected:
            best["score"] = selection["score"]
            best["rank"] = selection["rank"]
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
                desired_speed_m_s=task.desired_speed_m_s,
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
        "policy_distribution": "normal",
        "policy_noise_std_type": "log",
        "policy_state_dependent_std": False,
        "policy_mean_kernel_init": "uniform",
        "policy_mean_kernel_init_scale": WALKING_ACTOR_MEAN_INIT_SCALE,
        "policy_mean_clip_scale": WALKING_ACTOR_MEAN_CLIP_SCALE,
        "init_noise_std": args.init_noise_std,
        "observation_normalization": False,
        "observation_scaling": "fixed_task_scales",
        "bootstrap_on_timeout": True,
        "deterministic_eval": args.deterministic_eval,
        "clipping_epsilon": args.clipping_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "desired_kl": args.desired_kl,
        "learning_rate_schedule": args.learning_rate_schedule,
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "seed": args.seed,
        "task": asdict(task),
        "evaluation_task": asdict(eval_task),
        "reward": asdict(reward_config),
        "domain_randomization": (
            None if args.no_domain_randomization else asdict(domain_randomization)
        ),
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
        f"speed={task.desired_speed_m_s:.3f}m/s "
        f"target_distance={target_distance_m:.3f}m\n"
        f"  control=direct_joint_position reset={task.reset_keyframe_name} "
        f"abduction_scale={args.action_scale_abduction:.2f}rad "
        f"hip_scale={args.action_scale_hip:.2f}rad "
        f"knee_scale={args.action_scale_knee:.2f}rad\n"
        f"  reset_noise joint={task.reset_joint_noise_rad:g}rad "
        f"qvel={task.reset_velocity_noise:g} "
        f"root_xy={task.reset_root_xy_velocity_noise_m_s:g}m/s "
        f"root_yaw={task.reset_root_yaw_rate_noise_rad_s:g}rad/s\n"
        f"  noise observation={task.observation_noise_enabled} "
        f"domain_randomization={randomization_fn is not None}\n"
        f"  lr={args.learning_rate:g} entropy={args.entropy_cost:g} "
        f"discount={args.discounting:g} seed={args.seed}\n"
        f"  ppo_clip={args.clipping_epsilon:g} "
        f"grad_norm={args.max_grad_norm:g} "
        f"desired_kl={args.desired_kl:g} "
        f"lr_schedule={args.learning_rate_schedule}\n"
        f"  policy=normal state_dependent_std=false "
        f"mean_init_scale={WALKING_ACTOR_MEAN_INIT_SCALE:g} "
        f"mean_clip={WALKING_ACTOR_MEAN_CLIP_SCALE:g} "
        f"init_std={args.init_noise_std:g} "
        f"deterministic_eval={args.deterministic_eval}\n"
        f"  observation=fixed_task_scaling bootstrap_timeout=true\n"
        f"  output={args.out.resolve()}",
        flush=True,
    )

    checkpoint_kwargs = {}
    train_parameters = inspect.signature(ppo.train).parameters
    required_stability_parameters = {
        "clipping_epsilon",
        "max_grad_norm",
        "desired_kl",
        "learning_rate_schedule",
        "deterministic_eval",
        "bootstrap_on_timeout",
    }
    missing_stability_parameters = sorted(
        required_stability_parameters - set(train_parameters)
    )
    if missing_stability_parameters:
        raise SystemExit(
            "Installed Brax PPO lacks required stability parameters: "
            + ", ".join(missing_stability_parameters)
        )
    if randomization_fn is not None:
        if "randomization_fn" not in train_parameters:
            raise SystemExit("Installed Brax does not support randomization_fn.")
        checkpoint_kwargs["randomization_fn"] = randomization_fn
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
        normalize_observations=False,
        bootstrap_on_timeout=True,
        clipping_epsilon=args.clipping_epsilon,
        max_grad_norm=args.max_grad_norm,
        desired_kl=args.desired_kl,
        learning_rate_schedule=args.learning_rate_schedule,
        deterministic_eval=args.deterministic_eval,
        network_factory=_walking_network_factory(
            args.hidden_layers,
            args.activation,
            args.init_noise_std,
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
        "best_selection_rank": (
            list(best["rank"]) if best["rank"] is not None else None
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
