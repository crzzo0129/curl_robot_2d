"""Train a reference-free PPO walking policy for the 3-D curl robot."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from curl_robot_2d_mjx.environment_walking_3d import (
    WALKING_ACTION_GROUP_LABELS_3D,
    WALKING_ACTION_SATURATION_THRESHOLD_3D,
    WALKING_FOOT_LABELS_3D,
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
WALKING_ACTION_P95_RANGE_LIMIT_THRESHOLD_3D = 0.90
WALKING_ACTION_SATURATION_FRACTION_THRESHOLD_3D = 0.05


@contextmanager
def _running_statistics_update_scope(module, *, freeze):
    """Temporarily keep restored observation-normalizer statistics fixed."""

    if not freeze:
        yield
        return

    original_update = module.update

    def keep_existing_statistics(state, batch, **kwargs):
        del batch, kwargs
        return state

    module.update = keep_existing_statistics
    try:
        yield
    finally:
        module.update = original_update


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
            "gait_phase_enabled": False,
            "gait_cycle_time": 0.625,
            "gait_duty_factor": 0.68,
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
            "unroll_length": 20,
            "learning_rate": 3e-4,
            "adaptive_kl_min_lr": 3e-5,
            "adaptive_kl_max_lr": 3e-4,
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
            "gait_phase_enabled": False,
            "gait_cycle_time": 0.625,
            "gait_duty_factor": 0.68,
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
            "unroll_length": 20,
            "learning_rate": 2e-4,
            "adaptive_kl_min_lr": 2e-5,
            "adaptive_kl_max_lr": 2e-4,
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
            "gait_phase_enabled": False,
            "gait_cycle_time": 0.625,
            "gait_duty_factor": 0.68,
            "action_scale_abduction": 0.06,
            "action_scale_hip": 0.50,
            "action_scale_knee": 0.65,
            "reset_joint_noise": 0.0,
            "reset_velocity_noise": 0.0,
            "reset_root_xy_velocity_noise": 0.0,
            "reset_root_yaw_rate_noise": 0.0,
            "terminate_airborne_duration": 0.25,
            "terminate_nonfoot_contact_duration": 0.12,
            "terminate_self_contact_duration": 0.10,
            "updates_per_batch": 1,
            "unroll_length": 40,
            "learning_rate": 2e-5,
            "adaptive_kl_min_lr": 2e-6,
            "adaptive_kl_max_lr": 2e-5,
            "entropy_cost": 0.01,
            "discounting": 0.99,
            "reward_scaling": 0.05,
            "init_noise_std": 0.10,
            "clipping_epsilon": 0.20,
            "max_grad_norm": 1.0,
            "desired_kl": 0.003,
            "learning_rate_schedule": "ADAPTIVE_KL",
        },
        "reward": {
            "velocity_tracking": 4.0,
            "velocity_tracking_sigma_m_s": 0.05,
            "overspeed": 1.0,
            "overspeed_margin_m_s": 0.05,
            "overspeed_scale_m_s": 0.15,
            "yaw_rate_tracking": 0.25,
            "forward_progress": 0.0,
            "upright": 0.2,
            "upright_sigma_rad": 0.20,
            "stagnation": 0.2,
            "stagnation_window_s": 1.0,
            "stagnation_min_progress_m": 0.05,
            "upright_stagnation_gate": 1.0,
            "angular_velocity": 0.15,
            "foot_air_time": 0.8,
            "swing_clearance": 0.15,
            "swing_clearance_m": 0.025,
            "swing_clearance_speed_m_s": 0.10,
            "action_rate": 0.04,
            "termination": 20.0,
            "severe_extra_termination": 0.0,
            "early_termination_scale": 0.5,
        },
    },
    "forward_phase_bootstrap_v1": {
        "description": (
            "Stage-A 0.10 m/s bootstrap with an observable 0.625 s diagonal "
            "trot phase, dense contact scheduling, and small reset noise."
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
            "gait_phase_enabled": True,
            "gait_cycle_time": 0.625,
            "gait_duty_factor": 0.68,
            "action_scale_abduction": 0.06,
            "action_scale_hip": 0.50,
            "action_scale_knee": 0.65,
            "reset_joint_noise": 0.01,
            "reset_velocity_noise": 0.02,
            "reset_root_xy_velocity_noise": 0.03,
            "reset_root_yaw_rate_noise": 0.05,
            "terminate_airborne_duration": 0.25,
            "terminate_nonfoot_contact_duration": 0.12,
            "terminate_self_contact_duration": 0.10,
            "updates_per_batch": 4,
            "unroll_length": 40,
            "learning_rate": 1e-4,
            "adaptive_kl_min_lr": 1e-5,
            "adaptive_kl_max_lr": 1e-4,
            "entropy_cost": 0.01,
            "discounting": 0.99,
            "reward_scaling": 0.05,
            "init_noise_std": 0.30,
            "clipping_epsilon": 0.20,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "learning_rate_schedule": "ADAPTIVE_KL",
        },
        "reward": {
            "velocity_tracking": 4.0,
            "velocity_tracking_sigma_m_s": 0.05,
            "velocity_tracking_upright_gate": 0.0,
            "overspeed": 1.0,
            "overspeed_margin_m_s": 0.05,
            "overspeed_scale_m_s": 0.15,
            "yaw_rate_tracking": 0.25,
            "forward_progress": 0.0,
            "upright": 0.4,
            "upright_sigma_rad": 0.20,
            "stagnation": 0.2,
            "stagnation_window_s": 1.0,
            "stagnation_min_progress_m": 0.05,
            "upright_stagnation_gate": 1.0,
            "angular_velocity": 0.15,
            "foot_air_time": 0.2,
            "gait_contact": 1.0,
            "swing_clearance": 0.25,
            "swing_clearance_m": 0.025,
            "swing_clearance_target_sigma_m": 0.0075,
            "swing_clearance_target_tracking": 1.0,
            "swing_clearance_speed_m_s": 0.10,
            "action_rate": 0.04,
            "termination": 20.0,
            "severe_extra_termination": 0.0,
            "early_termination_scale": 0.5,
        },
    },
    "unitree_mjlab_velocity_discovery_v1": {
        "description": (
            "Route-B discovery: learn nominal 0.10 m/s forward locomotion "
            "before enabling command diversity, observation noise, domain "
            "randomization, or swing-foot clearance regularization."
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
            "gait_phase_enabled": True,
            "gait_cycle_time": 0.60,
            "gait_duty_factor": 0.56,
            "asymmetric_observations": True,
            "normalize_observations": True,
            "small_actor_mean_init": False,
            "hidden_layers": [512, 256, 128],
            "critic_hidden_layers": [512, 256, 128],
            # The action is relative to this robot's stand keyframe.  Phase
            # never maps to a joint target, so front-knee geometry is not
            # borrowed from Unitree.
            "action_scale_abduction": 0.08,
            "action_scale_hip": 0.25,
            "action_scale_knee": 0.25,
            "reset_joint_noise": 0.0,
            "reset_velocity_noise": 0.0,
            "reset_root_xy_velocity_noise": 0.0,
            "reset_root_yaw_rate_noise": 0.0,
            "terminate_upright_tilt": 1.22,
            "terminate_upright_tilt_duration": 0.02,
            "terminate_airborne_duration": 0.25,
            "terminate_nonfoot_force_min": 1.0,
            "terminate_nonfoot_contact_duration": 0.02,
            "terminate_self_contact_duration": 0.04,
            "terminate_low_progress_enabled": False,
            "eval_terminate_low_progress_enabled": False,
            "terminate_low_progress_window": 0.50,
            "terminate_low_progress_duration": 2.0,
            "terminate_low_progress_command_ratio": 0.50,
            "terminate_low_progress_cap": 0.05,
            "updates_per_batch": 5,
            "unroll_length": 24,
            "learning_rate": 3e-4,
            "adaptive_kl_min_lr": 3e-5,
            "adaptive_kl_max_lr": 3e-4,
            "entropy_cost": 0.01,
            "discounting": 0.99,
            # Unitree MjLab integrates reward rates by its 0.02 s control dt.
            # Brax applies this equivalent multiplier inside the PPO loss.
            "reward_scaling": 0.02,
            "init_noise_std": 0.50,
            "clipping_epsilon": 0.20,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "learning_rate_schedule": "ADAPTIVE_KL",
        },
        "reward": {
            "velocity_tracking": 1.0,
            "velocity_tracking_sigma_m_s": 0.10,
            "velocity_tracking_vertical_weight": 2.0,
            "velocity_tracking_upright_gate": 0.0,
            "overspeed": 0.0,
            "yaw_rate_tracking": 0.25,
            "yaw_rate_tracking_sigma_rad_s": 0.50,
            "yaw_rate_tracking_roll_pitch_weight": 0.05,
            "yaw_rate_tracking_progress_gate": 1.0,
            "forward_progress": 0.0,
            "upright": 0.0,
            "stagnation": 0.0,
            "height": 0.0,
            "heading": 0.0,
            "lateral_velocity": 0.0,
            "lateral_drift": 0.0,
            "vertical_velocity": 0.0,
            "angular_velocity": 0.05,
            "angular_velocity_sigma_rad_s": 1.0,
            "orientation": 1.0,
            "angular_momentum": 0.025,
            "foot_air_time": 0.0,
            "gait_contact": 0.5,
            "swing_clearance": 0.0,
            # During discovery this term opposes the first tentative swing:
            # it is restored in the robust continuation recipe below.
            "foot_clearance": 0.0,
            "foot_clearance_target_m": 0.025,
            "foot_slip": 0.25,
            "foot_slip_sigma_m_s": 1.0,
            "soft_landing": 0.001,
            "action_rate": 0.05,
            "action_magnitude": 0.0,
            "joint_velocity": 0.0,
            "joint_acceleration": 2.5e-7,
            "joint_limits": 10.0,
            "stand_still": 1.0,
            "torque": 0.0,
            "nonfoot_contact": 0.0,
            "nonfoot_depth": 0.0,
            "self_contact": 0.0,
            "self_contact_depth": 0.0,
            "termination": 200.0,
            "severe_extra_termination": 0.0,
            "nonfinite_termination": 200.0,
            "early_termination_scale": 0.0,
        },
    },
    "unitree_mjlab_velocity_v1": {
        "description": (
            "Route B: Unitree/MjLab-style phase-guided velocity locomotion "
            "with an asymmetric critic and no joint-pose trajectory reference."
        ),
        "args": {
            "desired_speed_m_s": 0.20,
            "command_forward_min": 0.10,
            "command_forward_max": 0.30,
            "command_lateral_max": 0.0,
            "command_yaw_rate_max": 0.0,
            "command_resample_time": 4.0,
            "command_stop_probability": 0.0,
            "no_observation_noise": False,
            "no_domain_randomization": False,
            "gait_phase_enabled": True,
            "gait_cycle_time": 0.60,
            "gait_duty_factor": 0.56,
            "asymmetric_observations": True,
            "normalize_observations": True,
            "small_actor_mean_init": False,
            "hidden_layers": [512, 256, 128],
            "critic_hidden_layers": [512, 256, 128],
            # Robot-specific action radii.  These are not phase poses and do
            # not assume Unitree's rear-folding front knees.
            "action_scale_abduction": 0.08,
            "action_scale_hip": 0.25,
            "action_scale_knee": 0.25,
            "reset_joint_noise": 0.015,
            "reset_velocity_noise": 0.05,
            "reset_root_xy_velocity_noise": 0.10,
            "reset_root_yaw_rate_noise": 0.15,
            "terminate_upright_tilt": 1.22,
            "terminate_upright_tilt_duration": 0.02,
            "terminate_airborne_duration": 0.25,
            "terminate_nonfoot_force_min": 1.0,
            "terminate_nonfoot_contact_duration": 0.02,
            "terminate_self_contact_duration": 0.04,
            "terminate_low_progress_enabled": False,
            "eval_terminate_low_progress_enabled": False,
            "terminate_low_progress_window": 0.50,
            "terminate_low_progress_duration": 2.0,
            "terminate_low_progress_command_ratio": 0.50,
            "terminate_low_progress_cap": 0.05,
            "updates_per_batch": 5,
            "unroll_length": 24,
            "learning_rate": 3e-4,
            "adaptive_kl_min_lr": 3e-5,
            "adaptive_kl_max_lr": 3e-4,
            "entropy_cost": 0.01,
            "discounting": 0.99,
            # Match MjLab's default reward-rate integration by control dt.
            "reward_scaling": 0.02,
            "init_noise_std": 0.50,
            "clipping_epsilon": 0.20,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "learning_rate_schedule": "ADAPTIVE_KL",
        },
        "reward": {
            "velocity_tracking": 1.0,
            "velocity_tracking_sigma_m_s": 0.10,
            "velocity_tracking_vertical_weight": 2.0,
            "velocity_tracking_upright_gate": 0.0,
            "overspeed": 0.0,
            "yaw_rate_tracking": 0.75,
            "yaw_rate_tracking_sigma_rad_s": 0.50,
            "yaw_rate_tracking_roll_pitch_weight": 0.05,
            "yaw_rate_tracking_progress_gate": 1.0,
            "forward_progress": 0.0,
            "upright": 0.0,
            "stagnation": 0.0,
            "height": 0.0,
            "heading": 0.0,
            "lateral_velocity": 0.0,
            "lateral_drift": 0.0,
            "vertical_velocity": 0.0,
            "angular_velocity": 0.05,
            "angular_velocity_sigma_rad_s": 1.0,
            "orientation": 1.0,
            "angular_momentum": 0.025,
            "foot_air_time": 0.0,
            "gait_contact": 0.5,
            "swing_clearance": 0.0,
            "foot_clearance": 1.0,
            "foot_clearance_target_m": 0.025,
            "foot_slip": 0.25,
            "foot_slip_sigma_m_s": 1.0,
            "soft_landing": 0.001,
            "action_rate": 0.05,
            "action_magnitude": 0.0,
            "joint_velocity": 0.0,
            "joint_acceleration": 2.5e-7,
            "joint_limits": 10.0,
            "stand_still": 1.0,
            "torque": 0.0,
            "nonfoot_contact": 0.0,
            "nonfoot_depth": 0.0,
            "self_contact": 0.0,
            "self_contact_depth": 0.0,
            "termination": 200.0,
            "severe_extra_termination": 0.0,
            "nonfinite_termination": 200.0,
            "early_termination_scale": 0.0,
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
    "yaw_rate_rad_s",
    "foot_contact_count",
    "foot_air_time_reward",
    "foot_air_time_mean_s",
    "gait_contact_reward",
    "swing_clearance_reward",
    "foot_clearance_cost",
    "gait_phase",
    "foot_slip_rms_m_s",
    "nonfoot_ground_contact_count",
    "nonfoot_ground_depth_m",
    "nonfoot_ground_max_force_n",
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
    "overspeed_m_s",
    "yaw_rate_error_rad_s",
    "progress_window_m",
    "stagnation_fraction",
    "low_progress_window_m",
    "low_progress_required_m",
    "low_progress_active",
) + tuple(
    f"{prefix}_{label}{suffix}"
    for prefix, suffix in (
        ("foot_contact", ""),
        ("foot_air_time", "_s"),
        ("foot_slip", "_m_s"),
        ("action_rms", ""),
    )
    for label in WALKING_FOOT_LABELS_3D
) + tuple(
    f"action_{metric}_{label}"
    for metric in ("rms", "saturation")
    for label in WALKING_ACTION_GROUP_LABELS_3D
) + ("action_rms_left_right_delta",)


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
    "failure_low_progress",
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
    critic_hidden_layers,
    activation_name,
    init_noise_std,
    *,
    asymmetric_observations,
    small_actor_mean_init,
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
        network_kwargs = dict(
            policy_hidden_layer_sizes=tuple(hidden_layers),
            value_hidden_layer_sizes=tuple(critic_hidden_layers),
            activation=activation,
            distribution_type="normal",
            noise_std_type="log",
            init_noise_std=init_noise_std,
            state_dependent_std=False,
            mean_clip_scale=WALKING_ACTOR_MEAN_CLIP_SCALE,
            **kwargs,
        )
        if asymmetric_observations:
            network_kwargs.update(
                policy_obs_key="state",
                value_obs_key="privileged_state",
            )
        if small_actor_mean_init:
            network_kwargs.update(
                mean_kernel_init_fn=jnn.initializers.uniform,
                mean_kernel_init_kwargs={
                    "scale": WALKING_ACTOR_MEAN_INIT_SCALE
                },
            )
        return networks.make_ppo_networks(*args, **network_kwargs)

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
    final_heading_change = abs(
        metrics.get(
            "eval/episode_heading_change_rad",
            metrics.get("eval/avg_heading_error_rad", 0.0),
        )
    )
    final_lateral_drift = abs(
        metrics.get("eval/episode_lateral_progress_m", lateral_drift)
    )
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
    heading_quality = 1.0 - min(final_heading_change / 0.50, 1.0)
    final_lateral_quality = 1.0 - min(
        final_lateral_drift / 0.10, 1.0
    )
    direction_quality = 0.75 * heading_quality + 0.25 * final_lateral_quality
    nonfailure_quality = 1.0 - min(max(failed_rate, 0.0), 1.0)
    upright_quality = 1.0 - min(max(upright_tilt / 0.30, 0.0), 1.0)
    lateral_quality = 1.0 - min(max(lateral_drift / 0.05, 0.0), 1.0)
    contact_quality = 1.0 - min(
        max(nonfoot / 0.02, 0.0) + max(self_contact / 0.02, 0.0),
        1.0,
    )
    meaningful_progress = float(raw_progress_quality >= 0.25)
    upright_nonfailure_quality = 1.0 - min(
        max(upright_failure_rate, 0.0), 1.0
    )
    completed = float(
        survival >= 0.999
        and failed_rate <= 0.001
        and raw_progress_quality >= 0.50
        and planar_tracking_quality >= 0.25
    )
    rank = (
        completed,
        meaningful_progress,
        survival,
        upright_nonfailure_quality,
        nonfailure_quality,
        tracking_quality,
        direction_quality,
        progress_quality,
        contact_quality,
        upright_quality,
        lateral_quality,
    )
    # A readable scalar for logs.  Actual selection uses the lexicographic
    # rank above so contact quality can never outweigh a survival regression.
    score = (
        1000.0 * completed
        + 100.0 * meaningful_progress
        + 100.0 * survival
        + 20.0 * upright_nonfailure_quality
        + 10.0 * nonfailure_quality
        + 5.0 * tracking_quality
        + 2.0 * direction_quality
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
        or not math.isfinite(final_heading_change)
        or not math.isfinite(final_lateral_drift)
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
        "meaningful_progress": meaningful_progress,
        "survival": survival,
        "distance_m": distance,
        "raw_progress_quality": raw_progress_quality,
        "progress_quality": progress_quality,
        "forward_velocity_m_s": velocity,
        "velocity_quality": velocity_quality,
        "planar_tracking_error_m_s": planar_tracking_error,
        "yaw_tracking_error_rad_s": yaw_tracking_error,
        "tracking_quality": tracking_quality,
        "final_heading_change_rad": final_heading_change,
        "final_lateral_drift_m": final_lateral_drift,
        "direction_quality": direction_quality,
        "upright_failure_rate": upright_failure_rate,
        "upright_tilt_rad": upright_tilt,
        "lateral_drift_m": lateral_drift,
        "contact_quality": contact_quality,
    }


def _checkpoint_is_selectable_walking_3d(selection, current_best_rank):
    """Reject stationary evals before applying lexicographic ranking."""

    return (
        not selection["rejected"]
        and selection["meaningful_progress"] > 0.0
        and (
            current_best_rank is None
            or selection["rank"] > current_best_rank
        )
    )


def _per_foot_eval_stats(metrics, label):
    contact_steps = _metric(metrics, f"eval/episode_foot_contact_{label}")
    touchdown_count = _metric(
        metrics, f"eval/episode_touchdown_count_{label}"
    )
    return {
        "contact": _metric(metrics, f"eval/avg_foot_contact_{label}"),
        "touchdown_air_time_s": (
            _metric(
                metrics,
                f"eval/episode_touchdown_air_time_sum_{label}_s",
            )
            / max(touchdown_count, 1.0)
        ),
        "contact_slip_m_s": (
            _metric(metrics, f"eval/episode_foot_slip_{label}_m_s")
            / max(contact_steps, 1.0)
        ),
        "action_rms": _metric(metrics, f"eval/avg_action_rms_{label}"),
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
    reward_scaling,
    selection,
    selected,
):
    marker = " new_best" if selected else ""
    touchdown_count = _metric(metrics, "eval/episode_touchdown_count")
    touchdown_air_time_s = (
        _metric(metrics, "eval/episode_touchdown_air_time_sum_s")
        / max(touchdown_count, 1.0)
    )
    foot_stats = {
        label: _per_foot_eval_stats(metrics, label)
        for label in WALKING_FOOT_LABELS_3D
    }
    left_contact = 0.5 * (
        foot_stats["fl"]["contact"] + foot_stats["rl"]["contact"]
    )
    right_contact = 0.5 * (
        foot_stats["fr"]["contact"] + foot_stats["rr"]["contact"]
    )
    lines = [
        (
            f"[eval {eval_index}/{total_evals}] step={int(step)} "
            f"physical_score={selection['score']:.4f}{marker}"
        ),
        (
            f"  outcome reward_raw="
            f"{_metric(metrics, 'eval/episode_reward'):+.3f} "
            f"avg_raw/step={_metric(metrics, 'eval/avg_reward'):+.4f} "
            f"avg_ppo/step="
            f"{_metric(metrics, 'eval/avg_reward') * reward_scaling:+.4f} "
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
            f"overspeed={_metric(metrics, 'eval/avg_overspeed_m_s'):.3f}m/s "
            f"progress_1s={_metric(metrics, 'eval/avg_progress_window_m'):.3f}m "
            f"stagnation={_metric(metrics, 'eval/avg_stagnation_fraction'):.2f} "
            f"progress_0.5s="
            f"{_metric(metrics, 'eval/avg_low_progress_window_m'):.3f}/"
            f"{_metric(metrics, 'eval/avg_low_progress_required_m'):.3f}m "
            f"low_progress="
            f"{_metric(metrics, 'eval/avg_low_progress_active'):.1%} "
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
            f"  direction yaw_signed="
            f"{_metric(metrics, 'eval/avg_yaw_rate_rad_s'):+.3f}rad/s "
            f"heading_final="
            f"{_metric(metrics, 'eval/episode_heading_change_rad'):+.3f}rad "
            f"heading_peak="
            f"{_metric(metrics, 'eval/episode_heading_abs_peak_increment_rad'):.3f}rad "
            f"lateral_final="
            f"{_metric(metrics, 'eval/episode_lateral_progress_m'):+.3f}m "
            f"quality={selection['direction_quality']:.2f}"
        ),
        (
            f"  feet    contacts="
            f"{_metric(metrics, 'eval/avg_foot_contact_count'):.2f} "
            f"air_time={touchdown_air_time_s:.3f}s "
            f"air_mean={_metric(metrics, 'eval/avg_foot_air_time_mean_s'):.3f}s "
            f"clearance_cost="
            f"{_metric(metrics, 'eval/avg_foot_clearance_cost'):.3f} "
            f"slip={_metric(metrics, 'eval/avg_foot_slip_rms_m_s'):.3f}m/s"
        ),
        (
            f"  safety  nonfoot="
            f"{_metric(metrics, 'eval/avg_nonfoot_ground_contact_count'):.3f} "
            f"nonfoot_force_avg="
            f"{_metric(metrics, 'eval/avg_nonfoot_ground_max_force_n'):.2f}N "
            f"nonfoot_force_peak="
            f"{_metric(metrics, 'eval/episode_nonfoot_ground_peak_force_n'):.2f}N "
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
        (
            "  action/by_joint "
            + " | ".join(
                f"{label} rms="
                f"{_metric(metrics, f'eval/avg_action_rms_{label}'):.3f} "
                f"sat="
                f"{_metric(metrics, f'eval/avg_action_saturation_{label}'):.1%}"
                for label in WALKING_ACTION_GROUP_LABELS_3D
            )
        ),
        (
            "  feet/by_leg "
            + " | ".join(
                f"{label.upper()} c={stats['contact']:.2f} "
                f"air={stats['touchdown_air_time_s']:.3f}s "
                f"slip={stats['contact_slip_m_s']:.3f} "
                f"act={stats['action_rms']:.3f}"
                for label, stats in foot_stats.items()
            )
        ),
        (
            f"  symmetry contact_L-R={left_contact - right_contact:+.3f} "
            f"action_L-R="
            f"{_metric(metrics, 'eval/avg_action_rms_left_right_delta'):+.3f}"
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


def _action_group_diagnostics_walking_3d(action_rows):
    actions = np.asarray(action_rows, dtype=np.float64)
    if actions.size == 0:
        return {
            label: {
                "rms": 0.0,
                "p95_absolute": 0.0,
                "peak_absolute": 0.0,
                "saturation_fraction": 0.0,
                "saturation_threshold": (
                    WALKING_ACTION_SATURATION_THRESHOLD_3D
                ),
            }
            for label in WALKING_ACTION_GROUP_LABELS_3D
        }
    if actions.shape[-1] != 12:
        raise ValueError(
            f"walking action rows must end in 12 values, got {actions.shape}"
        )
    applied_actions = np.clip(actions.reshape((-1, 4, 3)), -1.0, 1.0)
    diagnostics = {}
    for index, label in enumerate(WALKING_ACTION_GROUP_LABELS_3D):
        values = applied_actions[:, :, index]
        absolute = np.abs(values)
        diagnostics[label] = {
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "p95_absolute": float(np.percentile(absolute, 95.0)),
            "peak_absolute": float(np.max(absolute)),
            "saturation_fraction": float(
                np.mean(
                    absolute >= WALKING_ACTION_SATURATION_THRESHOLD_3D
                )
            ),
            "saturation_threshold": (
                WALKING_ACTION_SATURATION_THRESHOLD_3D
            ),
        }
    return diagnostics


def _evaluate_policy_walking_3d(
    env,
    make_inference_fn,
    params,
    *,
    seed,
    episode_length,
    output_dir,
    command_forward_velocity_m_s=None,
    initial_gait_phase=0.0,
    symmetry_mirrored=False,
):
    import jax

    try:
        policy = make_inference_fn(params, deterministic=True)
    except TypeError:
        policy = make_inference_fn(params)
    policy_step = jax.jit(policy)
    env_reset = jax.jit(env.reset)
    env_reset_for_evaluation = jax.jit(env.reset_for_evaluation)
    env_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    if command_forward_velocity_m_s is None:
        state = env_reset(rng)
        commanded_speed = env.config.desired_speed_m_s
    else:
        commanded_speed = float(command_forward_velocity_m_s)
        state = env_reset_for_evaluation(
            rng,
            np.asarray((commanded_speed, 0.0, 0.0), dtype=np.float32),
            np.asarray(initial_gait_phase, dtype=np.float32),
            np.asarray(symmetry_mirrored, dtype=np.bool_),
        )
    initial_x = _float(state.pipeline_state.qpos[0])
    initial_y = _float(state.pipeline_state.qpos[1])
    qpos_rows = []
    action_rows = []
    reward_rows = []
    metric_totals = {}
    reward_totals = {
        name: 0.0 for name in WALKING_REWARD_TERM_NAMES_3D
    }
    rollout_metric_names = (
        "forward_velocity_m_s",
        "lateral_velocity_m_s",
        "yaw_rate_rad_s",
        "yaw_rate_error_rad_s",
        "heading_error_rad",
        "lateral_drift_m",
        "gait_phase",
    ) + tuple(
        f"{prefix}_{label}{suffix}"
        for prefix, suffix in (
            ("foot_contact", ""),
            ("foot_air_time", "_s"),
            ("foot_slip", "_m_s"),
            ("action_rms", ""),
        )
        for label in WALKING_FOOT_LABELS_3D
    )
    rollout_metric_rows = {name: [] for name in rollout_metric_names}
    minimum_root_z = math.inf
    maximum_tilt = 0.0
    maximum_abs_heading = 0.0

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
        maximum_abs_heading = max(
            maximum_abs_heading,
            abs(_float(state.metrics["heading_error_rad"])),
        )
        for name in rollout_metric_names:
            rollout_metric_rows[name].append(_float(state.metrics[name]))
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
    action_group_diagnostics = _action_group_diagnostics_walking_3d(
        action_rows
    )
    final_x = _float(state.pipeline_state.qpos[0])
    final_y = _float(state.pipeline_state.qpos[1])
    episode_only_metric_names = {
        "nonfoot_ground_peak_force_n",
        "heading_change_rad",
        "heading_abs_peak_increment_rad",
        "lateral_progress_m",
        "touchdown_count",
        "touchdown_air_time_sum_s",
        *(
            f"touchdown_count_{label}"
            for label in WALKING_FOOT_LABELS_3D
        ),
        *(
            f"touchdown_air_time_sum_{label}_s"
            for label in WALKING_FOOT_LABELS_3D
        ),
    }
    averages = {
        name: value / max(steps, 1)
        for name, value in metric_totals.items()
        if name not in episode_only_metric_names
    }
    episode_metrics = {
        name: metric_totals.get(name, 0.0)
        for name in episode_only_metric_names
    }
    failures = {
        name.removeprefix("failure_"): bool(metric_totals.get(name, 0.0))
        for name in WALKING_FAILURE_METRICS_3D
    }
    failed = any(failures.values())
    feet = {}
    for label in WALKING_FOOT_LABELS_3D:
        contact_steps = metric_totals.get(f"foot_contact_{label}", 0.0)
        touchdown_count = metric_totals.get(
            f"touchdown_count_{label}", 0.0
        )
        feet[label] = {
            "contact_fraction": contact_steps / max(steps, 1),
            "average_air_time_state_s": averages.get(
                f"foot_air_time_{label}_s", 0.0
            ),
            "touchdown_count": touchdown_count,
            "average_touchdown_air_time_s": (
                metric_totals.get(
                    f"touchdown_air_time_sum_{label}_s", 0.0
                )
                / max(touchdown_count, 1.0)
            ),
            "contact_conditioned_slip_m_s": (
                metric_totals.get(f"foot_slip_{label}_m_s", 0.0)
                / max(contact_steps, 1.0)
            ),
            "average_action_rms": averages.get(
                f"action_rms_{label}", 0.0
            ),
        }
    summary = {
        "coordinate_mode": "mirrored" if symmetry_mirrored else "normal",
        "command_forward_velocity_m_s": commanded_speed,
        "initial_gait_phase": float(initial_gait_phase) % 1.0,
        "episode_steps": steps,
        "episode_duration_s": steps * env.config.control_timestep,
        "total_reward": float(sum(reward_rows)),
        "root_x_displacement_m": final_x - initial_x,
        "final_lateral_drift_m": final_y - initial_y,
        "final_heading_error_rad": _float(
            state.metrics["heading_error_rad"]
        ),
        "unwrapped_heading_change_rad": metric_totals.get(
            "heading_change_rad", 0.0
        ),
        "maximum_abs_heading_error_rad": maximum_abs_heading,
        "average_signed_yaw_rate_rad_s": averages.get(
            "yaw_rate_rad_s", 0.0
        ),
        "average_abs_yaw_rate_error_rad_s": averages.get(
            "yaw_rate_error_rad_s", 0.0
        ),
        "average_forward_velocity_m_s": averages.get(
            "forward_velocity_m_s", 0.0
        ),
        "forward_velocity_error_m_s": (
            averages.get("forward_velocity_m_s", 0.0) - commanded_speed
        ),
        "minimum_root_z_m": minimum_root_z,
        "maximum_upright_tilt_rad": maximum_tilt,
        "terminated": bool(_float(state.done) > 0.5),
        "failed": failed,
        "timed_out": steps >= episode_length and not failed,
        "failure_reasons": failures,
        "feet": feet,
        "control": {
            "normalized_action_by_joint_type": action_group_diagnostics,
            "average_action_rms_by_leg": {
                label: averages.get(f"action_rms_{label}", 0.0)
                for label in WALKING_FOOT_LABELS_3D
            },
            "average_action_rms_left_right_delta": averages.get(
                "action_rms_left_right_delta", 0.0
            ),
        },
        "reward_breakdown": {
            "terms": reward_totals,
            "per_step": {
                name: value / max(steps, 1)
                for name, value in reward_totals.items()
            },
        },
        "metrics": {
            "totals": metric_totals,
            "per_step": averages,
            "episode": episode_metrics,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "evaluation_rollout.npz",
        qpos=np.asarray(qpos_rows),
        action=np.asarray(action_rows),
        reward=np.asarray(reward_rows),
        command_forward_velocity_m_s=np.asarray(commanded_speed),
        initial_gait_phase=np.asarray(float(initial_gait_phase) % 1.0),
        **{
            name: np.asarray(rows)
            for name, rows in rollout_metric_rows.items()
        },
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _evaluation_case_token(value: float) -> str:
    return f"{float(value):+.3f}".replace("+", "p").replace("-", "m").replace(
        ".", "p"
    )


def _signed_bias_summary_walking_3d(values, *, deadband):
    if not values:
        return {
            "direction": "none",
            "active_case_count": 0,
            "consistency": 0.0,
            "mean": 0.0,
            "maximum_absolute": 0.0,
        }
    active = [value for value in values if abs(value) >= deadband]
    positive = sum(value > 0.0 for value in active)
    negative = sum(value < 0.0 for value in active)
    dominant_count = max(positive, negative)
    direction = "none"
    if dominant_count:
        direction = "positive" if positive >= negative else "negative"
    return {
        "direction": direction,
        "active_case_count": len(active),
        "consistency": dominant_count / max(len(active), 1),
        "mean": float(np.mean(values)),
        "maximum_absolute": max(abs(value) for value in values),
    }


def _case_is_locomoting_walking_3d(case):
    command = float(case["command_forward_velocity_m_s"])
    achieved = float(case["average_forward_velocity_m_s"])
    if abs(command) < 1.0e-4:
        return False
    directional_speed = math.copysign(1.0, command) * achieved
    required_speed = max(0.05, 0.5 * abs(command))
    return directional_speed >= required_speed


def _action_group_grid_diagnostics_walking_3d(cases):
    result = {}
    for label in WALKING_ACTION_GROUP_LABELS_3D:
        values = []
        for case in cases:
            stats = (
                case.get("control", {})
                .get("normalized_action_by_joint_type", {})
                .get(label)
            )
            if stats is not None:
                values.append(stats)
        limited_cases = [
            stats
            for stats in values
            if (
                float(stats.get("p95_absolute", 0.0))
                >= WALKING_ACTION_P95_RANGE_LIMIT_THRESHOLD_3D
                or float(stats.get("saturation_fraction", 0.0))
                >= WALKING_ACTION_SATURATION_FRACTION_THRESHOLD_3D
            )
        ]
        result[label] = {
            "case_count": len(values),
            "maximum_rms": max(
                (float(stats.get("rms", 0.0)) for stats in values),
                default=0.0,
            ),
            "maximum_p95_absolute": max(
                (
                    float(stats.get("p95_absolute", 0.0))
                    for stats in values
                ),
                default=0.0,
            ),
            "maximum_peak_absolute": max(
                (
                    float(stats.get("peak_absolute", 0.0))
                    for stats in values
                ),
                default=0.0,
            ),
            "maximum_saturation_fraction": max(
                (
                    float(stats.get("saturation_fraction", 0.0))
                    for stats in values
                ),
                default=0.0,
            ),
            "range_limited_case_count": len(limited_cases),
        }
    return result


def _diagnose_evaluation_grid_walking_3d(cases, _phase_sensitivity):
    locomoting_cases = [
        case for case in cases if _case_is_locomoting_walking_3d(case)
    ]
    heading = _signed_bias_summary_walking_3d(
        [
            case["unwrapped_heading_change_rad"]
            for case in locomoting_cases
        ],
        deadband=0.15,
    )
    signed_yaw = _signed_bias_summary_walking_3d(
        [
            case["average_signed_yaw_rate_rad_s"]
            for case in locomoting_cases
        ],
        deadband=0.02,
    )
    maximum_phase_heading_span = 0.0
    phase_locomotion_mismatch = False
    speeds = sorted({case["command_forward_velocity_m_s"] for case in cases})
    for speed in speeds:
        matching = [
            case
            for case in cases
            if abs(case["command_forward_velocity_m_s"] - speed) <= 1.0e-9
        ]
        locomotion_flags = [
            _case_is_locomoting_walking_3d(case) for case in matching
        ]
        if any(locomotion_flags) and not all(locomotion_flags):
            phase_locomotion_mismatch = True
        moving_at_speed = [
            case
            for case, is_moving in zip(matching, locomotion_flags)
            if is_moving
        ]
        if len(moving_at_speed) >= 2:
            heading_values = [
                case["unwrapped_heading_change_rad"]
                for case in moving_at_speed
            ]
            maximum_phase_heading_span = max(
                maximum_phase_heading_span,
                max(heading_values) - min(heading_values),
            )
    contact_deltas = []
    action_deltas = []
    for case in locomoting_cases:
        feet = case.get("feet", {})
        if all(label in feet for label in WALKING_FOOT_LABELS_3D):
            left_contact = 0.5 * (
                feet["fl"]["contact_fraction"]
                + feet["rl"]["contact_fraction"]
            )
            right_contact = 0.5 * (
                feet["fr"]["contact_fraction"]
                + feet["rr"]["contact_fraction"]
            )
            contact_deltas.append(left_contact - right_contact)
        control = case.get("control", {})
        action_delta = control.get("average_action_rms_left_right_delta")
        if action_delta is not None:
            action_deltas.append(float(action_delta))

    maximum_contact_delta = max(
        (abs(value) for value in contact_deltas), default=0.0
    )
    maximum_action_delta = max(
        (abs(value) for value in action_deltas), default=0.0
    )
    action_range = _action_group_grid_diagnostics_walking_3d(
        locomoting_cases
    )
    flags = []
    if not locomoting_cases:
        flags.append("insufficient_locomoting_cases")
    elif heading["active_case_count"] and heading["consistency"] >= 0.75:
        flags.append("systematic_direction_bias")
    if phase_locomotion_mismatch or maximum_phase_heading_span >= 0.15:
        flags.append("initial_phase_sensitive")
    if maximum_contact_delta >= 0.10:
        flags.append("left_right_contact_imbalance")
    if maximum_action_delta >= 0.10:
        flags.append("left_right_action_imbalance")
    for label, stats in action_range.items():
        if stats["range_limited_case_count"]:
            flags.append(f"{label}_action_range_saturation")
    if not flags:
        flags.append("no_large_diagnostic_asymmetry")
    return {
        "observed_pattern_flags": flags,
        "locomoting_case_count": len(locomoting_cases),
        "nonlocomoting_case_count": len(cases) - len(locomoting_cases),
        "locomotion_gate": (
            "signed achieved speed >= max(0.05 m/s, 50% of command)"
        ),
        "heading_change_bias": heading,
        "signed_yaw_rate_bias": signed_yaw,
        "maximum_phase_heading_span_rad": maximum_phase_heading_span,
        "phase_locomotion_mismatch": phase_locomotion_mismatch,
        "maximum_abs_left_right_contact_delta": maximum_contact_delta,
        "maximum_abs_left_right_action_rms_delta": maximum_action_delta,
        "normalized_action_range_by_joint_type": action_range,
        "thresholds": {
            "heading_change_deadband_rad": 0.15,
            "yaw_rate_deadband_rad_s": 0.02,
            "systematic_sign_consistency": 0.75,
            "phase_heading_span_rad": 0.15,
            "left_right_contact_delta": 0.10,
            "left_right_action_rms_delta": 0.10,
            "action_p95_absolute": (
                WALKING_ACTION_P95_RANGE_LIMIT_THRESHOLD_3D
            ),
            "action_saturation_fraction": (
                WALKING_ACTION_SATURATION_FRACTION_THRESHOLD_3D
            ),
        },
    }


def _summarize_evaluation_grid_walking_3d(cases):
    if not cases:
        raise ValueError("evaluation grid must contain at least one case")
    speed_errors = [
        abs(case["forward_velocity_error_m_s"]) for case in cases
    ]
    heading_changes = [
        abs(case["unwrapped_heading_change_rad"]) for case in cases
    ]
    lateral_drifts = [abs(case["final_lateral_drift_m"]) for case in cases]
    yaw_errors = [case["average_abs_yaw_rate_error_rad_s"] for case in cases]
    phase_sensitivity = []
    speeds = sorted({case["command_forward_velocity_m_s"] for case in cases})
    for speed in speeds:
        matching = [
            case
            for case in cases
            if abs(case["command_forward_velocity_m_s"] - speed) <= 1.0e-9
        ]
        if len(matching) < 2:
            continue
        phase_sensitivity.append(
            {
                "command_forward_velocity_m_s": speed,
                "velocity_span_m_s": max(
                    case["average_forward_velocity_m_s"] for case in matching
                )
                - min(
                    case["average_forward_velocity_m_s"] for case in matching
                ),
                "heading_change_span_rad": max(
                    case["unwrapped_heading_change_rad"] for case in matching
                )
                - min(
                    case["unwrapped_heading_change_rad"] for case in matching
                ),
                "lateral_drift_span_m": max(
                    case["final_lateral_drift_m"] for case in matching
                )
                - min(case["final_lateral_drift_m"] for case in matching),
            }
        )
    diagnosis = _diagnose_evaluation_grid_walking_3d(
        cases, phase_sensitivity
    )
    return {
        "case_count": len(cases),
        "all_cases_full_length": all(case["timed_out"] for case in cases),
        "failed_case_count": sum(bool(case["failed"]) for case in cases),
        "mean_absolute_velocity_error_m_s": float(np.mean(speed_errors)),
        "maximum_absolute_velocity_error_m_s": max(speed_errors),
        "maximum_absolute_heading_change_rad": max(heading_changes),
        "maximum_absolute_lateral_drift_m": max(lateral_drifts),
        "maximum_average_abs_yaw_rate_error_rad_s": max(yaw_errors),
        "phase_sensitivity": phase_sensitivity,
        "diagnosis": diagnosis,
        "cases": cases,
    }


def _format_evaluation_grid_walking_3d(label, grid):
    diagnosis = grid["diagnosis"]
    heading_bias = diagnosis["heading_change_bias"]
    action_range = diagnosis["normalized_action_range_by_joint_type"]
    lines = [
        (
            f"[evaluation grid {label}] cases={grid['case_count']} "
            f"failed={grid['failed_case_count']} "
            f"full_length={grid['all_cases_full_length']} "
            f"mean_speed_error="
            f"{grid['mean_absolute_velocity_error_m_s']:.3f}m/s "
            f"worst_heading="
            f"{grid['maximum_absolute_heading_change_rad']:.3f}rad"
        ),
        (
            "  diagnosis flags="
            f"{','.join(diagnosis['observed_pattern_flags'])} "
            f"moving={diagnosis['locomoting_case_count']}/"
            f"{grid['case_count']} "
            f"heading_sign={heading_bias['direction']} "
            f"consistency={heading_bias['consistency']:.2f} "
            f"phase_span="
            f"{diagnosis['maximum_phase_heading_span_rad']:.3f}rad "
            f"contact_L-R_max="
            f"{diagnosis['maximum_abs_left_right_contact_delta']:.3f} "
            f"action_L-R_max="
            f"{diagnosis['maximum_abs_left_right_action_rms_delta']:.3f}"
        ),
        (
            "  action/range "
            + " | ".join(
                f"{joint} rms={stats['maximum_rms']:.3f} "
                f"p95={stats['maximum_p95_absolute']:.3f} "
                f"peak={stats['maximum_peak_absolute']:.3f} "
                f"sat={stats['maximum_saturation_fraction']:.1%}"
                for joint, stats in action_range.items()
            )
        ),
        (
            "  command phase achieved error yaw_signed heading lateral "
            "contacts air slip contact_L-R action_L-R"
        ),
    ]
    for case in grid["cases"]:
        feet = case["feet"]
        contacts = sum(
            foot["contact_fraction"] for foot in feet.values()
        )
        touchdown_counts = sum(
            foot["touchdown_count"] for foot in feet.values()
        )
        air_time = sum(
            foot["average_touchdown_air_time_s"]
            * foot["touchdown_count"]
            for foot in feet.values()
        ) / max(touchdown_counts, 1.0)
        contact_steps = sum(
            foot["contact_fraction"] * case["episode_steps"]
            for foot in feet.values()
        )
        slip = sum(
            foot["contact_conditioned_slip_m_s"]
            * foot["contact_fraction"]
            * case["episode_steps"]
            for foot in feet.values()
        ) / max(contact_steps, 1.0)
        left_contact = 0.5 * (
            feet["fl"]["contact_fraction"]
            + feet["rl"]["contact_fraction"]
        )
        right_contact = 0.5 * (
            feet["fr"]["contact_fraction"]
            + feet["rr"]["contact_fraction"]
        )
        action_delta = case["control"][
            "average_action_rms_left_right_delta"
        ]
        lines.append(
            f"  {case['command_forward_velocity_m_s']:+.3f}  "
            f"{case['initial_gait_phase']:.2f}  "
            f"{case['average_forward_velocity_m_s']:+.3f}  "
            f"{case['forward_velocity_error_m_s']:+.3f}  "
            f"{case['average_signed_yaw_rate_rad_s']:+.3f}  "
            f"{case['unwrapped_heading_change_rad']:+.3f}  "
            f"{case['final_lateral_drift_m']:+.3f}  "
            f"{contacts:.2f}  {air_time:.3f}  {slip:.3f}  "
            f"{left_contact - right_contact:+.3f}  {action_delta:+.3f}"
        )
    return "\n".join(lines)


def _evaluate_policy_grid_walking_3d(
    env,
    make_inference_fn,
    params,
    *,
    seed,
    episode_length,
    output_dir,
    forward_speeds,
    gait_phases,
    symmetry_mirrored=False,
):
    cases = []
    for speed in forward_speeds:
        for phase in gait_phases:
            case_dir = output_dir / (
                f"vx_{_evaluation_case_token(speed)}_"
                f"phase_{_evaluation_case_token(phase)}"
            )
            cases.append(
                _evaluate_policy_walking_3d(
                    env,
                    make_inference_fn,
                    params,
                    seed=seed,
                    episode_length=episode_length,
                    output_dir=case_dir,
                    command_forward_velocity_m_s=speed,
                    initial_gait_phase=phase,
                    symmetry_mirrored=symmetry_mirrored,
                )
            )
    grid = _summarize_evaluation_grid_walking_3d(cases)
    grid["coordinate_mode"] = (
        "mirrored" if symmetry_mirrored else "normal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_grid_summary.json").write_text(
        json.dumps(grid, indent=2) + "\n", encoding="utf-8"
    )
    return grid


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
    parser.add_argument(
        "--eval-forward-speeds",
        type=float,
        nargs="+",
        help=(
            "Fixed forward commands for the post-training evaluation grid. "
            "Defaults to the unique training minimum, desired speed, and "
            "training maximum."
        ),
    )
    parser.add_argument(
        "--eval-gait-phases",
        type=float,
        nargs="+",
        help=(
            "Normalized initial gait phases for the post-training grid. "
            "Defaults to 0 and 0.5 when phase observations are enabled."
        ),
    )
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
    gait_phase_group = parser.add_mutually_exclusive_group()
    gait_phase_group.add_argument(
        "--gait-phase",
        dest="gait_phase_enabled",
        action="store_true",
    )
    gait_phase_group.add_argument(
        "--no-gait-phase",
        dest="gait_phase_enabled",
        action="store_false",
    )
    parser.set_defaults(gait_phase_enabled=None)
    parser.add_argument("--gait-cycle-time", type=float)
    parser.add_argument("--gait-duty-factor", type=float)
    asymmetric_group = parser.add_mutually_exclusive_group()
    asymmetric_group.add_argument(
        "--asymmetric-observations",
        dest="asymmetric_observations",
        action="store_true",
    )
    asymmetric_group.add_argument(
        "--symmetric-observations",
        dest="asymmetric_observations",
        action="store_false",
    )
    parser.set_defaults(asymmetric_observations=None)
    symmetry_group = parser.add_mutually_exclusive_group()
    symmetry_group.add_argument(
        "--symmetry-augmentation",
        dest="symmetry_augmentation_enabled",
        action="store_true",
    )
    symmetry_group.add_argument(
        "--no-symmetry-augmentation",
        dest="symmetry_augmentation_enabled",
        action="store_false",
    )
    parser.set_defaults(symmetry_augmentation_enabled=False)
    parser.add_argument(
        "--symmetry-mirror-probability", type=float, default=0.5
    )
    normalization_group = parser.add_mutually_exclusive_group()
    normalization_group.add_argument(
        "--normalize-observations",
        dest="normalize_observations",
        action="store_true",
    )
    normalization_group.add_argument(
        "--no-normalize-observations",
        dest="normalize_observations",
        action="store_false",
    )
    parser.set_defaults(normalize_observations=None)
    normalizer_update_group = parser.add_mutually_exclusive_group()
    normalizer_update_group.add_argument(
        "--freeze-observation-normalizer",
        dest="freeze_observation_normalizer",
        action="store_true",
    )
    normalizer_update_group.add_argument(
        "--update-observation-normalizer",
        dest="freeze_observation_normalizer",
        action="store_false",
    )
    parser.set_defaults(freeze_observation_normalizer=False)
    actor_init_group = parser.add_mutually_exclusive_group()
    actor_init_group.add_argument(
        "--small-actor-mean-init",
        dest="small_actor_mean_init",
        action="store_true",
    )
    actor_init_group.add_argument(
        "--standard-actor-mean-init",
        dest="small_actor_mean_init",
        action="store_false",
    )
    parser.set_defaults(small_actor_mean_init=None)
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
    parser.add_argument("--terminate-upright-tilt", type=float)
    parser.add_argument(
        "--terminate-upright-tilt-duration", type=float
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
        "--terminate-nonfoot-force-min", type=float, default=1.0
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
    low_progress_group = parser.add_mutually_exclusive_group()
    low_progress_group.add_argument(
        "--terminate-low-progress",
        dest="terminate_low_progress_enabled",
        action="store_true",
    )
    low_progress_group.add_argument(
        "--no-terminate-low-progress",
        dest="terminate_low_progress_enabled",
        action="store_false",
    )
    parser.set_defaults(terminate_low_progress_enabled=None)
    eval_low_progress_group = parser.add_mutually_exclusive_group()
    eval_low_progress_group.add_argument(
        "--eval-terminate-low-progress",
        dest="eval_terminate_low_progress_enabled",
        action="store_true",
    )
    eval_low_progress_group.add_argument(
        "--no-eval-terminate-low-progress",
        dest="eval_terminate_low_progress_enabled",
        action="store_false",
    )
    parser.set_defaults(eval_terminate_low_progress_enabled=None)
    parser.add_argument(
        "--terminate-low-progress-window", type=float, default=0.50
    )
    parser.add_argument(
        "--terminate-low-progress-duration", type=float, default=2.0
    )
    parser.add_argument(
        "--terminate-low-progress-command-ratio", type=float, default=0.50
    )
    parser.add_argument(
        "--terminate-low-progress-cap", type=float, default=0.05
    )
    parser.add_argument("--unroll-length", type=int)
    parser.add_argument("--updates-per-batch", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--adaptive-kl-min-lr", type=float)
    parser.add_argument("--adaptive-kl-max-lr", type=float)
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
    parser.add_argument("--hidden-layers", type=int, nargs="+")
    parser.add_argument("--critic-hidden-layers", type=int, nargs="+")
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
    fallbacks = {
        "asymmetric_observations": False,
        "normalize_observations": False,
        "small_actor_mean_init": True,
        "hidden_layers": [256, 256, 128],
        "critic_hidden_layers": [256, 256, 128],
        "terminate_upright_tilt": 0.72,
        "terminate_upright_tilt_duration": 0.08,
        "terminate_low_progress_enabled": False,
        "eval_terminate_low_progress_enabled": False,
    }
    for name, value in fallbacks.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def _unique_finite_values(values, *, modulo=None) -> tuple[float, ...]:
    result = []
    for raw_value in values:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("evaluation grid values must be finite")
        if modulo is not None:
            value %= modulo
        if not any(abs(value - existing) <= 1.0e-9 for existing in result):
            result.append(value)
    return tuple(result)


def _resolve_evaluation_grid(args) -> tuple[tuple[float, ...], tuple[float, ...]]:
    speed_values = args.eval_forward_speeds
    if speed_values is None:
        speed_values = (
            args.command_forward_min,
            args.desired_speed_m_s,
            args.command_forward_max,
        )
    phase_values = args.eval_gait_phases
    if phase_values is None:
        phase_values = (0.0, 0.5) if args.gait_phase_enabled else (0.0,)
    speeds = _unique_finite_values(speed_values)
    phases = _unique_finite_values(phase_values, modulo=1.0)
    if not speeds or not phases:
        raise ValueError("evaluation grid must contain a speed and a phase")
    return speeds, phases


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_recipe_defaults(args)
    if args.ppo_checkpoint_dir is not None and not args.save_ppo_checkpoints:
        parser.error("--ppo-checkpoint-dir requires --save-ppo-checkpoints")
    if args.freeze_observation_normalizer and not args.normalize_observations:
        parser.error(
            "--freeze-observation-normalizer requires observation normalization"
        )
    if args.freeze_observation_normalizer and args.restore_checkpoint is None:
        parser.error(
            "--freeze-observation-normalizer requires --restore-checkpoint"
        )
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
        "adaptive_kl_min_lr",
        "adaptive_kl_max_lr",
        "init_noise_std",
        "clipping_epsilon",
        "max_grad_norm",
        "desired_kl",
        "reward_scaling",
        "gait_cycle_time",
        "terminate_nonfoot_force_min",
        "terminate_low_progress_window",
        "terminate_low_progress_duration",
        "terminate_low_progress_cap",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command_forward_max < args.command_forward_min:
        parser.error("command forward range must be ordered")
    try:
        args.eval_forward_speeds, args.eval_gait_phases = (
            _resolve_evaluation_grid(args)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.adaptive_kl_min_lr > args.adaptive_kl_max_lr:
        parser.error("adaptive KL learning-rate range must be ordered")
    if not (
        args.adaptive_kl_min_lr
        <= args.learning_rate
        <= args.adaptive_kl_max_lr
    ):
        parser.error("--learning-rate must lie inside the adaptive KL bounds")
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
    if not 0.0 < args.gait_duty_factor < 1.0:
        parser.error("--gait-duty-factor must be in (0, 1)")
    if not 0.0 < args.terminate_low_progress_command_ratio <= 1.0:
        parser.error(
            "--terminate-low-progress-command-ratio must be in (0, 1]"
        )
    if args.asymmetric_observations and not args.gait_phase_enabled:
        parser.error("--asymmetric-observations requires --gait-phase")
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
    from brax.training.acme import running_statistics
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
            asymmetric_observation_enabled=args.asymmetric_observations,
            symmetry_augmentation_enabled=(
                args.symmetry_augmentation_enabled
            ),
            symmetry_mirror_probability=args.symmetry_mirror_probability,
            gait_phase_enabled=args.gait_phase_enabled,
            gait_cycle_time_s=args.gait_cycle_time,
            gait_duty_factor=args.gait_duty_factor,
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
            terminate_nonfoot_force_min_n=args.terminate_nonfoot_force_min,
            terminate_nonfoot_contact_duration_s=(
                args.terminate_nonfoot_contact_duration
            ),
            terminate_self_contact_depth_m=(
                args.terminate_self_contact_depth
            ),
            terminate_self_contact_duration_s=(
                args.terminate_self_contact_duration
            ),
            terminate_low_progress_enabled=(
                args.terminate_low_progress_enabled
            ),
            terminate_low_progress_window_s=(
                args.terminate_low_progress_window
            ),
            terminate_low_progress_duration_s=(
                args.terminate_low_progress_duration
            ),
            terminate_low_progress_command_ratio=(
                args.terminate_low_progress_command_ratio
            ),
            terminate_low_progress_cap_m=args.terminate_low_progress_cap,
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
        symmetry_augmentation_enabled=False,
        reset_joint_noise_rad=0.0,
        reset_velocity_noise=0.0,
        reset_root_xy_velocity_noise_m_s=0.0,
        reset_root_yaw_rate_noise_rad_s=0.0,
        terminate_low_progress_enabled=(
            args.eval_terminate_low_progress_enabled
        ),
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
        selected = _checkpoint_is_selectable_walking_3d(
            selection,
            best["rank"],
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
                reward_scaling=args.reward_scaling,
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
        "adaptive_kl_min_lr": args.adaptive_kl_min_lr,
        "adaptive_kl_max_lr": args.adaptive_kl_max_lr,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "reward_scaling": args.reward_scaling,
        "policy_distribution": "normal",
        "policy_noise_std_type": "log",
        "policy_state_dependent_std": False,
        "policy_mean_kernel_init": (
            "small_uniform" if args.small_actor_mean_init else "lecun_uniform"
        ),
        "policy_mean_kernel_init_scale": (
            WALKING_ACTOR_MEAN_INIT_SCALE
            if args.small_actor_mean_init
            else None
        ),
        "policy_mean_clip_scale": WALKING_ACTOR_MEAN_CLIP_SCALE,
        "init_noise_std": args.init_noise_std,
        "observation_normalization": args.normalize_observations,
        "observation_normalizer_frozen": (
            args.freeze_observation_normalizer
        ),
        "asymmetric_observations": args.asymmetric_observations,
        "observation_scaling": "fixed_task_scales",
        "bootstrap_on_timeout": True,
        "deterministic_eval": args.deterministic_eval,
        "clipping_epsilon": args.clipping_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "desired_kl": args.desired_kl,
        "learning_rate_schedule": args.learning_rate_schedule,
        "hidden_layers": args.hidden_layers,
        "critic_hidden_layers": args.critic_hidden_layers,
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
        "evaluation_grid": {
            "forward_speeds_m_s": list(args.eval_forward_speeds),
            "initial_gait_phases": list(args.eval_gait_phases),
        },
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
        f"  symmetry_augmentation={task.symmetry_augmentation_enabled} "
        f"mirror_probability={task.symmetry_mirror_probability:g}\n"
        f"  lr={args.learning_rate:g} entropy={args.entropy_cost:g} "
        f"discount={args.discounting:g} "
        f"reward_scale={args.reward_scaling:g} seed={args.seed}\n"
        f"  ppo_clip={args.clipping_epsilon:g} "
        f"grad_norm={args.max_grad_norm:g} "
        f"desired_kl={args.desired_kl:g} "
        f"lr_schedule={args.learning_rate_schedule} "
        f"lr_bounds=[{args.adaptive_kl_min_lr:g}, "
        f"{args.adaptive_kl_max_lr:g}]\n"
        f"  policy=normal state_dependent_std=false "
        f"mean_init_scale={WALKING_ACTOR_MEAN_INIT_SCALE:g} "
        f"mean_clip={WALKING_ACTOR_MEAN_CLIP_SCALE:g} "
        f"init_std={args.init_noise_std:g} "
        f"deterministic_eval={args.deterministic_eval}\n"
        f"  observation=fixed_task_scaling "
        f"normalize={args.normalize_observations} "
        f"normalizer_frozen={args.freeze_observation_normalizer} "
        f"asymmetric={args.asymmetric_observations} "
        f"bootstrap_timeout=true\n"
        f"  final_eval_grid speeds="
        f"{','.join(f'{value:g}' for value in args.eval_forward_speeds)} "
        f"phases="
        f"{','.join(f'{value:g}' for value in args.eval_gait_phases)}\n"
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
        "learning_rate_schedule_min_lr",
        "learning_rate_schedule_max_lr",
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
    with _running_statistics_update_scope(
        running_statistics,
        freeze=args.freeze_observation_normalizer,
    ):
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
            normalize_observations=args.normalize_observations,
            bootstrap_on_timeout=True,
            clipping_epsilon=args.clipping_epsilon,
            max_grad_norm=args.max_grad_norm,
            desired_kl=args.desired_kl,
            learning_rate_schedule=args.learning_rate_schedule,
            learning_rate_schedule_min_lr=args.adaptive_kl_min_lr,
            learning_rate_schedule_max_lr=args.adaptive_kl_max_lr,
            deterministic_eval=args.deterministic_eval,
            network_factory=_walking_network_factory(
                args.hidden_layers,
                args.critic_hidden_layers,
                args.activation,
                args.init_noise_std,
                asymmetric_observations=args.asymmetric_observations,
                small_actor_mean_init=args.small_actor_mean_init,
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
    best_status = (
        f"best_step={best['step']} score={best['score']:.4f}"
        if best["step"] is not None
        else "best_step=none score=none (params_best falls back to final)"
    )
    print(
        "[training complete]\n"
        f"  elapsed={elapsed / 60.0:.1f}min "
        f"throughput={throughput:,.0f} steps/s\n"
        f"  {best_status}\n"
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
        best_grid = _evaluate_policy_grid_walking_3d(
            eval_env,
            make_inference_fn,
            best_params,
            seed=args.seed + 30_000,
            episode_length=args.episode_length,
            output_dir=args.out / "evaluation_grid_best",
            forward_speeds=args.eval_forward_speeds,
            gait_phases=args.eval_gait_phases,
        )
        comparison = {
            "selection": {
                "best_step": best["step"],
                "best_selection_score": summary["best_selection_score"],
            },
            "best": best_eval,
            "final": final_eval,
            "best_evaluation_grid": best_grid,
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
        print(
            _format_evaluation_grid_walking_3d("best", best_grid),
            flush=True,
        )


if __name__ == "__main__":
    main()
