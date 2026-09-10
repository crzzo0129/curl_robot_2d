#!/usr/bin/env python3
"""Train a straight-tracking, mirror-consistent deploy walking policy.

The actor contract remains identical to neural_controller:
36 values per frame, 20 newest-first frames, and 12 position-residual actions.
This entrypoint adds explicit low-speed command curricula, deterministic
fixed-command checkpoint selection, and an Actor mirror-consistency loss.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from curl_robot_2d_mjx.deployment_walking_v2 import (
    DeployEvaluation,
    bounded_normalized_square,
    deploy_checkpoint_rank,
    deploy_checkpoint_rank_v3,
    deploy_checkpoint_rank_v4,
    mirror_deploy_action,
    mirror_deploy_observation,
    quaternion_yaw,
    signed_cross_track_error,
    smooth_normalized_square,
    straight_speed_terms,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "smoke": {
        "steps": 131_072,
        "envs": 8,
        "eval_envs": 1,
        "num_evals": 4,
        "batch_size": 16,
        "num_minibatches": 1,
    },
    "quick": {
        "steps": 2_000_000,
        "envs": 64,
        "eval_envs": 8,
        "num_evals": 6,
        "batch_size": 64,
        "num_minibatches": 4,
    },
    "5090": {
        "steps": 20_000_000,
        "envs": 64,
        "eval_envs": 8,
        "num_evals": 10,
        "batch_size": 64,
        "num_minibatches": 4,
    },
}

STRAIGHT_SPEEDS = (0.10, 0.15, 0.20)
DEFAULT_TRAINING_XML = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad.xml"
)


def mirror_action_consistency_mse(
    array_module,
    canonical_action,
    mirrored_policy_action,
    *,
    action_scale=None,
):
    """Return mirror error in normalized or physical residual coordinates."""

    if action_scale is None:
        canonical_comparison = canonical_action
        mirrored_comparison = mirror_deploy_action(
            array_module,
            mirrored_policy_action,
        )
    else:
        scale = array_module.asarray(action_scale)
        canonical_comparison = canonical_action * scale
        mirrored_comparison = mirror_deploy_action(
            array_module,
            mirrored_policy_action * scale,
        )
    return array_module.mean(
        array_module.square(
            mirrored_comparison - canonical_comparison
        )
    )


@contextmanager
def actor_mirror_consistency_scope(
    loss_module,
    *,
    weight,
    array_module,
    stop_gradient,
    action_scale=None,
):
    """Add mirror equivariance to the PPO Actor loss."""

    if weight <= 0.0:
        yield
        return

    original = loss_module.compute_ppo_loss

    def compute_with_mirror(
        params,
        normalizer_params,
        data,
        rng,
        ppo_network,
        **kwargs,
    ):
        base_loss, metrics = original(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network,
            **kwargs,
        )
        observation = data.observation
        mirrored_observation = mirror_deploy_observation(
            array_module, observation
        )
        policy_apply = ppo_network.policy_network.apply
        distribution = ppo_network.parametric_action_distribution
        canonical_logits = policy_apply(
            normalizer_params, params.policy, observation
        )
        mirrored_logits = policy_apply(
            normalizer_params, params.policy, mirrored_observation
        )
        canonical_action = stop_gradient(
            distribution.mode(canonical_logits)
        )
        mirrored_policy_action = distribution.mode(mirrored_logits)
        consistency = mirror_action_consistency_mse(
            array_module,
            canonical_action,
            mirrored_policy_action,
            action_scale=action_scale,
        )
        weighted = weight * consistency
        metrics = dict(metrics)
        metrics["ppo_total_loss"] = metrics.get("total_loss", base_loss)
        metrics["total_loss"] = base_loss + weighted
        metrics["actor_mirror_consistency_loss"] = consistency
        metrics["actor_mirror_consistency_rms"] = array_module.sqrt(
            consistency
        )
        metrics["actor_mirror_consistency_weighted_loss"] = weighted
        return base_loss + weighted, metrics

    loss_module.compute_ppo_loss = compute_with_mirror
    try:
        yield
    finally:
        loss_module.compute_ppo_loss = original


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("train", "probe"),
        default="train",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument(
        "--stage",
        choices=("straight", "mixed", "dr"),
        default="straight",
        help=(
            "command curriculum; legacy 'dr' means mixed commands plus DR, "
            "new runs should use --domain-randomization independently"
        ),
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help=(
            "randomize deploy physics, motor calibration, sensing, and "
            "control timing without changing the command curriculum"
        ),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--seed", type=int, default=260909)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--training-xml",
        type=Path,
        default=DEFAULT_TRAINING_XML,
        help="MJCF used for PPO; exported policy ABI is unchanged",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument(
        "--self-collision",
        action="store_true",
        help=(
            "enable expensive CAD self-collision during training; by default "
            "mesh-ground contact remains exact and self-collision is reserved "
            "for post-training validation"
        ),
    )
    parser.add_argument("--mirror-weight", type=float, default=0.02)
    parser.add_argument(
        "--scale-aware-mirror",
        action="store_true",
        help=(
            "compare mirrored physical action residuals after action scaling; "
            "needed when left/right joint scales differ"
        ),
    )
    parser.add_argument("--observation-noise", type=float, default=0.25)
    parser.add_argument(
        "--train-speeds",
        type=float,
        nargs="+",
        default=None,
        help="positive straight speeds sampled during training",
    )
    parser.add_argument(
        "--train-speed-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="sample straight speed continuously and uniformly in [MIN, MAX]",
    )
    parser.add_argument(
        "--command-deadzone",
        type=float,
        default=0.0,
        help="reserve (0, deadzone) for zero command instead of training it",
    )
    parser.add_argument(
        "--stand-probability",
        type=float,
        default=0.10,
        help="probability that a sampled command is exactly zero",
    )
    parser.add_argument("--straight-yaw-weight", type=float, default=1.5)
    parser.add_argument("--straight-yaw-scale-rad-s", type=float, default=0.08)
    parser.add_argument("--straight-lateral-weight", type=float, default=2.0)
    parser.add_argument(
        "--straight-lateral-scale-m-s",
        type=float,
        default=0.05,
    )
    parser.add_argument("--heading-drift-weight", type=float, default=0.5)
    parser.add_argument(
        "--heading-drift-scale-rad",
        type=float,
        default=0.12,
    )
    parser.add_argument("--lateral-drift-weight", type=float, default=0.0)
    parser.add_argument(
        "--lateral-drift-scale-m",
        type=float,
        default=0.05,
    )
    parser.add_argument("--speed-error-weight", type=float, default=0.75)
    parser.add_argument("--forward-progress-weight", type=float, default=0.50)
    parser.add_argument("--fore-aft-pose-weight", type=float, default=0.0)
    parser.add_argument(
        "--fore-aft-pose-scale-rad",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--fore-aft-leg-length-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--fore-aft-leg-length-scale-m",
        type=float,
        default=0.012,
    )
    parser.add_argument("--action-rate-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-rate-scale-rad",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--stand-action-rate-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stand-action-rate-scale-rad",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--stand-joint-velocity-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stand-joint-velocity-scale-rad-s",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--stand-body-angular-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stand-body-angular-scale-rad-s",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--stand-body-linear-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stand-body-linear-scale-m-s",
        type=float,
        default=0.03,
    )
    parser.add_argument("--stand-action-weight", type=float, default=0.0)
    parser.add_argument(
        "--stand-action-scale-rad",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--stand-height-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stand-height-scale-m",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--normalized-stability-rewards",
        action="store_true",
        help="normalize stability errors by physical tolerances and clip them",
    )
    parser.add_argument(
        "--smooth-stability-rewards",
        action="store_true",
        help=(
            "use pseudo-Huber normalized errors so large posture and drift "
            "errors retain a corrective gradient"
        ),
    )
    parser.add_argument("--reward-clip-min", type=float, default=-5.0)
    parser.add_argument("--reward-clip-max", type=float, default=10.0)
    parser.add_argument(
        "--front-right-hip-action-scale",
        type=float,
        default=0.50,
        help="front-right hip residual-action scale in radians",
    )
    parser.add_argument(
        "--front-hip-response-weight",
        type=float,
        default=0.0,
        help=(
            "penalty on the difference between running RMS front-left and "
            "front-right physical hip displacement"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--clipping-epsilon", type=float, default=0.20)
    parser.add_argument("--value-clipping-epsilon", type=float, default=0.20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--normalization",
        choices=("running", "identity"),
        default="running",
        help=(
            "use identity when resuming an actor imported from RTNeural JSON, "
            "whose normalization is already folded into layer 0"
        ),
    )
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--selection-speed", type=float, default=0.15)
    parser.add_argument(
        "--selection-speeds",
        type=float,
        nargs="+",
        help="fixed straight speeds used by V4 checkpoint selection",
    )
    parser.add_argument(
        "--selection-duration-s",
        type=float,
        default=5.0,
        help="duration of each deterministic checkpoint-selection rollout",
    )
    parser.add_argument(
        "--selection-warmup-s",
        type=float,
        default=1.0,
        help="initial selection-rollout interval excluded from RMS metrics",
    )
    parser.add_argument(
        "--selection-deploy-perturbations",
        action="store_true",
        help=(
            "select checkpoints with deploy observation noise, calibration "
            "bias, latency, and missed control deadlines enabled"
        ),
    )
    parser.add_argument(
        "--eval-speeds",
        type=float,
        nargs="+",
        default=STRAIGHT_SPEEDS,
    )
    parser.add_argument("--memory-fraction", type=float, default=0.82)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--v3-selection-metrics",
        action="store_true",
        help=(
            "rank checkpoints using front/rear posture and stand smoothness "
            "in addition to the V2 walking gates"
        ),
    )
    parser.add_argument(
        "--v4-selection-metrics",
        action="store_true",
        help=(
            "rank checkpoints across --selection-speeds using speed, "
            "straightness, geometric leg length, posture, and stand stability"
        ),
    )
    args = parser.parse_args(argv)
    legacy_dr_stage = args.stage == "dr"
    args.command_stage = "mixed" if legacy_dr_stage else args.stage
    args.domain_randomization = (
        args.domain_randomization or legacy_dr_stage
    )
    if args.train_speed_range is not None and args.train_speeds is not None:
        parser.error("--train-speeds and --train-speed-range are exclusive")
    if args.train_speed_range is None and args.train_speeds is None:
        args.train_speeds = list(STRAIGHT_SPEEDS)
    if args.selection_speeds is None:
        args.selection_speeds = [args.selection_speed]

    values = PRESETS[args.preset].copy()
    for name in (
        "steps",
        "envs",
        "eval_envs",
        "num_evals",
        "batch_size",
        "num_minibatches",
    ):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    args.values = values
    if args.out is None:
        args.out = (
            PROJECT_ROOT
            / "results"
            / f"deploy_walk_v2_{args.stage}_{args.preset}_seed{args.seed}"
        )
    elif not args.out.is_absolute():
        args.out = PROJECT_ROOT / args.out
    if args.resume is not None and not args.resume.is_absolute():
        args.resume = PROJECT_ROOT / args.resume
    if not args.training_xml.is_absolute():
        args.training_xml = PROJECT_ROOT / args.training_xml
    if not args.training_xml.is_file():
        parser.error(f"--training-xml does not exist: {args.training_xml}")
    for name in (
        "mirror_weight",
        "straight_yaw_weight",
        "straight_lateral_weight",
        "heading_drift_weight",
        "lateral_drift_weight",
        "speed_error_weight",
        "forward_progress_weight",
        "front_hip_response_weight",
        "fore_aft_pose_weight",
        "fore_aft_leg_length_weight",
        "action_rate_weight",
        "stand_action_rate_weight",
        "stand_joint_velocity_weight",
        "stand_body_angular_weight",
        "stand_body_linear_weight",
        "stand_action_weight",
        "stand_height_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0.0 <= args.stand_probability <= 1.0:
        parser.error("--stand-probability must be in [0, 1]")
    if args.command_deadzone < 0.0:
        parser.error("--command-deadzone must be nonnegative")
    if args.train_speed_range is not None:
        speed_min, speed_max = args.train_speed_range
        if (
            not math.isfinite(speed_min)
            or not math.isfinite(speed_max)
            or speed_min <= 0.0
            or speed_max < speed_min
        ):
            parser.error(
                "--train-speed-range requires finite 0 < MIN <= MAX"
            )
        if speed_min < args.command_deadzone:
            parser.error(
                "--train-speed-range MIN must be >= --command-deadzone"
            )
    elif (
        not args.train_speeds
        or any(
            not math.isfinite(speed)
            or speed <= 0.0
            or speed < args.command_deadzone
            for speed in args.train_speeds
        )
    ):
        parser.error(
            "--train-speeds must contain finite positive values at or above "
            "--command-deadzone"
        )
    if (
        not args.selection_speeds
        or any(
            not math.isfinite(speed) or speed <= 0.0
            for speed in args.selection_speeds
        )
    ):
        parser.error("--selection-speeds must contain finite positive values")
    if args.selection_duration_s <= 0.0:
        parser.error("--selection-duration-s must be positive")
    if not 0.0 <= args.selection_warmup_s < args.selection_duration_s:
        parser.error(
            "--selection-warmup-s must be in [0, selection-duration-s)"
        )
    for name in (
        "straight_yaw_scale_rad_s",
        "straight_lateral_scale_m_s",
        "heading_drift_scale_rad",
        "lateral_drift_scale_m",
        "fore_aft_pose_scale_rad",
        "fore_aft_leg_length_scale_m",
        "action_rate_scale_rad",
        "stand_action_rate_scale_rad",
        "stand_joint_velocity_scale_rad_s",
        "stand_body_angular_scale_rad_s",
        "stand_body_linear_scale_m_s",
        "stand_action_scale_rad",
        "stand_height_scale_m",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.front_right_hip_action_scale <= 0.0:
        parser.error("--front-right-hip-action-scale must be positive")
    if args.observation_noise < 0.0:
        parser.error("--observation-noise must be nonnegative")
    if not 0.0 < args.clipping_epsilon <= 1.0:
        parser.error("--clipping-epsilon must be in (0, 1]")
    if not 0.0 < args.value_clipping_epsilon <= 1.0:
        parser.error("--value-clipping-epsilon must be in (0, 1]")
    if args.max_grad_norm <= 0.0:
        parser.error("--max-grad-norm must be positive")
    if args.reward_clip_min >= args.reward_clip_max:
        parser.error("--reward-clip-min must be below --reward-clip-max")
    if not 0.0 < args.memory_fraction <= 1.0:
        parser.error("--memory-fraction must be in (0, 1]")
    if any(value <= 0 for value in values.values()):
        parser.error("all preset and override values must be positive")
    return args


def _yaw(q):
    w, x, y, z = (float(value) for value in q[3:7])
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _json_dump(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(args):
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=False,
        mujoco_gl="disable",
        verbose=True,
    )

    import jax
    import jax.numpy as jp
    from brax.io import model
    from brax.training.agents.ppo import losses as ppo_losses
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks

    from scripts import train_ppo_deploy as base

    train_speeds = (
        None
        if args.train_speeds is None
        else jp.asarray(args.train_speeds, dtype=jp.float32)
    )
    train_speed_range = (
        None
        if args.train_speed_range is None
        else jp.asarray(args.train_speed_range, dtype=jp.float32)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    runtime_xml = args.out / "_runtime" / "rollingquad_2_deploy_v2.xml"
    base.w3.SRC_XML = str(args.training_xml.resolve())
    base.w3.RUN_XML = str(runtime_xml)
    base.w3.OBS_NOISE = 0.0
    base.w3.SELF_COLLISION = args.self_collision
    base.ACTION_SCALE = base.ACTION_SCALE.at[4].set(
        args.front_right_hip_action_scale
    )
    use_domain_randomization = args.domain_randomization
    if use_domain_randomization:
        base.enable_deploy_dr()
    else:
        base.DEPLOY_DR = False
        base.w3.DOMAIN_RANDOMIZE = False

    class DeployEnvV2(base.DeployEnv):
        def __init__(
            self,
            *,
            stage,
            fixed_command=None,
            training,
            deploy_dr,
            observation_noise,
        ):
            self._stage = stage
            self._fixed_command = (
                None
                if fixed_command is None
                else jp.asarray(fixed_command, dtype=jp.float32)
            )
            self._training = bool(training)
            super().__init__(
                deploy_dr=deploy_dr,
                observation_noise=observation_noise,
            )
            hip_body = np.asarray(
                [
                    base.mujoco.mj_name2id(
                        self._mj,
                        base.mujoco.mjtObj.mjOBJ_BODY,
                        f"{leg}_hip_abduction_body",
                    )
                    for leg in base.LEGS
                ],
                dtype=np.int32,
            )
            if (hip_body < 0).any():
                raise RuntimeError("hip abduction bodies not found")
            self._hip_body = jp.asarray(hip_body)

        def _sample_command(self, rng):
            if self._fixed_command is not None:
                return self._fixed_command
            keys = jax.random.split(rng, 8)
            if train_speed_range is None:
                speed = train_speeds[
                    jax.random.randint(
                        keys[0], (), 0, len(args.train_speeds)
                    )
                ]
            else:
                speed = jax.random.uniform(
                    keys[0],
                    (),
                    minval=train_speed_range[0],
                    maxval=train_speed_range[1],
                )
            straight = jp.array([speed, 0.0, 0.0])
            if self._stage == "straight":
                stand = (
                    jax.random.uniform(keys[1]) < args.stand_probability
                )
                return jp.where(stand, jp.zeros(3), straight)

            selector = jax.random.uniform(keys[1])
            turn_sign = jp.where(
                jax.random.bernoulli(keys[2]), 1.0, -1.0
            )
            turn = jp.array(
                [
                    jax.random.uniform(
                        keys[3], (), minval=0.08, maxval=0.25
                    ),
                    0.0,
                    turn_sign
                    * jax.random.uniform(
                        keys[4], (), minval=0.10, maxval=0.40
                    ),
                ]
            )
            reverse = jp.array(
                [
                    -jax.random.uniform(
                        keys[5], (), minval=0.05, maxval=0.20
                    ),
                    0.0,
                    0.0,
                ]
            )
            lateral_sign = jp.where(
                jax.random.bernoulli(keys[6]), 1.0, -1.0
            )
            lateral = jp.array(
                [
                    jax.random.uniform(
                        keys[7], (), minval=0.05, maxval=0.20
                    ),
                    lateral_sign * 0.08,
                    0.0,
                ]
            )
            command = jp.where(selector < 0.60, straight, turn)
            command = jp.where(selector < 0.75, command, reverse)
            command = jp.where(selector < 0.85, command, lateral)
            return jp.where(selector < 0.95, command, jp.zeros(3))

        def reset(self, rng):
            state = super().reset(rng)
            info = dict(state.info)
            info["heading_drift"] = jp.zeros(())
            info["heading_origin_yaw"] = quaternion_yaw(
                jp,
                state.pipeline_state.q[3:7],
            )
            info["lateral_drift"] = jp.zeros(())
            info["path_origin_xy"] = state.pipeline_state.q[:2]
            info["front_hip_displacement_energy"] = jp.zeros(2)
            metrics = dict(state.metrics)
            metrics.update(
                {
                    "straight_yaw_penalty": jp.zeros(()),
                    "straight_lateral_penalty": jp.zeros(()),
                    "heading_drift_penalty": jp.zeros(()),
                    "heading_drift": jp.zeros(()),
                    "lateral_drift_penalty": jp.zeros(()),
                    "lateral_drift": jp.zeros(()),
                    "speed_error": jp.zeros(()),
                    "speed_error_penalty": jp.zeros(()),
                    "forward_progress_reward": jp.zeros(()),
                    "front_hip_response_error": jp.zeros(()),
                    "front_hip_response_penalty": jp.zeros(()),
                    "front_rear_pose_rms": jp.zeros(()),
                    "fore_aft_pose_penalty": jp.zeros(()),
                    "front_rear_leg_length_error": jp.zeros(()),
                    "fore_aft_leg_length_penalty": jp.zeros(()),
                    "physical_action_delta_rms": jp.zeros(()),
                    "action_rate_penalty": jp.zeros(()),
                    "stand_action_delta_rms": jp.zeros(()),
                    "stand_action_rate_penalty": jp.zeros(()),
                    "stand_joint_velocity_rms": jp.zeros(()),
                    "stand_joint_velocity_penalty": jp.zeros(()),
                    "stand_body_angular_velocity_rms": jp.zeros(()),
                    "stand_body_angular_penalty": jp.zeros(()),
                    "stand_body_linear_velocity_rms": jp.zeros(()),
                    "stand_body_linear_penalty": jp.zeros(()),
                    "stand_action_rms": jp.zeros(()),
                    "stand_action_penalty": jp.zeros(()),
                    "stand_height_error": jp.zeros(()),
                    "stand_height_penalty": jp.zeros(()),
                }
            )
            return state.replace(info=info, metrics=metrics)

        def step(self, state, action):
            command = state.info["command"]
            next_state = super().step(state, action)
            moving = jp.linalg.norm(command) > 0.05
            stand = (~moving).astype(jp.float32)
            straight = (
                (jp.abs(command[1]) < 0.005)
                & (jp.abs(command[2]) < 0.005)
                & moving
            ).astype(jp.float32)
            fore_aft_gate = (
                (jp.abs(command[1]) < 0.005)
                & (jp.abs(command[2]) < 0.005)
            ).astype(jp.float32)
            yaw_rate = next_state.metrics["wz"]
            lateral_speed = next_state.metrics["vy"]

            def stability_error(value, scale):
                if args.smooth_stability_rewards:
                    return smooth_normalized_square(
                        jp,
                        value,
                        scale,
                    )
                if args.normalized_stability_rewards:
                    return bounded_normalized_square(
                        jp,
                        value,
                        scale,
                    )
                return jp.square(value)

            current_yaw = quaternion_yaw(
                jp,
                next_state.pipeline_state.q[3:7],
            )
            heading_origin_yaw = jp.where(
                straight > 0.0,
                state.info["heading_origin_yaw"],
                current_yaw,
            )
            heading_delta = current_yaw - heading_origin_yaw
            heading_drift = jp.where(
                straight > 0.0,
                jp.arctan2(jp.sin(heading_delta), jp.cos(heading_delta)),
                0.0,
            )
            heading_drift *= 1.0 - next_state.done
            path_origin_xy = jp.where(
                straight > 0.0,
                state.info["path_origin_xy"],
                next_state.pipeline_state.q[:2],
            )
            lateral_drift = jp.where(
                straight > 0.0,
                signed_cross_track_error(
                    jp,
                    next_state.pipeline_state.q[:2],
                    path_origin_xy,
                    heading_origin_yaw,
                ),
                0.0,
            )
            lateral_drift *= 1.0 - next_state.done
            p_yaw = (
                args.straight_yaw_weight
                * straight
                * stability_error(
                    yaw_rate,
                    args.straight_yaw_scale_rad_s,
                )
            )
            p_lateral = (
                args.straight_lateral_weight
                * straight
                * stability_error(
                    lateral_speed,
                    args.straight_lateral_scale_m_s,
                )
            )
            p_heading = (
                args.heading_drift_weight
                * straight
                * stability_error(
                    heading_drift,
                    args.heading_drift_scale_rad,
                )
            )
            p_lateral_drift = (
                args.lateral_drift_weight
                * straight
                * (
                    stability_error(
                        lateral_drift,
                        args.lateral_drift_scale_m,
                    )
                    if args.normalized_stability_rewards
                    else jp.square(
                        lateral_drift / args.lateral_drift_scale_m
                    )
                )
            )
            p_speed, r_progress = straight_speed_terms(
                jp,
                command[0],
                next_state.metrics["vx"],
                error_weight=args.speed_error_weight,
                progress_weight=args.forward_progress_weight,
            )
            p_speed *= straight
            r_progress *= straight
            joint_offset = (
                next_state.pipeline_state.q[self._joint_qpos]
                - base.DEFAULT_POSE
            ).reshape((4, 3))
            front_pose = jp.mean(joint_offset[:2, 1:3], axis=0)
            rear_pose = jp.mean(joint_offset[2:, 1:3], axis=0)
            front_rear_pose_rms = jp.sqrt(
                jp.mean(jp.square(front_pose - rear_pose)) + 1.0e-8
            )
            p_fore_aft_pose = (
                args.fore_aft_pose_weight
                * fore_aft_gate
                * stability_error(
                    front_rear_pose_rms,
                    args.fore_aft_pose_scale_rad,
                )
            )
            foot_position, _ = self._feet(next_state.pipeline_state)
            hip_position = next_state.pipeline_state.xpos[self._hip_body]
            leg_length = jp.linalg.norm(
                foot_position - hip_position,
                axis=1,
            )
            front_rear_leg_length_error = jp.sqrt(
                jp.mean(jp.square(leg_length[:2] - leg_length[2:]))
                + 1.0e-8
            )
            p_fore_aft_leg_length = (
                args.fore_aft_leg_length_weight
                * fore_aft_gate
                * stability_error(
                    front_rear_leg_length_error,
                    args.fore_aft_leg_length_scale_m,
                )
            )
            front_hip_displacement = joint_offset[:2, 1]
            response_alpha = 0.04
            front_hip_energy = (
                (1.0 - response_alpha)
                * state.info["front_hip_displacement_energy"]
                + response_alpha * jp.square(front_hip_displacement)
            )
            front_hip_rms = jp.sqrt(front_hip_energy + 1.0e-8)
            front_hip_response_error = jp.abs(
                front_hip_rms[0] - front_hip_rms[1]
            )
            p_front_hip_response = (
                args.front_hip_response_weight
                * straight
                * jp.square(front_hip_response_error)
            )
            clipped_action = jp.clip(action, -1.0, 1.0)
            normalized_action_delta = (
                clipped_action - state.info["last_act"]
            )
            physical_action_delta_rms = jp.sqrt(
                jp.mean(
                    jp.square(
                        normalized_action_delta * base.ACTION_SCALE
                    )
                )
                + 1.0e-8
            )
            action_delta_rms = jp.where(
                args.normalized_stability_rewards,
                physical_action_delta_rms,
                jp.sqrt(
                    jp.mean(jp.square(normalized_action_delta)) + 1.0e-8
                ),
            )
            joint_velocity_rms = jp.sqrt(
                jp.mean(
                    jp.square(
                        next_state.pipeline_state.qd[self._joint_dof]
                    )
                )
                + 1.0e-8
            )
            body_angular_velocity_rms = jp.sqrt(
                jp.mean(
                    jp.square(next_state.pipeline_state.xd.ang[0])
                )
                + 1.0e-8
            )
            body_linear_velocity_rms = jp.sqrt(
                jp.mean(
                    jp.square(next_state.pipeline_state.xd.vel[0])
                )
                + 1.0e-8
            )
            normalized_action_rms = jp.sqrt(
                jp.mean(jp.square(clipped_action)) + 1.0e-8
            )
            physical_action_rms = jp.sqrt(
                jp.mean(
                    jp.square(clipped_action * base.ACTION_SCALE)
                )
                + 1.0e-8
            )
            action_rms = jp.where(
                args.normalized_stability_rewards,
                physical_action_rms,
                normalized_action_rms,
            )
            stand_height_error = jp.abs(
                next_state.pipeline_state.q[2] - self._nom_h
            )
            p_action_rate = (
                args.action_rate_weight
                * stability_error(
                    physical_action_delta_rms,
                    args.action_rate_scale_rad,
                )
            )
            p_stand_action_rate = (
                args.stand_action_rate_weight
                * stand
                * stability_error(
                    action_delta_rms,
                    args.stand_action_rate_scale_rad,
                )
            )
            p_stand_joint_velocity = (
                args.stand_joint_velocity_weight
                * stand
                * stability_error(
                    joint_velocity_rms,
                    args.stand_joint_velocity_scale_rad_s,
                )
            )
            p_stand_body_angular = (
                args.stand_body_angular_weight
                * stand
                * stability_error(
                    body_angular_velocity_rms,
                    args.stand_body_angular_scale_rad_s,
                )
            )
            p_stand_body_linear = (
                args.stand_body_linear_weight
                * stand
                * stability_error(
                    body_linear_velocity_rms,
                    args.stand_body_linear_scale_m_s,
                )
            )
            p_stand_action = (
                args.stand_action_weight
                * stand
                * stability_error(
                    action_rms,
                    args.stand_action_scale_rad,
                )
            )
            p_stand_height = (
                args.stand_height_weight
                * stand
                * stability_error(
                    stand_height_error,
                    args.stand_height_scale_m,
                )
            )
            front_hip_energy *= 1.0 - next_state.done
            info = dict(next_state.info)
            info["heading_drift"] = heading_drift
            info["heading_origin_yaw"] = heading_origin_yaw
            info["lateral_drift"] = lateral_drift
            info["path_origin_xy"] = path_origin_xy
            info["front_hip_displacement_energy"] = front_hip_energy
            metrics = dict(next_state.metrics)
            metrics.update(
                {
                    "straight_yaw_penalty": p_yaw,
                    "straight_lateral_penalty": p_lateral,
                    "heading_drift_penalty": p_heading,
                    "heading_drift": jp.abs(heading_drift),
                    "lateral_drift_penalty": p_lateral_drift,
                    "lateral_drift": jp.abs(lateral_drift),
                    "speed_error": jp.abs(
                        command[0] - next_state.metrics["vx"]
                    ),
                    "speed_error_penalty": p_speed,
                    "forward_progress_reward": r_progress,
                    "front_hip_response_error": front_hip_response_error,
                    "front_hip_response_penalty": p_front_hip_response,
                    "front_rear_pose_rms": front_rear_pose_rms,
                    "fore_aft_pose_penalty": p_fore_aft_pose,
                    "front_rear_leg_length_error": (
                        front_rear_leg_length_error
                    ),
                    "fore_aft_leg_length_penalty": (
                        p_fore_aft_leg_length
                    ),
                    "physical_action_delta_rms": (
                        physical_action_delta_rms
                    ),
                    "action_rate_penalty": p_action_rate,
                    "stand_action_delta_rms": (
                        stand * action_delta_rms
                    ),
                    "stand_action_rate_penalty": p_stand_action_rate,
                    "stand_joint_velocity_rms": (
                        stand * joint_velocity_rms
                    ),
                    "stand_joint_velocity_penalty": (
                        p_stand_joint_velocity
                    ),
                    "stand_body_angular_velocity_rms": (
                        stand * body_angular_velocity_rms
                    ),
                    "stand_body_angular_penalty": (
                        p_stand_body_angular
                    ),
                    "stand_body_linear_velocity_rms": (
                        stand * body_linear_velocity_rms
                    ),
                    "stand_body_linear_penalty": p_stand_body_linear,
                    "stand_action_rms": stand * action_rms,
                    "stand_action_penalty": p_stand_action,
                    "stand_height_error": stand * stand_height_error,
                    "stand_height_penalty": p_stand_height,
                }
            )
            reward = jp.clip(
                next_state.reward
                + r_progress
                - p_yaw
                - p_lateral
                - p_heading
                - p_lateral_drift
                - p_speed
                - p_front_hip_response
                - p_fore_aft_pose
                - p_fore_aft_leg_length
                - p_action_rate
                - p_stand_action_rate
                - p_stand_joint_velocity
                - p_stand_body_angular
                - p_stand_body_linear
                - p_stand_action
                - p_stand_height,
                args.reward_clip_min,
                args.reward_clip_max,
            )
            return next_state.replace(
                reward=reward,
                info=info,
                metrics=metrics,
            )

    train_env = DeployEnvV2(
        stage=args.command_stage,
        training=True,
        deploy_dr=use_domain_randomization,
        observation_noise=args.observation_noise,
    )
    eval_env = DeployEnvV2(
        stage=args.command_stage,
        fixed_command=(args.selection_speed, 0.0, 0.0),
        training=False,
        deploy_dr=False,
        observation_noise=0.0,
    )
    selection_eval_envs = tuple(
        DeployEnvV2(
            stage=args.command_stage,
            fixed_command=(speed, 0.0, 0.0),
            training=False,
            deploy_dr=(
                use_domain_randomization
                and args.selection_deploy_perturbations
            ),
            observation_noise=(
                args.observation_noise
                if args.selection_deploy_perturbations
                else 0.0
            ),
        )
        for speed in args.selection_speeds
    )
    stand_eval_env = DeployEnvV2(
        stage=args.command_stage,
        fixed_command=(0.0, 0.0, 0.0),
        training=False,
        deploy_dr=(
            use_domain_randomization
            and args.selection_deploy_perturbations
        ),
        observation_noise=(
            args.observation_noise
            if args.selection_deploy_perturbations
            else 0.0
        ),
    )

    if args.mode == "probe":
        state = jax.jit(eval_env.reset)(jax.random.PRNGKey(args.seed))
        next_state = jax.jit(eval_env.step)(state, jp.zeros(12))
        mirrored = mirror_deploy_observation(np, np.asarray(state.obs))
        print(
            "[probe]\n"
            f"  backend={jax.default_backend()} devices={jax.devices()}\n"
            f"  obs={eval_env.observation_size} action={eval_env.action_size} "
            f"dt={float(eval_env.dt):.4f}s\n"
            f"  command={np.asarray(state.info['command']).tolist()}\n"
            f"  reward={float(next_state.reward):+.4f} "
            f"done={float(next_state.done):.0f}\n"
            f"  mirror_involution_error="
            f"{float(np.max(np.abs(mirror_deploy_observation(np, mirrored) - np.asarray(state.obs)))):.3g}"
        )
        return

    if (
        any(args.out.iterdir())
        and not args.allow_existing_output
        and set(args.out.iterdir()) != {runtime_xml.parent}
    ):
        raise SystemExit(
            f"output directory is not empty: {args.out}; "
            "use a new --out directory"
        )
    checkpoint_dir = args.out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runtime = describe_runtime()
    speed_sampling = (
        {
            "kind": "continuous_uniform",
            "minimum_m_s": args.train_speed_range[0],
            "maximum_m_s": args.train_speed_range[1],
        }
        if args.train_speed_range is not None
        else {
            "kind": "discrete_uniform",
            "speeds_m_s": list(args.train_speeds),
        }
    )
    domain_randomization = {
        "enabled": use_domain_randomization,
    }
    if use_domain_randomization:
        domain_randomization.update(
            {
                "friction_range": list(base.FRICTION_RANGE),
                "torso_mass_scale": list(base.TORSO_MASS_SCALE),
                "leg_mass_scale": list(base.LEG_MASS_SCALE),
                "inertia_scale": list(base.INERTIA_SCALE),
                "torso_com_xy_m": base.TORSO_COM_XY_M,
                "torso_com_z_m": base.TORSO_COM_Z_M,
                "motor_kp_scale": list(base.MOTOR_KP_SCALE),
                "motor_kd_scale": list(base.MOTOR_KD_SCALE),
                "motor_torque_scale": list(base.MOTOR_TORQUE_SCALE),
                "action_latency_probabilities": [
                    float(value)
                    for value in np.asarray(base.ACTION_LATENCY_PROBS)
                ],
                "action_latency_steps_50hz": [0, 1, 2],
                "control_deadline_miss_probability": (
                    base.CONTROL_DEADLINE_MISS_PROB
                ),
                "motor_zero_bias_rad": base.MOTOR_ZERO_BIAS_RAD,
                "encoder_fixed_bias_rad": base.ENCODER_FIXED_BIAS_RAD,
                "observation_noise_scale": args.observation_noise,
                "random_pushes": False,
            }
        )
    config = {
        "date": date.today().isoformat(),
        "preset": args.preset,
        "stage": args.stage,
        "command_stage": args.command_stage,
        "seed": args.seed,
        "values": args.values,
        "episode_length": args.episode_length,
        "unroll_length": args.unroll_length,
        "updates_per_batch": args.updates_per_batch,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "clipping_epsilon": args.clipping_epsilon,
        "value_clipping_epsilon": args.value_clipping_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "reward_clip": [args.reward_clip_min, args.reward_clip_max],
        "normalization": args.normalization,
        "observation_normalization": args.normalization,
        "mirror_weight": args.mirror_weight,
        "scale_aware_mirror": args.scale_aware_mirror,
        "observation_noise": args.observation_noise,
        "straight_speeds_m_s": (
            list(args.train_speeds)
            if args.train_speeds is not None
            else None
        ),
        "straight_speed_range_m_s": (
            list(args.train_speed_range)
            if args.train_speed_range is not None
            else None
        ),
        "speed_sampling": speed_sampling,
        "command_deadzone_m_s": args.command_deadzone,
        "stand_probability": args.stand_probability,
        "selection_speed_m_s": args.selection_speed,
        "selection_speeds_m_s": list(args.selection_speeds),
        "selection_duration_s": args.selection_duration_s,
        "selection_warmup_s": args.selection_warmup_s,
        "selection_deploy_perturbations": (
            args.selection_deploy_perturbations
        ),
        "eval_speeds_m_s": list(args.eval_speeds),
        "v3_selection_metrics": args.v3_selection_metrics,
        "v4_selection_metrics": args.v4_selection_metrics,
        "normalized_stability_rewards": (
            args.normalized_stability_rewards
        ),
        "smooth_stability_rewards": args.smooth_stability_rewards,
        "reward_additions": {
            "straight_yaw_weight": args.straight_yaw_weight,
            "straight_yaw_scale_rad_s": args.straight_yaw_scale_rad_s,
            "straight_lateral_weight": args.straight_lateral_weight,
            "straight_lateral_scale_m_s": (
                args.straight_lateral_scale_m_s
            ),
            "heading_drift_weight": args.heading_drift_weight,
            "heading_drift_scale_rad": args.heading_drift_scale_rad,
            "lateral_drift_weight": args.lateral_drift_weight,
            "lateral_drift_scale_m": args.lateral_drift_scale_m,
            "speed_error_weight": args.speed_error_weight,
            "forward_progress_weight": args.forward_progress_weight,
            "front_hip_response_weight": args.front_hip_response_weight,
            "fore_aft_pose_weight": args.fore_aft_pose_weight,
            "fore_aft_pose_scale_rad": args.fore_aft_pose_scale_rad,
            "fore_aft_leg_length_weight": (
                args.fore_aft_leg_length_weight
            ),
            "fore_aft_leg_length_scale_m": (
                args.fore_aft_leg_length_scale_m
            ),
            "action_rate_weight": args.action_rate_weight,
            "action_rate_scale_rad": args.action_rate_scale_rad,
            "stand_action_rate_weight": args.stand_action_rate_weight,
            "stand_action_rate_scale_rad": (
                args.stand_action_rate_scale_rad
            ),
            "stand_joint_velocity_weight": (
                args.stand_joint_velocity_weight
            ),
            "stand_joint_velocity_scale_rad_s": (
                args.stand_joint_velocity_scale_rad_s
            ),
            "stand_body_angular_weight": args.stand_body_angular_weight,
            "stand_body_angular_scale_rad_s": (
                args.stand_body_angular_scale_rad_s
            ),
            "stand_body_linear_weight": args.stand_body_linear_weight,
            "stand_body_linear_scale_m_s": (
                args.stand_body_linear_scale_m_s
            ),
            "stand_action_weight": args.stand_action_weight,
            "stand_action_scale_rad": args.stand_action_scale_rad,
            "stand_height_weight": args.stand_height_weight,
            "stand_height_scale_m": args.stand_height_scale_m,
        },
        "front_right_hip_action_scale_rad": (
            args.front_right_hip_action_scale
        ),
        "deploy_domain_randomization": use_domain_randomization,
        "domain_randomization": domain_randomization,
        "self_collision_during_training": args.self_collision,
        "training_xml": str(args.training_xml.resolve()),
        "training_xml_sha256": hashlib.sha256(
            args.training_xml.read_bytes()
        ).hexdigest(),
        "restore_checkpoint": (
            str(args.resume.resolve()) if args.resume is not None else None
        ),
        "runtime": runtime,
    }
    _json_dump(args.out / "training_config.json", config)
    print(
        "[training]\n"
        f"  output={args.out}\n"
        f"  command_stage={args.command_stage} preset={args.preset} "
        f"steps={args.values['steps']:,} envs={args.values['envs']}\n"
        f"  training_xml={args.training_xml}\n"
        f"  self_collision={args.self_collision} "
        "(mesh-ground contact remains enabled)\n"
        f"  fixed selection command=({args.selection_speed:.3f}, 0, 0)\n"
        f"  mirror_weight={args.mirror_weight:g} "
        f"scale_aware={args.scale_aware_mirror} "
        f"noise={args.observation_noise:g}\n"
        f"  domain_randomization={use_domain_randomization}\n"
        f"  speed_sampling={speed_sampling} "
        f"deadzone={args.command_deadzone:g}m/s\n"
        f"  selection_speeds={list(args.selection_speeds)} "
        f"stand_probability={args.stand_probability:g} "
        f"selection={args.selection_duration_s:g}s/"
        f"{args.selection_warmup_s:g}s warmup "
        f"deploy_perturbations={args.selection_deploy_perturbations}\n"
        f"  added reward yaw={args.straight_yaw_weight:g} "
        f"lateral={args.straight_lateral_weight:g} "
        f"heading={args.heading_drift_weight:g} "
        f"lateral_drift={args.lateral_drift_weight:g} "
        f"speed_error={args.speed_error_weight:g} "
        f"progress={args.forward_progress_weight:g} "
        f"front_hip_response={args.front_hip_response_weight:g}\n"
        f"  normalized_stability={args.normalized_stability_rewards} "
        f"smooth={args.smooth_stability_rewards}; "
        f"posture angle={args.fore_aft_pose_weight:g} "
        f"leg_length={args.fore_aft_leg_length_weight:g}; "
        f"action_rate={args.action_rate_weight:g}\n"
        f"  stand rate={args.stand_action_rate_weight:g} "
        f"joint_vel={args.stand_joint_velocity_weight:g} "
        f"body_ang={args.stand_body_angular_weight:g} "
        f"body_lin={args.stand_body_linear_weight:g} "
        f"action={args.stand_action_weight:g} "
        f"height={args.stand_height_weight:g}\n"
        f"  front-right hip action scale="
        f"{args.front_right_hip_action_scale:g}rad\n"
        f"  PPO clip={args.clipping_epsilon:g} "
        f"value_clip={args.value_clipping_epsilon:g} "
        f"reward_clip=[{args.reward_clip_min:g},{args.reward_clip_max:g}] "
        f"max_grad_norm={args.max_grad_norm:g} "
        f"normalization={args.normalization}",
        flush=True,
    )

    restore = {}
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        restore["restore_params"] = model.load_params(str(args.resume))

    selection_history = []
    best = {"rank": None, "params": None, "step": None, "report": None}
    eval_cache = {}

    def evaluate_candidate(step, make_policy, params):
        if not eval_cache:
            eval_cache["act"] = jax.jit(
                lambda policy_params, observation, key: make_policy(
                    policy_params, deterministic=True
                )(observation, key)[0]
            )
            eval_cache["stand_reset"] = jax.jit(stand_eval_env.reset)
            eval_cache["stand_step"] = jax.jit(stand_eval_env.step)
            eval_cache["moving"] = tuple(
                (jax.jit(env.reset), jax.jit(env.step))
                for env in selection_eval_envs
            )
        steps = max(
            int(round(args.selection_duration_s / float(eval_env.dt))),
            1,
        )
        warmup = max(
            int(round(args.selection_warmup_s / float(eval_env.dt))),
            0,
        )

        stand_survival = 1.0
        stand_action_delta_rms = 0.0
        stand_joint_velocity_rms = 0.0
        stand_body_angular_velocity_rms = 0.0
        stand_body_linear_velocity_rms = 0.0
        stand_action_rms = 0.0
        stand_front_rear_pose_rms = 0.0
        stand_front_rear_leg_length_error = 0.0
        stand_height_std = 0.0
        if args.v3_selection_metrics or args.v4_selection_metrics:
            stand_state = eval_cache["stand_reset"](
                jax.random.PRNGKey(args.seed + 193)
            )
            stand_key = jax.random.PRNGKey(args.seed + 194)
            stand_actions = []
            stand_joint_positions = []
            stand_joint_velocities = []
            stand_body_angular_velocities = []
            stand_body_linear_velocities = []
            stand_pose_errors = []
            stand_leg_length_errors = []
            stand_heights = []
            stand_completed = 0
            for index in range(steps):
                stand_key, action_key = jax.random.split(stand_key)
                stand_action = eval_cache["act"](
                    params, stand_state.obs, action_key
                )
                stand_state = eval_cache["stand_step"](
                    stand_state, stand_action
                )
                stand_completed += 1
                if index >= warmup:
                    stand_actions.append(np.asarray(stand_action))
                    stand_joint_positions.append(
                        np.asarray(stand_state.pipeline_state.q)[
                            np.asarray(stand_eval_env._joint_qpos)
                        ]
                    )
                    stand_joint_velocities.append(
                        np.asarray(stand_state.pipeline_state.qd)[
                            np.asarray(stand_eval_env._joint_dof)
                        ]
                    )
                    stand_body_angular_velocities.append(
                        np.asarray(stand_state.pipeline_state.xd.ang[0])
                    )
                    stand_body_linear_velocities.append(
                        np.asarray(stand_state.pipeline_state.xd.vel[0])
                    )
                    stand_pose_errors.append(
                        float(stand_state.metrics["front_rear_pose_rms"])
                    )
                    stand_leg_length_errors.append(
                        float(
                            stand_state.metrics[
                                "front_rear_leg_length_error"
                            ]
                        )
                    )
                    stand_heights.append(
                        float(stand_state.pipeline_state.q[2])
                    )
                if bool(stand_state.done):
                    break

            stand_survival = stand_completed / steps
            stand_action_array = np.asarray(stand_actions)
            stand_joint_array = np.asarray(stand_joint_positions)
            stand_joint_velocity_array = np.asarray(
                stand_joint_velocities
            )
            stand_body_angular_array = np.asarray(
                stand_body_angular_velocities
            )
            stand_body_linear_array = np.asarray(
                stand_body_linear_velocities
            )
            if len(stand_action_array):
                if args.normalized_stability_rewards:
                    stand_action_array = (
                        stand_action_array * np.asarray(base.ACTION_SCALE)
                    )
                stand_action_rms = float(
                    np.sqrt(np.mean(np.square(stand_action_array)))
                )
                if len(stand_action_array) >= 2:
                    stand_action_delta_rms = float(
                        np.sqrt(
                            np.mean(
                                np.square(np.diff(stand_action_array, axis=0))
                            )
                        )
                    )
                else:
                    stand_action_delta_rms = float("nan")
            else:
                stand_action_rms = float("nan")
                stand_action_delta_rms = float("nan")
            if len(stand_joint_velocity_array):
                stand_joint_velocity_rms = float(
                    np.sqrt(
                        np.mean(np.square(stand_joint_velocity_array))
                    )
                )
            else:
                stand_joint_velocity_rms = float("nan")
            if len(stand_body_angular_array):
                stand_body_angular_velocity_rms = float(
                    np.sqrt(np.mean(np.square(stand_body_angular_array)))
                )
            else:
                stand_body_angular_velocity_rms = float("nan")
            if len(stand_body_linear_array):
                stand_body_linear_velocity_rms = float(
                    np.sqrt(np.mean(np.square(stand_body_linear_array)))
                )
            else:
                stand_body_linear_velocity_rms = float("nan")
            if stand_pose_errors:
                stand_front_rear_pose_rms = float(
                    np.sqrt(np.mean(np.square(stand_pose_errors)))
                )
            else:
                stand_front_rear_pose_rms = float("nan")
            if stand_leg_length_errors:
                stand_front_rear_leg_length_error = float(
                    np.sqrt(
                        np.mean(np.square(stand_leg_length_errors))
                    )
                )
            else:
                stand_front_rear_leg_length_error = float("nan")
            stand_height_std = (
                float(np.std(stand_heights))
                if stand_heights
                else float("nan")
            )

        reports = []
        for speed_index, (speed, env) in enumerate(
            zip(args.selection_speeds, selection_eval_envs)
        ):
            reset_fn, step_fn = eval_cache["moving"][speed_index]
            state = reset_fn(
                jax.random.PRNGKey(args.seed + 91 + speed_index * 17)
            )
            initial_yaw = _yaw(np.asarray(state.pipeline_state.q))
            initial_lateral_position = float(state.pipeline_state.q[1])
            actions = []
            joint_positions = []
            forward = []
            lateral = []
            yaw_rates = []
            all_yaw_rates = []
            moving_pose_errors = []
            moving_leg_length_errors = []
            key = jax.random.PRNGKey(
                args.seed + 92 + speed_index * 17
            )
            completed = 0
            for index in range(steps):
                key, action_key = jax.random.split(key)
                action = eval_cache["act"](
                    params,
                    state.obs,
                    action_key,
                )
                state = step_fn(state, action)
                completed += 1
                all_yaw_rates.append(float(state.metrics["wz"]))
                if index >= warmup:
                    actions.append(np.asarray(action))
                    joint_positions.append(
                        np.asarray(state.pipeline_state.q)[
                            np.asarray(env._joint_qpos)
                        ]
                    )
                    forward.append(float(state.metrics["vx"]))
                    lateral.append(float(state.metrics["vy"]))
                    yaw_rates.append(float(state.metrics["wz"]))
                    moving_pose_errors.append(
                        float(state.metrics["front_rear_pose_rms"])
                    )
                    moving_leg_length_errors.append(
                        float(
                            state.metrics[
                                "front_rear_leg_length_error"
                            ]
                        )
                    )
                if bool(state.done):
                    break

            action_array = np.asarray(actions)
            if len(action_array):
                amplitude = (
                    np.percentile(action_array, 95, axis=0)
                    - np.percentile(action_array, 5, axis=0)
                )
                amplitude_ratio = float(
                    amplitude[4] / max(amplitude[1], 1.0e-9)
                )
            else:
                amplitude_ratio = float("nan")
            joint_array = np.asarray(joint_positions)
            if len(joint_array):
                joint_amplitude = (
                    np.percentile(joint_array, 95, axis=0)
                    - np.percentile(joint_array, 5, axis=0)
                )
                measured_amplitude_ratio = float(
                    joint_amplitude[4] / max(joint_amplitude[1], 1.0e-9)
                )
            else:
                measured_amplitude_ratio = float("nan")
            moving_front_rear_pose_rms = (
                float(np.sqrt(np.mean(np.square(moving_pose_errors))))
                if moving_pose_errors
                else float("nan")
            )
            moving_front_rear_leg_length_error = (
                float(
                    np.sqrt(
                        np.mean(np.square(moving_leg_length_errors))
                    )
                )
                if moving_leg_length_errors
                else float("nan")
            )
            report = DeployEvaluation(
                survived_fraction=min(
                    completed / steps,
                    stand_survival,
                ),
                command_speed_m_s=float(speed),
                forward_speed_m_s=(
                    float(np.mean(forward)) if forward else float("nan")
                ),
                lateral_speed_m_s=(
                    float(np.mean(lateral)) if lateral else float("nan")
                ),
                yaw_rate_rad_s=(
                    float(np.mean(yaw_rates))
                    if yaw_rates
                    else float("nan")
                ),
                heading_change_rad=(
                    float(
                        np.sum(all_yaw_rates) * float(env.dt)
                    )
                    if all_yaw_rates
                    else _yaw(np.asarray(state.pipeline_state.q))
                    - initial_yaw
                ),
                lateral_drift_m=(
                    float(state.pipeline_state.q[1])
                    - initial_lateral_position
                ),
                front_hip_amplitude_ratio=amplitude_ratio,
                front_hip_measured_amplitude_ratio=(
                    measured_amplitude_ratio
                ),
                front_rear_pose_rms_rad=max(
                    moving_front_rear_pose_rms,
                    stand_front_rear_pose_rms,
                ),
                stand_action_delta_rms=stand_action_delta_rms,
                stand_joint_velocity_rms_rad_s=stand_joint_velocity_rms,
                stand_body_angular_velocity_rms_rad_s=(
                    stand_body_angular_velocity_rms
                ),
                stand_action_rms=stand_action_rms,
                front_rear_leg_length_error_m=max(
                    moving_front_rear_leg_length_error,
                    stand_front_rear_leg_length_error,
                ),
                stand_body_linear_velocity_rms_m_s=(
                    stand_body_linear_velocity_rms
                ),
                stand_height_std_m=stand_height_std,
            )
            reports.append(report)
            print(
                "[fixed eval] "
                f"step={int(step):,} speed={speed:.2f} "
                f"survive={report.survived_fraction:.1%} "
                f"vx={report.forward_speed_m_s:+.3f} "
                f"vy={report.lateral_speed_m_s:+.3f} "
                f"wz={report.yaw_rate_rad_s:+.3f} "
                f"heading={report.heading_change_rad:+.3f} "
                f"drift={report.lateral_drift_m:+.3f}m "
                f"hip measured={report.front_hip_measured_amplitude_ratio:.3f} "
                f"pose={report.front_rear_pose_rms_rad:.3f} "
                f"leg={report.front_rear_leg_length_error_m:.4f}m",
                flush=True,
            )

        if args.v4_selection_metrics:
            rank = deploy_checkpoint_rank_v4(reports)
            report_result = reports
            row = {
                "step": int(step),
                "speed_reports": [asdict(report) for report in reports],
                "rank": list(rank),
            }
            print(
                "[fixed suite] "
                f"step={int(step):,} "
                f"stand da={stand_action_delta_rms:.4f} "
                f"action={stand_action_rms:.4f} "
                f"qd={stand_joint_velocity_rms:.4f} "
                f"omega={stand_body_angular_velocity_rms:.4f} "
                f"linear={stand_body_linear_velocity_rms:.4f} "
                f"height_std={stand_height_std:.5f}",
                flush=True,
            )
        else:
            report_result = reports[0]
            rank = (
                deploy_checkpoint_rank_v3(report_result)
                if args.v3_selection_metrics
                else deploy_checkpoint_rank(report_result)
            )
            row = {
                "step": int(step),
                **asdict(report_result),
                "rank": list(rank),
            }
        selection_history.append(row)
        _json_dump(args.out / "selection_history.json", selection_history)
        return report_result, rank

    def policy_params_fn(step, make_policy, params):
        checkpoint = checkpoint_dir / f"{int(step):012d}.bin"
        model.save_params(str(checkpoint), params)
        report, rank = evaluate_candidate(step, make_policy, params)
        if best["rank"] is None or rank > best["rank"]:
            best.update(
                {
                    "rank": rank,
                    "params": params,
                    "step": int(step),
                    "report": report,
                }
            )
            model.save_params(str(args.out / "params_best.bin"), params)
            print(f"  selected new best checkpoint at step {int(step):,}")

    started = time.perf_counter()

    def progress_fn(step, metrics):
        elapsed = time.perf_counter() - started

        def metric(name):
            value = metrics.get(f"eval/episode_{name}", float("nan"))
            try:
                return float(value)
            except TypeError:
                return float(value.item())

        print(
            "[ppo eval] "
            f"step={int(step):,} "
            f"reward={float(metrics.get('eval/episode_reward', float('nan'))):+.3f} "
            f"length={float(metrics.get('eval/avg_episode_length', float('nan'))):.1f} "
            f"speed_error={metric('speed_error'):.3f} "
            f"speed_penalty={metric('speed_error_penalty'):.3f} "
            f"progress={metric('forward_progress_reward'):.3f} "
            f"yaw_penalty={metric('straight_yaw_penalty'):.3f} "
            f"heading={metric('heading_drift'):.3f} "
            f"lateral_drift_sum={metric('lateral_drift'):.3f} "
            f"fore/aft_sum={metric('front_rear_pose_rms'):.3f} "
            f"leg_error_sum={metric('front_rear_leg_length_error'):.3f} "
            f"action_rate={metric('action_rate_penalty'):.3f} "
            f"elapsed={elapsed / 60.0:.1f}min",
            flush=True,
        )

    train_kwargs = {
        "environment": train_env,
        "eval_env": eval_env,
        "num_timesteps": args.values["steps"],
        "num_evals": args.values["num_evals"],
        "episode_length": args.episode_length,
        "num_envs": args.values["envs"],
        "batch_size": args.values["batch_size"],
        "num_minibatches": args.values["num_minibatches"],
        "num_updates_per_batch": args.updates_per_batch,
        "unroll_length": args.unroll_length,
        "discounting": args.discounting,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "clipping_epsilon": args.clipping_epsilon,
        "clipping_epsilon_value": args.value_clipping_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "reward_scaling": 1.0,
        "normalize_observations": args.normalization == "running",
        "action_repeat": 1,
        "network_factory": functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=base.POLICY_HIDDEN,
            value_hidden_layer_sizes=base.VALUE_HIDDEN,
            activation=base.ACTIVATION,
        ),
        "randomization_fn": (
            base.deploy_domain_randomize
            if use_domain_randomization
            else None
        ),
        "policy_params_fn": policy_params_fn,
        "progress_fn": progress_fn,
        "deterministic_eval": True,
        "seed": args.seed,
        **restore,
    }
    parameters = inspect.signature(ppo.train).parameters
    if "num_eval_envs" in parameters:
        train_kwargs["num_eval_envs"] = args.values["eval_envs"]

    with actor_mirror_consistency_scope(
        ppo_losses,
        weight=args.mirror_weight,
        array_module=jp,
        stop_gradient=jax.lax.stop_gradient,
        action_scale=(
            base.ACTION_SCALE if args.scale_aware_mirror else None
        ),
    ):
        make_inference_fn, final_params, final_metrics = ppo.train(
            **train_kwargs
        )

    elapsed = time.perf_counter() - started
    model.save_params(str(args.out / "params_final.bin"), final_params)
    if best["params"] is None:
        best["params"] = final_params
        best["step"] = args.values["steps"]
        model.save_params(str(args.out / "params_best.bin"), final_params)

    best_report = best["report"]
    if isinstance(best_report, (list, tuple)):
        serialized_best_report = [
            asdict(report) for report in best_report
        ]
    elif best_report is not None:
        serialized_best_report = asdict(best_report)
    else:
        serialized_best_report = None
    summary = {
        "elapsed_s": elapsed,
        "throughput_requested_steps_s": args.values["steps"] / max(elapsed, 1e-9),
        "best_step": best["step"],
        "best_report": serialized_best_report,
        "best_rank": list(best["rank"]) if best["rank"] is not None else None,
        "final_metrics": {
            name: float(value)
            for name, value in (final_metrics or {}).items()
            if np.asarray(value).size == 1
        },
    }
    _json_dump(args.out / "training_summary.json", summary)

    exported_policy = None
    if not args.skip_export:
        controller_config = {
            "use_imu": True,
            "control_orientation": False,
            "observation_history": base.HISTORY,
            "observation_normalization": args.normalization,
            "kp": base.SERVO_KP,
            "kd": base.SERVO_KD,
            "action_scale": [
                float(value) for value in np.asarray(base.ACTION_SCALE)
            ],
            "default_joint_pos": [
                float(value) for value in np.asarray(base.DEFAULT_POSE)
            ],
            "joint_lower_limits": [
                float(value) for value in np.asarray(base.CTRL_LO)
            ],
            "joint_upper_limits": [
                float(value) for value in np.asarray(base.CTRL_HI)
            ],
        }
        config_path = args.out / "controller_config.json"
        _json_dump(config_path, controller_config)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "export_rtneural.py"),
                str(args.out / "params_best.bin"),
                str(args.out / "policy_best.json"),
                "--activation",
                base.ACTIVATION_NAME,
                "--config",
                str(config_path),
                "--obs-history",
                str(base.HISTORY),
                "--observation-normalization",
                args.normalization,
            ],
            check=True,
        )
        exported_policy = args.out / "policy_best.json"

    print(
        "[complete]\n"
        f"  elapsed={elapsed / 60.0:.1f}min\n"
        f"  best_step={best['step']}\n"
        f"  best_checkpoint={args.out / 'params_best.bin'}\n"
        f"  exported_policy="
        f"{exported_policy if exported_policy is not None else 'skipped'}",
        flush=True,
    )
    return make_inference_fn


def main(argv=None):
    args = parse_args(argv)
    _run(args)


if __name__ == "__main__":
    main()
