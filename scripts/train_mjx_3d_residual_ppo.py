"""Train a 3-D CEM-reference residual PPO rolling policy."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import inspect
import json
import math
import shutil
from dataclasses import asdict, fields, replace
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.config_3d import (
    GEOMETRY_NAMES_3D,
    PHYSICS_PROFILE_NAMES_3D,
    Rolling3DConfig,
    physics_profile_3d,
)
from curl_robot_2d_mjx.curriculum_3d import (
    CURRICULUM_NAMES_3D,
    CURRICULUM_STAGE_NAMES_3D,
    curriculum_stages_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    OBSERVATION_SIZE_3D,
    PHASE_FEEDBACK_SIZE_3D,
    cem_controller_path_3d,
    mirror_rolling_observation_3d,
)
from curl_robot_2d_mjx.randomization_3d import (
    make_domain_randomization_fn_3d,
)
from curl_robot_2d_mjx.reward_3d import (
    REWARD_3D_TERM_NAMES,
    Rolling3DRewardConfig,
)
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    describe_runtime,
)
from curl_robot_2d_mjx.startup_rolling_3d import add_stand_startup_arguments, with_stand_startup
from scripts.train_mjx_ppo import (
    _float,
    _network_factory,
    _resolve_restore_checkpoint,
    _split_metrics,
    _training_step_schedule,
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

TANH_NORMAL_MIN_STD = 1e-3
MAX_CHECKPOINT_FAILURE_RATE = 0.05


def _allocate_weighted_counts(total, weights, *, minimum=1):
    """Allocate integer work units while preserving the requested weights."""

    count = len(weights)
    if count == 0:
        return []
    total = max(int(total), count * minimum)
    remaining = total - count * minimum
    weight_sum = sum(weights)
    raw = [remaining * weight / weight_sum for weight in weights]
    allocated = [minimum + int(math.floor(value)) for value in raw]
    leftover = total - sum(allocated)
    order = sorted(
        range(count),
        key=lambda index: raw[index] - math.floor(raw[index]),
        reverse=True,
    )
    for index in order[:leftover]:
        allocated[index] += 1
    return allocated


def _curriculum_training_plan(args, values):
    stages = curriculum_stages_3d(
        args.curriculum,
        only_stage=args.curriculum_stage,
    )
    if len(stages) == 1 and stages[0].name == "nominal":
        stage_schedule = _training_step_schedule(
            requested_steps=values["steps"],
            num_evals=values["num_evals"],
            batch_size=values["batch_size"],
            unroll_length=args.unroll_length,
            num_minibatches=values["num_minibatches"],
        )
        return (
            {
                "stage": stages[0],
                "num_evals": values["num_evals"],
                "schedule": stage_schedule,
            },
        )

    rollout_quantum = (
        values["batch_size"]
        * args.unroll_length
        * values["num_minibatches"]
    )
    weights = [stage.weight for stage in stages]
    requested_quanta = math.ceil(values["steps"] / rollout_quantum)
    stage_quanta = _allocate_weighted_counts(requested_quanta, weights)
    # Brax uses the first evaluation at stage step zero.  Each stage therefore
    # needs a second evaluation to measure the policy after that stage trains.
    stage_evals = _allocate_weighted_counts(
        values["num_evals"], weights, minimum=2
    )
    plan = []
    for stage, quanta, num_evals in zip(
        stages, stage_quanta, stage_evals, strict=True
    ):
        requested_steps = quanta * rollout_quantum
        stage_schedule = _training_step_schedule(
            requested_steps=requested_steps,
            num_evals=num_evals,
            batch_size=values["batch_size"],
            unroll_length=args.unroll_length,
            num_minibatches=values["num_minibatches"],
        )
        plan.append(
            {
                "stage": stage,
                "num_evals": num_evals,
                "schedule": stage_schedule,
            }
        )
    return tuple(plan)


def _ppo_update_counts(plan, *, updates_per_batch, num_minibatches):
    """Count fresh-rollout PPO cycles and the gradient work they trigger."""

    rollout_updates = sum(
        item["schedule"]["eval_intervals"]
        * item["schedule"]["updates_per_interval"]
        for item in plan
    )
    data_reuse_passes = rollout_updates * updates_per_batch
    return {
        "rollout_updates": rollout_updates,
        "data_reuse_passes": data_reuse_passes,
        "optimizer_steps": data_reuse_passes * num_minibatches,
    }


RECIPES_3D = {
    "anchored_v1": {
        "description": "Original conservative CEM-anchored residual run.",
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.05,
            "phase_rate_scale": 1.0,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "selection_target_turns": 1.0,
            "zero_residual_policy_init": False,
            "initial_policy_std": 1.0,
        },
        "reward": {},
    },
    "push_v2": {
        "description": (
            "Give residual enough authority and make forward rolling dominate "
            "the early learning signal."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.30,
            "phase_rate_scale": 1.0,
            "learning_rate": 1e-4,
            "entropy_cost": 3e-3,
            "selection_target_turns": 1.0,
            "zero_residual_policy_init": False,
            "initial_policy_std": 1.0,
        },
        "reward": {
            "roll_progress": 12.0,
            "roll_mismatch": 0.25,
            "backward": 0.4,
            "lateral_drift": 1.0,
            "yaw": 1.0,
            "axis_tilt": 10.0,
            "action_rate": 0.01,
            "residual_action": 0.003,
        },
    },
    "phase_locked_v3": {
        "description": (
            "Learn independent left/right residuals around the restored "
            "phase-locked 3-D CEM reference."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "learning_rate": 5e-5,
            "entropy_cost": 1e-3,
            "selection_target_turns": 10.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.20,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_drift": 1.0,
            "yaw": 1.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
        },
    },
    "phase_locked_safe_v4": {
        "description": (
            "Keep the phase-locked residual policy centered while making "
            "late lateral failure worse than stable forward rolling."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "learning_rate": 5e-5,
            "entropy_cost": 1e-3,
            "selection_target_turns": 10.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.20,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 8.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_safe_v5": {
        "description": (
            "Use a partial failed-progress clawback so stable rolling beats "
            "lateral failure without collapsing exploration."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "learning_rate": 5e-5,
            "entropy_cost": 1e-3,
            "selection_target_turns": 10.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.20,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 2.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_coupled_v6": {
        "description": (
            "Explore mainly through symmetric rolling residuals while "
            "retaining lower-authority left/right correction channels."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "explicit_phase_observation": True,
            "learning_rate": 5e-5,
            "entropy_cost": 1e-3,
            "selection_target_turns": 10.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.20,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 2.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_coupled_v7": {
        "description": (
            "Retain coupled residual exploration while making late failed "
            "rolling less valuable than the validated reference baseline."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "explicit_phase_observation": True,
            "learning_rate": 5e-5,
            "entropy_cost": 1e-3,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.20,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_coupled_v8": {
        "description": (
            "Learn lower-variance differential corrections conservatively "
            "after coupled v7 exposed directional PPO overshoot."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "explicit_phase_observation": True,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_equivariant_v9": {
        "description": (
            "Enforce reflection-even common residuals and reflection-odd "
            "differential corrections while retaining conservative updates."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "explicit_phase_observation": True,
            "reflection_equivariant_policy": True,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_meanzero_v10": {
        "description": (
            "Drop the hard reflection projection and instead penalize the "
            "batch-mean differential, so the residual policy can learn "
            "reflection-odd lateral corrections without a constant left/right "
            "bias."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "explicit_phase_observation": True,
            "reflection_equivariant_policy": False,
            "differential_mean_zero_weight": 50.0,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_reflex_v11": {
        "description": (
            "Fixed lateral-stability reflex on the differential channels with "
            "a common-only residual policy for straight rolling."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.0,
            "lateral_reflex_gain": 0.25,
            "lateral_reflex_position_gain": 2.0,
            "lateral_reflex_velocity_gain": 2.0,
            "explicit_phase_observation": True,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "phase_locked_turn_v12": {
        "description": (
            "Fixed lateral reflex plus sampled turning commands, with the "
            "differential residual re-enabled so the policy can refine "
            "turning beyond the reflex envelope."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "lateral_reflex_gain": 0.25,
            "lateral_reflex_position_gain": 2.0,
            "lateral_reflex_velocity_gain": 2.0,
            "lateral_command_enabled": True,
            "lateral_command_max": 0.15,
            "lateral_command_probability": 0.20,
            "lateral_command_error_limit": 0.20,
            "explicit_phase_observation": True,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "robust_low_friction_v13": {
        "description": (
            "Fixed lateral reflex plus a low-friction domain-randomization "
            "curriculum; the residual policy learns rolling compensation "
            "under shell slip."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "lateral_reflex_gain": 0.25,
            "lateral_reflex_position_gain": 2.0,
            "lateral_reflex_velocity_gain": 2.0,
            "explicit_phase_observation": True,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "learn_yaw_v14": {
        "description": (
            "No reflex. The policy learns lateral stabilization from separate "
            "cross-track, lateral-velocity, yaw, and yaw-rate rewards."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.15,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "lateral_reflex_gain": 0.0,
            "explicit_phase_observation": True,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 0.5,
            "lateral_drift": 0.5,
            "yaw_rate": 0.5,
            "yaw": 0.5,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.01,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "robust_recovery_v15": {
        "description": (
            "Preserve the CEM reference while learning recovery from bounded "
            "Huber stability costs and error-reduction potential shaping."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.30,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "lateral_reflex_gain": 0.0,
            "explicit_phase_observation": True,
            "learning_rate": 1e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 8.0,
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.10,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.8,
            "backward": 1.0,
            "lateral_velocity": 0.0,
            "lateral_drift": 0.0,
            "yaw_rate": 0.0,
            "yaw": 0.0,
            "lateral_velocity_cost": 0.25,
            "lateral_drift_cost": 1.0,
            "yaw_rate_cost": 0.25,
            "yaw_cost": 0.5,
            "stability_huber_clip": 1.0,
            "recovery": 4.0,
            "recovery_clip": 0.25,
            "axis_tilt": 10.0,
            "action_rate": 0.02,
            "residual_action": 0.05,
            "failure_progress_clawback": 4.0,
            "termination": 40.0,
            "severe_extra_termination": 40.0,
        },
    },
    "real_geometry_contact_v1": {
        "description": (
            "Keep the real-geometry 2-D CEM reference, learn symmetric "
            "phase-local residual tucking, and select only policies that "
            "retain five conservative turns while reducing forbidden contact."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.20,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.25,
            "explicit_phase_observation": True,
            "learning_rate": 5e-5,
            "entropy_cost": 7.5e-4,
            "selection_target_turns": 5.0,
            "selection_objective": "contact",
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.15,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.6,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.03,
            "residual_action": 0.005,
            "torque": 0.02,
            "forbidden_contact_time": 8.0,
            "first_turn_forbidden_contact_multiplier": 3.0,
            "forbidden_penetration_integral": 30000.0,
            "maximum_forbidden_penetration": 4000.0,
            "failure_progress_clawback": 4.0,
            "termination": 50.0,
            "severe_extra_termination": 50.0,
        },
    },
    "real_geometry_contact_v2": {
        "description": (
            "Conservative real-geometry contact cleanup: preserve left/right "
            "symmetry, keep residual authority small, and strengthen the "
            "phase-local collision learning signal."
        ),
        "args": {
            "reference_weight": 1.0,
            "minimum_residual_gain": 0.08,
            "phase_rate_scale": 1.0,
            "residual_pair_differential_scale": 0.0,
            "explicit_phase_observation": True,
            "learning_rate": 3e-5,
            "entropy_cost": 2.5e-4,
            "selection_target_turns": 5.0,
            "selection_objective": "contact",
            "zero_residual_policy_init": True,
            "initial_policy_std": 0.05,
        },
        "reward": {
            "roll_progress": 8.0,
            "roll_mismatch": 0.6,
            "backward": 1.0,
            "lateral_velocity": 2.0,
            "lateral_drift": 3.0,
            "yaw_rate": 2.0,
            "yaw": 3.0,
            "axis_tilt": 10.0,
            "action_rate": 0.03,
            "residual_action": 0.01,
            "torque": 0.02,
            "forbidden_contact_time": 24.0,
            "first_turn_forbidden_contact_multiplier": 4.0,
            "forbidden_penetration_integral": 60000.0,
            "maximum_forbidden_penetration": 8000.0,
            "failure_progress_clawback": 4.0,
            "termination": 50.0,
            "severe_extra_termination": 50.0,
        },
    },
}


def _tanh_normal_scale_logit(
    initial_std: float,
    minimum_std: float = TANH_NORMAL_MIN_STD,
) -> float:
    """Invert Brax's softplus scale transform for a requested initial std."""

    adjusted_std = initial_std - minimum_std
    if adjusted_std <= 0.0:
        raise ValueError("initial_std must exceed minimum_std")
    if adjusted_std > 20.0:
        return adjusted_std
    return math.log(math.expm1(adjusted_std))


def _observation_width(observation_size) -> int:
    """Return the final state width from Brax's int or shape tuple."""

    if isinstance(observation_size, (int, np.integer)):
        width = int(observation_size)
    elif isinstance(observation_size, (tuple, list)) and observation_size:
        width = int(observation_size[-1])
    else:
        raise ValueError(
            f"Unsupported policy observation size: {observation_size!r}"
        )
    if width <= 0:
        raise ValueError("Policy observation width must be positive")
    return width


def _zero_centered_residual_network_factory(
    hidden_layers,
    activation_name,
    initial_std,
    *,
    reflection_equivariant=False,
):
    """Build PPO networks whose initial residual policy is centered at zero."""

    import jax.numpy as jnp
    import jax.nn as jnn
    from brax.training import networks as training_networks
    from brax.training import types as training_types
    from brax.training.agents.ppo import networks as ppo_networks
    from flax import linen

    activation = {
        "elu": jnn.elu,
        "relu": jnn.relu,
        "swish": jnn.swish,
        "tanh": jnn.tanh,
    }[activation_name]
    hidden_layer_sizes = tuple(hidden_layers)
    scale_logit = _tanh_normal_scale_logit(initial_std)

    class ResidualPolicyModule(linen.Module):
        action_size: int

        @linen.compact
        def __call__(self, observation):
            hidden = observation
            for index, layer_size in enumerate(hidden_layer_sizes):
                hidden = linen.Dense(
                    layer_size,
                    kernel_init=jnn.initializers.lecun_uniform(),
                    name=f"hidden_{index}",
                )(hidden)
                hidden = activation(hidden)
            location = linen.Dense(
                self.action_size,
                kernel_init=jnn.initializers.zeros,
                bias_init=jnn.initializers.zeros,
                name="location",
            )(hidden)
            scale = linen.Dense(
                self.action_size,
                kernel_init=jnn.initializers.zeros,
                bias_init=jnn.initializers.constant(scale_logit),
                name="scale",
            )(hidden)
            return jnp.concatenate((location, scale), axis=-1)

    def factory(
        observation_size,
        action_size,
        preprocess_observations_fn=(
            training_types.identity_observation_preprocessor
        ),
    ):
        base_networks = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
        )
        policy_module = ResidualPolicyModule(action_size=action_size)
        dummy_observation = jnp.zeros(
            (1, _observation_width(observation_size))
        )

        def apply(processor_params, policy_params, observation):
            processed_observation = preprocess_observations_fn(
                observation, processor_params
            )
            policy_parameters = policy_module.apply(
                policy_params, processed_observation
            )
            if not reflection_equivariant:
                return policy_parameters
            mirrored_observation = mirror_rolling_observation_3d(
                jnp, observation
            )
            processed_mirrored_observation = preprocess_observations_fn(
                mirrored_observation, processor_params
            )
            mirrored_policy_parameters = policy_module.apply(
                policy_params, processed_mirrored_observation
            )
            return _reflection_equivariant_policy_parameters_3d(
                jnp,
                policy_parameters,
                mirrored_policy_parameters,
                action_size,
            )

        policy_network = training_networks.FeedForwardNetwork(
            init=lambda key: policy_module.init(key, dummy_observation),
            apply=apply,
        )
        return ppo_networks.PPONetworks(
            policy_network=policy_network,
            value_network=base_networks.value_network,
            parametric_action_distribution=(
                base_networks.parametric_action_distribution
            ),
        )

    return factory


def _reflection_equivariant_policy_parameters_3d(
    xp,
    policy_parameters,
    mirrored_policy_parameters,
    action_size=8,
):
    """Project raw policy parameters onto the rolling reflection symmetry."""

    if action_size % 2:
        raise ValueError("rolling action_size must be even")
    common_size = action_size // 2
    location = policy_parameters[..., :action_size]
    mirrored_location = mirrored_policy_parameters[..., :action_size]
    common_location = 0.5 * (
        location[..., :common_size]
        + mirrored_location[..., :common_size]
    )
    differential_location = 0.5 * (
        location[..., common_size:]
        - mirrored_location[..., common_size:]
    )
    scale_parameters = 0.5 * (
        policy_parameters[..., action_size:]
        + mirrored_policy_parameters[..., action_size:]
    )
    return xp.concatenate(
        (common_location, differential_location, scale_parameters),
        axis=-1,
    )


@contextmanager
def _differential_mean_zero_loss_scope(module, *, weight, array_module):
    """Temporarily add a mean-zero differential PPO loss.

    The rolling residual action is ``[common(4), differential(4)]``. The
    differential half maps left/right hip and knee in opposite directions, so
    a constant differential bias steers the robot into lateral drift (the
    v6-v8 failure). This scope penalizes the squared batch-mean of the
    differential mode, which suppresses that constant bias while leaving
    reflection-odd feedback such as ``differential = +k * lateral_y`` (whose
    batch mean is zero) free to learn.
    """

    if weight <= 0.0:
        yield
        return

    original_compute_ppo_loss = module.compute_ppo_loss

    def compute_ppo_loss_with_differential_mean_zero(
        params,
        normalizer_params,
        data,
        rng,
        ppo_network,
        **kwargs,
    ):
        base_loss, metrics = original_compute_ppo_loss(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network,
            **kwargs,
        )
        policy_apply = ppo_network.policy_network.apply
        action_distribution = ppo_network.parametric_action_distribution
        logits = policy_apply(
            normalizer_params, params.policy, data.observation
        )
        mode = action_distribution.mode(logits)
        action_size = mode.shape[-1]
        differential_mode = mode[..., action_size // 2 :]
        differential_mean = array_module.mean(differential_mode, axis=0)
        mean_zero_loss = array_module.mean(
            array_module.square(differential_mean)
        )
        weighted_loss = weight * mean_zero_loss
        total_loss = base_loss + weighted_loss
        metrics = dict(metrics)
        metrics["ppo_total_loss"] = metrics.get("total_loss", base_loss)
        metrics["total_loss"] = total_loss
        metrics["differential_mean_zero_loss"] = mean_zero_loss
        metrics["differential_mean_rms"] = array_module.sqrt(
            array_module.mean(array_module.square(differential_mean))
        )
        metrics["differential_mean_zero_weighted_loss"] = weighted_loss
        return total_loss, metrics

    module.compute_ppo_loss = compute_ppo_loss_with_differential_mean_zero
    try:
        yield
    finally:
        module.compute_ppo_loss = original_compute_ppo_loss


PER_STEP_EVAL_METRICS_3D = (
    "root_x_m",
    "root_y_m",
    "root_z_m",
    "lateral_drift_m",
    "lateral_drift_abs_m",
    "lateral_velocity_m_s",
    "stability_error_cost",
    "axis_tilt_rad",
    "axis_tilt_step_count",
    "root_low_active",
    "root_low_step_count",
    "shell_floor_contact_count",
    "foot_floor_contact_count",
    "same_side_foot_contact_count",
    "same_side_foot_penetration_m",
    "same_side_foot_contact_active",
    "same_side_foot_contact_start",
    "forbidden_contact_count",
    "first_turn_forbidden_contact_count",
    "forbidden_penetration_m",
    "forbidden_contact_step_count",
    "cross_side_foot_contact_count",
    "front_rear_leg_contact_count",
    "front_rear_leg_penetration_m",
    "leg_torso_contact_count",
    "leg_torso_penetration_m",
    "foot_leg_contact_count",
    "foot_leg_penetration_m",
    "foot_foot_contact_count",
    "foot_foot_penetration_m",
    "action_rms",
    "action_rate_rms",
    "startup_action_ramp",
    "normalized_torque_rms",
    "reference_action_rms",
    "residual_action_rms",
    "residual_common_rms",
    "residual_differential_rms",
    "reference_weight",
    "residual_gain",
    "rolling_phase_rad",
    "oscillator_phase_rad",
    "phase_error_rad",
    "oscillator_rate_rad_s",
    "roll_progress_rad",
    "rotation_progress_rad",
    "translation_progress_rad",
    "mismatch_progress_rad",
)


def _add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose 3-D reward dataclass fields as optional overrides."""

    for field in fields(Rolling3DRewardConfig):
        parser.add_argument(
            f"--reward-{field.name.replace('_', '-')}",
            dest=f"reward_{field.name}",
            type=float,
            default=None,
            help=(
                f"Override Rolling3DRewardConfig.{field.name}; "
                "the default comes from reward_3d.py."
            ),
        )


def _reward_config_from_args(args) -> Rolling3DRewardConfig:
    overrides = dict(RECIPES_3D[args.recipe]["reward"])
    overrides.update(
        {
            field.name: value
            for field in fields(Rolling3DRewardConfig)
            if (value := getattr(args, f"reward_{field.name}", None))
            is not None
        }
    )
    return replace(Rolling3DRewardConfig(), **overrides)


def _add_per_step_eval_metrics_3d(metrics):
    average_length = metrics.get("eval/avg_episode_length")
    if average_length is None or average_length <= 0:
        return
    for name in PER_STEP_EVAL_METRICS_3D:
        key = f"eval/episode_{name}"
        if key in metrics:
            metrics[f"eval/avg_{name}"] = metrics[key] / average_length
    for name in REWARD_3D_TERM_NAMES:
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


def _checkpoint_selection_3d(
    metrics,
    episode_length,
    *,
    target_turns=1.0,
    maximum_failure_rate=MAX_CHECKPOINT_FAILURE_RATE,
    objective="balanced",
):
    """Score eval points by 3-D physical behavior, not raw reward scale."""

    average_length = metrics.get("eval/avg_episode_length", 0.0)
    failed_rate = metrics.get("eval/episode_failed", 1.0)
    nonfinite_rate = metrics.get("eval/episode_failure_nonfinite", 0.0)
    roll_total = metrics.get("eval/episode_roll_progress_rad", -math.inf)
    lateral_drift = metrics.get(
        "eval/avg_lateral_drift_abs_m",
        abs(metrics.get("eval/avg_lateral_drift_m", math.inf)),
    )
    axis_tilt = metrics.get("eval/avg_axis_tilt_rad", math.inf)
    forbidden_depth = metrics.get("eval/avg_forbidden_penetration_m", math.inf)
    forbidden_contact = metrics.get("eval/avg_forbidden_contact_count", math.inf)
    first_turn_forbidden_contact = metrics.get(
        "eval/avg_first_turn_forbidden_contact_count",
        forbidden_contact,
    )
    survival = min(max(average_length / episode_length, 0.0), 1.0)
    turns = roll_total / (2.0 * math.pi)
    progress_quality = min(max(turns / target_turns, -1.0), 1.0)
    nonfailure_quality = 1.0 - min(max(failed_rate, 0.0), 1.0)
    lateral_quality = 1.0 - min(max(lateral_drift / 0.05, 0.0), 1.0)
    tilt_quality = 1.0 - min(max(axis_tilt / 0.25, 0.0), 1.0)
    contact_quality = 1.0 - min(
        max(forbidden_depth / 0.001, 0.0)
        + max(forbidden_contact / 0.05, 0.0),
        1.0,
    )
    if objective == "contact":
        contact_quality = 1.0 - min(
            max(forbidden_depth / 0.001, 0.0)
            + max(forbidden_contact / 0.05, 0.0)
            + max(first_turn_forbidden_contact / 0.02, 0.0),
            1.0,
        )
        score = (
            0.15 * survival
            + 0.20 * progress_quality
            + 0.05 * nonfailure_quality
            + 0.05 * lateral_quality
            + 0.05 * tilt_quality
            + 0.50 * contact_quality
        )
    elif objective == "balanced":
        score = (
            0.20 * survival
            + 0.25 * progress_quality
            + 0.30 * nonfailure_quality
            + 0.15 * lateral_quality
            + 0.05 * tilt_quality
            + 0.05 * contact_quality
        )
    else:
        raise ValueError(f"unknown checkpoint selection objective: {objective}")
    rejected = (
        nonfinite_rate > 0.0
        or failed_rate > maximum_failure_rate
        or not math.isfinite(turns)
        or not math.isfinite(lateral_drift)
        or not math.isfinite(axis_tilt)
        or not math.isfinite(forbidden_depth)
        or not math.isfinite(forbidden_contact)
        or not math.isfinite(first_turn_forbidden_contact)
        or not math.isfinite(score)
        or (objective == "contact" and turns < target_turns)
    )
    return {
        "score": -1_000_000.0 if rejected else score,
        "rejected": rejected,
        "failure_rate": failed_rate,
        "passes_acceptance_failure_rate": (
            failed_rate <= MAX_CHECKPOINT_FAILURE_RATE
        ),
        "survival": survival,
        "turns": turns,
        "lateral_drift_m": lateral_drift,
        "axis_tilt_rad": axis_tilt,
        "contact_quality": contact_quality,
        "forbidden_contact_count": forbidden_contact,
        "first_turn_forbidden_contact_count": first_turn_forbidden_contact,
        "objective": objective,
    }


def _resolve_best_params(
    best,
    final_params,
    metric_history,
    *,
    final_step,
    initial_evaluated=None,
):
    """Resolve an exact evaluated parameter set despite callback ordering."""

    last_eval_step = (
        int(metric_history[-1]["step"]) if metric_history else None
    )
    if (
        best["step"] is not None
        and best["step"] == last_eval_step
        and last_eval_step == int(final_step)
    ):
        return final_params, "final_eval"
    if (
        best["params"] is not None
        and best.get("params_step") == best["step"]
    ):
        return best["params"], f"callback_step_{best['params_step']}"
    if initial_evaluated is not None and best["step"] is not None:
        if (
            initial_evaluated.get("step") == best["step"]
            and initial_evaluated.get("params") is not None
            and initial_evaluated.get("params_step") == best["step"]
        ):
            return (
                initial_evaluated["params"],
                f"initial_eval_step_{best['step']}",
            )
    if initial_evaluated is not None and best["step"] is None:
        if (
            initial_evaluated.get("params") is not None
            and initial_evaluated.get("params_step")
            == initial_evaluated.get("step")
        ):
            return (
                initial_evaluated["params"],
                f"initial_eval_step_{initial_evaluated['step']}",
            )
    return final_params, "final_unresolved_fallback"


def _best_and_final_share_checkpoint(best_params_source):
    return best_params_source in {
        "final_eval",
        "final_unresolved_fallback",
    }


def _format_eval_report_3d(
    eval_index,
    total_evals,
    step,
    metrics,
    *,
    episode_length,
    control_dt,
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
            f"failed={_metric(metrics, 'eval/episode_failed'):.1%} "
            f"timeout={_metric(metrics, 'eval/episode_timeout'):.1%}"
        ),
        (
            "  motion (rad/step) "
            f"translation="
            f"{_metric(metrics, 'eval/avg_translation_progress_rad'):+.5f} "
            f"rotation="
            f"{_metric(metrics, 'eval/avg_rotation_progress_rad'):+.5f} "
            f"roll={_metric(metrics, 'eval/avg_roll_progress_rad'):+.5f} "
            f"mismatch="
            f"{_metric(metrics, 'eval/avg_mismatch_progress_rad'):+.5f} "
            f"turns/episode={selection['turns']:+.3f}"
        ),
        (
            "  mean pose "
            f"mean_x={_metric(metrics, 'eval/avg_root_x_m'):+.3f}m "
            f"mean_y={_metric(metrics, 'eval/avg_root_y_m'):+.3f}m "
            f"mean_z={_metric(metrics, 'eval/avg_root_z_m'):.3f}m "
            f"mean_lateral={selection['lateral_drift_m']:.3f}m "
            f"stability_cost="
            f"{_metric(metrics, 'eval/avg_stability_error_cost'):.3f} "
            f"mean_axis_tilt={selection['axis_tilt_rad']:.3f}rad"
        ),
        (
            "  contact "
            f"shell={_metric(metrics, 'eval/avg_shell_floor_contact_count'):.2f} "
            f"foot={_metric(metrics, 'eval/avg_foot_floor_contact_count'):.2f} "
            f"same={_metric(metrics, 'eval/avg_same_side_foot_contact_count'):.2f} "
            f"cross={_metric(metrics, 'eval/avg_cross_side_foot_contact_count'):.2f} "
            f"forbidden="
            f"{_metric(metrics, 'eval/avg_forbidden_contact_count'):.2f} "
            f"depth="
            f"{1e3 * _metric(metrics, 'eval/avg_forbidden_penetration_m'):.3f}mm"
        ),
        (
            "  self    "
            f"front-rear="
            f"{_metric(metrics, 'eval/avg_front_rear_leg_contact_count'):.3f} "
            f"leg-torso="
            f"{_metric(metrics, 'eval/avg_leg_torso_contact_count'):.3f} "
            f"foot-leg="
            f"{_metric(metrics, 'eval/avg_foot_leg_contact_count'):.3f} "
            f"foot-foot="
            f"{_metric(metrics, 'eval/avg_foot_foot_contact_count'):.3f}"
        ),
        (
            "  action  "
            f"rms={_metric(metrics, 'eval/avg_action_rms'):.3f} "
            f"rate={_metric(metrics, 'eval/avg_action_rate_rms'):.3f} "
            f"torque={_metric(metrics, 'eval/avg_normalized_torque_rms'):.3f} "
            f"ref={_metric(metrics, 'eval/avg_reference_action_rms'):.3f} "
            f"residual={_metric(metrics, 'eval/avg_residual_action_rms'):.3f} "
            f"common={_metric(metrics, 'eval/avg_residual_common_rms'):.3f} "
            f"diff={_metric(metrics, 'eval/avg_residual_differential_rms'):.3f} "
            f"gain={_metric(metrics, 'eval/avg_residual_gain'):.3f}"
        ),
        (
            "  phase   "
            f"theta={_metric(metrics, 'eval/avg_rolling_phase_rad'):+.3f} "
            f"phi={_metric(metrics, 'eval/avg_oscillator_phase_rad'):+.3f} "
            f"error={_metric(metrics, 'eval/avg_phase_error_rad'):+.3f} "
            f"rate={_metric(metrics, 'eval/avg_oscillator_rate_rad_s'):+.3f}rad/s"
        ),
    ]
    for group, reward_labels in (
        (
            "progress",
            (
                ("roll", "roll_progress"),
                ("mismatch", "roll_mismatch"),
                ("back", "backward"),
            ),
        ),
        (
            "3d      ",
            (
                ("lat_vel", "lateral_velocity"),
                ("lat", "lateral_drift"),
                ("yaw_rate", "yaw_rate"),
                ("yaw", "yaw"),
                ("tilt", "axis_tilt"),
            ),
        ),
        (
            "recovery",
            (
                ("vy_cost", "lateral_velocity_cost"),
                ("y_cost", "lateral_drift_cost"),
                ("wz_cost", "yaw_rate_cost"),
                ("yaw_cost", "yaw_cost"),
                ("delta", "recovery"),
            ),
        ),
        (
            "control ",
            (
                ("rate", "action_rate"),
                ("residual", "residual_action"),
                ("torque", "torque"),
            ),
        ),
        (
            "safety  ",
            (
                ("collision", "collision"),
                ("clawback", "failure_progress_clawback"),
                ("terminal", "termination"),
                ("early", "early_termination"),
            ),
        ),
    ):
        lines.append(
            f"  reward/step {group} "
            + " ".join(
                f"{label}="
                f"{_metric(metrics, f'eval/avg_reward_{name}'):+.4f}"
                for label, name in reward_labels
            )
        )
    lines.append(
        "  failures "
        f"low={_metric(metrics, 'eval/episode_failure_root_low'):.1%} "
        f"high={_metric(metrics, 'eval/episode_failure_root_high'):.1%} "
        f"lat={_metric(metrics, 'eval/episode_failure_lateral_drift'):.1%} "
        f"lat+={_metric(metrics, 'eval/episode_failure_lateral_positive'):.1%} "
        f"lat-={_metric(metrics, 'eval/episode_failure_lateral_negative'):.1%} "
        f"tilt={_metric(metrics, 'eval/episode_failure_axis_tilt'):.1%} "
        f"depth={_metric(metrics, 'eval/episode_failure_forbidden_depth'):.1%} "
        f"contact="
        f"{_metric(metrics, 'eval/episode_failure_forbidden_contact'):.1%}"
    )
    lines.append(
        "  numerics "
        f"nan={_metric(metrics, 'eval/episode_failure_nonfinite'):.1%} "
        f"action_nan="
        f"{_metric(metrics, 'eval/episode_failure_nonfinite_action'):.1%} "
        f"physics_nan="
        f"{_metric(metrics, 'eval/episode_failure_nonfinite_physics'):.1%}"
    )
    if "training/sps" in metrics:
        lines.append(
            "  ppo     "
            f"sps={_metric(metrics, 'training/sps'):.0f} "
            f"kl={_metric(metrics, 'training/kl_mean'):.4f} "
            f"policy_loss={_metric(metrics, 'training/policy_loss'):+.4f} "
            f"value_loss={_metric(metrics, 'training/v_loss'):.4f} "
            f"mean_std={_metric(metrics, 'training/policy_dist_mean_std'):.3f}"
        )
    return "\n".join(lines)


def _format_rollout_report_3d(label, summary):
    failures = [
        name
        for name, failed in summary["failure_reasons"].items()
        if failed
    ]
    failure_text = ",".join(failures) if failures else "none"
    terms = summary["reward_breakdown"]["terms"]
    return "\n".join(
        [
            f"[policy {label}]",
            (
                f"  outcome reward={summary['total_reward']:+.3f} "
                f"steps={summary['episode_steps']} "
                f"time={summary['episode_duration_s']:.2f}s "
                f"failure={failure_text}"
            ),
            (
                f"  motion  turns={summary['net_turns']:+.3f} "
                f"final_dx={summary['root_x_displacement_m']:+.3f}m "
                f"final_dy={summary['final_lateral_drift_m']:+.3f}m "
                f"mean_axis_tilt={summary['mean_axis_tilt_rad']:.3f}rad"
            ),
            (
                "  reward  "
                + " ".join(
                    f"{name}={float(terms.get(name, 0.0)):+.3f}"
                    for name in REWARD_3D_TERM_NAMES
                )
            ),
        ]
    )


def _evaluate_policy_3d(
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
    reward_term_totals = {name: 0.0 for name in REWARD_3D_TERM_NAMES}

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
            if name.startswith("reward_") and name != "reward_total":
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

    final_x = _float(state.pipeline_state.qpos[0])
    final_y = _float(state.pipeline_state.qpos[1])
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
            "failure_lateral_drift",
            "failure_axis_tilt",
            "failure_forbidden_depth",
            "failure_forbidden_contact",
        )
    }
    roll_progress = metric_totals.get("roll_progress_rad", 0.0)
    summary = {
        "episode_steps": steps,
        "episode_duration_s": (
            steps
            * float(env.mj_model.opt.timestep)
            * env.config.action_repeat
        ),
        "total_reward": float(sum(reward_rows)),
        "net_turns": roll_progress / (2.0 * math.pi),
        "translation_turns": (
            metric_totals.get("translation_progress_rad", 0.0)
            / (2.0 * math.pi)
        ),
        "rotation_turns": (
            metric_totals.get("rotation_progress_rad", 0.0)
            / (2.0 * math.pi)
        ),
        "root_x_displacement_m": final_x - initial_x,
        "final_lateral_drift_m": final_y - initial_y,
        "mean_lateral_drift_m": metric_averages.get(
            "lateral_drift_m", 0.0
        ),
        "mean_axis_tilt_rad": metric_averages.get("axis_tilt_rad", 0.0),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument(
        "--geometry",
        choices=GEOMETRY_NAMES_3D,
        default="rollingquad_2_primitive",
        help=(
            "Select the corrected 12-joint RollingQuad geometry, the older "
            "Pupper geometry, or a legacy 8-joint rolling geometry. The "
            "default rollingquad_2_primitive uses analytic capsule/cylinder/"
            "sphere collision primitives (a 300-degree capsule-arc shell plus "
            "leg primitives) for the fastest large-scale training; "
            "rollingquad_2_simple_convex uses simplified convex-hull meshes; "
            "rollingquad_2 uses the full STL collision meshes."
        ),
    )
    parser.add_argument(
        "--recipe",
        choices=tuple(RECIPES_3D),
        default="anchored_v1",
        help=(
            "3-D training recipe. anchored_v1 reproduces the first run; "
            "phase_locked_v3 starts residual learning from the restored "
            "phase-locked reference; phase_locked_equivariant_v9 enforces "
            "left/right reflection symmetry in residual learning; "
            "robust_recovery_v15 learns recovery from independent reset "
            "noise while preserving the CEM reference."
        ),
    )
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILE_NAMES_3D,
        default="cg12",
    )
    parser.add_argument(
        "--curriculum",
        choices=CURRICULUM_NAMES_3D,
        default="none",
        help=(
            "Robustness curriculum. reset_v1 preserves the original reset "
            "schedule; reset_v2 resolves the measured axis-tilt failure "
            "cliff; nominal_reset_v3 gradually introduces the nominal "
            "eight-joint independent reset noise through paired stages; "
            "independent_reset_v4 keeps all eight joints independent while "
            "ramping only q/qdot noise magnitude; floor_friction_v2 "
            "continues from the accepted independent_reset_v4 checkpoint "
            "and expands only floor-contact friction; floor_mass_v2 "
            "continues from floor_friction_v2 and expands coupled per-body "
            "mass and inertia while retaining its reset and floor-friction "
            "distributions; floor_mass_gain_v3 continues from floor_mass_v2 "
            "and expands independent per-actuator position gain while "
            "retaining the accepted mass and friction distributions; "
            "friction_v1 warm-starts "
            "from an accepted reset_v2 "
            "checkpoint and expands only global geom friction; mass_v1 "
            "continues from friction_v1 and expands coupled per-body mass "
            "and inertia while retaining friction randomization; "
            "robustness_v1 is the legacy reset_v1 plus physics schedule."
        ),
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=CURRICULUM_STAGE_NAMES_3D,
        help="Run one stage for an ablation instead of the full curriculum.",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    add_stand_startup_arguments(parser)
    parser.add_argument("--reset-joint-noise-rad", type=float, default=0.005)
    parser.add_argument("--reset-velocity-noise", type=float, default=0.005)
    parser.add_argument("--reset-root-velocity-noise", type=float, default=0.0)
    parser.add_argument("--reset-pair-differential-scale", type=float)
    parser.add_argument(
        "--reset-axis-tilt-noise-rad", type=float, default=0.0
    )
    parser.add_argument("--terminate-root-z-min", type=float, default=0.025)
    parser.add_argument(
        "--terminate-root-z-low-duration", type=float, default=0.20
    )
    parser.add_argument(
        "--no-root-low-termination",
        action="store_true",
        help="Disable continuous low-root termination for compatibility runs.",
    )
    parser.add_argument("--terminate-root-z-max", type=float, default=0.80)
    parser.add_argument("--terminate-lateral-drift", type=float, default=0.20)
    parser.add_argument("--terminate-axis-tilt", type=float, default=0.50)
    parser.add_argument(
        "--terminate-axis-tilt-duration", type=float, default=0.10
    )
    parser.add_argument(
        "--terminate-forbidden-depth", type=float, default=0.004
    )
    parser.add_argument(
        "--terminate-forbidden-contact-duration",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--controller",
        type=Path,
        default=None,
        help="CEM reference; defaults to the controller matched to --geometry.",
    )
    parser.add_argument("--minimum-foot-gap-mm", type=float)
    parser.add_argument("--foot-gap-tracking-margin-mm", type=float)
    parser.add_argument("--reference-weight", type=float)
    parser.add_argument("--minimum-residual-gain", type=float)
    parser.add_argument("--phase-rate-scale", type=float)
    parser.add_argument("--reference-action-scale", type=float, default=1.0)
    parser.add_argument("--reference-ramp-start-scale", type=float, default=0.0)
    parser.add_argument("--reference-ramp-duration-s", type=float, default=0.25)
    parser.add_argument("--reference-startup-boost", type=float, default=0.0)
    parser.add_argument(
        "--reference-startup-boost-duration-s",
        type=float,
        default=0.25,
    )
    parser.add_argument("--residual-pair-differential-scale", type=float)
    parser.add_argument(
        "--lateral-reflex-gain",
        type=float,
        help=(
            "Fixed lateral-stability reflex differential authority. 0 "
            "disables the reflex; 0.25 matches the validated reference "
            "authority."
        ),
    )
    parser.add_argument(
        "--lateral-reflex-position-gain",
        type=float,
        help="Reflex position feedback k in d = k*y + kw*vy (per metre).",
    )
    parser.add_argument(
        "--lateral-reflex-velocity-gain",
        type=float,
        help="Reflex velocity feedback kw in d = k*y + kw*vy (per m/s).",
    )
    parser.add_argument(
        "--lateral-command-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Sample a lateral velocity command at reset (turning). Defaults "
            "from the selected recipe."
        ),
    )
    parser.add_argument(
        "--lateral-command-max",
        type=float,
        help="Maximum turning command magnitude (m/s).",
    )
    parser.add_argument(
        "--lateral-command-probability",
        type=float,
        help="Probability a reset samples a non-zero turning command.",
    )
    parser.add_argument(
        "--lateral-command-error-limit",
        type=float,
        help="Turning termination threshold on |vy - vy_cmd| (m/s).",
    )
    parser.add_argument(
        "--explicit-phase-observation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-cost", type=float)
    parser.add_argument(
        "--zero-residual-policy-init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Initialize the residual policy with exactly zero deterministic "
            "action. The selected recipe supplies the default."
        ),
    )
    parser.add_argument(
        "--reflection-equivariant-policy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Project common residual outputs to be reflection-even and "
            "differential outputs to be reflection-odd."
        ),
    )
    parser.add_argument(
        "--differential-mean-zero-weight",
        type=float,
        help=(
            "Penalize the squared batch-mean differential mode, suppressing a "
            "constant left/right bias while allowing reflection-odd lateral "
            "feedback to learn."
        ),
    )
    parser.add_argument(
        "--initial-policy-std",
        type=float,
        help="Initial pre-tanh policy standard deviation.",
    )
    parser.add_argument(
        "--deterministic-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the deterministic policy for periodic evaluation and best "
            "checkpoint selection."
        ),
    )
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
        choices=("auto", "egl", "glfw", "osmesa", "disable"),
        default="auto",
        help="Defaults to EGL on a headless Linux instance.",
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
        "--restore-params",
        type=Path,
        help=(
            "Warm-start from a Brax params_best/params_final file. Unlike "
            "--restore-checkpoint, this does not require an Orbax directory."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "mjx_3d_residual_nominal",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Explicitly allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--save-ppo-checkpoints",
        action="store_true",
        help=(
            "Save Brax/Orbax checkpoints during training. This is disabled "
            "by default because those checkpoints can exceed tight disk quotas."
        ),
    )
    parser.add_argument(
        "--ppo-checkpoint-dir",
        type=Path,
        help=(
            "Optional directory for periodic Brax/Orbax checkpoints. "
            "Requires --save-ppo-checkpoints."
        ),
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--selection-target-turns", type=float)
    parser.add_argument(
        "--selection-objective",
        choices=("balanced", "contact"),
        default=None,
        help=(
            "Checkpoint ranking. The real_geometry_contact_v1 recipe uses "
            "a five-turn safety gate followed by contact-priority ranking."
        ),
    )
    _add_reward_arguments(parser)
    return parser


def _apply_recipe_defaults(args) -> None:
    for name, value in RECIPES_3D[args.recipe]["args"].items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_recipe_defaults(args)
    if args.reflection_equivariant_policy is None:
        args.reflection_equivariant_policy = False
    if args.differential_mean_zero_weight is None:
        args.differential_mean_zero_weight = 0.0
    if args.lateral_reflex_gain is None:
        args.lateral_reflex_gain = 0.0
    if args.lateral_reflex_position_gain is None:
        args.lateral_reflex_position_gain = 2.0
    if args.lateral_reflex_velocity_gain is None:
        args.lateral_reflex_velocity_gain = 2.0
    if args.lateral_reflex_gain < 0.0:
        parser.error("--lateral-reflex-gain must be nonnegative")
    if args.lateral_command_enabled is None:
        args.lateral_command_enabled = False
    if args.lateral_command_max is None:
        args.lateral_command_max = 0.15
    if args.lateral_command_probability is None:
        args.lateral_command_probability = 0.20
    if args.lateral_command_error_limit is None:
        args.lateral_command_error_limit = 0.20
    if args.differential_mean_zero_weight < 0.0:
        parser.error("--differential-mean-zero-weight must be nonnegative")
    if (
        args.reflection_equivariant_policy
        and args.differential_mean_zero_weight > 0.0
    ):
        parser.error(
            "--reflection-equivariant-policy and "
            "--differential-mean-zero-weight are mutually exclusive."
        )
    if args.controller is None:
        args.controller = cem_controller_path_3d(args.geometry)
    if args.selection_objective is None:
        args.selection_objective = "balanced"
    if not 0.0 <= args.reference_weight <= 1.0:
        parser.error("--reference-weight must be in [0, 1]")
    if not 0.0 <= args.minimum_residual_gain <= 1.0:
        parser.error("--minimum-residual-gain must be in [0, 1]")
    if (
        not math.isfinite(args.reference_action_scale)
        or args.reference_action_scale <= 0.0
    ):
        parser.error("--reference-action-scale must be positive")
    if args.reference_ramp_start_scale is not None:
        if (
            not math.isfinite(args.reference_ramp_start_scale)
            or args.reference_ramp_start_scale < 0.0
        ):
            parser.error("--reference-ramp-start-scale must be nonnegative")
    if (
        not math.isfinite(args.reference_ramp_duration_s)
        or args.reference_ramp_duration_s <= 0.0
    ):
        parser.error("--reference-ramp-duration-s must be positive")
    if (
        not math.isfinite(args.reference_startup_boost)
        or args.reference_startup_boost < 0.0
    ):
        parser.error("--reference-startup-boost must be nonnegative")
    if (
        not math.isfinite(args.reference_startup_boost_duration_s)
        or args.reference_startup_boost_duration_s <= 0.0
    ):
        parser.error("--reference-startup-boost-duration-s must be positive")
    if (
        args.residual_pair_differential_scale is not None
        and not 0.0 <= args.residual_pair_differential_scale <= 1.0
    ):
        parser.error("--residual-pair-differential-scale must be in [0, 1]")
    if args.ppo_checkpoint_dir is not None and not args.save_ppo_checkpoints:
        parser.error("--ppo-checkpoint-dir requires --save-ppo-checkpoints")
    if args.restore_checkpoint is not None and args.restore_params is not None:
        parser.error(
            "--restore-checkpoint and --restore-params are mutually exclusive"
        )
    try:
        curriculum_stages_3d(
            args.curriculum,
            only_stage=args.curriculum_stage,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.selection_target_turns <= 0.0:
        parser.error("--selection-target-turns must be positive")
    for value, name in (
        (args.minimum_foot_gap_mm, "--minimum-foot-gap-mm"),
        (
            args.foot_gap_tracking_margin_mm,
            "--foot-gap-tracking-margin-mm",
        ),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(f"{name} must be finite and nonnegative")
    for value, name in (
        (args.reset_joint_noise_rad, "--reset-joint-noise-rad"),
        (args.reset_velocity_noise, "--reset-velocity-noise"),
        (args.reset_root_velocity_noise, "--reset-root-velocity-noise"),
        (args.reset_axis_tilt_noise_rad, "--reset-axis-tilt-noise-rad"),
    ):
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{name} must be finite and nonnegative")
    if (
        args.reset_pair_differential_scale is not None
        and not 0.0 <= args.reset_pair_differential_scale <= 1.0
    ):
        parser.error("--reset-pair-differential-scale must be in [0, 1]")
    if args.initial_policy_std <= TANH_NORMAL_MIN_STD:
        parser.error(
            "--initial-policy-std must be greater than "
            f"{TANH_NORMAL_MIN_STD:g}"
        )
    for name in (
        "episode_length",
        "unroll_length",
        "updates_per_batch",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
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
    curriculum_plan = _curriculum_training_plan(args, values)
    update_counts = _ppo_update_counts(
        curriculum_plan,
        updates_per_batch=args.updates_per_batch,
        num_minibatches=values["num_minibatches"],
    )
    total_curriculum_evals = sum(
        item["num_evals"] for item in curriculum_plan
    )
    schedule = {
        "requested_steps": values["steps"],
        "effective_steps": sum(
            item["schedule"]["effective_steps"]
            for item in curriculum_plan
        ),
        "rollout_quantum": curriculum_plan[0]["schedule"][
            "rollout_quantum"
        ],
        **update_counts,
        "stage_count": len(curriculum_plan),
        "stages": [
            {
                "name": item["stage"].name,
                "weight": item["stage"].weight,
                "num_evals": item["num_evals"],
                **item["schedule"],
            }
            for item in curriculum_plan
        ],
    }
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
        verbose=False,
    )
    import jax
    from brax.io import model as model_io
    from brax.training.agents.ppo import losses as ppo_losses
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d

    args.out.mkdir(parents=True, exist_ok=True)
    runtime = describe_runtime()
    if args.runtime_diagnostics:
        print(
            "[runtime]\n"
            f"  python={runtime['python_version']} "
            f"jax={runtime['jax_version']} backend={runtime['backend']}\n"
            f"  devices={', '.join(runtime['devices'])}\n"
            f"  mujoco_gl={runtime['mujoco_gl']} "
            f"memory_fraction={runtime['memory_fraction']}\n"
            f"  compilation_cache={runtime['compilation_cache']}",
            flush=True,
        )

    task = physics_profile_3d(
        args.physics_profile,
        Rolling3DConfig(
            geometry=args.geometry,
            episode_length=args.episode_length,
            reset_joint_noise_rad=args.reset_joint_noise_rad,
            reset_velocity_noise=args.reset_velocity_noise,
            reset_root_velocity_noise=args.reset_root_velocity_noise,
            reset_pair_differential_scale=(
                args.reset_pair_differential_scale
            ),
            reset_axis_tilt_noise_rad=args.reset_axis_tilt_noise_rad,
            reference_phase_rate_scale=args.phase_rate_scale,
            reference_action_scale=args.reference_action_scale,
            reference_ramp_start_scale=args.reference_ramp_start_scale,
            reference_ramp_duration_s=args.reference_ramp_duration_s,
            reference_startup_boost=args.reference_startup_boost,
            reference_startup_boost_duration_s=(
                args.reference_startup_boost_duration_s
            ),
            residual_pair_differential_scale=(
                args.residual_pair_differential_scale
            ),
            lateral_reflex_gain=args.lateral_reflex_gain,
            lateral_reflex_position_gain=args.lateral_reflex_position_gain,
            lateral_reflex_velocity_gain=args.lateral_reflex_velocity_gain,
            lateral_command_enabled=args.lateral_command_enabled,
            lateral_command_max=args.lateral_command_max,
            lateral_command_probability=args.lateral_command_probability,
            lateral_command_error_limit=args.lateral_command_error_limit,
            explicit_phase_observation=bool(
                args.explicit_phase_observation
            ),
            terminate_root_z_min=(
                None
                if args.no_root_low_termination
                else args.terminate_root_z_min
            ),
            terminate_root_z_low_duration_s=(
                args.terminate_root_z_low_duration
            ),
            terminate_root_z_max=args.terminate_root_z_max,
            terminate_lateral_drift_m=args.terminate_lateral_drift,
            terminate_axis_tilt_rad=args.terminate_axis_tilt,
            terminate_axis_tilt_duration_s=(
                args.terminate_axis_tilt_duration
            ),
            terminate_forbidden_depth_m=args.terminate_forbidden_depth,
            terminate_forbidden_contact_duration_s=(
                args.terminate_forbidden_contact_duration
            ),
        ),
    )
    task = with_stand_startup(task, args)
    print(f"[startup] reset={task.reset_pose} rolling_start={task.rolling_start_time_s:g}s; "
          "startup INCLUDED in episode, compact action origin unchanged", flush=True)
    reward_config = _reward_config_from_args(args)
    reference = load_cem_reference(
        args.controller,
        reference_weight=args.reference_weight,
        minimum_residual_gain=args.minimum_residual_gain,
        minimum_foot_surface_gap_m=(
            None
            if args.minimum_foot_gap_mm is None
            else args.minimum_foot_gap_mm / 1000.0
        ),
        foot_gap_tracking_margin_m=(
            None
            if args.foot_gap_tracking_margin_mm is None
            else args.foot_gap_tracking_margin_mm / 1000.0
        ),
    )
    metric_history = []
    reward_history = []

    config_payload = {
        "preset": args.preset,
        "recipe": args.recipe,
        "geometry": args.geometry,
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
        "zero_residual_policy_init": args.zero_residual_policy_init,
        "reflection_equivariant_policy": (
            args.reflection_equivariant_policy
        ),
        "initial_policy_std": args.initial_policy_std,
        "deterministic_eval": args.deterministic_eval,
        "seed": args.seed,
        "restore_checkpoint": (
            str(args.restore_checkpoint)
            if args.restore_checkpoint is not None
            else None
        ),
        "restore_params": (
            str(args.restore_params)
            if args.restore_params is not None
            else None
        ),
        "save_ppo_checkpoints": args.save_ppo_checkpoints,
        "ppo_checkpoint_dir": (
            str(args.ppo_checkpoint_dir)
            if args.ppo_checkpoint_dir is not None
            else None
        ),
        "task": asdict(task),
        "reward": asdict(reward_config),
        "reference": asdict(reference),
        "runtime": runtime,
        "evaluation": {
            "skip": args.skip_evaluation,
            "post_training_domain_randomization": False,
        },
        "curriculum": {
            "name": args.curriculum,
            "only_stage": args.curriculum_stage,
            "selection_scope": "final_stage",
            "reset_resampling": (
                "stage_start"
                if args.curriculum == "none"
                else "stage_start_and_each_eval_interval"
            ),
            "stages": [
                {
                    "index": index,
                    "name": item["stage"].name,
                    "weight": item["stage"].weight,
                    "task": asdict(item["stage"].task_config(task)),
                    "domain_randomization": asdict(
                        item["stage"].domain_randomization
                    ),
                    "num_evals": item["num_evals"],
                    "schedule": item["schedule"],
                }
                for index, item in enumerate(curriculum_plan)
            ],
        },
        "checkpoint_selection": {
            "description": (
                "hard target-turn gate, then contact-priority physical "
                "ranking"
                if args.selection_objective == "contact"
                else (
                    "0.20 survival + 0.25 forward turns + 0.30 "
                    "non-failure + 0.15 lateral stability + 0.05 axis "
                    "stability + 0.05 forbidden-contact safety"
                )
            ),
            "target_turns": args.selection_target_turns,
            "objective": args.selection_objective,
            "lateral_full_score_m": 0.05,
            "axis_tilt_full_score_rad": 0.25,
            "forbidden_depth_limit_m": 0.001,
            "ranking_maximum_failure_rate": 1.0,
            "acceptance_maximum_failure_rate": (
                MAX_CHECKPOINT_FAILURE_RATE
            ),
            "reject_nonfinite": True,
        },
        "training_step_schedule": schedule,
    }
    (args.out / "training_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "reward_config.json").write_text(
        json.dumps(asdict(reward_config), indent=2) + "\n",
        encoding="utf-8",
    )
    root_low_text = (
        "disabled"
        if task.terminate_root_z_min is None
        else (
            f"{task.terminate_root_z_min:g}m/"
            f"{task.terminate_root_z_low_duration_s:g}s"
        )
    )
    policy_init_text = (
        f"zero-residual/std={args.initial_policy_std:g}"
        if args.zero_residual_policy_init
        else "brax-default"
    )
    if args.reflection_equivariant_policy:
        policy_symmetry_text = "reflection-equivariant"
    elif args.differential_mean_zero_weight > 0.0:
        policy_symmetry_text = (
            f"mean-zero-differential/"
            f"{args.differential_mean_zero_weight:g}"
        )
    else:
        policy_symmetry_text = "unconstrained"
    residual_pair_text = (
        "direct"
        if task.residual_pair_differential_scale is None
        else (
            "common-differential/"
            f"{task.residual_pair_differential_scale:g}"
        )
    )
    reflex_text = (
        "disabled"
        if task.lateral_reflex_gain == 0.0
        else (
            f"gain={task.lateral_reflex_gain:g} "
            f"k={task.lateral_reflex_position_gain:g} "
            f"kw={task.lateral_reflex_velocity_gain:g}"
        )
    )
    observation_size = OBSERVATION_SIZE_3D + (
        PHASE_FEEDBACK_SIZE_3D if task.explicit_phase_observation else 0
    )
    curriculum_text = ", ".join(
        (
            f"{item['stage'].name}:"
            f"{item['schedule']['effective_steps']:,}/"
            f"{item['num_evals']}eval"
        )
        for item in curriculum_plan
    )
    print(
        "[training]\n"
        f"  preset={args.preset} recipe={args.recipe} "
        f"geometry={args.geometry} "
        f"physics={args.physics_profile} "
        f"requested_steps={schedule['requested_steps']:,} "
        f"effective_steps={schedule['effective_steps']:,}\n"
        f"  rollout_quantum={schedule['rollout_quantum']:,} "
        f"stages={len(curriculum_plan)} "
        f"evals={total_curriculum_evals}\n"
        f"  ppo_rollout_updates={schedule['rollout_updates']:,} "
        f"data_reuse_passes={schedule['data_reuse_passes']:,} "
        f"optimizer_steps={schedule['optimizer_steps']:,}\n"
        f"  curriculum={args.curriculum} [{curriculum_text}]\n"
        f"  envs={values['envs']} eval_envs={values['eval_envs']} "
        f"batch={values['batch_size']} "
        f"minibatches={values['num_minibatches']}\n"
        f"  episode={args.episode_length} steps "
        f"({args.episode_length * task.control_timestep:.2f}s) "
        f"root_low={root_low_text} "
        f"lat_limit={task.terminate_lateral_drift_m:g}m "
        f"axis_limit={task.terminate_axis_tilt_rad:g}rad\n"
        f"  recipe_note={RECIPES_3D[args.recipe]['description']}\n"
        f"  reference_weight={reference.reference_weight:.2f} "
        f"residual_gain={reference.residual_gain:.3f} "
        f"phase_rate_scale={task.reference_phase_rate_scale:g}\n"
        f"  reference_action_scale={task.reference_action_scale:g} "
        f"ramp_start={task.reference_ramp_start_scale} "
        f"ramp_duration={task.reference_ramp_duration_s:g}s "
        f"startup_boost={task.reference_startup_boost:g} "
        f"startup_boost_duration={task.reference_startup_boost_duration_s:g}s\n"
        f"  residual_channels={residual_pair_text}\n"
        f"  lateral_reflex={reflex_text}\n"
        f"  explicit_phase_obs={task.explicit_phase_observation} "
        f"obs_size={observation_size}\n"
        f"  lr={args.learning_rate:g} entropy={args.entropy_cost:g} "
        f"discount={args.discounting:g} seed={args.seed}\n"
        f"  policy_init={policy_init_text}\n"
        f"  policy_symmetry={policy_symmetry_text}\n"
        f"  eval_policy={'deterministic' if args.deterministic_eval else 'stochastic'}\n"
        f"  reward roll={reward_config.roll_progress:g} "
        f"lat_cost={reward_config.lateral_drift_cost:g} "
        f"yaw_cost={reward_config.yaw_cost:g} "
        f"recovery={reward_config.recovery:g} "
        f"residual={reward_config.residual_action:g} "
        f"tilt={reward_config.axis_tilt:g} "
        f"collision_event={reward_config.foot_contact_event:g} "
        f"clawback={reward_config.failure_progress_clawback:g} "
        f"termination={reward_config.termination:g}\n"
        f"  selection={args.selection_objective}: survival, turns, lateral, "
        "axis, contact safety "
        f"(target_turns={args.selection_target_turns:g})\n"
        f"  controller={reference.source}\n"
        f"  output={args.out.resolve()}",
        flush=True,
    )

    start = time.perf_counter()
    train_parameters = inspect.signature(ppo.train).parameters
    if (
        len(curriculum_plan) > 1 or args.restore_params is not None
    ) and "restore_params" not in train_parameters:
        raise SystemExit(
            "Installed Brax does not support restore_params, which is "
            "required for staged curriculum training."
        )
    if any(
        item["stage"].domain_randomization.enabled
        for item in curriculum_plan
    ) and "randomization_fn" not in train_parameters:
        raise SystemExit(
            "Installed Brax does not support randomization_fn."
        )
    if (
        args.curriculum != "none"
        and "num_resets_per_eval" not in train_parameters
    ):
        raise SystemExit(
            "Installed Brax does not support num_resets_per_eval, which is "
            "required to refresh curriculum resets between eval intervals."
        )
    restore_checkpoint_path = None
    if args.restore_checkpoint is not None:
        if "restore_checkpoint_path" not in train_parameters:
            raise SystemExit(
                "Installed Brax does not support restore_checkpoint_path."
            )
        restore_checkpoint_path = str(
            _resolve_restore_checkpoint(args.restore_checkpoint)
        )
        print(
            "[checkpoint]\n"
            f"  restoring={restore_checkpoint_path}",
            flush=True,
        )

    network_factory = (
        _zero_centered_residual_network_factory(
            args.hidden_layers,
            args.activation,
            args.initial_policy_std,
            reflection_equivariant=args.reflection_equivariant_policy,
        )
        if args.zero_residual_policy_init
        else _network_factory(args.hidden_layers, args.activation)
    )
    restored_params = (
        model_io.load_params(args.restore_params)
        if args.restore_params is not None
        else None
    )
    if args.restore_params is not None:
        print(
            "[checkpoint]\n"
            f"  restoring_params={args.restore_params.resolve()}",
            flush=True,
        )
    make_inference_fn = None
    final_params = None
    final_metrics = {}
    best_params = None
    best_params_source = None
    best = None
    reward_peak = None
    eval_env = None
    stage_history = []
    eval_counter = {"value": 0}
    global_step_offset = 0

    for stage_index, stage_item in enumerate(curriculum_plan):
        stage = stage_item["stage"]
        stage_schedule = stage_item["schedule"]
        stage_task = stage.task_config(task)
        train_env = make_brax_env_3d(
            stage_task,
            reward_config=reward_config,
            cem_reference=reference,
            seed=args.seed + 100 * stage_index,
        )
        eval_env = make_brax_env_3d(
            stage_task,
            reward_config=reward_config,
            cem_reference=reference,
            seed=args.seed + 10_000 + 100 * stage_index,
        )
        randomization_fn = make_domain_randomization_fn_3d(
            stage.domain_randomization,
            floor_geom_id=train_env.floor_geom_id,
        )
        best = {
            "score": float("-inf"),
            "reward": float("-inf"),
            "step": None,
            "params": None,
            "params_step": None,
            "candidate_step": None,
            "candidate_params": None,
            "selection": None,
        }
        initial_evaluated = {
            "step": None,
            "params": None,
            "params_step": None,
        }
        reward_peak = {"reward": float("-inf"), "step": None}
        stage_metric_history = []
        domain = stage.domain_randomization
        print(
            f"[curriculum stage {stage_index + 1}/{len(curriculum_plan)}] "
            f"{stage.name}\n"
            f"  steps={stage_schedule['effective_steps']:,} "
            f"evals={stage_item['num_evals']} "
            f"global_start={global_step_offset:,}\n"
            f"  reset joint={stage_task.reset_joint_noise_rad:g}rad "
            f"joint_velocity={stage_task.reset_velocity_noise:g} "
            f"root_velocity={stage_task.reset_root_velocity_noise:g} "
            f"differential={stage_task.reset_pair_differential_scale} "
            f"tilt={stage_task.reset_axis_tilt_noise_rad:g}rad\n"
            f"  randomization global_friction={domain.geom_friction_scale} "
            f"floor_friction={domain.floor_friction_scale} "
            f"mass={domain.body_mass_scale} "
            f"actuator_gain={domain.actuator_gain_scale}",
            flush=True,
        )

        def policy_params_fn(step, make_policy, params):
            del make_policy
            global_step = global_step_offset + int(step)
            params_snapshot = jax.tree_util.tree_map(
                lambda value: np.asarray(value).copy(), params
            )
            best["candidate_step"] = global_step
            best["candidate_params"] = params_snapshot
            if best["step"] == global_step:
                best["params"] = params_snapshot
                best["params_step"] = global_step
            if initial_evaluated["step"] == global_step:
                initial_evaluated["params"] = params_snapshot
                initial_evaluated["params_step"] = global_step

        def progress_fn(step, metrics):
            global_step = global_step_offset + int(step)
            clean = {name: _float(value) for name, value in metrics.items()}
            _add_per_step_eval_metrics_3d(clean)
            reward_metrics, ordinary_metrics = _split_metrics(clean)
            record_prefix = {
                "step": global_step,
                "stage_index": stage_index,
                "stage_name": stage.name,
                "stage_step": int(step),
            }
            reward_history.append({**record_prefix, **reward_metrics})
            metric_record = {**record_prefix, **ordinary_metrics}
            metric_history.append(metric_record)
            stage_metric_history.append(metric_record)
            reward = clean.get(
                "eval/episode_reward",
                clean.get("eval/episode_reward_mean"),
            )
            if reward is not None and reward > reward_peak["reward"]:
                reward_peak["reward"] = reward
                reward_peak["step"] = global_step
            selection = _checkpoint_selection_3d(
                clean,
                args.episode_length,
                target_turns=args.selection_target_turns,
                # Rank finite intermediate policies even before they meet the
                # independent five-percent deployment acceptance threshold.
                maximum_failure_rate=1.0,
                objective=args.selection_objective,
            )
            if initial_evaluated["step"] is None:
                initial_evaluated["step"] = global_step
                if best["candidate_step"] == global_step:
                    initial_evaluated["params"] = best["candidate_params"]
                    initial_evaluated["params_step"] = global_step
            selected = (
                not selection["rejected"]
                and selection["score"] > best["score"]
            )
            if selected:
                best["score"] = selection["score"]
                best["reward"] = reward
                best["step"] = global_step
                best["selection"] = selection
                if best["candidate_step"] == global_step:
                    best["params"] = best["candidate_params"]
                    best["params_step"] = global_step
            eval_counter["value"] += 1
            print(
                _format_eval_report_3d(
                    eval_counter["value"],
                    total_curriculum_evals,
                    global_step,
                    clean,
                    episode_length=args.episode_length,
                    control_dt=stage_task.control_timestep,
                    selection=selection,
                    selected=selected,
                ),
                flush=True,
            )

        train_kwargs = {}
        if restored_params is not None:
            train_kwargs["restore_params"] = restored_params
        elif restore_checkpoint_path is not None:
            train_kwargs["restore_checkpoint_path"] = (
                restore_checkpoint_path
            )
        if randomization_fn is not None:
            train_kwargs["randomization_fn"] = randomization_fn
        if args.curriculum != "none":
            train_kwargs["num_resets_per_eval"] = 1
        if args.save_ppo_checkpoints:
            if "save_checkpoint_path" not in train_parameters:
                raise SystemExit(
                    "Installed Brax does not support save_checkpoint_path."
                )
            checkpoint_dir = args.ppo_checkpoint_dir or (
                args.out / "ppo_checkpoint"
            )
            if len(curriculum_plan) > 1:
                checkpoint_dir = (
                    checkpoint_dir / f"stage_{stage_index}_{stage.name}"
                )
            train_kwargs["save_checkpoint_path"] = str(
                checkpoint_dir.resolve()
            )
            print(
                "[checkpoint]\n"
                f"  periodic_save={train_kwargs['save_checkpoint_path']}",
                flush=True,
            )

        with _differential_mean_zero_loss_scope(
            ppo_losses,
            weight=args.differential_mean_zero_weight,
            array_module=jax.numpy,
        ):
            (
                make_inference_fn,
                stage_final_params,
                stage_final_metrics,
            ) = ppo.train(
                environment=train_env,
                eval_env=eval_env,
                num_timesteps=stage_schedule["requested_steps"],
                episode_length=args.episode_length,
                action_repeat=1,
                num_envs=values["envs"],
                num_evals=stage_item["num_evals"],
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
                deterministic_eval=args.deterministic_eval,
                network_factory=network_factory,
                seed=args.seed + stage_index,
                progress_fn=progress_fn,
                policy_params_fn=policy_params_fn,
                **train_kwargs,
            )
        stage_best_params, stage_best_source = _resolve_best_params(
            best,
            stage_final_params,
            stage_metric_history,
            final_step=(
                global_step_offset + stage_schedule["effective_steps"]
            ),
            initial_evaluated=initial_evaluated,
        )
        clean_stage_final_metrics = {
            name: _float(value)
            for name, value in (stage_final_metrics or {}).items()
        }
        _add_per_step_eval_metrics_3d(clean_stage_final_metrics)
        stage_history.append(
            {
                "index": stage_index,
                "name": stage.name,
                "global_start_step": global_step_offset,
                "effective_steps": stage_schedule["effective_steps"],
                "best_step": best["step"],
                "best_score": (
                    best["score"] if math.isfinite(best["score"]) else None
                ),
                "best_reward": (
                    best["reward"] if math.isfinite(best["reward"]) else None
                ),
                "best_params_source": stage_best_source,
                "retained_initial_eval_step": initial_evaluated["step"],
                "best_failure_rate": (
                    best["selection"]["failure_rate"]
                    if best["selection"] is not None
                    else None
                ),
                "best_passes_acceptance": (
                    best["selection"][
                        "passes_acceptance_failure_rate"
                    ]
                    if best["selection"] is not None
                    else False
                ),
                "final_metrics": clean_stage_final_metrics,
            }
        )
        if len(curriculum_plan) > 1:
            stage_prefix = args.out / f"params_stage_{stage_index}_{stage.name}"
            model_io.save_params(
                Path(f"{stage_prefix}_final"), stage_final_params
            )
            model_io.save_params(
                Path(f"{stage_prefix}_best"), stage_best_params
            )
        restored_params = stage_best_params
        final_params = stage_final_params
        final_metrics = stage_final_metrics or {}
        best_params = stage_best_params
        best_params_source = stage_best_source
        global_step_offset += stage_schedule["effective_steps"]

    elapsed = time.perf_counter() - start
    model_io.save_params(args.out / "params_final", final_params)
    model_io.save_params(args.out / "params_best", best_params)
    (args.out / "curriculum_history.json").write_text(
        json.dumps(stage_history, indent=2) + "\n", encoding="utf-8"
    )
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
    _add_per_step_eval_metrics_3d(clean_final_metrics)
    final_reward_metrics, final_ordinary_metrics = _split_metrics(
        clean_final_metrics
    )
    best_reward_value = (
        best["reward"] if math.isfinite(best["reward"]) else None
    )
    best_score_value = (
        best["score"] if math.isfinite(best["score"]) else None
    )
    reward_peak_value = (
        reward_peak["reward"]
        if math.isfinite(reward_peak["reward"])
        else None
    )
    train_summary = {
        "elapsed_s": elapsed,
        "curriculum": args.curriculum,
        "final_stage": stage_history[-1]["name"],
        "stages": stage_history,
        "best_eval_reward": best_reward_value,
        "best_step": best["step"],
        "best_selection_score": best_score_value,
        "best_params_source": best_params_source,
        "best_passes_acceptance": stage_history[-1][
            "best_passes_acceptance"
        ],
        "reward_peak": reward_peak_value,
        "reward_peak_step": reward_peak["step"],
        "final_metrics": final_ordinary_metrics,
        "final_reward_metrics": final_reward_metrics,
    }
    (args.out / "training_summary.json").write_text(
        json.dumps(train_summary, indent=2) + "\n", encoding="utf-8"
    )
    throughput = schedule["effective_steps"] / max(elapsed, 1e-9)
    best_source = (
        f"step={best['step']} score={best['score']:.4f} "
        f"reward={best['reward']:+.3f}"
        if best["step"] is not None
        else (
            "none (all eval points rejected; params_best retains "
            f"initial eval step={initial_evaluated['step']})"
        )
    )
    reward_peak_source = (
        f"step={reward_peak['step']} reward={reward_peak['reward']:+.3f}"
        if reward_peak["step"] is not None
        else "unavailable"
    )
    print(
        "[training complete]\n"
        f"  elapsed={elapsed / 60.0:.1f}min "
        f"throughput={throughput:,.0f} steps/s\n"
        f"  physical_best {best_source} "
        f"acceptance="
        f"{'PASS' if stage_history[-1]['best_passes_acceptance'] else 'NOT_YET'}\n"
        f"  best_params_source={best_params_source}\n"
        f"  reward_peak {reward_peak_source}\n"
        f"  checkpoints best={args.out / 'params_best'} "
        f"final={args.out / 'params_final'}",
        flush=True,
    )

    if not args.skip_evaluation:
        evaluation_best_dir = args.out / "evaluation_best"
        evaluation_final_dir = args.out / "evaluation_final"
        evaluation_best = _evaluate_policy_3d(
            eval_env,
            make_inference_fn,
            best_params,
            seed=args.seed + 20_000,
            episode_length=args.episode_length,
            output_dir=evaluation_best_dir,
        )
        shared_checkpoint = _best_and_final_share_checkpoint(
            best_params_source
        )
        if shared_checkpoint:
            evaluation_final_dir.mkdir(parents=True, exist_ok=True)
            for artifact_name in (
                "evaluation_rollout.npz",
                "evaluation_summary.json",
            ):
                shutil.copy2(
                    evaluation_best_dir / artifact_name,
                    evaluation_final_dir / artifact_name,
                )
            evaluation_final = evaluation_best
        else:
            evaluation_final = _evaluate_policy_3d(
                eval_env,
                make_inference_fn,
                final_params,
                seed=args.seed + 20_000,
                episode_length=args.episode_length,
                output_dir=evaluation_final_dir,
            )
        comparison = {
            "selection": {
                "best_step": best["step"],
                "best_selection_score": best_score_value,
                "reward_peak_step": reward_peak["step"],
                "best_params_source": best_params_source,
                "best_and_final_share_checkpoint": shared_checkpoint,
                "final_evaluation_reused": shared_checkpoint,
            },
            "best": evaluation_best,
            "final": evaluation_final,
        }
        (args.out / "policy_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(_format_rollout_report_3d("best", evaluation_best), flush=True)
        final_label = (
            "final (same checkpoint; reused rollout)"
            if shared_checkpoint
            else "final"
        )
        print(
            _format_rollout_report_3d(final_label, evaluation_final),
            flush=True,
        )
        print(
            "[artifacts]\n"
            f"  summary={args.out / 'training_summary.json'}\n"
            f"  comparison={args.out / 'policy_comparison.json'}\n"
            f"  best={evaluation_best_dir}\n"
            f"  final={evaluation_final_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
