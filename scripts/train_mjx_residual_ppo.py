"""Train residual PPO with a performance-gated CEM reference curriculum."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
import math
from pathlib import Path
import time

from curl_robot_2d_mjx.cem_reference import (
    DEFAULT_CEM_CONTROLLER,
    expected_budget_steps,
    load_cem_reference,
)
from curl_robot_2d_mjx.config import (
    PHYSICS_PROFILE_NAMES,
    NominalRLConfig,
    physics_profile,
    validate_nominal_rl_config,
)
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    describe_runtime,
)
from scripts.train_mjx_ppo import (
    _add_per_step_eval_metrics,
    _add_reward_arguments,
    _checkpoint_selection,
    _evaluate_policy,
    _float,
    _format_eval_report,
    _format_rollout_report,
    _network_factory,
    _reward_config_from_args,
    _split_metrics,
)


PRESETS = {
    "smoke": {
        "steps": 131_072,
        "envs": 64,
        "eval_envs": 8,
        "batch_size": 64,
        "num_minibatches": 4,
        "unroll_length": 16,
        "updates_per_batch": 2,
    },
    "h200": {
        "steps": 2_000_000,
        "envs": 2048,
        "eval_envs": 256,
        "batch_size": 512,
        "num_minibatches": 16,
        "unroll_length": 16,
        "updates_per_batch": 2,
    },
}


class _AdvanceCurriculum(RuntimeError):
    pass


def _gate_assessment(
    metrics,
    *,
    episode_length,
    minimum_survival,
    minimum_turns,
    maximum_failure_rate,
):
    avg_length = float(metrics.get("eval/avg_episode_length", 0.0))
    survival = min(max(avg_length / episode_length, 0.0), 1.0)
    turns = float(
        metrics.get("eval/episode_roll_progress_rad", -math.inf)
    ) / (2.0 * math.pi)
    failed = float(metrics.get("eval/episode_failed", 1.0))
    nonfinite = float(
        metrics.get("eval/episode_failure_nonfinite", 0.0)
    )
    checks = {
        "survival": survival >= minimum_survival,
        "turns": turns >= minimum_turns,
        "failure_rate": failed <= maximum_failure_rate,
        "finite": nonfinite == 0.0 and math.isfinite(turns),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "survival": survival,
        "turns": turns,
        "failure_rate": failed,
        "nonfinite_rate": nonfinite,
    }


def _exact_stage_eval_schedule(
    remaining_steps,
    rollout_quantum,
    gate_check_steps,
):
    """Choose equal Brax eval intervals without exceeding the fixed budget."""

    if remaining_steps % rollout_quantum:
        raise ValueError("remaining budget must align to rollout quantum")
    remaining_units = remaining_steps // rollout_quantum
    minimum_interval_units = max(
        1, math.ceil(gate_check_steps / rollout_quantum)
    )
    divisors = [
        units
        for units in range(minimum_interval_units, remaining_units + 1)
        if remaining_units % units == 0
    ]
    interval_units = divisors[0] if divisors else remaining_units
    intervals = remaining_units // interval_units
    return {
        "num_evals": intervals + 1,
        "eval_interval_steps": interval_units * rollout_quantum,
        "eval_intervals": intervals,
    }


def _validate_weights(parser, weights):
    if not weights or weights[-1] != 0.0:
        parser.error("--reference-weights must end at exactly 0")
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        parser.error("--reference-weights must stay in [0, 1]")
    if any(left <= right for left, right in zip(weights, weights[1:])):
        parser.error("--reference-weights must be strictly decreasing")


def _validate_residual_scales(parser, scales):
    if not scales:
        parser.error("--residual-scales must not be empty")
    if any(not 0.0 < scale <= 1.0 for scale in scales):
        parser.error("--residual-scales must stay in (0, 1]")
    if any(left >= right for left, right in zip(scales, scales[1:])):
        parser.error("--residual-scales must be strictly increasing")


def _stage_plan(args):
    if args.retain_cem:
        return [
            {
                "reference_weight": 1.0,
                "minimum_residual_gain": scale,
            }
            for scale in args.residual_scales
        ]
    return [
        {
            "reference_weight": weight,
            "minimum_residual_gain": args.minimum_residual_gain,
        }
        for weight in args.reference_weights
    ]


def _target_eval_eligible(
    stage_index, stage_count, local_step, minimum_stage_steps
):
    return (
        stage_index == stage_count - 1
        and int(local_step) >= minimum_stage_steps
    )


def _residual_checkpoint_rank(selection, metrics):
    """Rank safe residual checkpoints by motion before tie breakers."""

    return (
        float(selection["turns"]),
        float(selection["survival"]),
        -float(metrics.get("eval/episode_failed", 1.0)),
        -float(
            metrics.get("eval/avg_forbidden_penetration_m", math.inf)
        ),
    )


def _safe_stage_checkpoint(gate):
    return all(
        passed
        for name, passed in gate["checks"].items()
        if name != "turns"
    )


def _can_advance_stage(
    stage_index,
    stage_count,
    local_step,
    minimum_stage_steps,
    current_gate,
    best_gate,
):
    return (
        stage_index < stage_count - 1
        and int(local_step) >= minimum_stage_steps
        and (
            current_gate["passed"]
            or (best_gate is not None and best_gate["passed"])
        )
    )


def _last_trained_stage_spec(stage_plan, stage_history):
    if not stage_history:
        raise ValueError("stage history is empty")
    return stage_plan[int(stage_history[-1]["stage_index"])]


def _distribution_summary(values):
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("distribution must not be empty")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _evaluate_policy_distribution(
    env,
    make_inference_fn,
    params,
    *,
    seed,
    episode_length,
    batch_size,
):
    import jax
    import jax.numpy as jp
    import numpy as np

    try:
        policy = make_inference_fn(params, deterministic=True)
    except TypeError:
        policy = make_inference_fn(params)

    def policy_action(observation, key):
        action, _ = policy(observation, key)
        return action

    policy_batch = jax.jit(jax.vmap(policy_action))
    reset_batch = jax.jit(jax.vmap(env.reset))

    def step_one(state, action, active):
        return jax.lax.cond(
            active,
            lambda _: env.step(state, action),
            lambda _: state,
            operand=None,
        )

    step_batch = jax.jit(jax.vmap(step_one))
    reset_keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    state = reset_batch(reset_keys)
    active = jp.ones((batch_size,), dtype=bool)
    steps = jp.zeros((batch_size,), dtype=jp.int32)
    reward_total = jp.zeros((batch_size,), dtype=jp.float32)
    roll_progress = jp.zeros((batch_size,), dtype=jp.float32)
    disturbance_count = jp.zeros((batch_size,), dtype=jp.float32)

    for step_index in range(episode_length):
        action_keys = jax.random.split(
            jax.random.fold_in(
                jax.random.PRNGKey(seed + 1), step_index
            ),
            batch_size,
        )
        actions = policy_batch(state.obs, action_keys)
        was_active = active
        state = step_batch(state, actions, active)
        active_float = was_active.astype(jp.float32)
        steps = steps + was_active.astype(jp.int32)
        reward_total = reward_total + active_float * state.reward
        roll_progress = (
            roll_progress
            + active_float * state.metrics["roll_progress_rad"]
        )
        disturbance_count = (
            disturbance_count
            + active_float * state.metrics["disturbance_applied"]
        )
        active = active & (state.done < 0.5)

    jax.block_until_ready(state.obs)
    arrays = {
        "turns": np.asarray(
            jax.device_get(roll_progress / (2.0 * math.pi))
        ),
        "reward": np.asarray(jax.device_get(reward_total)),
        "steps": np.asarray(jax.device_get(steps)),
        "failed": np.asarray(
            jax.device_get(state.metrics["failed"])
        ),
        "disturbance_count": np.asarray(
            jax.device_get(disturbance_count)
        ),
    }
    return {
        "batch_size": batch_size,
        "seed": seed,
        "turns": _distribution_summary(arrays["turns"]),
        "reward": _distribution_summary(arrays["reward"]),
        "steps": _distribution_summary(arrays["steps"]),
        "disturbance_count": _distribution_summary(
            arrays["disturbance_count"]
        ),
        "failure_rate": float(np.mean(arrays["failed"])),
        "samples": [
            {
                "turns": float(arrays["turns"][index]),
                "reward": float(arrays["reward"][index]),
                "steps": int(arrays["steps"][index]),
                "failed": bool(arrays["failed"][index]),
                "disturbance_count": float(
                    arrays["disturbance_count"][index]
                ),
            }
            for index in range(batch_size)
        ],
    }


def _eval_visualization_dir(output_dir, eval_index, step, weight):
    weight_label = f"{weight:.2f}".replace(".", "p")
    return (
        Path(output_dir)
        / "eval_visualizations"
        / (
            f"eval_{eval_index:03d}_step_{step:09d}_"
            f"ref_{weight_label}"
        )
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=tuple(PRESETS), default="h200")
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILE_NAMES,
        default="cg12",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--unroll-length", type=int)
    parser.add_argument("--updates-per-batch", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument(
        "--disturbance-root-x-velocity",
        type=float,
        default=0.0,
        help="Maximum signed root-x velocity impulse in m/s per episode.",
    )
    parser.add_argument(
        "--disturbance-root-pitch-velocity",
        type=float,
        default=0.0,
        help="Maximum signed root-pitch velocity impulse in rad/s per episode.",
    )
    parser.add_argument("--disturbance-min-step", type=int, default=100)
    parser.add_argument("--disturbance-max-step", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--entropy-cost", type=float, default=1e-3)
    parser.add_argument("--discounting", type=float, default=0.995)
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
    parser.add_argument("--controller", type=Path, default=DEFAULT_CEM_CONTROLLER)
    parser.add_argument(
        "--retain-cem",
        action="store_true",
        help=(
            "Keep CEM weight at 1 and curriculum only the bounded residual "
            "scale."
        ),
    )
    parser.add_argument(
        "--reference-weights",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.0],
    )
    parser.add_argument("--minimum-residual-gain", type=float, default=0.05)
    parser.add_argument(
        "--residual-scales",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
        help="Increasing residual scales used by --retain-cem.",
    )
    parser.add_argument("--minimum-stage-steps", type=int, default=500_000)
    parser.add_argument("--gate-check-steps", type=int, default=500_000)
    parser.add_argument("--gate-min-survival", type=float, default=0.80)
    parser.add_argument("--gate-min-turns", type=float, default=3.0)
    parser.add_argument(
        "--target-min-turns",
        type=float,
        help=(
            "Final-stage turns gate. Defaults to --gate-min-turns for "
            "backward compatibility."
        ),
    )
    parser.add_argument("--gate-max-failure-rate", type=float, default=0.20)
    parser.add_argument(
        "--robust-eval-envs",
        type=int,
        default=128,
        help="Final deterministic evaluation batch; set to 0 to disable.",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa", "disable"),
        default="auto",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "mjx_residual_cem_curriculum",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--visualization-diagnostics", action="store_true")
    _add_reward_arguments(parser)
    args = parser.parse_args(argv)
    if args.retain_cem:
        _validate_residual_scales(parser, args.residual_scales)
    else:
        _validate_weights(parser, args.reference_weights)
    if not 0.0 <= args.minimum_residual_gain <= 1.0:
        parser.error("--minimum-residual-gain must be in [0, 1]")
    if args.target_min_turns is None:
        args.target_min_turns = args.gate_min_turns
    if args.robust_eval_envs < 0:
        parser.error("--robust-eval-envs must be nonnegative")
    try:
        validate_nominal_rl_config(
            NominalRLConfig(
                episode_length=args.episode_length,
                disturbance_root_x_velocity_m_s=(
                    args.disturbance_root_x_velocity
                ),
                disturbance_root_pitch_velocity_rad_s=(
                    args.disturbance_root_pitch_velocity
                ),
                disturbance_min_step=args.disturbance_min_step,
                disturbance_max_step=args.disturbance_max_step,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = _parse_args()
    stage_plan = _stage_plan(args)
    values = PRESETS[args.preset].copy()
    for name in (
        "steps",
        "envs",
        "eval_envs",
        "batch_size",
        "num_minibatches",
        "unroll_length",
        "updates_per_batch",
    ):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    if (
        args.out.exists()
        and any(args.out.iterdir())
        and not args.allow_existing_output
    ):
        raise SystemExit(
            f"Output directory is not empty: {args.out}. Use a new --out."
        )

    rollout_quantum = (
        values["batch_size"]
        * values["unroll_length"]
        * values["num_minibatches"]
    )
    effective_budget = expected_budget_steps(
        values["steps"], rollout_quantum
    )
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=True,
        xla_triton=True,
        mujoco_gl=args.mujoco_gl,
    )
    import jax
    from brax.io import model as model_io
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment import make_brax_env

    if "restore_params" not in inspect.signature(ppo.train).parameters:
        raise SystemExit("Installed Brax does not support restore_params.")

    args.out.mkdir(parents=True, exist_ok=True)
    task = physics_profile(
        args.physics_profile,
        NominalRLConfig(
            episode_length=args.episode_length,
            disturbance_root_x_velocity_m_s=(
                args.disturbance_root_x_velocity
            ),
            disturbance_root_pitch_velocity_rad_s=(
                args.disturbance_root_pitch_velocity
            ),
            disturbance_min_step=args.disturbance_min_step,
            disturbance_max_step=args.disturbance_max_step,
        ),
    )
    reward_config = _reward_config_from_args(args)
    base_reference = load_cem_reference(
        args.controller,
        minimum_residual_gain=stage_plan[0]["minimum_residual_gain"],
    )
    runtime = describe_runtime()
    config_payload = {
        "preset": args.preset,
        **values,
        "requested_steps": values["steps"],
        "effective_budget_steps": effective_budget,
        "rollout_quantum": rollout_quantum,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "reward_scaling": args.reward_scaling,
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "seed": args.seed,
        "task": asdict(task),
        "reward": asdict(reward_config),
        "reference": asdict(base_reference),
        "curriculum_mode": (
            "retain_cem" if args.retain_cem else "withdraw_reference"
        ),
        "stage_plan": stage_plan,
        "reference_weights": args.reference_weights,
        "residual_scales": args.residual_scales,
        "minimum_stage_steps": args.minimum_stage_steps,
        "gate_check_steps": args.gate_check_steps,
        "gate": {
            "minimum_survival": args.gate_min_survival,
            "stage_minimum_turns": args.gate_min_turns,
            "target_minimum_turns": args.target_min_turns,
            "maximum_failure_rate": args.gate_max_failure_rate,
        },
        "robust_eval_envs": args.robust_eval_envs,
        "runtime": runtime,
    }
    (args.out / "training_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[residual curriculum]\n"
        f"  mode={'retain_cem' if args.retain_cem else 'withdraw_reference'}\n"
        f"  requested_steps={values['steps']:,} "
        f"fixed_budget={effective_budget:,} "
        f"rollout_quantum={rollout_quantum:,}\n"
        f"  stage_plan={stage_plan} "
        f"minimum_stage={args.minimum_stage_steps:,}\n"
        f"  physics={args.physics_profile} root_damping="
        f"{'disabled' if task.disable_root_damping else 'xml'}\n"
        f"  disturbance root_x<=+/-"
        f"{task.disturbance_root_x_velocity_m_s:g}m/s "
        f"root_pitch<=+/-"
        f"{task.disturbance_root_pitch_velocity_rad_s:g}rad/s "
        f"step=[{task.disturbance_min_step},"
        f"{task.disturbance_max_step}]\n"
        f"  gate survival>={args.gate_min_survival:.0%} "
        f"stage_turns>={args.gate_min_turns:g} "
        f"target_turns>={args.target_min_turns:g} "
        f"failure<={args.gate_max_failure_rate:.0%}\n"
        f"  controller={base_reference.source}\n"
        f"  output={args.out.resolve()}",
        flush=True,
    )

    start = time.perf_counter()
    consumed_steps = 0
    stage_index = 0
    restored_params = None
    make_inference_fn = None
    final_params = None
    final_metrics = {}
    history = []
    stage_history = []
    eval_visualizations = []
    eval_counter = 0
    best_target = {
        "rank": None,
        "score": float("-inf"),
        "turns": None,
        "step": None,
        "params": None,
    }
    target_stage_steps = 0
    target_stage_gate_passed = False

    while consumed_steps < effective_budget:
        stage_spec = stage_plan[stage_index]
        weight = stage_spec["reference_weight"]
        reference = load_cem_reference(
            args.controller,
            reference_weight=weight,
            minimum_residual_gain=stage_spec["minimum_residual_gain"],
        )
        remaining = effective_budget - consumed_steps
        stage_schedule = _exact_stage_eval_schedule(
            remaining, rollout_quantum, args.gate_check_steps
        )
        train_env = make_brax_env(
            task,
            reward_config=reward_config,
            cem_reference=reference,
            seed=args.seed + 100 * stage_index,
        )
        eval_env = make_brax_env(
            task,
            reward_config=reward_config,
            cem_reference=reference,
            seed=args.seed + 10_000 + 100 * stage_index,
        )
        stage = {
            "local_step": 0,
            "params": restored_params,
            "params_step": None,
            "make_inference_fn": None,
            "last_metrics": {},
            "gate": None,
            "eval_index": 0,
            "eval_records": {},
            "visualized_steps": set(),
            "best": {
                "rank": None,
                "step": None,
                "params": None,
                "metrics": None,
                "gate": None,
            },
        }
        stage_start = consumed_steps
        print(
            f"[stage {stage_index + 1}/{len(stage_plan)}]\n"
            f"  reference_weight={weight:.2f} "
            f"residual_gain={reference.residual_gain:.3f}\n"
            f"  global_start={stage_start:,} remaining={remaining:,} "
            f"eval_interval={stage_schedule['eval_interval_steps']:,}",
            flush=True,
        )

        def visualize_eval(local_step):
            if args.skip_visualization:
                return
            local_step = int(local_step)
            record = stage["eval_records"].get(local_step)
            if (
                record is None
                or local_step in stage["visualized_steps"]
                or stage["params_step"] != local_step
                or stage["params"] is None
                or stage["make_inference_fn"] is None
            ):
                return
            stage["visualized_steps"].add(local_step)
            output_dir = _eval_visualization_dir(
                args.out,
                record["global_eval"],
                record["global_step"],
                weight,
            )
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                model_io.save_params(output_dir / "params", stage["params"])
                rollout = _evaluate_policy(
                    eval_env,
                    stage["make_inference_fn"],
                    stage["params"],
                    seed=args.seed + 30_000 + record["global_eval"],
                    episode_length=args.episode_length,
                    output_dir=output_dir,
                )
                from scripts.render_mjx_policy import render_rollout

                gif_path = output_dir / "policy_rollout.gif"
                render_rollout(
                    output_dir / "evaluation_rollout.npz",
                    gif_path,
                    control_dt=task.control_timestep,
                    fps=20,
                    width=640,
                    height=480,
                    camera_distance=0.75,
                    diagnostics=args.visualization_diagnostics,
                )
                record["visualization"] = {
                    "output_dir": str(output_dir),
                    "gif": str(gif_path),
                    "rollout": rollout,
                }
                print(
                    f"[eval visualization {record['global_eval']}]\n"
                    f"  step={record['global_step']:,} "
                    f"ref={weight:.2f} "
                    f"seed={args.seed + 30_000 + record['global_eval']}\n"
                    f"  single_rollout turns={rollout['net_turns']:+.3f} "
                    f"reward={rollout['total_reward']:+.3f} "
                    f"steps={rollout['episode_steps']}\n"
                    f"  gif={gif_path.resolve()}",
                    flush=True,
                )
            except Exception as exc:
                record["visualization_error"] = str(exc)
                print(
                    f"[eval visualization warning "
                    f"{record['global_eval']}]\n"
                    f"  step={record['global_step']:,} "
                    f"ref={weight:.2f} error={exc}",
                    flush=True,
                )

        def policy_params_fn(local_step, make_policy, params):
            stage["local_step"] = int(local_step)
            stage["params"] = params
            stage["params_step"] = int(local_step)
            stage["make_inference_fn"] = make_policy
            visualize_eval(local_step)

        def progress_fn(local_step, metrics):
            nonlocal eval_counter, target_stage_gate_passed
            clean = {name: _float(value) for name, value in metrics.items()}
            _add_per_step_eval_metrics(clean)
            global_step = stage_start + int(local_step)
            is_target_stage = stage_index == len(stage_plan) - 1
            required_turns = (
                args.target_min_turns
                if is_target_stage
                else args.gate_min_turns
            )
            gate = _gate_assessment(
                clean,
                episode_length=args.episode_length,
                minimum_survival=args.gate_min_survival,
                minimum_turns=required_turns,
                maximum_failure_rate=args.gate_max_failure_rate,
            )
            selection = _checkpoint_selection(
                clean, args.episode_length
            )
            stage["local_step"] = int(local_step)
            stage["last_metrics"] = clean
            stage["gate"] = gate
            stage["eval_index"] += 1
            reward_metrics, ordinary_metrics = _split_metrics(clean)
            history.append(
                {
                    "global_step": global_step,
                    "stage_index": stage_index,
                    "reference_weight": weight,
                    "gate": gate,
                    "reward_metrics": reward_metrics,
                    "metrics": ordinary_metrics,
                }
            )
            checkpoint_rank = _residual_checkpoint_rank(selection, clean)
            if (
                int(local_step) > 0
                and stage["params"] is not None
                and not selection["rejected"]
                and _safe_stage_checkpoint(gate)
                and (
                    stage["best"]["rank"] is None
                    or checkpoint_rank > stage["best"]["rank"]
                )
            ):
                stage["best"] = {
                    "rank": checkpoint_rank,
                    "step": int(local_step),
                    "params": stage["params"],
                    "metrics": clean,
                    "gate": gate,
                }
            target_eval_eligible = _target_eval_eligible(
                stage_index,
                len(stage_plan),
                local_step,
                args.minimum_stage_steps,
            )
            selected = (
                target_eval_eligible
                and gate["passed"]
                and not selection["rejected"]
                and (
                    best_target["rank"] is None
                    or checkpoint_rank > best_target["rank"]
                )
            )
            if target_eval_eligible:
                target_stage_gate_passed = (
                    target_stage_gate_passed or gate["passed"]
                )
            if selected and stage["params"] is not None:
                best_target["rank"] = checkpoint_rank
                best_target["score"] = selection["score"]
                best_target["turns"] = selection["turns"]
                best_target["step"] = global_step
                best_target["params"] = stage["params"]
            eval_counter += 1
            eval_record = {
                "global_eval": eval_counter,
                "global_step": global_step,
                "stage_index": stage_index,
                "stage_eval_index": stage["eval_index"],
                "reference_weight": weight,
            }
            stage["eval_records"][int(local_step)] = eval_record
            eval_visualizations.append(eval_record)
            print(
                _format_eval_report(
                    stage["eval_index"],
                    stage_schedule["num_evals"],
                    global_step,
                    clean,
                    episode_length=args.episode_length,
                    control_dt=task.control_timestep,
                    selection=selection,
                    selected=selected,
                ),
                flush=True,
            )
            policy_rms = clean.get("eval/avg_residual_action_rms", 0.0)
            print(
                "  residual "
                f"ref_weight={weight:.2f} "
                f"gain={reference.residual_gain:.3f} "
                f"ref_rms="
                f"{clean.get('eval/avg_reference_action_rms', 0.0):.3f} "
                f"policy_rms={policy_rms:.3f} "
                f"effective_residual_rms="
                f"{reference.residual_gain * policy_rms:.3f}",
                flush=True,
            )
            visualize_eval(local_step)
            failed_checks = [
                name for name, passed in gate["checks"].items() if not passed
            ]
            print(
                "  curriculum "
                f"global_eval={eval_counter} "
                f"stage={stage_index + 1}/{len(stage_plan)} "
                f"ref={weight:.2f} local_steps={int(local_step):,} "
                f"gate={'PASS' if gate['passed'] else 'WAIT'} "
                f"missing={','.join(failed_checks) or 'none'}",
                flush=True,
            )
            can_advance = _can_advance_stage(
                stage_index,
                len(stage_plan),
                local_step,
                args.minimum_stage_steps,
                gate,
                stage["best"]["gate"],
            )
            if can_advance:
                if not gate["passed"]:
                    print(
                        "  curriculum advance_from_best "
                        f"step={stage['best']['step']:,} "
                        f"turns={stage['best']['rank'][0]:+.3f}",
                        flush=True,
                    )
                raise _AdvanceCurriculum

        train_kwargs = {}
        if restored_params is not None:
            train_kwargs["restore_params"] = restored_params
        checkpoint_path = args.out / "ppo_checkpoints" / f"stage_{stage_index}"
        if "save_checkpoint_path" in inspect.signature(ppo.train).parameters:
            train_kwargs["save_checkpoint_path"] = str(
                checkpoint_path.resolve()
            )
        try:
            (
                stage_make_inference_fn,
                stage_final_params,
                stage_final_metrics,
            ) = ppo.train(
                environment=train_env,
                eval_env=eval_env,
                num_timesteps=remaining,
                episode_length=args.episode_length,
                action_repeat=1,
                num_envs=values["envs"],
                num_evals=stage_schedule["num_evals"],
                num_eval_envs=values["eval_envs"],
                learning_rate=args.learning_rate,
                entropy_cost=args.entropy_cost,
                discounting=args.discounting,
                reward_scaling=args.reward_scaling,
                unroll_length=values["unroll_length"],
                batch_size=values["batch_size"],
                num_minibatches=values["num_minibatches"],
                num_updates_per_batch=values["updates_per_batch"],
                normalize_observations=True,
                deterministic_eval=True,
                network_factory=_network_factory(
                    args.hidden_layers, args.activation
                ),
                seed=args.seed + stage_index,
                progress_fn=progress_fn,
                policy_params_fn=policy_params_fn,
                **train_kwargs,
            )
            local_trained = remaining
            make_inference_fn = stage_make_inference_fn
            stage_last_params = stage_final_params
            stage_last_metrics = stage_final_metrics or {}
            advanced = False
        except _AdvanceCurriculum:
            local_trained = int(stage["local_step"])
            if local_trained <= 0 or stage["params"] is None:
                raise RuntimeError("curriculum advanced without policy params")
            make_inference_fn = stage["make_inference_fn"]
            stage_last_params = stage["params"]
            stage_last_metrics = stage["last_metrics"]
            advanced = True

        if stage["best"]["params"] is not None:
            final_params = stage["best"]["params"]
            final_metrics = stage["best"]["metrics"]
        else:
            final_params = stage_last_params
            final_metrics = stage_last_metrics
        restored_params = final_params

        consumed_steps += local_trained
        if stage_index == len(stage_plan) - 1:
            target_stage_steps += local_trained
        stage_history.append(
            {
                "stage_index": stage_index,
                "reference_weight": weight,
                "residual_gain": reference.residual_gain,
                "global_start_step": stage_start,
                "global_end_step": consumed_steps,
                "trained_steps": local_trained,
                "gate": stage["gate"],
                "advanced": advanced,
                "best_local_step": stage["best"]["step"],
                "best_turns": (
                    stage["best"]["rank"][0]
                    if stage["best"]["rank"] is not None
                    else None
                ),
            }
        )
        model_io.save_params(
            args.out / f"params_stage_{stage_index}_last",
            stage_last_params,
        )
        if stage["best"]["params"] is not None:
            model_io.save_params(
                args.out / f"params_stage_{stage_index}_best",
                stage["best"]["params"],
            )
        model_io.save_params(
            args.out / f"params_stage_{stage_index}_final",
            final_params,
        )
        selected_step = (
            stage["best"]["step"]
            if stage["best"]["step"] is not None
            else int(stage["local_step"])
        )
        selected_turns = stage_history[-1]["best_turns"]
        print(
            f"[stage result {stage_index + 1}/{len(stage_plan)}]\n"
            f"  trained={local_trained:,} "
            f"last_step={int(stage['local_step']):,} "
            f"selected_step={selected_step:,} "
            f"best_turns="
            f"{selected_turns if selected_turns is not None else float('nan'):+.3f}\n"
            f"  action={'advance' if advanced else 'stop'} "
            f"checkpoint=params_stage_{stage_index}_final",
            flush=True,
        )
        if advanced:
            stage_index += 1
        else:
            break

    last_trained_stage_spec = _last_trained_stage_spec(
        stage_plan, stage_history
    )
    last_trained_weight = last_trained_stage_spec["reference_weight"]
    last_trained_reference = load_cem_reference(
        args.controller,
        reference_weight=last_trained_stage_spec["reference_weight"],
        minimum_residual_gain=(
            last_trained_stage_spec["minimum_residual_gain"]
        ),
    )
    reached_target_stage = target_stage_steps > 0
    curriculum_success = (
        reached_target_stage
        and target_stage_gate_passed
        and stage_index == len(stage_plan) - 1
        and consumed_steps == effective_budget
    )
    model_io.save_params(args.out / "params_final", final_params)
    best_target_name = (
        "params_best_retained_cem"
        if args.retain_cem
        else "params_best_zero_reference"
    )
    if best_target["params"] is not None:
        model_io.save_params(
            args.out / best_target_name,
            best_target["params"],
        )
    elapsed = time.perf_counter() - start
    clean_final_metrics = {
        name: _float(value) for name, value in final_metrics.items()
    }
    summary = {
        "curriculum_mode": (
            "retain_cem" if args.retain_cem else "withdraw_reference"
        ),
        "curriculum_success": curriculum_success,
        "reached_target_stage": reached_target_stage,
        "target_stage_gate_passed": target_stage_gate_passed,
        "target_stage_training_steps": target_stage_steps,
        "reached_zero_reference": (
            reached_target_stage and not args.retain_cem
        ),
        "zero_reference_gate_passed": (
            target_stage_gate_passed and not args.retain_cem
        ),
        "zero_reference_training_steps": (
            target_stage_steps if not args.retain_cem else 0
        ),
        "requested_steps": values["steps"],
        "effective_budget_steps": effective_budget,
        "consumed_steps": consumed_steps,
        "final_reference_weight": last_trained_weight,
        "final_residual_gain": last_trained_reference.residual_gain,
        "best_target_step": best_target["step"],
        "best_target_turns": best_target["turns"],
        "best_target_score": (
            best_target["score"]
            if math.isfinite(best_target["score"])
            else None
        ),
        "elapsed_s": elapsed,
        "stages": stage_history,
        "final_metrics": clean_final_metrics,
    }
    (args.out / "curriculum_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "eval_visualizations.json").write_text(
        json.dumps(eval_visualizations, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[curriculum complete]\n"
        f"  mode={'retain_cem' if args.retain_cem else 'withdraw_reference'}\n"
        f"  status={'SUCCESS' if curriculum_success else 'FAILED'} "
        f"consumed={consumed_steps:,}/{effective_budget:,}\n"
        f"  final_reference_weight="
        f"{last_trained_weight:.2f} "
        f"final_residual_gain={last_trained_reference.residual_gain:.3f} "
        f"target_stage_steps={target_stage_steps:,}\n"
        f"  elapsed={elapsed / 60.0:.1f}min",
        flush=True,
    )

    evaluation_reference = load_cem_reference(
        args.controller,
        reference_weight=last_trained_stage_spec["reference_weight"],
        minimum_residual_gain=(
            last_trained_stage_spec["minimum_residual_gain"]
        ),
    )
    final_eval_env = make_brax_env(
        task,
        reward_config=reward_config,
        cem_reference=evaluation_reference,
        seed=args.seed + 20_000,
    )
    evaluation_label = (
        f"retained-cem scale={evaluation_reference.residual_gain:.3f}"
        if args.retain_cem
        else "zero-reference"
    )
    evaluation_dir = args.out / (
        "evaluation_retained_cem"
        if args.retain_cem
        else "evaluation_zero_reference"
    )
    evaluation_params = (
        best_target["params"]
        if reached_target_stage and best_target["params"] is not None
        else final_params
    )
    evaluation = _evaluate_policy(
        final_eval_env,
        make_inference_fn,
        evaluation_params,
        seed=args.seed + 20_000,
        episode_length=args.episode_length,
        output_dir=evaluation_dir,
    )
    print(_format_rollout_report(evaluation_label, evaluation), flush=True)
    robust_evaluation = None
    if args.robust_eval_envs > 0:
        robust_evaluation = _evaluate_policy_distribution(
            final_eval_env,
            make_inference_fn,
            evaluation_params,
            seed=args.seed + 40_000,
            episode_length=args.episode_length,
            batch_size=args.robust_eval_envs,
        )
        robust_path = evaluation_dir / "robust_evaluation.json"
        robust_path.write_text(
            json.dumps(robust_evaluation, indent=2) + "\n",
            encoding="utf-8",
        )
        turns = robust_evaluation["turns"]
        print(
            "[robust evaluation]\n"
            f"  scale={evaluation_reference.residual_gain:.3f} "
            f"batch={args.robust_eval_envs} "
            f"failure={robust_evaluation['failure_rate']:.1%}\n"
            f"  turns mean={turns['mean']:+.3f} "
            f"min={turns['min']:+.3f} p10={turns['p10']:+.3f} "
            f"median={turns['median']:+.3f} "
            f"p90={turns['p90']:+.3f} max={turns['max']:+.3f}\n"
            f"  pushes/episode="
            f"{robust_evaluation['disturbance_count']['mean']:.2f} "
            f"output={robust_path.resolve()}",
            flush=True,
        )
        summary["robust_evaluation"] = robust_evaluation
        (args.out / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    if not args.skip_visualization:
        try:
            from scripts.render_mjx_policy import render_rollout

            gif_path = evaluation_dir / "policy_rollout.gif"
            render_rollout(
                evaluation_dir / "evaluation_rollout.npz",
                gif_path,
                control_dt=task.control_timestep,
                fps=20,
                width=640,
                height=480,
                camera_distance=0.75,
                diagnostics=args.visualization_diagnostics,
            )
            print(f"[visualization]\n  gif={gif_path.resolve()}", flush=True)
        except Exception as exc:
            print(
                f"[visualization warning]\n  rendering failed: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
