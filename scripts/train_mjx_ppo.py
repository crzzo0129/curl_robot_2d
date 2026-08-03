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
    "reference_action_rms",
    "residual_action_rms",
    "reference_weight",
    "residual_gain",
    "forbidden_contact_count",
    "forbidden_penetration_m",
    "allowed_foot_penetration_m",
    "foot_contact_active",
    "foot_contact_start",
    "ground_contact_count",
    "root_low_active",
    "root_low_step_count",
    "roll_progress_rad",
    "phase_progress_rad",
    "translation_progress_rad",
    "stuck_active",
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


def _training_step_schedule(
    *,
    requested_steps,
    num_evals,
    batch_size,
    unroll_length,
    num_minibatches,
):
    """Mirror Brax PPO's integer rollout scheduling."""

    rollout_quantum = batch_size * unroll_length * num_minibatches
    eval_intervals = max(num_evals - 1, 1)
    updates_per_interval = math.ceil(
        requested_steps / (eval_intervals * rollout_quantum)
    )
    eval_interval_steps = updates_per_interval * rollout_quantum
    return {
        "requested_steps": requested_steps,
        "effective_steps": eval_intervals * eval_interval_steps,
        "rollout_quantum": rollout_quantum,
        "eval_intervals": eval_intervals,
        "updates_per_interval": updates_per_interval,
        "eval_interval_steps": eval_interval_steps,
    }


def _checkpoint_selection(
    metrics,
    episode_length,
    *,
    target_turns=3.0,
    minimum_turns=None,
    minimum_tail_turns=None,
):
    """Score an eval point by physical behavior, independent of reward scale."""

    avg_length = metrics.get("eval/avg_episode_length", 0.0)
    failed_rate = metrics.get("eval/episode_failed", 1.0)
    nonfinite_rate = metrics.get(
        "eval/episode_failure_nonfinite", 0.0
    )
    roll_total = metrics.get("eval/episode_roll_progress_rad", -math.inf)
    tail_metric_present = "eval/episode_tail_roll_progress_rad" in metrics
    tail_total = metrics.get("eval/episode_tail_roll_progress_rad", 0.0)
    stuck_failure_rate = metrics.get("eval/episode_failure_stuck", 0.0)
    penetration = metrics.get(
        "eval/avg_forbidden_penetration_m", math.inf
    )
    survival = min(max(avg_length / episode_length, 0.0), 1.0)
    turns = roll_total / (2.0 * math.pi)
    tail_turns = tail_total / (2.0 * math.pi)
    progress_quality = min(max(turns / target_turns, -1.0), 1.0)
    safety_quality = 1.0 - min(
        max(penetration / 0.001, 0.0), 1.0
    )
    score = (
        0.50 * survival
        + 0.35 * progress_quality
        + 0.10 * (1.0 - min(max(failed_rate, 0.0), 1.0))
        + 0.05 * safety_quality
    )
    rejected = (
        nonfinite_rate > 0.0
        or stuck_failure_rate > 0.0
        or (minimum_turns is not None and turns < minimum_turns)
        or (
            minimum_tail_turns is not None
            and (
                not tail_metric_present
                or tail_turns < minimum_tail_turns
            )
        )
        or not math.isfinite(turns)
        or not math.isfinite(penetration)
        or not math.isfinite(score)
    )
    return {
        "score": -1_000_000.0 if rejected else score,
        "rejected": rejected,
        "survival": survival,
        "turns": turns,
        "tail_turns": tail_turns,
    }


def _metric(metrics, name, default=0.0):
    return float(metrics.get(name, default))


def _format_eval_report(
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
    reward = _metric(metrics, "eval/episode_reward")
    avg_reward = _metric(metrics, "eval/avg_reward")
    avg_length = _metric(metrics, "eval/avg_episode_length")
    failed = _metric(metrics, "eval/episode_failed")
    timeout = _metric(metrics, "eval/episode_timeout")
    marker = " new_best" if selected else ""
    lines = [
        (
            f"[eval {eval_index}/{total_evals}] step={int(step)} "
            f"physical_score={selection['score']:.4f}{marker}"
        ),
        (
            f"  outcome reward={reward:+.3f} avg/step={avg_reward:+.4f} "
            f"length={avg_length:.1f}/{episode_length} "
            f"time={avg_length * control_dt:.2f}s "
            f"failed={failed:.1%} timeout={timeout:.1%}"
        ),
        (
            "  motion (rad/step) "
            f"phase={_metric(metrics, 'eval/avg_phase_progress_rad'):+.5f} "
            f"translation="
            f"{_metric(metrics, 'eval/avg_translation_progress_rad'):+.5f} "
            f"roll={_metric(metrics, 'eval/avg_roll_progress_rad'):+.5f} "
            f"turns/episode={selection['turns']:+.3f} "
            f"tail_turns={selection['tail_turns']:+.3f}"
        ),
        (
            "  state   "
            f"avg_root_z={_metric(metrics, 'eval/avg_root_height_m'):.3f}m "
            f"avg_foot_gap="
            f"{_metric(metrics, 'eval/avg_foot_center_distance_m'):.3f}m "
            f"action_rms={_metric(metrics, 'eval/avg_action_rms'):.3f} "
            f"torque_rms="
            f"{_metric(metrics, 'eval/avg_normalized_torque_rms'):.3f} "
            f"ramp={_metric(metrics, 'eval/avg_startup_action_ramp'):.3f} "
            f"pushes/episode="
            f"{_metric(metrics, 'eval/episode_disturbance_applied'):.2f}"
        ),
    ]
    for group, reward_labels in (
        (
            "progress",
            (
                ("roll", "roll_progress"),
                ("phase", "phase_progress"),
                ("x", "translation_progress"),
                ("mismatch", "roll_mismatch"),
                ("back", "backward"),
            ),
        ),
        (
            "control ",
            (
                ("action", "action_rate"),
                ("residual", "residual_action"),
                ("torque", "torque"),
            ),
        ),
        (
            "safety  ",
            (
                ("air", "airborne"),
                ("gap", "foot_gap"),
                ("collision", "collision"),
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
        f"stuck={_metric(metrics, 'eval/episode_failure_stuck'):.1%} "
        f"gap={_metric(metrics, 'eval/episode_failure_foot_gap'):.1%} "
        f"high={_metric(metrics, 'eval/episode_failure_root_high'):.1%} "
        f"cross={_metric(metrics, 'eval/episode_failure_leg_crossing'):.1%} "
        f"forbidden_depth="
        f"{1e3 * _metric(metrics, 'eval/avg_forbidden_penetration_m'):.3f}mm"
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


def _format_rollout_report(label, summary):
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
                f"x={summary['root_x_displacement_m']:+.3f}m"
            ),
            (
                "  reward  "
                + " ".join(
                    f"{name}={float(terms.get(name, 0.0)):+.3f}"
                    for name in REWARD_TERM_NAMES
                )
            ),
        ]
    )


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
    net_phase_rad = final_phase - initial_phase
    translation_equivalent_rad = (
        final_x - initial_x
    ) / env.rolling_radius
    conservative_roll_rad = min(
        net_phase_rad, translation_equivalent_rad
    )
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
            "failure_stuck",
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
        "net_phase_rad": net_phase_rad,
        "net_phase_turns": net_phase_rad / (2.0 * math.pi),
        "translation_equivalent_turns": (
            translation_equivalent_rad / (2.0 * math.pi)
        ),
        "net_turns": conservative_roll_rad / (2.0 * math.pi),
        "tail_turns": metric_totals.get(
            "tail_roll_progress_rad", 0.0
        )
        / (2.0 * math.pi),
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
    parser.add_argument(
        "--model-xml",
        type=Path,
        help=(
            "MuJoCo XML to load. Relative paths are resolved from the "
            "curl_robot_2d project root."
        ),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--terminate-root-z-min", type=float, default=0.05)
    parser.add_argument(
        "--terminate-root-z-low-duration", type=float, default=0.30
    )
    parser.add_argument(
        "--no-root-low-termination",
        action="store_true",
        help="Disable continuous low-root termination for compatibility runs.",
    )
    parser.add_argument("--stuck-root-z-max", type=float, default=0.10)
    parser.add_argument("--stuck-progress-window", type=float, default=1.0)
    parser.add_argument("--stuck-min-progress-rad", type=float, default=0.20)
    parser.add_argument("--stuck-duration", type=float, default=0.75)
    parser.add_argument("--stuck-grace", type=float, default=1.50)
    parser.add_argument("--tail-progress-window", type=float, default=2.0)
    parser.add_argument(
        "--no-stuck-termination",
        action="store_true",
        help="Disable low-root plus stalled-progress termination.",
    )
    parser.add_argument("--terminate-root-z-max", type=float, default=0.70)
    parser.add_argument(
        "--no-root-high-termination",
        action="store_true",
        help="Disable high-root termination during early curriculum stages.",
    )
    parser.add_argument(
        "--maximum-foot-center-distance", type=float, default=0.28
    )
    parser.add_argument(
        "--no-foot-gap-termination",
        action="store_true",
        help="Disable hard front/rear foot-distance termination.",
    )
    parser.add_argument(
        "--no-leg-crossing-termination",
        action="store_true",
        help="Keep logging leg crossings but do not terminate on them.",
    )
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--selection-target-turns", type=float, default=3.0)
    parser.add_argument("--selection-min-turns", type=float)
    parser.add_argument("--selection-min-tail-turns", type=float)
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
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Do not render best/final deterministic rollout GIFs.",
    )
    parser.add_argument("--visualization-fps", type=int, default=20)
    parser.add_argument("--visualization-width", type=int, default=640)
    parser.add_argument("--visualization-height", type=int, default=480)
    parser.add_argument(
        "--visualization-camera-distance", type=float, default=0.75
    )
    parser.add_argument(
        "--visualization-diagnostics",
        action="store_true",
        help="Show MuJoCo COM and contact points in generated GIFs.",
    )
    _add_reward_arguments(parser)
    args = parser.parse_args()
    if (
        not math.isfinite(args.selection_target_turns)
        or args.selection_target_turns <= 0.0
    ):
        raise SystemExit("--selection-target-turns must be finite and positive")
    if args.selection_min_turns is not None and not math.isfinite(
        args.selection_min_turns
    ):
        raise SystemExit("--selection-min-turns must be finite")
    if (
        args.selection_min_tail_turns is not None
        and (
            not math.isfinite(args.selection_min_tail_turns)
            or args.selection_min_tail_turns < 0.0
        )
    ):
        raise SystemExit(
            "--selection-min-tail-turns must be finite and nonnegative"
        )

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
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment import make_brax_env

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
    task = physics_profile(
        args.physics_profile,
        NominalRLConfig(
            model_xml=(
                str(args.model_xml)
                if args.model_xml is not None
                else None
            ),
            episode_length=args.episode_length,
            terminate_root_z_min=(
                None
                if args.no_root_low_termination
                else args.terminate_root_z_min
            ),
            terminate_root_z_low_duration_s=(
                args.terminate_root_z_low_duration
            ),
            terminate_stuck_root_z_max=(
                None
                if args.no_stuck_termination
                else args.stuck_root_z_max
            ),
            terminate_stuck_progress_window_s=(
                args.stuck_progress_window
            ),
            terminate_stuck_min_progress_rad=(
                args.stuck_min_progress_rad
            ),
            terminate_stuck_duration_s=args.stuck_duration,
            terminate_stuck_grace_s=args.stuck_grace,
            tail_progress_window_s=args.tail_progress_window,
            terminate_root_z_max=(
                None
                if args.no_root_high_termination
                else args.terminate_root_z_max
            ),
            maximum_foot_center_distance_m=(
                None
                if args.no_foot_gap_termination
                else args.maximum_foot_center_distance
            ),
            terminate_leg_crossing=(
                not args.no_leg_crossing_termination
            ),
        ),
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
        "score": float("-inf"),
        "reward": float("-inf"),
        "step": None,
        "params": None,
        "candidate_step": None,
        "candidate_params": None,
    }
    reward_peak = {
        "reward": float("-inf"),
        "step": None,
        "params": None,
    }
    eval_counter = {"value": 0}

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
        if reward is not None and reward > reward_peak["reward"]:
            reward_peak["reward"] = reward
            reward_peak["step"] = int(step)
            if best["candidate_step"] == int(step):
                reward_peak["params"] = best["candidate_params"]
        selection = _checkpoint_selection(
            clean,
            args.episode_length,
            target_turns=args.selection_target_turns,
            minimum_turns=args.selection_min_turns,
            minimum_tail_turns=args.selection_min_tail_turns,
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
            _format_eval_report(
                eval_counter["value"],
                values["num_evals"],
                step,
                clean,
                episode_length=args.episode_length,
                control_dt=task.control_timestep,
                selection=selection,
                selected=selected,
            ),
            flush=True,
        )

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
        "model_xml_resolved": str(train_env.model_path.resolve()),
        "evaluation": {
            "skip": args.skip_evaluation,
            "skip_visualization": args.skip_visualization,
            "visualization_fps": args.visualization_fps,
            "visualization_width": args.visualization_width,
            "visualization_height": args.visualization_height,
            "visualization_camera_distance": (
                args.visualization_camera_distance
            ),
            "visualization_diagnostics": (
                args.visualization_diagnostics
            ),
        },
        "checkpoint_selection": {
            "description": (
                "0.50 survival + 0.35 forward turns + "
                "0.10 non-failure + 0.05 contact safety"
            ),
            "target_turns": args.selection_target_turns,
            "minimum_turns": args.selection_min_turns,
            "minimum_tail_turns": args.selection_min_tail_turns,
            "tail_window_s": task.tail_progress_window_s,
            "penetration_limit_m": 0.001,
            "reject_nonfinite": True,
            "reject_stuck": True,
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
    root_high_text = (
        "disabled"
        if task.terminate_root_z_max is None
        else f"{task.terminate_root_z_max:g}m"
    )
    stuck_text = (
        "disabled"
        if task.terminate_stuck_root_z_max is None
        else (
            f"z<{task.terminate_stuck_root_z_max:g}m "
            f"progress<{task.terminate_stuck_min_progress_rad:g}rad/"
            f"{task.terminate_stuck_progress_window_s:g}s "
            f"for {task.terminate_stuck_duration_s:g}s "
            f"after {task.terminate_stuck_grace_s:g}s"
        )
    )
    foot_gap_text = (
        "disabled"
        if task.maximum_foot_center_distance_m is None
        else f"{task.maximum_foot_center_distance_m:g}m"
    )
    leg_crossing_text = (
        "enabled" if task.terminate_leg_crossing else "disabled"
    )

    print(
        "[training]\n"
        f"  preset={args.preset} physics={args.physics_profile} "
        f"model={train_env.model_path} "
        f"requested_steps={schedule['requested_steps']:,} "
        f"effective_steps={schedule['effective_steps']:,}\n"
        f"  rollout_quantum={schedule['rollout_quantum']:,} "
        f"eval_interval={schedule['eval_interval_steps']:,} "
        f"evals={values['num_evals']}\n"
        f"  envs={values['envs']} eval_envs={values['eval_envs']}\n"
        f"  episode={args.episode_length} steps "
        f"({args.episode_length * task.control_timestep:.2f}s) "
        f"batch={values['batch_size']} "
        f"minibatches={values['num_minibatches']}\n"
        f"  root_damping="
        f"{'disabled' if task.disable_root_damping else 'xml'} "
        f"root_low={root_low_text} "
        f"stuck={stuck_text} "
        f"root_high={root_high_text} "
        f"foot_gap={foot_gap_text} "
        f"leg_crossing={leg_crossing_text}\n"
        f"  lr={args.learning_rate:g} entropy={args.entropy_cost:g} "
        f"discount={args.discounting:g} seed={args.seed}\n"
        f"  reward roll={reward_config.roll_progress:g} "
        f"mismatch={reward_config.roll_mismatch:g} "
        f"termination={reward_config.termination:g} "
        f"early={reward_config.early_termination_scale:g}\n"
        "  selection=physical: 50% survival, 35% turns, "
        "10% non-failure, 5% contact safety "
        f"target_turns={args.selection_target_turns:g} "
        f"min_turns={args.selection_min_turns} "
        f"min_tail_turns={args.selection_min_tail_turns} "
        f"tail_window={task.tail_progress_window_s:g}s\n"
        f"  output={args.out.resolve()}",
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
            "[checkpoint]\n"
            f"  restoring={checkpoint_kwargs['restore_checkpoint_path']}",
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
    best_params = best["params"]
    reward_peak_params = (
        reward_peak["params"]
        if reward_peak["params"] is not None
        else final_params
    )
    model_io.save_params(args.out / "params_final", final_params)
    if best_params is not None:
        model_io.save_params(args.out / "params_best", best_params)
    model_io.save_params(args.out / "params_reward_peak", reward_peak_params)
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
        "best_eval_reward": best_reward_value,
        "best_step": best["step"],
        "best_selection_score": best_score_value,
        "valid_best_found": best_params is not None,
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
        else "none (all eval points rejected)"
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
        f"  physical_best {best_source}\n"
        f"  reward_peak {reward_peak_source}\n"
        f"  checkpoints best="
        f"{args.out / 'params_best' if best_params is not None else 'unavailable'} "
        f"reward_peak={args.out / 'params_reward_peak'} "
        f"final={args.out / 'params_final'}",
        flush=True,
    )

    if not args.skip_evaluation:
        evaluation_best_dir = args.out / "evaluation_best"
        evaluation_final_dir = args.out / "evaluation_final"
        evaluation_best = None
        if best_params is not None:
            evaluation_best = _evaluate_policy(
                eval_env,
                make_inference_fn,
                best_params,
                seed=args.seed + 20_000,
                episode_length=args.episode_length,
                output_dir=evaluation_best_dir,
            )
        evaluation_final = _evaluate_policy(
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
            },
            "best": evaluation_best,
            "final": evaluation_final,
        }
        (args.out / "policy_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        if evaluation_best is not None:
            print(_format_rollout_report("best", evaluation_best), flush=True)
        print(_format_rollout_report("final", evaluation_final), flush=True)

        if not args.skip_visualization:
            try:
                from scripts.render_mjx_policy import render_rollout

                render_targets = [("final", evaluation_final_dir)]
                if evaluation_best is not None:
                    render_targets.insert(0, ("best", evaluation_best_dir))
                for label, directory in render_targets:
                    gif_path = directory / "policy_rollout.gif"
                    render_summary = render_rollout(
                        directory / "evaluation_rollout.npz",
                        gif_path,
                        model_path=train_env.model_path,
                        control_dt=task.control_timestep,
                        fps=args.visualization_fps,
                        width=args.visualization_width,
                        height=args.visualization_height,
                        camera_distance=(
                            args.visualization_camera_distance
                        ),
                        diagnostics=args.visualization_diagnostics,
                    )
                    print(
                        f"[visualization {label}]\n"
                        f"  gif={gif_path.resolve()}\n"
                        f"  frames={render_summary['frames']} "
                        f"duration={render_summary['duration_s']:.2f}s",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    "[visualization warning]\n"
                    f"  automatic GIF rendering failed: {exc}\n"
                    "  rollout data was saved and can be rendered later.",
                    flush=True,
                )

        print(
            "[artifacts]\n"
            f"  summary={args.out / 'training_summary.json'}\n"
            f"  comparison={args.out / 'policy_comparison.json'}\n"
            f"  best="
            f"{evaluation_best_dir if evaluation_best is not None else 'unavailable'}\n"
            f"  final={evaluation_final_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
