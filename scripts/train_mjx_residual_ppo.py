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


def _parse_args():
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
        "--reference-weights",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.0],
    )
    parser.add_argument("--minimum-residual-gain", type=float, default=0.05)
    parser.add_argument("--minimum-stage-steps", type=int, default=500_000)
    parser.add_argument("--gate-check-steps", type=int, default=500_000)
    parser.add_argument("--gate-min-survival", type=float, default=0.80)
    parser.add_argument("--gate-min-turns", type=float, default=3.0)
    parser.add_argument("--gate-max-failure-rate", type=float, default=0.20)
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
    args = parser.parse_args()
    _validate_weights(parser, args.reference_weights)
    if not 0.0 <= args.minimum_residual_gain <= 1.0:
        parser.error("--minimum-residual-gain must be in [0, 1]")
    return args


def main() -> None:
    args = _parse_args()
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
        NominalRLConfig(episode_length=args.episode_length),
    )
    reward_config = _reward_config_from_args(args)
    base_reference = load_cem_reference(
        args.controller,
        minimum_residual_gain=args.minimum_residual_gain,
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
        "reference_weights": args.reference_weights,
        "minimum_stage_steps": args.minimum_stage_steps,
        "gate_check_steps": args.gate_check_steps,
        "gate": {
            "minimum_survival": args.gate_min_survival,
            "minimum_turns": args.gate_min_turns,
            "maximum_failure_rate": args.gate_max_failure_rate,
        },
        "runtime": runtime,
    }
    (args.out / "training_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[residual curriculum]\n"
        f"  requested_steps={values['steps']:,} "
        f"fixed_budget={effective_budget:,} "
        f"rollout_quantum={rollout_quantum:,}\n"
        f"  reference_weights={args.reference_weights} "
        f"minimum_stage={args.minimum_stage_steps:,}\n"
        f"  physics={args.physics_profile} root_damping="
        f"{'disabled' if task.disable_root_damping else 'xml'}\n"
        f"  gate survival>={args.gate_min_survival:.0%} "
        f"turns>={args.gate_min_turns:g} "
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
    best_zero = {"score": float("-inf"), "step": None, "params": None}
    zero_reference_steps = 0
    zero_reference_gate_passed = False

    while consumed_steps < effective_budget:
        weight = args.reference_weights[stage_index]
        reference = base_reference.with_weight(weight)
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
        }
        stage_start = consumed_steps
        print(
            f"[stage {stage_index + 1}/{len(args.reference_weights)}]\n"
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
                    f"ref={weight:.2f}\n"
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
            nonlocal eval_counter, zero_reference_gate_passed
            clean = {name: _float(value) for name, value in metrics.items()}
            _add_per_step_eval_metrics(clean)
            global_step = stage_start + int(local_step)
            gate = _gate_assessment(
                clean,
                episode_length=args.episode_length,
                minimum_survival=args.gate_min_survival,
                minimum_turns=args.gate_min_turns,
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
            selected = (
                weight == 0.0
                and gate["passed"]
                and not selection["rejected"]
                and selection["score"] > best_zero["score"]
            )
            if weight == 0.0 and gate["passed"]:
                zero_reference_gate_passed = True
            if selected and stage["params"] is not None:
                best_zero["score"] = selection["score"]
                best_zero["step"] = global_step
                best_zero["params"] = stage["params"]
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
            print(
                "  residual "
                f"ref_weight={weight:.2f} "
                f"gain={reference.residual_gain:.3f} "
                f"ref_rms="
                f"{clean.get('eval/avg_reference_action_rms', 0.0):.3f} "
                f"policy_rms="
                f"{clean.get('eval/avg_residual_action_rms', 0.0):.3f} "
                f"effective_residual_rms="
                f"{reference.residual_gain * clean.get('eval/avg_residual_action_rms', 0.0):.3f}",
                flush=True,
            )
            visualize_eval(local_step)
            failed_checks = [
                name for name, passed in gate["checks"].items() if not passed
            ]
            print(
                "  curriculum "
                f"global_eval={eval_counter} "
                f"stage={stage_index + 1}/{len(args.reference_weights)} "
                f"ref={weight:.2f} local_steps={int(local_step):,} "
                f"gate={'PASS' if gate['passed'] else 'WAIT'} "
                f"missing={','.join(failed_checks) or 'none'}",
                flush=True,
            )
            can_advance = (
                stage_index < len(args.reference_weights) - 1
                and int(local_step) >= args.minimum_stage_steps
                and gate["passed"]
            )
            if can_advance:
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
            final_params = stage_final_params
            final_metrics = stage_final_metrics or {}
            restored_params = stage_final_params
            advanced = False
        except _AdvanceCurriculum:
            local_trained = int(stage["local_step"])
            if local_trained <= 0 or stage["params"] is None:
                raise RuntimeError("curriculum advanced without policy params")
            make_inference_fn = stage["make_inference_fn"]
            final_params = stage["params"]
            final_metrics = stage["last_metrics"]
            restored_params = stage["params"]
            advanced = True

        consumed_steps += local_trained
        if weight == 0.0:
            zero_reference_steps += local_trained
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
            }
        )
        model_io.save_params(
            args.out / f"params_stage_{stage_index}_final",
            final_params,
        )
        if advanced:
            stage_index += 1
        else:
            break

    reached_zero = zero_reference_steps > 0
    last_trained_weight = stage_history[-1]["reference_weight"]
    curriculum_success = (
        reached_zero
        and zero_reference_gate_passed
        and stage_index == len(args.reference_weights) - 1
        and consumed_steps == effective_budget
    )
    model_io.save_params(args.out / "params_final", final_params)
    if best_zero["params"] is not None:
        model_io.save_params(
            args.out / "params_best_zero_reference",
            best_zero["params"],
        )
    elapsed = time.perf_counter() - start
    clean_final_metrics = {
        name: _float(value) for name, value in final_metrics.items()
    }
    summary = {
        "curriculum_success": curriculum_success,
        "reached_zero_reference": reached_zero,
        "zero_reference_gate_passed": zero_reference_gate_passed,
        "zero_reference_training_steps": zero_reference_steps,
        "requested_steps": values["steps"],
        "effective_budget_steps": effective_budget,
        "consumed_steps": consumed_steps,
        "final_reference_weight": last_trained_weight,
        "best_zero_reference_step": best_zero["step"],
        "best_zero_reference_score": (
            best_zero["score"]
            if math.isfinite(best_zero["score"])
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
        f"  status={'SUCCESS' if curriculum_success else 'FAILED'} "
        f"consumed={consumed_steps:,}/{effective_budget:,}\n"
        f"  final_reference_weight="
        f"{last_trained_weight:.2f} "
        f"zero_reference_steps={zero_reference_steps:,}\n"
        f"  elapsed={elapsed / 60.0:.1f}min",
        flush=True,
    )

    zero_reference = base_reference.with_weight(0.0)
    zero_eval_env = make_brax_env(
        task,
        reward_config=reward_config,
        cem_reference=zero_reference,
        seed=args.seed + 20_000,
    )
    evaluation_dir = args.out / "evaluation_zero_reference"
    evaluation = _evaluate_policy(
        zero_eval_env,
        make_inference_fn,
        (
            best_zero["params"]
            if best_zero["params"] is not None
            else final_params
        ),
        seed=args.seed + 20_000,
        episode_length=args.episode_length,
        output_dir=evaluation_dir,
    )
    print(_format_rollout_report("zero-reference", evaluation), flush=True)
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
