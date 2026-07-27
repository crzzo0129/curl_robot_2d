"""Compare the reference and light physics profiles with the frozen CEM policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco

from curl_robot_2d_mjx.config import PHYSICS_PROFILE_NAMES, physics_profile
from curl_robot_2d_mjx.environment import apply_physics_options
from scripts.optimize_phase_controller import (
    _load_controller_parameters,
    rollout_controller,
)
from scripts.run_release_baseline import MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "collision_constrained_cem"
    / "best_phase_controller.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "physics_profile_comparison"
    / "comparison_solver_matrix.json"
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PHYSICS_PROFILE_NAMES,
        default=list(PHYSICS_PROFILE_NAMES),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    parameters = _load_controller_parameters(args.controller)
    coefficients = parameters[:8]
    oscillator_rate = float(parameters[8])
    oscillator_coupling = float(parameters[9])
    results = {}
    for name in args.profiles:
        task = physics_profile(name)
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        apply_physics_options(model, task)
        start = time.perf_counter()
        rollout = rollout_controller(
            model,
            coefficients,
            duration=args.duration,
            oscillator_rate=oscillator_rate,
            oscillator_coupling=oscillator_coupling,
            objective="sustained",
            detailed=False,
        )
        wall_time = time.perf_counter() - start
        results[name] = {
            "physics": {
                "timestep": task.physics_timestep,
                "control_timestep": task.control_timestep,
                "action_repeat": task.action_repeat,
                "solver": task.solver_name,
                "integrator": task.integrator_name,
                "cone": task.cone_name,
                "jacobian": task.jacobian_name,
                "iterations": task.solver_iterations,
                "ls_iterations": task.solver_ls_iterations,
            },
            "wall_time_s": wall_time,
            "rollout": rollout.summary,
        }
        print(
            f"profile={name} wall={wall_time:.3f}s "
            f"turns={rollout.summary['conservative_rolling_turns']:.3f} "
            f"forbidden={rollout.summary['forbidden_contact_fraction']:.4f} "
            f"max_penetration_mm="
            f"{1000.0 * rollout.summary['maximum_forbidden_penetration_m']:.3f}",
            flush=True,
        )

    reference = results.get("reference")
    comparison = {}
    if reference is not None:
        reference_rollout = reference["rollout"]
        for name, candidate in results.items():
            if name == "reference":
                continue
            candidate_rollout = candidate["rollout"]
            comparison[name] = {
                "cpu_wall_speedup": (
                    reference["wall_time_s"]
                    / max(candidate["wall_time_s"], 1e-12)
                ),
                "conservative_turn_difference": (
                    candidate_rollout["conservative_rolling_turns"]
                    - reference_rollout["conservative_rolling_turns"]
                ),
                "forbidden_contact_fraction_difference": (
                    candidate_rollout["forbidden_contact_fraction"]
                    - reference_rollout["forbidden_contact_fraction"]
                ),
                "maximum_forbidden_penetration_difference_m": (
                    candidate_rollout["maximum_forbidden_penetration_m"]
                    - reference_rollout["maximum_forbidden_penetration_m"]
                ),
            }

    payload = {
        "controller": str(args.controller),
        "model": str(MODEL_PATH),
        "duration_s": args.duration,
        "profiles": results,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.output
    suffix = 1
    while output_path.exists():
        output_path = args.output.with_name(
            f"{args.output.stem}_{suffix}{args.output.suffix}"
        )
        suffix += 1
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path, flush=True)


if __name__ == "__main__":
    main()
