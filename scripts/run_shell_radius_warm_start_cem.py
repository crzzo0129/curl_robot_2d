"""Warm-start strict-collision CEM after enlarging the segmented shell."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco

from curl_robot_2d.model import write_mjcf
from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts.evaluate_fixed_reference_shell_radius import (
    DEFAULT_CONTROLLER,
    _controller_settings,
)
from scripts.optimize_phase_controller import (
    _load_controller_parameters,
    optimize_controller,
    rollout_controller,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "shell_radius_160mm_warm_start_cem"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--shell-radius-mm", type=float, default=160.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warm-start-std-scale", type=float, default=0.35)
    parser.add_argument("--collision-penalty-scale", type=float, default=3.0)
    return parser.parse_args(argv)


def _rollout(model_path, parameters, controller, duration, minimum_gap_m,
             tracking_margin_m, collision_penalty_scale):
    from scripts import optimize_phase_controller as phase_controller

    phase_controller._activate_geometry(parameters)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return rollout_controller(
        model,
        controller[:8],
        duration=duration,
        oscillator_rate=float(controller[8]),
        oscillator_coupling=float(controller[9]),
        minimum_foot_surface_gap_m=minimum_gap_m,
        foot_gap_tracking_margin_m=tracking_margin_m,
        enforce_leg_crossing_constraint=True,
        allow_foot_contact=True,
        collision_penalty_scale=collision_penalty_scale,
        detailed=True,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.shell_radius_mm <= 0.0:
        raise SystemExit("--shell-radius-mm must be positive")
    if args.generations <= 0 or args.population <= 0 or args.workers <= 0:
        raise SystemExit("generations, population, and workers must be positive")

    controller_path = args.controller.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    geometry = replace(
        FIXED_PARAMETERS,
        shell_contact_radius_override=args.shell_radius_mm / 1000.0,
        shell_arc_coverage_angle_override=FIXED_PARAMETERS.shell_arc_coverage_angle,
    )
    model_path = write_mjcf(
        output_dir / "model.xml",
        geometry,
        disable_shell_shell_collision=True,
    )
    initial = _load_controller_parameters(controller_path)
    minimum_gap_m, tracking_margin_m, allow_foot_contact = _controller_settings(
        controller_path
    )
    if not allow_foot_contact:
        raise SystemExit("This runner expects the warm reference to allow foot contact")

    before = _rollout(
        model_path,
        geometry,
        initial,
        args.duration,
        minimum_gap_m,
        tracking_margin_m,
        args.collision_penalty_scale,
    )
    best, history, _ = optimize_controller(
        model_path=model_path,
        generations=args.generations,
        population=args.population,
        elite_count=args.elite_count,
        duration=args.duration,
        seed=args.seed,
        barrier_generations=0,
        initial_parameters=initial,
        workers=args.workers,
        minimum_foot_surface_gap_m=minimum_gap_m,
        foot_gap_tracking_margin_m=tracking_margin_m,
        enforce_leg_crossing_constraint=True,
        allow_foot_contact=True,
        collision_penalty_scale=args.collision_penalty_scale,
        warm_start_std_scale=args.warm_start_std_scale,
        geometry_parameters=geometry,
    )
    after = _rollout(
        model_path,
        geometry,
        best,
        args.duration,
        minimum_gap_m,
        tracking_margin_m,
        args.collision_penalty_scale,
    )
    write_outputs(
        output_dir,
        best,
        history,
        before,
        after,
        minimum_gap_m,
        tracking_margin_m,
        allow_foot_contact=True,
        collision_penalty_scale=args.collision_penalty_scale,
    )
    result = {
        "warm_start_controller": str(controller_path),
        "model_path": str(model_path.resolve()),
        "shell_contact_radius_m": geometry.shell_contact_radius,
        "shell_arc_coverage_angle_rad": geometry.shell_arc_coverage_angle,
        "shell_shell_collision_disabled": True,
        "minimum_foot_surface_gap_m": minimum_gap_m,
        "foot_gap_tracking_margin_m": tracking_margin_m,
        "collision_penalty_scale": args.collision_penalty_scale,
        "generations": args.generations,
        "population": args.population,
        "seed": args.seed,
        "warm_start_std_scale": args.warm_start_std_scale,
        "before": before.summary,
        "after": after.summary,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
