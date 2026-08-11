"""A/B replay of the old 2-D reference with a larger segmented shell."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco

from curl_robot_2d.model import write_mjcf
from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts import optimize_phase_controller as phase_controller


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "collision_constrained_cem_foot_gap_2mm_short_contact"
    / "best_phase_controller.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "old_reference_shell_radius_160mm"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--shell-radius-mm", type=float, default=160.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _controller_settings(path: Path) -> tuple[float, float, bool]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    objective = payload.get("collision_objective", {})
    return (
        float(payload.get("minimum_foot_surface_gap_m", 0.0)),
        float(payload.get("foot_gap_tracking_margin_m", 0.004)),
        not bool(objective.get("all_robot_self_contact_forbidden", False)),
    )


def _evaluate(
    *,
    name: str,
    parameters,
    disable_shell_shell_collision: bool,
    controller,
    duration: float,
    minimum_gap_m: float,
    tracking_margin_m: float,
    allow_foot_contact: bool,
    output_dir: Path,
) -> dict[str, object]:
    variant_dir = output_dir / name
    model_path = write_mjcf(
        variant_dir / "model.xml",
        parameters,
        disable_shell_shell_collision=disable_shell_shell_collision,
    )
    phase_controller._activate_geometry(parameters)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    baseline = phase_controller.rollout_controller(
        model,
        controller[:8] * 0.0,
        duration=duration,
        minimum_foot_surface_gap_m=minimum_gap_m,
        foot_gap_tracking_margin_m=tracking_margin_m,
        allow_foot_contact=allow_foot_contact,
        detailed=True,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    controlled = phase_controller.rollout_controller(
        model,
        controller[:8],
        duration=duration,
        oscillator_rate=float(controller[8]),
        oscillator_coupling=float(controller[9]),
        minimum_foot_surface_gap_m=minimum_gap_m,
        foot_gap_tracking_margin_m=tracking_margin_m,
        allow_foot_contact=allow_foot_contact,
        detailed=True,
    )
    phase_controller.write_outputs(
        variant_dir,
        controller,
        [
            {
                "generation": 0,
                "objective": "fixed_reference",
                "generation_best_score": float(controlled.score),
            }
        ],
        baseline,
        controlled,
        minimum_gap_m,
        tracking_margin_m,
        allow_foot_contact=allow_foot_contact,
    )
    return {
        "variant": name,
        "model_path": str(model_path.resolve()),
        "edge_length_m": parameters.edge_length,
        "foot_diameter_m": 2.0 * parameters.foot_radius,
        "shell_contact_radius_m": parameters.shell_contact_radius,
        "shell_arc_coverage_angle_rad": parameters.shell_arc_coverage_angle,
        "shell_shell_collision_disabled": disable_shell_shell_collision,
        **controlled.summary,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.shell_radius_mm <= 0.0:
        raise SystemExit("--shell-radius-mm must be positive")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    controller_path = args.controller.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    controller = phase_controller._load_controller_parameters(controller_path)
    minimum_gap_m, tracking_margin_m, allow_foot_contact = (
        _controller_settings(controller_path)
    )
    enlarged = replace(
        FIXED_PARAMETERS,
        shell_contact_radius_override=args.shell_radius_mm / 1000.0,
        shell_arc_coverage_angle_override=(
            FIXED_PARAMETERS.shell_arc_coverage_angle
        ),
    )
    variants = (
        ("original_collision", FIXED_PARAMETERS, False),
        ("original_no_shell_shell", FIXED_PARAMETERS, True),
        (f"shell_{args.shell_radius_mm:g}mm", enlarged, True),
    )
    results = [
        _evaluate(
            name=name,
            parameters=parameters,
            disable_shell_shell_collision=disable_collision,
            controller=controller,
            duration=args.duration,
            minimum_gap_m=minimum_gap_m,
            tracking_margin_m=tracking_margin_m,
            allow_foot_contact=allow_foot_contact,
            output_dir=output_dir,
        )
        for name, parameters, disable_collision in variants
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for result in results:
        print(
            f"variant={result['variant']} "
            f"radius={1000.0 * float(result['shell_contact_radius_m']):.3f}mm "
            f"turns={float(result['conservative_rolling_turns']):.3f} "
            f"forbidden={float(result['forbidden_contact_total_s']):.3f}s "
            f"score={float(result['score']):.3f}",
            flush=True,
        )
    print(f"output={summary_path}")


if __name__ == "__main__":
    main()
