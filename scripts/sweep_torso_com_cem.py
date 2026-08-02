"""Sweep 2-D torso COM positions and re-optimize the CEM rolling controller."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.model import write_mjcf
from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts.optimize_phase_controller import (
    FOOT_GAP_TRACKING_MARGIN_M,
    _load_controller_parameters,
    optimize_controller,
    rollout_controller,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "torso_com_cem_sweep"
DEFAULT_COM_X = (0.000, 0.025, 0.050)
DEFAULT_COM_Z = (0.000, 0.015, 0.030)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso-com-x", type=float, nargs="+", default=list(DEFAULT_COM_X))
    parser.add_argument("--torso-com-z", type=float, nargs="+", default=list(DEFAULT_COM_Z))
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--final-duration", type=float, default=4.0)
    parser.add_argument("--barrier-generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimum-foot-gap-mm", type=float, default=2.0)
    parser.add_argument(
        "--foot-gap-tracking-margin-mm",
        type=float,
        default=1000.0 * FOOT_GAP_TRACKING_MARGIN_M,
    )
    parser.add_argument("--initial-controller", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def variant_name(com_x: float, com_z: float) -> str:
    return f"torso_com_x_{com_x:+.3f}_z_{com_z:+.3f}".replace("+", "p").replace("-", "m")


def run_variant(args, com_x: float, com_z: float, initial_parameters):
    variant = variant_name(com_x, com_z)
    variant_dir = args.output_dir / variant
    model_dir = args.output_dir / "models"
    model_path = model_dir / f"{variant}.xml"
    model_parameters = replace(
        FIXED_PARAMETERS,
        torso_com_x=float(com_x),
        torso_com_z=float(com_z),
    )
    write_mjcf(model_path, model_parameters)

    minimum_foot_surface_gap_m = args.minimum_foot_gap_mm / 1000.0
    foot_gap_tracking_margin_m = args.foot_gap_tracking_margin_mm / 1000.0
    parameters, history, _ = optimize_controller(
        model_path=model_path,
        generations=args.generations,
        population=args.population,
        elite_count=args.elite_count,
        duration=args.duration,
        seed=args.seed,
        barrier_generations=args.barrier_generations,
        initial_parameters=initial_parameters,
        workers=args.workers,
        minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
    )
    baseline_model = mujoco.MjModel.from_xml_path(str(model_path))
    baseline = rollout_controller(
        baseline_model,
        np.zeros(8),
        duration=args.final_duration,
        detailed=True,
    )
    controlled_model = mujoco.MjModel.from_xml_path(str(model_path))
    controlled = rollout_controller(
        controlled_model,
        parameters[:8],
        duration=args.final_duration,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
        detailed=True,
    )
    outputs = write_outputs(
        variant_dir,
        parameters,
        history,
        baseline,
        controlled,
        minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m,
    )
    result = {
        "variant": variant,
        "torso_com_x_m": float(com_x),
        "torso_com_z_m": float(com_z),
        "model_path": str(model_path.resolve()),
        "controller_path": str(outputs[0].resolve()),
        "score": float(controlled.score),
        "net_turns": float(controlled.summary["net_turns"]),
        "rolling_turns": float(controlled.summary["rolling_progress_turns"]),
        "conservative_rolling_turns": float(
            controlled.summary["conservative_rolling_turns"]
        ),
        "root_x_displacement_m": float(controlled.summary["root_x_displacement_m"]),
        "rolling_mismatch_rad": float(controlled.summary["rolling_mismatch_rad"]),
        "actuator_positive_work_J": float(controlled.summary["actuator_positive_work_J"]),
        "actuator_net_work_J": float(controlled.summary["actuator_net_work_J"]),
        "maximum_actuator_torque_Nm": float(controlled.summary["maximum_actuator_torque_Nm"]),
        "self_contact_total_s": float(controlled.summary["self_contact_total_s"]),
        "forbidden_contact_total_s": float(
            controlled.summary["forbidden_contact_total_s"]
        ),
        "maximum_forbidden_penetration_m": float(
            controlled.summary["maximum_forbidden_penetration_m"]
        ),
        "foot_contact_total_s": float(controlled.summary["foot_contact_total_s"]),
        "minimum_foot_surface_gap_m": float(
            controlled.summary["minimum_foot_surface_gap_m"]
        ),
        "leg_crossing_detected": bool(controlled.summary["leg_crossing_detected"]),
        "completed_two_turns": bool(controlled.summary["completed_two_turns"]),
    }
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.minimum_foot_gap_mm < 0.0:
        raise SystemExit("--minimum-foot-gap-mm cannot be negative")
    if args.foot_gap_tracking_margin_mm < 0.0:
        raise SystemExit("--foot-gap-tracking-margin-mm cannot be negative")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_parameters = (
        _load_controller_parameters(args.initial_controller)
        if args.initial_controller is not None
        else None
    )

    results = []
    total = len(args.torso_com_x) * len(args.torso_com_z)
    for index, com_x in enumerate(args.torso_com_x, start=1):
        for com_z in args.torso_com_z:
            run_index = len(results) + 1
            print(
                f"variant={run_index}/{total} torso_com=({com_x:+.3f},{com_z:+.3f})",
                flush=True,
            )
            result = run_variant(args, com_x, com_z, initial_parameters)
            results.append(result)
            print(
                f"  turns={result['conservative_rolling_turns']:.3f} "
                f"x={result['root_x_displacement_m']:.3f}m "
                f"collision={result['forbidden_contact_total_s']:.3f}s "
                f"score={result['score']:.3f}",
                flush=True,
            )

    results.sort(
        key=lambda item: (
            item["completed_two_turns"],
            item["conservative_rolling_turns"],
            -item["forbidden_contact_total_s"],
            -item["actuator_positive_work_J"],
        ),
        reverse=True,
    )
    csv_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"summary_csv={csv_path.resolve()}")
    print(f"summary_json={json_path.resolve()}")


if __name__ == "__main__":
    main()
