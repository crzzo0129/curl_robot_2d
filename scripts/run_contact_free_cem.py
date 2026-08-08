"""Warm-start a contact-free CEM cleanup stage on the real 2-D geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS
from scripts.optimize_phase_controller import (
    FOOT_GAP_TRACKING_MARGIN_M,
    _load_controller_parameters,
    optimize_controller,
    rollout_controller,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results/staged_cem_real_geometry_180_d50_foot60"
    / "03_foot_gap_2mm/best_phase_controller.json"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "results/staged_cem_real_geometry_180_d50_foot60"
    / "models/collision.xml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/contact_free_cem_real_geometry"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--minimum-turns", type=float, default=5.0)
    args = parser.parse_args()

    initial = _load_controller_parameters(args.controller)
    parameters, history, _ = optimize_controller(
        model_path=args.model,
        generations=args.generations,
        population=args.population,
        elite_count=10,
        duration=10.0,
        seed=args.seed,
        barrier_generations=0,
        initial_parameters=initial,
        workers=args.workers,
        minimum_foot_surface_gap_m=0.002,
        foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
        enforce_leg_crossing_constraint=True,
        allow_foot_contact=False,
        geometry_parameters=REAL_GEOMETRY_PARAMETERS,
    )
    model = mujoco.MjModel.from_xml_path(str(args.model))
    before = rollout_controller(
        model,
        initial[:8],
        duration=10.0,
        oscillator_rate=float(initial[8]),
        oscillator_coupling=float(initial[9]),
        minimum_foot_surface_gap_m=0.002,
        foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
        enforce_leg_crossing_constraint=True,
        allow_foot_contact=False,
        detailed=True,
    )
    model = mujoco.MjModel.from_xml_path(str(args.model))
    after = rollout_controller(
        model,
        parameters[:8],
        duration=10.0,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        minimum_foot_surface_gap_m=0.002,
        foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
        enforce_leg_crossing_constraint=True,
        allow_foot_contact=False,
        detailed=True,
    )
    accepted = (
        float(after.summary["conservative_rolling_turns"])
        >= args.minimum_turns
        and float(after.summary["forbidden_contact_total_s"]) == 0.0
        and not bool(after.summary["leg_crossing_detected"])
    )
    chosen_parameters = parameters if accepted else initial
    chosen_rollout = after if accepted else before
    write_outputs(
        args.output_dir,
        chosen_parameters,
        history,
        before,
        chosen_rollout,
        0.002,
        FOOT_GAP_TRACKING_MARGIN_M,
        allow_foot_contact=False,
    )
    result = {
        "accepted": accepted,
        "selection": "optimized" if accepted else "rollback_stage3",
        "minimum_turns": args.minimum_turns,
        "all_robot_self_contact_forbidden": True,
        "candidate": after.summary,
        "selected": chosen_rollout.summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
