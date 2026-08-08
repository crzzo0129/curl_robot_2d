"""Gradually tighten all-robot self-contact time to zero."""

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
SOURCE_ROOT = PROJECT_ROOT / "results/staged_cem_real_geometry_180_d50_foot60"


def evaluate(model_path: Path, parameters, contact_limit: float):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return rollout_controller(
        model,
        parameters[:8],
        duration=10.0,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        minimum_foot_surface_gap_m=0.002,
        foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
        enforce_leg_crossing_constraint=True,
        allow_foot_contact=False,
        maximum_self_contact_time_s=contact_limit,
        detailed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        type=Path,
        default=SOURCE_ROOT / "03_foot_gap_2mm/best_phase_controller.json",
    )
    parser.add_argument(
        "--model", type=Path, default=SOURCE_ROOT / "models/collision.xml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/contact_free_curriculum_real_geometry",
    )
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-turns", type=float, default=5.0)
    args = parser.parse_args()

    stages = (
        ("01_contact_260ms", 0.260),
        ("02_contact_200ms", 0.200),
        ("03_contact_150ms", 0.150),
        ("04_contact_100ms", 0.100),
        ("05_contact_50ms", 0.050),
        ("06_contact_20ms", 0.020),
        ("07_contact_free", 0.0),
    )
    selected = _load_controller_parameters(args.controller)
    results = []
    for index, (name, limit) in enumerate(stages):
        before = evaluate(args.model, selected, limit)
        candidate, history, _ = optimize_controller(
            model_path=args.model,
            generations=args.generations,
            population=args.population,
            elite_count=10,
            duration=10.0,
            seed=71 + index,
            barrier_generations=0,
            initial_parameters=selected,
            workers=args.workers,
            minimum_foot_surface_gap_m=0.002,
            foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
            enforce_leg_crossing_constraint=True,
            allow_foot_contact=False,
            maximum_self_contact_time_s=limit,
            geometry_parameters=REAL_GEOMETRY_PARAMETERS,
        )
        after = evaluate(args.model, candidate, limit)
        accepted = (
            float(after.summary["conservative_rolling_turns"])
            >= args.minimum_turns
            and float(after.summary["forbidden_contact_total_s"])
            <= limit + 1e-12
            and not bool(after.summary["leg_crossing_detected"])
        )
        stage_dir = args.output_dir / name
        write_outputs(
            stage_dir,
            candidate if accepted else selected,
            history,
            before,
            after if accepted else before,
            0.002,
            FOOT_GAP_TRACKING_MARGIN_M,
            allow_foot_contact=False,
        )
        result = {
            "stage": name,
            "contact_limit_s": limit,
            "accepted": accepted,
            "candidate": after.summary,
        }
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(result)
        print(
            f"stage={name} accepted={accepted} "
            f"turns={float(after.summary['conservative_rolling_turns']):.3f} "
            f"contact={float(after.summary['forbidden_contact_total_s']):.3f}s",
            flush=True,
        )
        if not accepted:
            break
        selected = candidate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
