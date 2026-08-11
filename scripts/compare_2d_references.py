"""Paired, controller-only evaluation of two 2-D CEM references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

from scripts.optimize_phase_controller import (
    FOOT_GAP_TRACKING_MARGIN_M,
    _load_controller_parameters,
    rollout_controller,
)


def _evaluate(
    model_path: Path, controller_path: Path, duration: float, minimum_foot_gap_m: float
) -> dict:
    parameters = _load_controller_parameters(controller_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    rollout = rollout_controller(
        model,
        parameters[:8],
        duration=duration,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        minimum_foot_surface_gap_m=minimum_foot_gap_m,
        foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
        enforce_leg_crossing_constraint=True,
        allow_foot_contact=False,
        detailed=True,
    )
    return {
        "controller": str(controller_path.resolve()),
        **rollout.summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--minimum-foot-gap-mm", type=float, default=2.0)
    parser.add_argument("--minimum-turn-gain", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    minimum_foot_gap_m = args.minimum_foot_gap_mm / 1000.0
    old = _evaluate(args.model, args.old, args.duration, minimum_foot_gap_m)
    new = _evaluate(args.model, args.new, args.duration, minimum_foot_gap_m)
    old_turns = float(old["conservative_rolling_turns"])
    new_turns = float(new["conservative_rolling_turns"])
    old_contact = float(old["forbidden_contact_total_s"])
    new_contact = float(new["forbidden_contact_total_s"])
    passed = (
        new_turns >= old_turns + args.minimum_turn_gain
        and new_contact < old_contact
        and not bool(new["leg_crossing_detected"])
    )
    result = {
        "protocol": "same model, reset, duration, foot-gap target, and zero-residual controller rollout",
        "gate": {
            "passed": passed,
            "minimum_turn_gain": args.minimum_turn_gain,
            "requires_lower_leg_contact": True,
        },
        "old": old,
        "new": new,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["gate"], ensure_ascii=False))
    print(f"old turns={old_turns:.3f} leg_contact={old_contact:.3f}s")
    print(f"new turns={new_turns:.3f} leg_contact={new_contact:.3f}s")


if __name__ == "__main__":
    main()
