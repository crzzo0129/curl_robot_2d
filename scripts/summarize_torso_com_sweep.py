"""Summarize completed torso-COM CEM sweep variants.

This is intentionally tolerant of interrupted sweeps: every variant directory
that already has a best_phase_controller.json is included.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = PROJECT_ROOT / "results" / "torso_com_cem_sweep"
VARIANT_RE = re.compile(r"^torso_com_x_([mp]\d+\.\d+)_z_([mp]\d+\.\d+)$")


def parse_signed_token(token: str) -> float:
    sign = -1.0 if token.startswith("m") else 1.0
    return sign * float(token[1:])


def parse_variant_name(name: str) -> tuple[float, float]:
    match = VARIANT_RE.match(name)
    if match is None:
        raise ValueError(f"not a torso-COM variant directory: {name}")
    return parse_signed_token(match.group(1)), parse_signed_token(match.group(2))


def read_variant(variant_dir: Path) -> dict:
    controller_path = variant_dir / "best_phase_controller.json"
    payload = json.loads(controller_path.read_text(encoding="utf-8"))
    summary = payload["rollout_summary"]
    com_x, com_z = parse_variant_name(variant_dir.name)
    return {
        "variant": variant_dir.name,
        "torso_com_x_m": com_x,
        "torso_com_z_m": com_z,
        "score": float(summary["score"]),
        "net_turns": float(summary["net_turns"]),
        "rolling_turns": float(summary["rolling_progress_turns"]),
        "conservative_rolling_turns": float(summary["conservative_rolling_turns"]),
        "root_x_displacement_m": float(summary["root_x_displacement_m"]),
        "rolling_mismatch_rad": float(summary["rolling_mismatch_rad"]),
        "actuator_positive_work_J": float(summary["actuator_positive_work_J"]),
        "actuator_net_work_J": float(summary["actuator_net_work_J"]),
        "maximum_actuator_torque_Nm": float(summary["maximum_actuator_torque_Nm"]),
        "self_contact_total_s": float(summary["self_contact_total_s"]),
        "forbidden_contact_total_s": float(summary["forbidden_contact_total_s"]),
        "maximum_forbidden_penetration_m": float(
            summary["maximum_forbidden_penetration_m"]
        ),
        "foot_contact_total_s": float(summary["foot_contact_total_s"]),
        "minimum_foot_surface_gap_m": float(summary["minimum_foot_surface_gap_m"]),
        "leg_crossing_detected": bool(summary["leg_crossing_detected"]),
        "completed_two_turns": bool(summary["completed_two_turns"]),
        "controller_path": str(controller_path.resolve()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--ranked-csv", type=Path, default=None)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    variant_dirs = [
        path
        for path in args.sweep_dir.iterdir()
        if path.is_dir() and (path / "best_phase_controller.json").exists()
    ]
    rows = [read_variant(path) for path in variant_dirs]
    if not rows:
        raise SystemExit(f"no completed variants found under {args.sweep_dir}")

    grid_rows = sorted(rows, key=lambda item: (item["torso_com_x_m"], item["torso_com_z_m"]))
    ranked_rows = sorted(
        rows,
        key=lambda item: (
            item["completed_two_turns"],
            item["conservative_rolling_turns"],
            -item["forbidden_contact_total_s"],
            -item["actuator_positive_work_J"],
        ),
        reverse=True,
    )

    csv_path = args.csv or (args.sweep_dir / "summary_completed.csv")
    json_path = args.json or (args.sweep_dir / "summary_completed.json")
    ranked_csv_path = args.ranked_csv or (args.sweep_dir / "summary_ranked.csv")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(grid_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, grid_rows)
    write_csv(ranked_csv_path, ranked_rows)

    print(f"completed_variants={len(rows)}")
    print(f"summary_csv={csv_path.resolve()}")
    print(f"summary_json={json_path.resolve()}")
    print(f"ranked_csv={ranked_csv_path.resolve()}")


if __name__ == "__main__":
    main()
