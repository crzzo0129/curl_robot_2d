"""Scan ``target-scale -> actual rolling speed`` for a 3-D CEM reference.

Each scale point runs the CPU evaluator for the full ``--duration`` (10 s by
default) with the rollingquad_2 geometry and the ``cg20`` physics profile, and
writes the per-point evaluator JSON plus a summary CSV / Markdown table.

This establishes the ``v_cmd -> target_scale`` lookup for command-conditioned
training.  The mapping is reference-specific: do not reuse the old mapping from
the original safe reference on the high-speed reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import evaluate_3d_symmetric_cem_reference as bridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad_abd10.xml"
)
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "rollingquad_abd10_high_speed_zero_contact_refine_smoke"
    / "01_zero_contact_speed_refine"
    / "best_phase_controller.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "rollingquad_abd10_target_scale_probe"

TABLE_FIELDS = (
    ("target_scale", "target_scale"),
    ("distance_x_m", "distance_x_m"),
    ("mean_x_speed_m_s", "mean_x_speed_m_s"),
    ("final_quarter_x_speed_m_s", "final_quarter_x_speed_m_s"),
    ("rolling_speed_m_s", "rolling_speed_m_s"),
    ("rolling_translation_slip_m_s", "rolling_translation_slip_m_s"),
    ("self_contact_fraction", "self_contact_fraction"),
    ("maximum_self_penetration_m", "maximum_self_penetration_m"),
    ("distance_y_m", "distance_y_m"),
    ("tracking_rmse_rad", "tracking_rmse_rad"),
    ("torque_saturation_fraction", "torque_saturation_fraction"),
    ("phase_error_rms_rad", "phase_error_rms_rad"),
    ("status", "status"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--min-scale", type=float, default=0.30)
    parser.add_argument("--max-scale", type=float, default=1.05)
    parser.add_argument("--scale-step", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--geometry", default="rollingquad_2")
    parser.add_argument("--physics-profile", default="cg20")
    parser.add_argument("--front-abduction-deg", type=float, default=-10.0)
    parser.add_argument("--rear-abduction-deg", type=float, default=10.0)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    return parser.parse_args(argv)


def scale_key(scale: float) -> str:
    return f"scale_{scale:.2f}".replace("-", "m").replace(".", "p")


def run_one(args: argparse.Namespace, scale: float) -> dict[str, object]:
    argv = [
        "--xml", str(args.xml),
        "--controller", str(args.controller),
        "--geometry", args.geometry,
        "--physics-profile", args.physics_profile,
        "--front-abduction-deg", str(args.front_abduction_deg),
        "--rear-abduction-deg", str(args.rear_abduction_deg),
        "--kp", str(args.kp),
        "--kd", str(args.kd),
        "--torque-limit", str(args.torque_limit),
        "--duration", str(args.duration),
        "--target-scale", f"{scale:.4f}",
    ]
    namespace = bridge.parse_args(argv)
    summary = bridge.run_smoke(namespace)
    out = args.output_dir / f"{scale_key(scale)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.min_scale > args.max_scale or args.scale_step <= 0.0:
        raise SystemExit("--min-scale <= --max-scale and --scale-step > 0 required")
    scales = list(
        np.round(np.arange(args.min_scale, args.max_scale + 1.0e-9, args.scale_step), 8)
    )

    rows: list[dict[str, object]] = []
    for scale in scales:
        summary = run_one(args, float(scale))
        rows.append(summary)
        speed = summary.get("mean_x_speed_m_s", float("nan"))
        contact = summary.get("self_contact_fraction", float("nan"))
        print(
            f"target_scale={float(scale):.2f}  "
            f"mean_speed={speed:.3f} m/s  "
            f"self_contact={contact:.3f}  "
            f"status={summary.get('status')}",
            flush=True,
        )

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name for name, _ in TABLE_FIELDS])
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(key) for name, key in TABLE_FIELDS})

    summary_md = args.output_dir / "summary.md"
    lines = [
        "# target-scale -> actual speed scan",
        "",
        f"controller: `{args.controller}`",
        f"xml: `{args.xml}`",
        f"duration: {args.duration} s | geometry: {args.geometry} | "
        f"physics: {args.physics_profile} | kp={args.kp} kd={args.kd} "
        f"torque_limit={args.torque_limit}",
        f"front abduction: {args.front_abduction_deg} deg | "
        f"rear abduction: {args.rear_abduction_deg} deg",
        "",
        "| " + " | ".join(name for name, _ in TABLE_FIELDS) + " |",
        "|" + "---|" * len(TABLE_FIELDS),
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row.get(key)) for _, key in TABLE_FIELDS) + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {len(rows)} points to {args.output_dir}")
    print(f"  CSV:      {summary_csv}")
    print(f"  Markdown: {summary_md}")


if __name__ == "__main__":
    main()
