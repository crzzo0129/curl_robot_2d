"""Roll representative rolling snapshots through phase-binned brake schedules."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS
from scripts.collect_rolling_snapshots import _contact_features, phase_bin_index
from scripts.optimize_phase_controller import _activate_geometry
from scripts.replay_active_controller import load_controller_options
from scripts.search_braking_schedule import (
    BrakingSchedule,
    DEFAULT_CONTROLLER,
    DEFAULT_MODEL,
    DEFAULT_SNAPSHOTS,
    evaluate_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "rolling_stop" / "braking_terminal_states.npz"
)
DEFAULT_REPORTS = {
    0: PROJECT_ROOT / "results" / "rolling_stop" / "braking_cem_phase00_target_aware_eval10.json",
    8: PROJECT_ROOT / "results" / "rolling_stop" / "braking_cem_phase08_target_aware.json",
    16: PROJECT_ROOT / "results" / "rolling_stop" / "braking_cem_phase16_target_aware.json",
    24: PROJECT_ROOT / "results" / "rolling_stop" / "braking_cem_phase24_target_aware.json",
}


def select_snapshot_indices(
    phase_bins: np.ndarray,
    contact_features: np.ndarray,
    requested_bins: list[int],
    samples_per_bin: int,
) -> list[tuple[int, int]]:
    if samples_per_bin <= 0:
        raise ValueError("samples_per_bin must be positive")
    valid = (contact_features[:, 2] == 0) & (contact_features[:, 3] == 0)
    selected: list[tuple[int, int]] = []
    for phase_bin in requested_bins:
        matches = np.flatnonzero((phase_bins == phase_bin) & valid)
        if len(matches) < samples_per_bin:
            raise RuntimeError(
                f"phase bin {phase_bin} has {len(matches)} valid snapshots; "
                f"need {samples_per_bin}"
            )
        selected.extend((phase_bin, int(index)) for index in matches[:samples_per_bin])
    return selected


def _load_schedule(path: Path) -> BrakingSchedule:
    report = json.loads(path.read_text(encoding="utf-8"))
    return BrakingSchedule(**report["best"]["schedule"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--samples-per-bin", type=int, default=3)
    parser.add_argument("--frequency-hz", type=float, default=0.40)
    parser.add_argument("--physics-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--schedule-report",
        action="append",
        nargs=2,
        metavar=("PHASE_BIN", "REPORT"),
        help="Override/add a phase-bin braking report; may be repeated.",
    )
    args = parser.parse_args()
    reports = dict(DEFAULT_REPORTS)
    if args.schedule_report:
        reports = {int(phase_bin): Path(path) for phase_bin, path in args.schedule_report}

    snapshots = np.load(args.snapshots)
    selected = select_snapshot_indices(
        snapshots["phase_bin"], snapshots["contact_features"],
        sorted(reports), args.samples_per_bin,
    )
    _activate_geometry(REAL_GEOMETRY_PARAMETERS)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    coefficients, native_rate, coupling, foot_gap, foot_margin, knee_bias = (
        load_controller_options(args.controller)
    )
    compact_id = model.key("compact").id
    compact_ctrl = np.asarray(model.key_ctrl[compact_id], dtype=float)
    initial_rate_scale = 2.0 * math.pi * args.frequency_hz / native_rate
    schedules = {phase_bin: _load_schedule(path) for phase_bin, path in reports.items()}

    rows: list[dict[str, object]] = []
    for source_bin, index in selected:
        terminal: dict[str, np.ndarray | float] = {}
        metrics = evaluate_schedule(
            model,
            snapshots["qpos"][index], snapshots["qvel"][index],
            float(snapshots["oscillator_phase_rad"][index]), schedules[source_bin],
            capture_time_s=float(snapshots["episode_time_s"][index]),
            coefficients=coefficients, native_rate_rad_s=native_rate,
            coupling_per_s=coupling, initial_phase_rate_scale=initial_rate_scale,
            minimum_foot_gap_m=foot_gap, foot_gap_margin_m=foot_margin,
            knee_bias_rad=knee_bias, compact_ctrl=compact_ctrl,
            physics_stride=args.physics_stride,
            nominal_body_roll_rate_rad_s=2.0 * math.pi * args.frequency_hz,
            terminal_state_out=terminal,
        )
        terminal_data = mujoco.MjData(model)
        terminal_data.qpos[:] = terminal["qpos"]
        terminal_data.qvel[:] = terminal["qvel"]
        terminal_data.ctrl[:] = terminal["ctrl"]
        terminal_data.time = float(terminal["time_s"])
        mujoco.mj_forward(model, terminal_data)
        rows.append({
            **terminal,
            "source_snapshot_index": index,
            "source_phase_bin": source_bin,
            "phase_bin": phase_bin_index(float(terminal_data.qpos[2]), 32),
            "contact_features": _contact_features(model, terminal_data),
            "deploy_entry_gate": metrics.deploy_entry_gate,
            "braking_score": metrics.score,
            "final_linear_speed_m_s": metrics.final_linear_speed_m_s,
            "final_angular_speed_rad_s": metrics.final_angular_speed_rad_s,
            "final_phase_error_rad": metrics.final_phase_error_rad,
        })
        print(json.dumps({
            "source_phase_bin": source_bin,
            "source_snapshot_index": index,
            "deploy_entry_gate": metrics.deploy_entry_gate,
            "linear_speed_m_s": metrics.final_linear_speed_m_s,
            "angular_speed_rad_s": metrics.final_angular_speed_rad_s,
            "phase_error_rad": metrics.final_phase_error_rad,
        }, sort_keys=True))

    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    gate_rate = float(np.mean(arrays["deploy_entry_gate"]))
    metadata = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "controller": str(args.controller.resolve()),
        "source_snapshots": str(args.snapshots.resolve()),
        "output": str(args.output.resolve()),
        "samples": len(rows),
        "samples_per_bin": args.samples_per_bin,
        "source_phase_bins": sorted(reports),
        "schedule_reports": {str(key): str(path.resolve()) for key, path in reports.items()},
        "deploy_entry_rate": gate_rate,
        "schedules": {str(key): asdict(value) for key, value in schedules.items()},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
