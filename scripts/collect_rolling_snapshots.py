"""Collect phase-balanced snapshots from the active 2-D rolling controller."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS, REAL_GEOMETRY_PARAMETERS
from scripts.optimize_phase_controller import _activate_geometry
from scripts.replay_active_controller import (
    DEFAULT_CONTROLLER_PATH,
    MODEL_PATH,
    REAL_GEOMETRY_MODEL_PATH,
    advance_controller,
    initialize_simulation,
    load_controller_options,
)
from scripts.run_release_baseline import _id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "rolling_stop" / "low_speed_snapshots.npz"


def phase_bin_index(phase_rad: float, bin_count: int) -> int:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    wrapped = float(phase_rad) % (2.0 * math.pi)
    return min(int(wrapped * bin_count / (2.0 * math.pi)), bin_count - 1)


@dataclass
class PhaseBalancedBuffer:
    bin_count: int
    samples_per_bin: int

    def __post_init__(self) -> None:
        if self.bin_count <= 0 or self.samples_per_bin <= 0:
            raise ValueError("bin count and quota must be positive")
        self.rows: list[dict[str, np.ndarray | float | int]] = []
        self.counts = np.zeros(self.bin_count, dtype=np.int32)

    @property
    def complete(self) -> bool:
        return bool(np.all(self.counts >= self.samples_per_bin))

    def add(self, phase_rad: float, row: dict[str, object]) -> bool:
        bin_index = phase_bin_index(phase_rad, self.bin_count)
        if self.counts[bin_index] >= self.samples_per_bin:
            return False
        stored = dict(row)
        stored["phase_bin"] = bin_index
        self.rows.append(stored)
        self.counts[bin_index] += 1
        return True


def _contact_features(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    floor = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    front_foot = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_foot_proxy")
    rear_foot = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "rear_foot_proxy")
    torso_proxy = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_proxy")
    front_ground = rear_ground = torso_ground = internal = foot_contact = False
    maximum_penetration = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        pair = {geom1, geom2}
        maximum_penetration = max(maximum_penetration, -float(contact.dist))
        if floor in pair:
            moving = geom2 if geom1 == floor else geom1
            front_ground |= moving == front_foot
            rear_ground |= moving == rear_foot
            torso_ground |= moving == torso_proxy
        else:
            allowed_foot_pair = pair == {front_foot, rear_foot}
            foot_contact |= allowed_foot_pair
            internal |= not allowed_foot_pair
    return np.asarray(
        [front_ground, rear_ground, torso_ground, internal, foot_contact,
         data.ncon, maximum_penetration],
        dtype=np.float64,
    )


def _stack_rows(buffer: PhaseBalancedBuffer) -> dict[str, np.ndarray]:
    if not buffer.rows:
        raise RuntimeError("no rolling snapshots were collected")
    keys = tuple(buffer.rows[0])
    return {key: np.asarray([row[key] for row in buffer.rows]) for key in keys}


def collect_snapshots(
    *,
    controller_path: Path,
    model_path: Path,
    geometry: str,
    frequency_hz: float,
    bin_count: int,
    samples_per_bin: int,
    sample_period_s: float,
    warmup_s: float,
    maximum_duration_s: float,
    rollout_duration_s: float = 10.0,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    for name, value in (("frequency_hz", frequency_hz), ("sample_period_s", sample_period_s),
                        ("maximum_duration_s", maximum_duration_s),
                        ("rollout_duration_s", rollout_duration_s)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if warmup_s < 0.0 or not math.isfinite(warmup_s):
        raise ValueError("warmup_s must be finite and nonnegative")
    geometry_parameters = REAL_GEOMETRY_PARAMETERS if geometry == "real" else FIXED_PARAMETERS
    _activate_geometry(geometry_parameters)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    coefficients, native_rate, coupling, foot_gap, foot_margin, knee_bias = load_controller_options(
        controller_path
    )
    phase_rate_scale = 2.0 * math.pi * frequency_hz / native_rate
    buffer = PhaseBalancedBuffer(bin_count, samples_per_bin)
    total_elapsed_s = total_turns = total_distance_m = 0.0
    episode_index = 0
    while total_elapsed_s < maximum_duration_s and not buffer.complete:
        data, pitch_address, _ = initialize_simulation(
            model, foot_gap, geometry_parameters
        )
        oscillator_phase = 0.0
        next_sample_s = warmup_s
        start_pitch = float(data.qpos[pitch_address])
        start_x = float(data.qpos[0])
        episode_limit_s = min(rollout_duration_s, maximum_duration_s - total_elapsed_s)
        while data.time < episode_limit_s and not buffer.complete:
            oscillator_phase = advance_controller(
                model, data, coefficients, native_rate, coupling, oscillator_phase,
                pitch_address, foot_gap, foot_margin, knee_bias, phase_rate_scale,
            )
            if data.time + 1.0e-12 < next_sample_s:
                continue
            body_phase = float(data.qpos[pitch_address])
            buffer.add(body_phase, {
                "time_s": total_elapsed_s + float(data.time),
                "episode_time_s": float(data.time),
                "episode_index": episode_index,
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(),
                "body_phase_rad": body_phase,
                "oscillator_phase_rad": float(oscillator_phase),
                "contact_features": _contact_features(model, data),
            })
            next_sample_s += sample_period_s
        episode_elapsed = float(data.time)
        total_elapsed_s += episode_elapsed
        total_turns += (float(data.qpos[pitch_address]) - start_pitch) / (2.0 * math.pi)
        total_distance_m += float(data.qpos[0]) - start_x
        episode_index += 1
    arrays = _stack_rows(buffer)
    elapsed = max(total_elapsed_s, 1.0e-12)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "controller_path": str(controller_path.resolve()),
        "model_path": str(model_path.resolve()),
        "geometry": geometry,
        "commanded_frequency_hz": frequency_hz,
        "actual_average_roll_frequency_hz": total_turns / elapsed,
        "distance_x_m": total_distance_m,
        "elapsed_s": elapsed,
        "bin_count": bin_count,
        "samples_per_bin": samples_per_bin,
        "bin_counts": buffer.counts.tolist(),
        "complete": buffer.complete,
        "sample_period_s": sample_period_s,
        "warmup_s": warmup_s,
        "rollout_duration_s": rollout_duration_s,
        "episode_count": episode_index,
        "friction_scale": 1.0,
        "ground_tilt_rad": 0.0,
        "left_right_differential": 0.0,
        "seed": 0,
        "contact_feature_names": [
            "front_foot_ground", "rear_foot_ground", "torso_ground",
            "forbidden_internal_contact", "allowed_foot_contact",
            "contact_count", "maximum_penetration_m",
        ],
    }
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER_PATH)
    parser.add_argument("--geometry", choices=("baseline", "real"), default="real")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--frequency-hz", type=float, default=0.40)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--samples-per-bin", type=int, default=50)
    parser.add_argument("--sample-period", type=float, default=0.02)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--maximum-duration", type=float, default=180.0)
    parser.add_argument("--rollout-duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model_path = args.model or (
        REAL_GEOMETRY_MODEL_PATH if args.geometry == "real" else MODEL_PATH
    )
    arrays, metadata = collect_snapshots(
        controller_path=args.controller, model_path=model_path, geometry=args.geometry,
        frequency_hz=args.frequency_hz, bin_count=args.bins,
        samples_per_bin=args.samples_per_bin, sample_period_s=args.sample_period,
        warmup_s=args.warmup, maximum_duration_s=args.maximum_duration,
        rollout_duration_s=args.rollout_duration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**metadata, "output": str(args.output)}, indent=2, sort_keys=True))
    if not metadata["complete"]:
        raise SystemExit("phase-bin quota was not reached before maximum duration")


if __name__ == "__main__":
    main()
