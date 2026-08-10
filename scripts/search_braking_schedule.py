"""Offline CEM feasibility search for phase-aware deterministic braking."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d_mjx.stop_task import (
    required_braking_phase_distance,
    select_reachable_target_phase_unwrapped,
    smoothstep01,
    wrap_to_pi,
)
from scripts.optimize_phase_controller import controller_targets, _activate_geometry
from scripts.replay_active_controller import load_controller_options
from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "curl_robot_2d_real_geometry.xml"
DEFAULT_CONTROLLER = (
    PROJECT_ROOT / "results" / "staged_cem_real_geometry_180_d50_foot60"
    / "03_foot_gap_2mm" / "best_phase_controller.json"
)
DEFAULT_SNAPSHOTS = (
    PROJECT_ROOT / "results" / "rolling_stop" / "low_speed_snapshots_0p40hz.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "rolling_stop" / "braking_cem.json"

# duration scale, brake start progress, end rate/amplitude, and two knots.
LOWER = np.asarray(
    [0.60, 0.20, 0.0, 0.0, *([-0.25] * 4), *([-0.20] * 4)], dtype=float
)
UPPER = np.asarray(
    [1.60, 0.80, 0.65, 1.0, *([0.25] * 4), *([0.20] * 4)], dtype=float
)


@dataclass(frozen=True)
class BrakingSchedule:
    duration_scale: float
    brake_start_progress: float
    end_phase_rate_scale: float
    end_amplitude_scale: float
    midpoint_joint_offset_rad: tuple[float, float, float, float]
    terminal_joint_offset_rad: tuple[float, float, float, float]

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "BrakingSchedule":
        values = np.clip(np.asarray(vector, dtype=float), LOWER, UPPER)
        return cls(
            duration_scale=float(values[0]),
            brake_start_progress=float(values[1]),
            end_phase_rate_scale=float(values[2]),
            end_amplitude_scale=float(values[3]),
            midpoint_joint_offset_rad=tuple(float(value) for value in values[4:8]),
            terminal_joint_offset_rad=tuple(float(value) for value in values[8:12]),
        )

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.duration_scale,
                self.brake_start_progress,
                self.end_phase_rate_scale,
                self.end_amplitude_scale,
                *self.midpoint_joint_offset_rad,
                *self.terminal_joint_offset_rad,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class BrakingMetrics:
    score: float
    final_linear_speed_m_s: float
    final_angular_speed_rad_s: float
    final_phase_error_rad: float
    final_joint_speed_rms_rad_s: float
    minimum_root_height_m: float
    maximum_torque_nm: float
    torso_contact: bool
    forbidden_internal_contact: bool
    final_forbidden_internal_contact: bool
    torso_contact_duration_s: float
    forbidden_internal_contact_duration_s: float
    numerical_failure: bool
    deploy_entry_gate: bool
    target_phase_distance_rad: float
    planned_duration_s: float


def schedule_scales(
    schedule: BrakingSchedule,
    phase_progress: float,
    initial_phase_rate_scale: float,
) -> tuple[float, float, np.ndarray]:
    progress = smoothstep01(
        (phase_progress - schedule.brake_start_progress)
        / max(1.0 - schedule.brake_start_progress, 1.0e-6)
    )
    rate = initial_phase_rate_scale + progress * (
        schedule.end_phase_rate_scale - initial_phase_rate_scale
    )
    amplitude = 1.0 + progress * (schedule.end_amplitude_scale - 1.0)
    midpoint = np.asarray(schedule.midpoint_joint_offset_rad)
    terminal = np.asarray(schedule.terminal_joint_offset_rad)
    if progress <= 0.5:
        local = smoothstep01(progress * 2.0)
        offsets = local * midpoint
    else:
        local = smoothstep01((progress - 0.5) * 2.0)
        offsets = midpoint + local * (terminal - midpoint)
    return rate, amplitude, offsets


def _unsafe_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool]:
    floor = model.geom("floor").id
    torso = model.geom("torso_proxy").id
    feet = {model.geom("front_foot_proxy").id, model.geom("rear_foot_proxy").id}
    torso_contact = forbidden_internal = False
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor in pair:
            torso_contact |= torso in pair
        elif pair != feet:
            forbidden_internal = True
    return torso_contact, forbidden_internal


def evaluate_schedule(
    model: mujoco.MjModel,
    capture_qpos: np.ndarray,
    capture_qvel: np.ndarray,
    oscillator_phase_rad: float,
    schedule: BrakingSchedule,
    *,
    coefficients: np.ndarray,
    native_rate_rad_s: float,
    coupling_per_s: float,
    initial_phase_rate_scale: float,
    minimum_foot_gap_m: float,
    foot_gap_margin_m: float,
    knee_bias_rad: float,
    compact_ctrl: np.ndarray,
    park_phase_rad: float = 0.0,
    physics_stride: int = 1,
    maximum_brake_deceleration_rad_s2: float = 8.0,
    brake_phase_margin_rad: float = math.radians(20.0),
    nominal_body_roll_rate_rad_s: float = 2.0 * math.pi * 0.40,
    minimum_root_height_m: float = 0.05,
) -> BrakingMetrics:
    data = mujoco.MjData(model)
    data.qpos[:] = capture_qpos
    data.qvel[:] = capture_qvel
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    initial_phase = float(capture_qpos[2])
    initial_angular_speed = float(capture_qvel[2])
    direction = 1.0 if initial_angular_speed >= 0.0 else -1.0
    required_distance = required_braking_phase_distance(
        initial_angular_speed,
        maximum_brake_deceleration_rad_s2,
        brake_phase_margin_rad,
    )
    _, target_distance = select_reachable_target_phase_unwrapped(
        initial_phase, park_phase_rad, required_distance, direction
    )
    # Instantaneous body speed is strongly periodic (roughly 1--9 rad/s) and
    # must not be mistaken for the net rate at which a full turn is covered.
    # It still determines braking distance above; duration uses the nominal
    # net rolling rate so peaks do not produce unrealistically short plans.
    nominal_duration = target_distance / max(nominal_body_roll_rate_rad_s, 0.5)
    planned_duration = float(
        np.clip(schedule.duration_scale * nominal_duration, 0.4, 4.0)
    )
    oscillator = float(oscillator_phase_rad)
    minimum_height = float(data.qpos[1])
    maximum_torque = 0.0
    torso_contact = forbidden_internal = numerical_failure = False
    torso_now = internal_now = False
    torso_contact_duration = forbidden_internal_duration = 0.0
    steps = int(math.ceil(planned_duration / dt))
    completed = 0
    while completed < steps:
        block = min(physics_stride, steps - completed)
        body_phase = float(data.qpos[2])
        phase_progress = (
            direction * (body_phase - initial_phase) / max(target_distance, 1.0e-6)
        )
        rate_scale, amplitude, offsets = schedule_scales(
            schedule, phase_progress, initial_phase_rate_scale
        )
        phase_speed = native_rate_rad_s + coupling_per_s * math.sin(body_phase - oscillator)
        oscillator += dt * block * rate_scale * max(0.1, phase_speed)
        rolling_target = controller_targets(
            body_phase,
            float(data.time),
            coefficients,
            oscillator_rate=native_rate_rad_s,
            control_phase=oscillator,
            knee_bias_rad=knee_bias_rad,
            minimum_foot_surface_gap_m=minimum_foot_gap_m,
            foot_gap_tracking_margin_m=foot_gap_margin_m,
        )
        data.ctrl[:] = compact_ctrl + amplitude * (rolling_target - compact_ctrl) + offsets
        data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data, nstep=block)
        completed += block
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.actuator_force))
        )
        if not finite:
            numerical_failure = True
            break
        minimum_height = min(minimum_height, float(data.qpos[1]))
        maximum_torque = max(maximum_torque, float(np.max(np.abs(data.actuator_force))))
        torso_now, internal_now = _unsafe_contacts(model, data)
        torso_contact |= torso_now
        forbidden_internal |= internal_now
        torso_contact_duration += dt * block * float(torso_now)
        forbidden_internal_duration += dt * block * float(internal_now)

    linear = abs(float(data.qvel[0])) if not numerical_failure else math.inf
    angular = abs(float(data.qvel[2])) if not numerical_failure else math.inf
    phase_error = abs(wrap_to_pi(float(data.qpos[2]) - park_phase_rad))
    joint_speed = float(np.sqrt(np.mean(np.asarray(data.qvel[3:]) ** 2)))
    deploy_gate = bool(
        not numerical_failure
        and linear <= 0.08
        and angular <= 0.50
        and phase_error <= math.radians(15.0)
        and joint_speed <= 1.0
        and minimum_height >= minimum_root_height_m
        and not torso_contact
        and not internal_now
    )
    score = (
        8.0 * angular
        + 12.0 * linear
        + 20.0 * phase_error
        + 0.5 * joint_speed
        + 20.0 * float(torso_contact)
        + 50.0 * torso_contact_duration
        + 35.0 * forbidden_internal_duration
        + 30.0 * float(minimum_height < minimum_root_height_m)
        + 50.0 * float(numerical_failure)
        + 0.1 * maximum_torque
    )
    return BrakingMetrics(
        score=score,
        final_linear_speed_m_s=linear,
        final_angular_speed_rad_s=angular,
        final_phase_error_rad=phase_error,
        final_joint_speed_rms_rad_s=joint_speed,
        minimum_root_height_m=minimum_height,
        maximum_torque_nm=maximum_torque,
        torso_contact=torso_contact,
        forbidden_internal_contact=forbidden_internal,
        final_forbidden_internal_contact=internal_now,
        torso_contact_duration_s=torso_contact_duration,
        forbidden_internal_contact_duration_s=forbidden_internal_duration,
        numerical_failure=numerical_failure,
        deploy_entry_gate=deploy_gate,
        target_phase_distance_rad=target_distance,
        planned_duration_s=planned_duration,
    )


def _aggregate(metrics: list[BrakingMetrics]) -> tuple[float, dict[str, float]]:
    scores = np.asarray([metric.score for metric in metrics])
    summary = {
        "mean_score": float(scores.mean()),
        "worst_score": float(scores.max()),
        "deploy_entry_rate": float(np.mean([m.deploy_entry_gate for m in metrics])),
        "median_linear_speed_m_s": float(np.median([m.final_linear_speed_m_s for m in metrics])),
        "median_angular_speed_rad_s": float(np.median([m.final_angular_speed_rad_s for m in metrics])),
        "median_phase_error_rad": float(np.median([m.final_phase_error_rad for m in metrics])),
        "median_planned_duration_s": float(np.median([m.planned_duration_s for m in metrics])),
        "median_target_phase_distance_rad": float(
            np.median([m.target_phase_distance_rad for m in metrics])
        ),
        "torso_contact_rate": float(np.mean([m.torso_contact for m in metrics])),
        "forbidden_internal_contact_rate": float(
            np.mean([m.forbidden_internal_contact for m in metrics])
        ),
        "mean_forbidden_internal_contact_duration_s": float(
            np.mean([m.forbidden_internal_contact_duration_s for m in metrics])
        ),
    }
    return summary["mean_score"] + 0.25 * summary["worst_score"], summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--phase-bins", type=int, nargs="*", default=(0, 8, 16, 24))
    parser.add_argument("--samples-per-bin", type=int, default=1)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frequency-hz", type=float, default=0.40)
    parser.add_argument(
        "--physics-stride", type=int, default=1,
        help="MuJoCo steps per control update; use 1 for authoritative results.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--initial-report", type=Path, default=None,
        help="Seed the first candidate from a previous braking CEM report.",
    )
    args = parser.parse_args()
    if args.population < 4 or args.generations < 1:
        parser.error("population must be >= 4 and generations must be positive")
    if args.samples_per_bin < 1:
        parser.error("samples per bin must be positive")
    elite_count = max(2, int(round(args.population * args.elite_fraction)))
    snapshots = np.load(args.snapshots)
    contact = snapshots["contact_features"]
    valid = (contact[:, 2] == 0) & (contact[:, 3] == 0)
    selected = []
    for phase_bin in args.phase_bins:
        matches = np.flatnonzero((snapshots["phase_bin"] == phase_bin) & valid)
        if not len(matches):
            raise RuntimeError(f"phase bin {phase_bin} has no valid snapshots")
        selected.extend(int(index) for index in matches[: args.samples_per_bin])

    _activate_geometry(REAL_GEOMETRY_PARAMETERS)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    coefficients, native_rate, coupling, foot_gap, foot_margin, knee_bias = (
        load_controller_options(args.controller)
    )
    compact_id = model.key("compact").id
    compact_ctrl = np.asarray(model.key_ctrl[compact_id], dtype=float)
    initial_rate_scale = 2.0 * math.pi * args.frequency_hz / native_rate
    rng = np.random.default_rng(args.seed)
    initial_vector = None
    if args.initial_report is not None:
        previous = json.loads(args.initial_report.read_text(encoding="utf-8"))
        initial_vector = BrakingSchedule(**previous["best"]["schedule"]).to_vector()
    mean = (LOWER + UPPER) * 0.5
    std = (UPPER - LOWER) * 0.30
    history: list[dict[str, object]] = []
    global_best: tuple[float, np.ndarray, dict[str, float]] | None = None
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, len(mean))), LOWER, UPPER
        )
        if generation == 0 and initial_vector is not None:
            population[0] = initial_vector
        elif global_best is not None:
            population[0] = global_best[1]
        evaluated = []
        for vector in population:
            schedule = BrakingSchedule.from_vector(vector)
            metrics = [
                evaluate_schedule(
                    model,
                    snapshots["qpos"][index], snapshots["qvel"][index],
                    float(snapshots["oscillator_phase_rad"][index]), schedule,
                    coefficients=coefficients, native_rate_rad_s=native_rate,
                    coupling_per_s=coupling, initial_phase_rate_scale=initial_rate_scale,
                    minimum_foot_gap_m=foot_gap, foot_gap_margin_m=foot_margin,
                    knee_bias_rad=knee_bias, compact_ctrl=compact_ctrl,
                    physics_stride=args.physics_stride,
                    nominal_body_roll_rate_rad_s=2.0 * math.pi * args.frequency_hz,
                )
                for index in selected
            ]
            objective, summary = _aggregate(metrics)
            evaluated.append((objective, vector.copy(), summary))
        evaluated.sort(key=lambda item: item[0])
        elites = np.asarray([item[1] for item in evaluated[:elite_count]])
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.03 * (UPPER - LOWER))
        if global_best is None or evaluated[0][0] < global_best[0]:
            global_best = evaluated[0]
        record = {
            "generation": generation,
            "objective": global_best[0],
            "schedule": asdict(BrakingSchedule.from_vector(global_best[1])),
            **global_best[2],
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    assert global_best is not None
    report = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "controller": str(args.controller.resolve()),
        "snapshots": str(args.snapshots.resolve()),
        "phase_bins": list(args.phase_bins),
        "samples_per_bin": args.samples_per_bin,
        "selected_snapshot_indices": selected,
        "population": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "history": history,
        "best": history[-1],
        "feasible": history[-1]["deploy_entry_rate"] > 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
