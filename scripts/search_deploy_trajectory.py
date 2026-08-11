"""Search deterministic state-conditioned parking trajectories on snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d_mjx.deploy_trajectory import deploy_trajectory_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "curl_robot_2d_real_geometry.xml"
DEFAULT_SNAPSHOTS = PROJECT_ROOT / "results" / "rolling_stop" / "low_speed_snapshots_0p40hz.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "rolling_stop" / "deploy_search.json"
# Symmetric sagittal projection of the validated 3-D ``park`` keyframe.
DEFAULT_PARK_JOINTS = np.asarray(
    [0.4983797927, 0.3491569025, 0.4340952810, 0.0107701707], dtype=float
)
COMPACT_JOINTS = np.asarray(
    [0.3141592654, 1.011720272, 0.3141592654, 1.011720272], dtype=float
)


def deployment_midpoint(
    strategy: str,
    capture_joints: np.ndarray,
    park_joints: np.ndarray,
) -> np.ndarray | None:
    """Return a waypoint; sequential modes unfold only one leg per segment."""

    capture = np.asarray(capture_joints, dtype=float)
    park = np.asarray(park_joints, dtype=float)
    if strategy == "direct":
        return None
    if strategy == "compact":
        return COMPACT_JOINTS.copy()
    if strategy == "front_first":
        return np.concatenate((park[:2], capture[2:]))
    if strategy == "rear_first":
        return np.concatenate((capture[:2], park[2:]))
    raise ValueError(f"unknown deployment strategy: {strategy}")


@dataclass(frozen=True)
class DeployRolloutMetrics:
    success: bool
    final_linear_speed_m_s: float
    final_angular_speed_rad_s: float
    final_pitch_error_rad: float
    final_joint_rms_error_rad: float
    grounded_feet: int
    settled_duration_s: float
    minimum_root_height_m: float
    maximum_torque_nm: float
    maximum_ground_force_n: float
    torso_contact: bool
    forbidden_internal_contact: bool
    numerical_failure: bool


def _contact_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, bool, bool, float]:
    floor = model.geom("floor").id
    front = model.geom("front_foot_proxy").id
    rear = model.geom("rear_foot_proxy").id
    torso = model.geom("torso_proxy").id
    feet = {front, rear}
    grounded: set[int] = set()
    torso_contact = False
    forbidden_internal = False
    maximum_force = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor in pair:
            grounded |= pair & feet
            torso_contact |= torso in pair
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, index, force)
            maximum_force = max(maximum_force, abs(float(force[0])))
        elif pair != feet:
            forbidden_internal = True
    return len(grounded), torso_contact, forbidden_internal, maximum_force


def rollout_deploy(
    model: mujoco.MjModel,
    capture_qpos: np.ndarray,
    capture_qvel: np.ndarray,
    park_joints: np.ndarray,
    *,
    deploy_duration_s: float,
    hold_duration_s: float = 3.0,
    midpoint: np.ndarray | None = None,
    physics_stride: int = 1,
    minimum_root_height_m: float = 0.05,
) -> DeployRolloutMetrics:
    data = mujoco.MjData(model)
    data.qpos[:] = capture_qpos
    data.qvel[:] = capture_qvel
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    if physics_stride <= 0:
        raise ValueError("physics_stride must be positive")
    total_steps = int(math.ceil((deploy_duration_s + hold_duration_s) / dt))
    minimum_height = float(data.qpos[1])
    maximum_torque = maximum_force = settled = 0.0
    torso_contact = forbidden_internal = numerical_failure = False
    completed_steps = 0
    while completed_steps < total_steps:
        elapsed = float(data.time)
        sample = deploy_trajectory_sample(
            np,
            capture_qpos[3:],
            capture_qvel[3:],
            park_joints,
            elapsed_s=elapsed,
            duration_s=deploy_duration_s,
            midpoint=midpoint,
        )
        data.ctrl[:] = sample.position
        block_steps = min(physics_stride, total_steps - completed_steps)
        mujoco.mj_step(model, data, nstep=block_steps)
        completed_steps += block_steps
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
        grounded, torso_now, internal_now, force = _contact_state(model, data)
        maximum_force = max(maximum_force, force)
        torso_contact |= torso_now
        forbidden_internal |= internal_now
        if elapsed >= deploy_duration_s:
            joint_rms = float(np.sqrt(np.mean((data.qpos[3:] - park_joints) ** 2)))
            is_settled = (
                abs(float(data.qvel[0])) <= 0.03
                and abs(float(data.qvel[2])) <= 0.10
                and abs(float(data.qpos[2])) <= math.radians(5.0)
                and joint_rms <= math.radians(5.0)
                and grounded >= 2
                and not torso_now
                and not internal_now
            )
            settled = settled + dt * block_steps if is_settled else 0.0
    grounded, torso_now, internal_now, force = _contact_state(model, data)
    maximum_force = max(maximum_force, force)
    joint_rms = float(np.sqrt(np.mean((data.qpos[3:] - park_joints) ** 2)))
    linear_speed = abs(float(data.qvel[0]))
    angular_speed = abs(float(data.qvel[2]))
    pitch_error = abs(float(math.atan2(math.sin(data.qpos[2]), math.cos(data.qpos[2]))))
    success = bool(
        not numerical_failure
        and minimum_height >= minimum_root_height_m
        and linear_speed <= 0.03
        and angular_speed <= 0.10
        and pitch_error <= math.radians(5.0)
        and joint_rms <= math.radians(5.0)
        and grounded >= 2
        and settled >= 2.0
        and not torso_contact
        and not forbidden_internal
        and maximum_torque <= 6.0
    )
    return DeployRolloutMetrics(
        success=success,
        final_linear_speed_m_s=linear_speed,
        final_angular_speed_rad_s=angular_speed,
        final_pitch_error_rad=pitch_error,
        final_joint_rms_error_rad=joint_rms,
        grounded_feet=grounded,
        settled_duration_s=settled,
        minimum_root_height_m=minimum_height,
        maximum_torque_nm=maximum_torque,
        maximum_ground_force_n=maximum_force,
        torso_contact=torso_contact,
        forbidden_internal_contact=forbidden_internal,
        numerical_failure=numerical_failure,
    )


def balanced_subset(phase_bins: np.ndarray, valid: np.ndarray, per_bin: int) -> np.ndarray:
    selected: list[int] = []
    for bin_index in range(int(phase_bins.max()) + 1):
        indices = np.flatnonzero((phase_bins == bin_index) & valid)
        selected.extend(indices[:per_bin].tolist())
    return np.asarray(selected, dtype=np.int32)


def _summarize(metrics: list[DeployRolloutMetrics]) -> dict[str, object]:
    successes = sum(metric.success for metric in metrics)
    return {
        "episodes": len(metrics),
        "successes": successes,
        "success_rate": successes / max(len(metrics), 1),
        "median_final_linear_speed_m_s": float(np.median([m.final_linear_speed_m_s for m in metrics])),
        "median_final_angular_speed_rad_s": float(np.median([m.final_angular_speed_rad_s for m in metrics])),
        "median_settled_duration_s": float(np.median([m.settled_duration_s for m in metrics])),
        "torso_contact_rate": float(np.mean([m.torso_contact for m in metrics])),
        "forbidden_internal_contact_rate": float(
            np.mean([m.forbidden_internal_contact for m in metrics])
        ),
        "maximum_torque_nm": max((m.maximum_torque_nm for m in metrics), default=0.0),
        "maximum_ground_force_n": max((m.maximum_ground_force_n for m in metrics), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--park-joints", type=float, nargs=4, default=DEFAULT_PARK_JOINTS,
        metavar=("FRONT_HIP", "FRONT_KNEE", "REAR_HIP", "REAR_KNEE"),
    )
    parser.add_argument("--durations", type=float, nargs="+", default=(0.8, 1.0, 1.2, 1.5))
    parser.add_argument("--samples-per-bin", type=int, default=3)
    parser.add_argument("--hold-duration", type=float, default=3.0)
    parser.add_argument("--minimum-root-height", type=float, default=0.05)
    parser.add_argument(
        "--strategies",
        choices=("direct", "compact", "front_first", "rear_first"),
        nargs="+", default=("direct", "compact", "front_first", "rear_first"),
    )
    parser.add_argument(
        "--physics-stride", type=int, default=1,
        help="MuJoCo steps per control update; use 1 for authoritative results.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    snapshots = np.load(args.snapshots)
    park_joints = np.asarray(args.park_joints, dtype=float)
    contact = snapshots["contact_features"]
    valid = (contact[:, 2] == 0) & (contact[:, 3] == 0)
    if "deploy_entry_gate" in snapshots:
        valid &= snapshots["deploy_entry_gate"].astype(bool)
    balance_bins = (
        snapshots["source_phase_bin"]
        if "source_phase_bin" in snapshots
        else snapshots["phase_bin"]
    )
    indices = balanced_subset(balance_bins, valid, args.samples_per_bin)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    candidates: list[dict[str, object]] = []
    for strategy in args.strategies:
        for duration in args.durations:
            metrics = [
                rollout_deploy(
                    model, snapshots["qpos"][index], snapshots["qvel"][index],
                    park_joints, deploy_duration_s=duration,
                    hold_duration_s=args.hold_duration,
                    midpoint=deployment_midpoint(
                        strategy, snapshots["qpos"][index][3:], park_joints
                    ),
                    physics_stride=args.physics_stride,
                    minimum_root_height_m=args.minimum_root_height,
                )
                for index in indices
            ]
            candidates.append({
                "strategy": strategy, "duration_s": duration, **_summarize(metrics)
            })
            print(json.dumps(candidates[-1], sort_keys=True))
    best = max(
        candidates,
        key=lambda item: (
            item["success_rate"], -item["forbidden_internal_contact_rate"],
            -item["torso_contact_rate"], -item["median_final_linear_speed_m_s"],
        ),
    )
    report = {
        "schema_version": 1,
        "snapshots": str(args.snapshots.resolve()),
        "model": str(args.model.resolve()),
        "park_joints_rad": park_joints.tolist(),
        "valid_snapshot_count": int(valid.sum()),
        "search_snapshot_count": len(indices),
        "candidates": candidates,
        "best": best,
        "gate_passed": best["success_rate"] >= 0.95,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
