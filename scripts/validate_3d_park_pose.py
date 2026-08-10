"""Validate a 3-D parking keyframe under gravity without policy training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curl_robot_2d_mjx.stop_evaluation import (  # noqa: E402
    ParkPoseStaticGate,
    ParkPoseStaticMetrics,
    park_pose_failure_reasons,
)


DEFAULT_MODEL = PROJECT_ROOT / "assets/curl_robot_3d_real_geometry.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/park_pose_validation_3d"
FOOT_GEOM_NAMES = (
    "front_left_foot_proxy",
    "front_right_foot_proxy",
    "rear_left_foot_proxy",
    "rear_right_foot_proxy",
)


def _torso_tilt(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    torso_id = model.body("torso").id
    rotation = np.asarray(data.xmat[torso_id]).reshape(3, 3)
    return float(math.acos(np.clip(rotation[2, 2], -1.0, 1.0)))


def _contact_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
    torso_id: int,
    foot_ids: set[int],
) -> tuple[set[int], bool, bool, bool, list[str]]:
    grounded: set[int] = set()
    internal = False
    torso_ground = False
    torso_internal = False
    pairs: list[str] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        pair = {geom1, geom2}
        if floor_id in pair:
            other = geom2 if geom1 == floor_id else geom1
            if other in foot_ids:
                grounded.add(other)
            if other == torso_id:
                torso_ground = True
        else:
            internal = True
            if torso_id in pair:
                torso_internal = True
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
        pairs.append(f"{name1 or geom1}|{name2 or geom2}")
    return grounded, internal, torso_ground, torso_internal, pairs


def validate_park_pose(
    model_path: Path,
    keyframe: str,
    duration_s: float,
) -> tuple[ParkPoseStaticMetrics, dict[str, float], list[dict[str, float]]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    try:
        key_id = model.key(keyframe).id
    except KeyError as error:
        available = [model.key(index).name for index in range(model.nkey)]
        raise ValueError(
            f"unknown keyframe {keyframe!r}; available: {available}"
        ) from error

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    initial_root_xy = np.asarray(data.qpos[:2], dtype=float).copy()
    joint_ids = np.unique(np.asarray(model.actuator_trnid[:, 0], dtype=int))
    joint_qpos_indices = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    desired_joint_qpos = np.asarray(data.qpos[joint_qpos_indices], dtype=float).copy()

    floor_id = model.geom("floor").id
    torso_geom_id = model.geom("torso_box_proxy").id
    foot_ids = {model.geom(name).id for name in FOOT_GEOM_NAMES}
    dt = float(model.opt.timestep)
    steps = max(1, int(math.ceil(duration_s / dt)))
    sample_stride = max(1, int(round(0.01 / dt)))

    internal_duration = 0.0
    torso_ground_duration = 0.0
    torso_internal_duration = 0.0
    maximum_tilt = _torso_tilt(model, data)
    minimum_root_height = float(data.qpos[2])
    maximum_torque = 0.0
    numerical_failure = False
    contact_pair_duration_s: dict[str, float] = {}
    initial_grounded, _, _, _, _ = _contact_summary(
        model, data, floor_id, torso_geom_id, foot_ids
    )
    history: list[dict[str, float]] = [
        {
            "time_s": float(data.time),
            "root_x_m": float(data.qpos[0]),
            "root_y_m": float(data.qpos[1]),
            "root_z_m": float(data.qpos[2]),
            "linear_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
            "angular_speed_rad_s": float(np.linalg.norm(data.qvel[3:6])),
            "torso_tilt_rad": maximum_tilt,
            "grounded_feet": float(len(initial_grounded)),
            "actuator_peak_nm": 0.0,
        }
    ]

    for step in range(steps):
        mujoco.mj_step(model, data)
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.actuator_force))
        )
        if not finite:
            numerical_failure = True
            break

        tilt = _torso_tilt(model, data)
        maximum_tilt = max(maximum_tilt, tilt)
        minimum_root_height = min(minimum_root_height, float(data.qpos[2]))
        if model.nu:
            maximum_torque = max(
                maximum_torque,
                float(np.max(np.abs(data.actuator_force))),
            )
        grounded, internal, torso_ground, torso_internal, pairs = _contact_summary(
            model,
            data,
            floor_id,
            torso_geom_id,
            foot_ids,
        )
        internal_duration += dt if internal else 0.0
        torso_ground_duration += dt if torso_ground else 0.0
        torso_internal_duration += dt if torso_internal else 0.0
        for pair in set(pairs):
            contact_pair_duration_s[pair] = (
                contact_pair_duration_s.get(pair, 0.0) + dt
            )

        if step % sample_stride == 0 or step == steps - 1:
            history.append(
                {
                    "time_s": float(data.time),
                    "root_x_m": float(data.qpos[0]),
                    "root_y_m": float(data.qpos[1]),
                    "root_z_m": float(data.qpos[2]),
                    "linear_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
                    "angular_speed_rad_s": float(np.linalg.norm(data.qvel[3:6])),
                    "torso_tilt_rad": tilt,
                    "grounded_feet": float(len(grounded)),
                    "actuator_peak_nm": float(
                        np.max(np.abs(data.actuator_force)) if model.nu else 0.0
                    ),
                }
            )

    final_grounded, _, _, _, _ = _contact_summary(
        model,
        data,
        floor_id,
        torso_geom_id,
        foot_ids,
    )
    joint_error = np.asarray(data.qpos[joint_qpos_indices]) - desired_joint_qpos
    final_tilt = _torso_tilt(model, data) if not numerical_failure else math.inf
    lateral_drift = float(
        np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - initial_root_xy)
    )
    survived = bool(
        not numerical_failure
        and minimum_root_height > 0.05
        and maximum_tilt < math.radians(90.0)
    )
    metrics = ParkPoseStaticMetrics(
        survived=survived,
        numerical_failure=numerical_failure,
        duration_s=float(data.time),
        final_linear_speed_m_s=float(np.linalg.norm(data.qvel[:3])),
        final_angular_speed_rad_s=float(np.linalg.norm(data.qvel[3:6])),
        final_torso_tilt_rad=final_tilt,
        maximum_torso_tilt_rad=maximum_tilt,
        final_joint_pose_rms_error_rad=float(np.sqrt(np.mean(joint_error**2))),
        grounded_feet=len(final_grounded),
        internal_contact_total_s=internal_duration,
        torso_ground_contact_total_s=torso_ground_duration,
        torso_internal_contact_total_s=torso_internal_duration,
        lateral_drift_m=lateral_drift,
        minimum_root_height_m=minimum_root_height,
        maximum_torque_nm=maximum_torque,
    )
    return metrics, contact_pair_duration_s, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--keyframe", default="stand")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")

    metrics, contact_pair_duration_s, history = validate_park_pose(
        args.model.resolve(),
        args.keyframe,
        args.duration,
    )
    gate = ParkPoseStaticGate()
    reasons = park_pose_failure_reasons(metrics, gate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "keyframe": args.keyframe,
        "passed": not reasons,
        "failure_reasons": list(reasons),
        "metrics": asdict(metrics),
        "gate": asdict(gate),
        "contact_pair_duration_s": dict(sorted(contact_pair_duration_s.items())),
    }
    report_path = args.output_dir / f"{args.keyframe}_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    history_path = args.output_dir / f"{args.keyframe}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {report_path}")
    print(f"history: {history_path}")
    raise SystemExit(0 if not reasons else 2)


if __name__ == "__main__":
    main()
