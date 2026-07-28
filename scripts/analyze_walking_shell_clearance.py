from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d.planar_geometry import segment_distance
from scripts.optimize_phase_controller import (
    PARAMETER_NAMES,
    rollout_controller,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "walking_shell_clearance"
    / "summary.json"
)
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "collision_constrained_cem"
    / "best_phase_controller.json"
)
SHELL_BODIES = (
    "torso",
    "front_thigh",
    "front_shank",
    "rear_thigh",
    "rear_shank",
)


def _id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"missing {object_type.name}: {name}")
    return value


def _shell_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    prefix: str,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    segments = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if not (name or "").startswith(f"{prefix}_shell_"):
            continue
        rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        axis = rotation[:, 2][[0, 2]]
        center = np.asarray(data.geom_xpos[geom_id])[[0, 2]]
        half_length = float(model.geom_size[geom_id, 1])
        segments.append(
            (
                center - half_length * axis,
                center + half_length * axis,
                float(model.geom_size[geom_id, 0]),
            )
        )
    return segments


def _shell_pair_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first_prefix: str,
    second_prefix: str,
) -> float:
    return min(
        segment_distance(
            first_start,
            first_end,
            second_start,
            second_end,
        )
        - first_radius
        - second_radius
        for first_start, first_end, first_radius in _shell_segments(
            model, data, first_prefix
        )
        for second_start, second_end, second_radius in _shell_segments(
            model, data, second_prefix
        )
    )


def _scan_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    open_qpos: np.ndarray,
    *,
    joint_name: str,
    angle_range: tuple[float, float],
    shell_pair: tuple[str, str],
    samples: int,
) -> dict[str, float]:
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = int(model.jnt_qposadr[joint_id])
    minimum_clearance = float("inf")
    minimum_angle = 0.0
    for angle in np.linspace(*angle_range, samples):
        data.qpos[:] = open_qpos
        data.qpos[qpos_address] = angle
        mujoco.mj_forward(model, data)
        clearance = _shell_pair_clearance(
            model, data, shell_pair[0], shell_pair[1]
        )
        if clearance < minimum_clearance:
            minimum_clearance = clearance
            minimum_angle = float(angle)
    return {
        "minimum_clearance_m": minimum_clearance,
        "angle_at_minimum_rad": minimum_angle,
        "range_min_rad": float(angle_range[0]),
        "range_max_rad": float(angle_range[1]),
    }


def _replay_controller(
    model: mujoco.MjModel,
    controller_path: Path,
    *,
    duration: float,
) -> dict[str, object]:
    payload = json.loads(controller_path.read_text(encoding="utf-8"))
    raw = payload["raw_coefficients"]
    coefficients = np.asarray(
        [float(raw[name]) for name in PARAMETER_NAMES],
        dtype=float,
    )
    rollout = rollout_controller(
        model,
        coefficients,
        duration=duration,
        oscillator_rate=float(payload["oscillator_rate_rad_s"]),
        oscillator_coupling=float(
            payload["oscillator_coupling_per_s"]
        ),
    )
    keys = (
        "duration_s",
        "net_turns",
        "root_x_displacement_m",
        "airborne_fraction",
        "longest_airborne_s",
        "maximum_foot_gap_m",
        "forbidden_contact_fraction",
        "maximum_forbidden_penetration_m",
        "maximum_allowed_foot_penetration_m",
        "leg_crossing_detected",
    )
    return {
        "controller": str(controller_path),
        "note": "Direct replay without retuning after shortening the shells.",
        **{key: rollout.summary[key] for key in keys},
    }


def analyze(
    model: mujoco.MjModel,
    *,
    samples: int,
    controller_path: Path | None = None,
    controller_duration: float = 10.0,
) -> dict[str, object]:
    data = mujoco.MjData(model)
    open_key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "open")
    walk_key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "walk")
    front_foot_site = _id(
        model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
    )
    rear_foot_site = _id(
        model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
    )

    mujoco.mj_resetDataKeyframe(model, data, open_key_id)
    mujoco.mj_forward(model, data)
    open_qpos = data.qpos.copy()

    hip_scan = _scan_joint(
        model,
        data,
        open_qpos,
        joint_name="front_hip",
        angle_range=FIXED_PARAMETERS.hip.safe_range,
        shell_pair=("torso", "front_thigh"),
        samples=samples,
    )
    knee_scan = _scan_joint(
        model,
        data,
        open_qpos,
        joint_name="front_knee",
        angle_range=FIXED_PARAMETERS.knee.safe_range,
        shell_pair=("front_thigh", "front_shank"),
        samples=samples,
    )

    mujoco.mj_resetDataKeyframe(model, data, walk_key_id)
    mujoco.mj_forward(model, data)
    pair_clearances = {}
    for first_index, first_name in enumerate(SHELL_BODIES):
        for second_name in SHELL_BODIES[first_index + 1 :]:
            pair_name = f"{first_name}__{second_name}"
            pair_clearances[pair_name] = _shell_pair_clearance(
                model, data, first_name, second_name
            )

    contact_pairs = []
    for contact in data.contact:
        contact_pairs.append(
            [
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ),
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ),
                float(contact.dist),
            ]
        )

    summary = {
        "model": str(MODEL_PATH),
        "samples_per_joint": samples,
        "shell_design_gap_m": FIXED_PARAMETERS.shell_design_gap,
        "shell_arc_trim_angle_rad": (
            FIXED_PARAMETERS.shell_arc_trim_angle
        ),
        "shell_arc_coverage_angle_rad": (
            FIXED_PARAMETERS.shell_arc_coverage_angle
        ),
        "safe_range_scans": {
            "front_hip": hip_scan,
            "front_knee": knee_scan,
        },
        "walk_keyframe": {
            "qpos": data.qpos.tolist(),
            "front_foot_xz_m": (
                np.asarray(data.site_xpos[front_foot_site])[[0, 2]].tolist()
            ),
            "rear_foot_xz_m": (
                np.asarray(data.site_xpos[rear_foot_site])[[0, 2]].tolist()
            ),
            "minimum_shell_clearance_m": min(pair_clearances.values()),
            "shell_pair_clearances_m": pair_clearances,
            "contact_pairs": contact_pairs,
        },
    }
    if controller_path is not None:
        summary["legacy_controller_replay"] = _replay_controller(
            model,
            controller_path,
            duration=controller_duration,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--controller",
        type=Path,
        default=DEFAULT_CONTROLLER,
    )
    parser.add_argument("--controller-duration", type=float, default=10.0)
    parser.add_argument("--skip-controller-replay", action="store_true")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.controller_duration <= 0.0:
        parser.error("--controller-duration must be positive")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    summary = analyze(
        model,
        samples=args.samples,
        controller_path=(
            None if args.skip_controller_replay else args.controller
        ),
        controller_duration=args.controller_duration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    hip = summary["safe_range_scans"]["front_hip"]
    knee = summary["safe_range_scans"]["front_knee"]
    walk = summary["walk_keyframe"]
    print(f"output={args.output}")
    print(
        "hip_minimum_clearance_mm="
        f"{1000.0 * hip['minimum_clearance_m']:.6f}"
    )
    print(
        "knee_minimum_clearance_mm="
        f"{1000.0 * knee['minimum_clearance_m']:.6f}"
    )
    print(
        "walk_minimum_shell_clearance_mm="
        f"{1000.0 * walk['minimum_shell_clearance_m']:.6f}"
    )
    print(f"walk_contacts={len(walk['contact_pairs'])}")
    if "legacy_controller_replay" in summary:
        replay = summary["legacy_controller_replay"]
        print(f"legacy_controller_net_turns={replay['net_turns']:.6f}")


if __name__ == "__main__":
    main()
