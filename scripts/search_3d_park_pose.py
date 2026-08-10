"""Search a symmetric static 3-D parking pose without training a policy."""

from __future__ import annotations

import argparse
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
from scripts.validate_3d_park_pose import (  # noqa: E402
    DEFAULT_MODEL,
    FOOT_GEOM_NAMES,
    _contact_summary,
    _torso_tilt,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results/park_pose_search_3d/search_result.json"


class StaticPoseEvaluator:
    """Reusable MuJoCo evaluator for four left/right-symmetric joint targets."""

    def __init__(self, model_path: Path, keyframe: str):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.key_id = self.model.key(keyframe).id
        self.floor_id = self.model.geom("floor").id
        self.torso_geom_id = self.model.geom("torso_box_proxy").id
        self.foot_ids = {self.model.geom(name).id for name in FOOT_GEOM_NAMES}
        self.joint_ids = np.asarray(self.model.actuator_trnid[:, 0], dtype=int)
        self.joint_qpos_indices = np.asarray(
            self.model.jnt_qposadr[self.joint_ids], dtype=int
        )
        self.dt = float(self.model.opt.timestep)
        self.lower = np.asarray(self.model.jnt_range[self.joint_ids, 0], dtype=float)[
            [0, 1, 4, 5]
        ]
        self.upper = np.asarray(self.model.jnt_range[self.joint_ids, 1], dtype=float)[
            [0, 1, 4, 5]
        ]

    @staticmethod
    def expand(targets: np.ndarray) -> np.ndarray:
        front_hip, front_knee, rear_hip, rear_knee = targets
        return np.asarray(
            [
                front_hip,
                front_knee,
                front_hip,
                front_knee,
                rear_hip,
                rear_knee,
                rear_hip,
                rear_knee,
            ],
            dtype=float,
        )

    def evaluate(self, targets: np.ndarray, duration_s: float) -> ParkPoseStaticMetrics:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        expanded = self.expand(targets)
        self.data.qpos[:3] = (0.0, 0.0, 0.5)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.joint_qpos_indices] = expanded
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = expanded
        mujoco.mj_forward(self.model, self.data)
        foot_bottoms = np.asarray(
            [
                self.data.geom(name).xpos[2] - self.model.geom(name).size[0]
                for name in FOOT_GEOM_NAMES
            ],
            dtype=float,
        )
        self.data.qpos[2] += 0.001 - float(np.min(foot_bottoms))
        mujoco.mj_forward(self.model, self.data)

        desired_joint_qpos = expanded.copy()
        initial_root_xy = np.asarray(self.data.qpos[:2], dtype=float).copy()
        maximum_tilt = _torso_tilt(self.model, self.data)
        minimum_root_height = float(self.data.qpos[2])
        maximum_torque = 0.0
        internal_duration = 0.0
        torso_ground_duration = 0.0
        torso_internal_duration = 0.0
        numerical_failure = False
        steps = max(1, int(math.ceil(duration_s / self.dt)))
        final_grounded: set[int] = set()

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
                and np.all(np.isfinite(self.data.actuator_force))
            ):
                numerical_failure = True
                break
            maximum_tilt = max(maximum_tilt, _torso_tilt(self.model, self.data))
            minimum_root_height = min(minimum_root_height, float(self.data.qpos[2]))
            maximum_torque = max(
                maximum_torque,
                float(np.max(np.abs(self.data.actuator_force))),
            )
            (
                final_grounded,
                internal,
                torso_ground,
                torso_internal,
                _,
            ) = _contact_summary(
                self.model,
                self.data,
                self.floor_id,
                self.torso_geom_id,
                self.foot_ids,
            )
            internal_duration += self.dt if internal else 0.0
            torso_ground_duration += self.dt if torso_ground else 0.0
            torso_internal_duration += self.dt if torso_internal else 0.0

        final_tilt = (
            _torso_tilt(self.model, self.data) if not numerical_failure else math.inf
        )
        joint_error = (
            np.asarray(self.data.qpos[self.joint_qpos_indices]) - desired_joint_qpos
        )
        survived = bool(
            not numerical_failure
            and minimum_root_height > 0.05
            and maximum_tilt < math.radians(90.0)
        )
        return ParkPoseStaticMetrics(
            survived=survived,
            numerical_failure=numerical_failure,
            duration_s=float(self.data.time),
            final_linear_speed_m_s=float(np.linalg.norm(self.data.qvel[:3])),
            final_angular_speed_rad_s=float(np.linalg.norm(self.data.qvel[3:6])),
            final_torso_tilt_rad=final_tilt,
            maximum_torso_tilt_rad=maximum_tilt,
            final_joint_pose_rms_error_rad=float(np.sqrt(np.mean(joint_error**2))),
            grounded_feet=len(final_grounded),
            internal_contact_total_s=internal_duration,
            torso_ground_contact_total_s=torso_ground_duration,
            torso_internal_contact_total_s=torso_internal_duration,
            lateral_drift_m=float(
                np.linalg.norm(np.asarray(self.data.qpos[:2]) - initial_root_xy)
            ),
            minimum_root_height_m=minimum_root_height,
            maximum_torque_nm=maximum_torque,
        )


def pose_cost(metrics: ParkPoseStaticMetrics) -> float:
    """Continuous search objective; hard gates remain the final authority."""

    return float(
        100.0 * (not metrics.survived)
        + 40.0 * metrics.numerical_failure
        + 12.0 * (4 - metrics.grounded_feet)
        + 30.0 * abs(metrics.final_linear_speed_m_s)
        + 12.0 * abs(metrics.final_angular_speed_rad_s)
        + 15.0 * abs(metrics.final_torso_tilt_rad)
        + 20.0 * metrics.lateral_drift_m
        + 8.0 * metrics.final_joint_pose_rms_error_rad
        + 12.0 * metrics.internal_contact_total_s
        + 30.0 * metrics.torso_ground_contact_total_s
        + 30.0 * metrics.torso_internal_contact_total_s
        + max(metrics.maximum_torque_nm - 2.1, 0.0) * 5.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--keyframe", default="stand")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--search-duration", type=float, default=1.5)
    parser.add_argument("--verify-duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.population < 4 or args.generations < 1:
        parser.error("population must be >= 4 and generations must be >= 1")

    evaluator = StaticPoseEvaluator(args.model.resolve(), args.keyframe)
    rng = np.random.default_rng(args.seed)
    mujoco.mj_resetDataKeyframe(evaluator.model, evaluator.data, evaluator.key_id)
    initial = np.asarray(
        evaluator.data.qpos[evaluator.joint_qpos_indices], dtype=float
    )[[0, 1, 4, 5]]
    mean = initial.copy()
    std = np.asarray([0.30, 0.40, 0.30, 0.40], dtype=float)
    elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))
    best_targets = mean.copy()
    best_metrics = evaluator.evaluate(best_targets, args.search_duration)
    best_cost = pose_cost(best_metrics)
    generations: list[dict[str, object]] = []

    for generation in range(args.generations):
        samples = rng.normal(mean, std, size=(args.population, 4))
        samples = np.clip(samples, evaluator.lower, evaluator.upper)
        samples[0] = mean
        evaluated = []
        for targets in samples:
            metrics = evaluator.evaluate(targets, args.search_duration)
            cost = pose_cost(metrics)
            evaluated.append((cost, targets.copy(), metrics))
            if cost < best_cost:
                best_cost = cost
                best_targets = targets.copy()
                best_metrics = metrics
        evaluated.sort(key=lambda item: item[0])
        elites = np.stack([item[1] for item in evaluated[:elite_count]])
        mean = np.mean(elites, axis=0)
        std = np.maximum(np.std(elites, axis=0), 0.02)
        generations.append(
            {
                "generation": generation,
                "best_cost": float(evaluated[0][0]),
                "global_best_cost": float(best_cost),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "best_grounded_feet": best_metrics.grounded_feet,
                "best_final_speed_m_s": best_metrics.final_linear_speed_m_s,
            }
        )
        print(json.dumps(generations[-1]))

    verified_metrics = evaluator.evaluate(best_targets, args.verify_duration)
    gate = ParkPoseStaticGate()
    reasons = park_pose_failure_reasons(verified_metrics, gate)
    result = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "source_keyframe": args.keyframe,
        "seed": args.seed,
        "passed": not reasons,
        "failure_reasons": list(reasons),
        "symmetric_targets_rad": {
            "front_hip": float(best_targets[0]),
            "front_knee": float(best_targets[1]),
            "rear_hip": float(best_targets[2]),
            "rear_knee": float(best_targets[3]),
        },
        "expanded_ctrl_rad": evaluator.expand(best_targets).tolist(),
        "search_cost": float(best_cost),
        "verification_metrics": asdict(verified_metrics),
        "gate": asdict(gate),
        "generation_history": generations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"result: {args.output}")


if __name__ == "__main__":
    main()
