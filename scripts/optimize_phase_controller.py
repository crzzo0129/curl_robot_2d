"""Search an interpretable phase-periodic active rolling controller with CEM."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d.planar_geometry import proper_segments_intersect
from scripts.run_release_baseline import (
    JOINT_TARGETS,
    MODEL_PATH,
    _draw_panel,
    _id,
    _shell_roundness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "collision_constrained_cem"
PARAMETER_NAMES = tuple(
    coefficient
    for joint_name, _ in JOINT_TARGETS
    for coefficient in (f"{joint_name}_sin", f"{joint_name}_cos")
)
COEFFICIENT_BOUNDS = np.asarray(
    [
        (-1.00, 1.00),
        (-1.00, 1.00),
        (-1.40, 1.40),
        (-1.40, 1.40),
        (-1.00, 1.00),
        (-1.00, 1.00),
        (-1.40, 1.40),
        (-1.40, 1.40),
    ],
    dtype=float,
)
OSCILLATOR_RATE_BOUNDS = (0.5, 6.0)
OSCILLATOR_COUPLING_BOUNDS = (0.0, 8.0)

# Foot-to-foot contact is intentional in the compact pose.  Every other
# robot-to-robot contact is undesirable.  MuJoCo contacts are soft numerical
# constraints, so the allowed foot pair receives a small penetration
# tolerance while forbidden contacts are penalized by both duration and depth.
ALLOWED_FOOT_PENETRATION_M = 0.0005
FORBIDDEN_CONTACT_TIME_WEIGHT = 6.0
FORBIDDEN_PENETRATION_INTEGRAL_WEIGHT = 20000.0
MAXIMUM_FORBIDDEN_PENETRATION_WEIGHT = 2500.0
ALLOWED_PENETRATION_EXCESS_INTEGRAL_WEIGHT = 12000.0
MAXIMUM_ALLOWED_PENETRATION_EXCESS_WEIGHT = 2500.0
LEG_CROSSING_FAILURE_PENALTY = 1000.0
FOOT_CONTACT_TIME_WEIGHT = 40.0
LONGEST_FOOT_CONTACT_WEIGHT = 100.0
FOOT_GAP_DEFICIT_INTEGRAL_WEIGHT = 3000.0
MAXIMUM_FOOT_GAP_DEFICIT_WEIGHT = 1200.0
FOOT_SURFACE_PENETRATION_WEIGHT = 5000.0
FOOT_GAP_MAXIMUM_AIRBORNE_FRACTION = 0.14
FOOT_GAP_MAXIMUM_FORBIDDEN_PENETRATION_M = 0.00065
FOOT_GAP_AIRBORNE_EXCESS_WEIGHT = 200.0
FOOT_GAP_FORBIDDEN_EXCESS_WEIGHT = 50000.0
FOOT_GAP_TRACKING_MARGIN_M = 0.004


@dataclass(frozen=True)
class ControllerRollout:
    score: float
    summary: dict[str, float | int | bool | None]
    columns: tuple[str, ...] | None = None
    rows: np.ndarray | None = None

    def column(self, name: str) -> np.ndarray:
        if self.columns is None or self.rows is None:
            raise ValueError("This rollout was not recorded in detail")
        return self.rows[:, self.columns.index(name)]


_WORKER_MODEL: mujoco.MjModel | None = None


def _initialize_rollout_worker(model_path: str) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(model_path)


def _rollout_worker(
    task: tuple[np.ndarray, float, str, float, float, bool],
) -> ControllerRollout:
    if _WORKER_MODEL is None:
        raise RuntimeError("rollout worker model was not initialized")
    (
        sample,
        duration,
        objective,
        minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m,
        enforce_leg_crossing_constraint,
    ) = task
    return rollout_controller(
        _WORKER_MODEL,
        sample[:8],
        duration=duration,
        oscillator_rate=float(sample[8]),
        oscillator_coupling=float(sample[9]),
        objective=objective,
        minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
        enforce_leg_crossing_constraint=enforce_leg_crossing_constraint,
        detailed=False,
    )


def knee_bias_for_foot_gap(minimum_foot_surface_gap_m: float) -> float:
    """Return the symmetric knee offset for a nominal compact foot gap."""

    if minimum_foot_surface_gap_m < 0.0:
        raise ValueError("minimum foot surface gap cannot be negative")
    separated = replace(
        FIXED_PARAMETERS,
        compact_foot_surface_gap=minimum_foot_surface_gap_m,
    )
    return separated.compact_knee_angle - FIXED_PARAMETERS.compact_knee_angle


def target_foot_center_distance(targets: np.ndarray) -> float:
    """Return planar foot-center distance for four effective joint targets."""

    front_hip, front_knee, rear_hip, rear_knee = map(float, targets)
    length = FIXED_PARAMETERS.edge_length
    delta_x = FIXED_PARAMETERS.torso_length + length * (
        math.sin(front_hip)
        + math.sin(front_hip - front_knee)
        + math.sin(rear_hip)
        + math.sin(rear_hip - rear_knee)
    )
    delta_z = length * (
        -math.cos(front_hip)
        - math.cos(front_knee - front_hip)
        + math.cos(rear_hip)
        + math.cos(rear_knee - rear_hip)
    )
    return math.hypot(delta_x, delta_z)


def project_targets_to_foot_gap(
    targets: np.ndarray,
    minimum_foot_surface_gap_m: float,
    *,
    tracking_margin_m: float = FOOT_GAP_TRACKING_MARGIN_M,
) -> np.ndarray:
    """Minimally shift both knee targets away from foot-foot contact."""

    if minimum_foot_surface_gap_m <= 0.0:
        return targets
    projected = np.asarray(targets, dtype=float).copy()
    target_distance = (
        2.0 * FIXED_PARAMETERS.foot_radius
        + minimum_foot_surface_gap_m
        + tracking_margin_m
    )
    epsilon = 1.0e-5
    for _ in range(6):
        distance = target_foot_center_distance(projected)
        deficit = target_distance - distance
        if deficit <= 1.0e-7:
            break
        gradient = np.zeros(2, dtype=float)
        for gradient_index, joint_index in enumerate((1, 3)):
            perturbed = projected.copy()
            perturbed[joint_index] += epsilon
            gradient[gradient_index] = (
                target_foot_center_distance(perturbed) - distance
            ) / epsilon
        gradient_norm_squared = float(gradient @ gradient)
        if gradient_norm_squared < 1.0e-10:
            break
        correction = deficit * gradient / gradient_norm_squared
        projected[1] += float(np.clip(correction[0], -0.20, 0.20))
        projected[3] += float(np.clip(correction[1], -0.20, 0.20))
        projected[1] = float(
            np.clip(projected[1], *FIXED_PARAMETERS.knee.safe_range)
        )
        projected[3] = float(
            np.clip(projected[3], *FIXED_PARAMETERS.knee.safe_range)
        )
    return projected


def _joint_setup(model: mujoco.MjModel) -> list[tuple[str, float, int, int, int]]:
    setup = []
    for joint_name, target in JOINT_TARGETS:
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_servo"
        )
        setup.append(
            (
                joint_name,
                target,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
                actuator_id,
            )
        )
    return setup


def _contact_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    floor_geom_id: int,
    allowed_foot_pair: frozenset[int],
) -> tuple[int, int, int, float, float]:
    """Return ground/self contact counts and penetration depths for one step."""

    ground_contact_count = 0
    self_pair_distances: dict[frozenset[int], float] = {}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = frozenset((int(contact.geom1), int(contact.geom2)))
        if floor_geom_id in pair:
            ground_contact_count += 1
            continue
        self_pair_distances[pair] = min(
            self_pair_distances.get(pair, math.inf),
            float(contact.dist),
        )

    allowed_penetration = max(
        -self_pair_distances.get(allowed_foot_pair, 0.0), 0.0
    )
    forbidden_distances = [
        distance
        for pair, distance in self_pair_distances.items()
        if pair != allowed_foot_pair
    ]
    forbidden_penetration = max(
        (-min(forbidden_distances, default=0.0)), 0.0
    )
    forbidden_pair_count = sum(
        pair != allowed_foot_pair for pair in self_pair_distances
    )
    return (
        ground_contact_count,
        len(self_pair_distances),
        forbidden_pair_count,
        allowed_penetration,
        forbidden_penetration,
    )


def _has_leg_crossing(
    data: mujoco.MjData,
    *,
    front_thigh_body_id: int,
    front_shank_body_id: int,
    rear_thigh_body_id: int,
    rear_shank_body_id: int,
    front_foot_site_id: int,
    rear_foot_site_id: int,
) -> bool:
    front_hip = np.asarray(data.xpos[front_thigh_body_id])[[0, 2]]
    front_knee = np.asarray(data.xpos[front_shank_body_id])[[0, 2]]
    front_foot = np.asarray(data.site_xpos[front_foot_site_id])[[0, 2]]
    rear_hip = np.asarray(data.xpos[rear_thigh_body_id])[[0, 2]]
    rear_knee = np.asarray(data.xpos[rear_shank_body_id])[[0, 2]]
    rear_foot = np.asarray(data.site_xpos[rear_foot_site_id])[[0, 2]]
    pairs = (
        (front_hip, front_knee, rear_hip, rear_knee),
        (front_hip, front_knee, rear_knee, rear_foot),
        (front_knee, front_foot, rear_hip, rear_knee),
        (front_knee, front_foot, rear_knee, rear_foot),
    )
    return any(proper_segments_intersect(*pair) for pair in pairs)


def controller_targets(
    phase: float,
    time: float,
    coefficients: np.ndarray,
    *,
    oscillator_rate: float | None = None,
    control_phase: float | None = None,
    ramp_duration: float = 0.25,
    knee_bias_rad: float = 0.0,
    minimum_foot_surface_gap_m: float = 0.0,
    foot_gap_tracking_margin_m: float = FOOT_GAP_TRACKING_MARGIN_M,
) -> np.ndarray:
    if coefficients.shape != (8,):
        raise ValueError("phase controller requires 8 coefficients")
    if ramp_duration <= 0.0:
        blend = 1.0
    else:
        normalized_time = min(max(time / ramp_duration, 0.0), 1.0)
        blend = normalized_time * normalized_time * (3.0 - 2.0 * normalized_time)

    if control_phase is None:
        control_phase = (
            phase if oscillator_rate is None else oscillator_rate * time
        )
    targets = []
    for joint_index, (_, compact_target) in enumerate(JOINT_TARGETS):
        sine = coefficients[2 * joint_index]
        cosine = coefficients[2 * joint_index + 1]
        nominal_offset = knee_bias_rad if joint_index in (1, 3) else 0.0
        target = compact_target + nominal_offset + blend * (
            sine * math.sin(control_phase)
            + cosine * math.cos(control_phase)
        )
        safe_range = (
            FIXED_PARAMETERS.hip.shell_compatible_range
            if "hip" in JOINT_TARGETS[joint_index][0]
            else FIXED_PARAMETERS.knee.shell_compatible_range
        )
        targets.append(float(np.clip(target, *safe_range)))
    return project_targets_to_foot_gap(
        np.asarray(targets),
        minimum_foot_surface_gap_m,
        tracking_margin_m=foot_gap_tracking_margin_m,
    )


def rollout_controller(
    model: mujoco.MjModel,
    coefficients: np.ndarray,
    *,
    duration: float,
    oscillator_rate: float | None = None,
    oscillator_coupling: float = 0.0,
    objective: str = "sustained",
    minimum_foot_surface_gap_m: float = 0.0,
    foot_gap_tracking_margin_m: float = FOOT_GAP_TRACKING_MARGIN_M,
    enforce_leg_crossing_constraint: bool = True,
    detailed: bool = False,
) -> ControllerRollout:
    if objective not in {"barrier", "sustained"}:
        raise ValueError(f"unknown objective: {objective}")
    knee_bias_rad = knee_bias_for_foot_gap(minimum_foot_surface_gap_m)
    data = mujoco.MjData(model)
    compact_key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
    root_pitch_joint_id = _id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch"
    )
    root_x_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
    root_z_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_z")
    root_x_qpos_address = int(model.jnt_qposadr[root_x_joint_id])
    root_z_qpos_address = int(model.jnt_qposadr[root_z_joint_id])
    root_pitch_qpos_address = int(model.jnt_qposadr[root_pitch_joint_id])
    root_pitch_dof_address = int(model.jnt_dofadr[root_pitch_joint_id])
    front_foot_site_id = _id(
        model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
    )
    rear_foot_site_id = _id(
        model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
    )
    floor_geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    front_foot_geom_id = _id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "front_foot_proxy"
    )
    rear_foot_geom_id = _id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "rear_foot_proxy"
    )
    allowed_foot_pair = frozenset((front_foot_geom_id, rear_foot_geom_id))
    leg_body_ids = {
        name: _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (
            "front_thigh",
            "front_shank",
            "rear_thigh",
            "rear_shank",
        )
    }
    shell_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if "_shell_"
        in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
    ]
    joint_setup = _joint_setup(model)

    mujoco.mj_resetDataKeyframe(model, data, compact_key_id)
    if minimum_foot_surface_gap_m > 0.0:
        separated = replace(
            FIXED_PARAMETERS,
            compact_foot_surface_gap=minimum_foot_surface_gap_m,
        )
        data.qpos[root_z_qpos_address] = separated.compact_root_height
        for joint_name, _, qpos_address, _, _ in joint_setup:
            if "knee" in joint_name:
                data.qpos[qpos_address] = separated.compact_knee_angle
    data.qvel[:] = 0.0
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    for root_joint_name in ("root_x", "root_z", "root_pitch"):
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
        model.dof_damping[int(model.jnt_dofadr[joint_id])] = 0.0
    mujoco.mj_forward(model, data)

    timestep = float(model.opt.timestep)
    steps = int(math.ceil(duration / timestep))
    previous_phase = float(data.qpos[root_pitch_qpos_address])
    initial_root_x = float(data.qpos[root_x_qpos_address])
    maximum_phase = previous_phase
    forward_travel = 0.0
    backward_travel = 0.0
    actuator_positive_work = 0.0
    actuator_absorbed_work = 0.0
    airborne_steps = 0
    consecutive_airborne_steps = 0
    longest_airborne_steps = 0
    maximum_foot_gap = 0.0
    minimum_foot_surface_gap = math.inf
    foot_contact_steps = 0
    consecutive_foot_contact_steps = 0
    longest_foot_contact_steps = 0
    foot_gap_deficit_integral = 0.0
    maximum_foot_gap_deficit = 0.0
    maximum_joint_error = 0.0
    maximum_torque = 0.0
    saturated_steps = 0
    self_contact_steps = 0
    forbidden_contact_steps = 0
    forbidden_penetration_integral = 0.0
    allowed_penetration_excess_integral = 0.0
    maximum_forbidden_penetration = 0.0
    maximum_allowed_foot_penetration = 0.0
    leg_crossing_detected = False
    first_leg_crossing_time: float | None = None
    tail_start_phase = previous_phase
    tail_start_root_x = initial_root_x
    latest_contact_metrics = _contact_metrics(
        model,
        data,
        floor_geom_id=floor_geom_id,
        allowed_foot_pair=allowed_foot_pair,
    )

    columns = (
        "time_s",
        "phase_rad",
        "phase_deg",
        "pitch_rate_rad_s",
        "root_x_m",
        "front_hip_target_rad",
        "front_knee_target_rad",
        "rear_hip_target_rad",
        "rear_knee_target_rad",
        "front_hip_position_rad",
        "front_knee_position_rad",
        "rear_hip_position_rad",
        "rear_knee_position_rad",
        "front_hip_torque_Nm",
        "front_knee_torque_Nm",
        "rear_hip_torque_Nm",
        "rear_knee_torque_Nm",
        "actuator_power_W",
        "actuator_net_work_J",
        "foot_gap_m",
        "roundness_peak_to_peak_m",
        "ground_contact_count",
        "airborne",
        "self_contact_pair_count",
        "forbidden_contact_pair_count",
        "allowed_foot_penetration_m",
        "forbidden_penetration_m",
        "leg_crossing",
    )
    rows: list[list[float]] = []
    actuator_net_work = 0.0
    oscillator_phase = 0.0

    def record(targets: np.ndarray) -> None:
        nonlocal maximum_foot_gap
        actual_positions = [
            float(data.qpos[qpos_address])
            for _, _, qpos_address, _, _ in joint_setup
        ]
        torques = [
            float(data.actuator_force[actuator_id])
            for _, _, _, _, actuator_id in joint_setup
        ]
        power = sum(
            torque * float(data.qvel[dof_address])
            for torque, (_, _, _, dof_address, _) in zip(torques, joint_setup)
        )
        foot_gap = float(
            np.linalg.norm(
                np.asarray(data.site_xpos[front_foot_site_id])[[0, 2]]
                - np.asarray(data.site_xpos[rear_foot_site_id])[[0, 2]]
            )
        )
        maximum_foot_gap = max(maximum_foot_gap, foot_gap)
        _, roundness_peak_to_peak = _shell_roundness(
            model, data, shell_geom_ids
        )
        (
            ground_contact_count,
            self_contact_pair_count,
            forbidden_contact_pair_count,
            allowed_foot_penetration,
            forbidden_penetration,
        ) = latest_contact_metrics
        rows.append(
            [
                float(data.time),
                float(data.qpos[root_pitch_qpos_address]),
                math.degrees(float(data.qpos[root_pitch_qpos_address])),
                float(data.qvel[root_pitch_dof_address]),
                float(data.qpos[root_x_qpos_address]),
                *targets,
                *actual_positions,
                *torques,
                power,
                actuator_net_work,
                foot_gap,
                roundness_peak_to_peak,
                float(ground_contact_count),
                float(ground_contact_count == 0),
                float(self_contact_pair_count),
                float(forbidden_contact_pair_count),
                allowed_foot_penetration,
                forbidden_penetration,
                float(leg_crossing_detected),
            ]
        )

    initial_targets = controller_targets(
        previous_phase,
        float(data.time),
        coefficients,
        oscillator_rate=oscillator_rate,
        control_phase=oscillator_phase if oscillator_rate is not None else None,
        knee_bias_rad=knee_bias_rad,
        minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
    )
    data.ctrl[:] = initial_targets
    if detailed:
        record(initial_targets)

    for step in range(steps):
        if step == int(0.75 * steps):
            tail_start_phase = float(data.qpos[root_pitch_qpos_address])
            tail_start_root_x = float(data.qpos[root_x_qpos_address])
        phase = float(data.qpos[root_pitch_qpos_address])
        if oscillator_rate is not None:
            oscillator_phase_rate = oscillator_rate + oscillator_coupling * math.sin(
                phase - oscillator_phase
            )
            oscillator_phase += timestep * max(0.1, oscillator_phase_rate)
        targets = controller_targets(
            phase,
            float(data.time),
            coefficients,
            oscillator_rate=oscillator_rate,
            control_phase=(
                oscillator_phase if oscillator_rate is not None else None
            ),
            knee_bias_rad=knee_bias_rad,
            minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
            foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
        )
        data.ctrl[:] = targets
        mujoco.mj_step(model, data)
        latest_contact_metrics = _contact_metrics(
            model,
            data,
            floor_geom_id=floor_geom_id,
            allowed_foot_pair=allowed_foot_pair,
        )

        new_phase = float(data.qpos[root_pitch_qpos_address])
        phase_change = new_phase - previous_phase
        forward_travel += max(phase_change, 0.0)
        backward_travel += max(-phase_change, 0.0)
        maximum_phase = max(maximum_phase, new_phase)
        previous_phase = new_phase

        actual_positions = np.asarray(
            [
                data.qpos[qpos_address]
                for _, _, qpos_address, _, _ in joint_setup
            ]
        )
        maximum_joint_error = max(
            maximum_joint_error,
            float(np.max(np.abs(actual_positions - targets))),
        )
        torques = np.asarray(
            [
                data.actuator_force[actuator_id]
                for _, _, _, _, actuator_id in joint_setup
            ]
        )
        maximum_torque = max(maximum_torque, float(np.max(np.abs(torques))))
        if np.any(np.abs(torques) >= 0.999 * 6.0):
            saturated_steps += 1
        power = sum(
            float(torque) * float(data.qvel[dof_address])
            for torque, (_, _, _, dof_address, _) in zip(torques, joint_setup)
        )
        if power >= 0.0:
            actuator_positive_work += power * timestep
        else:
            actuator_absorbed_work += -power * timestep
        actuator_net_work += power * timestep
        (
            ground_contact_count,
            self_contact_pair_count,
            forbidden_contact_pair_count,
            allowed_foot_penetration,
            forbidden_penetration,
        ) = latest_contact_metrics
        if self_contact_pair_count > 0:
            self_contact_steps += 1
        if forbidden_contact_pair_count > 0:
            forbidden_contact_steps += 1
        maximum_forbidden_penetration = max(
            maximum_forbidden_penetration, forbidden_penetration
        )
        maximum_allowed_foot_penetration = max(
            maximum_allowed_foot_penetration, allowed_foot_penetration
        )
        forbidden_penetration_integral += forbidden_penetration * timestep
        allowed_penetration_excess_integral += max(
            allowed_foot_penetration - ALLOWED_FOOT_PENETRATION_M, 0.0
        ) * timestep

        if ground_contact_count == 0:
            airborne_steps += 1
            consecutive_airborne_steps += 1
            longest_airborne_steps = max(
                longest_airborne_steps, consecutive_airborne_steps
            )
        else:
            consecutive_airborne_steps = 0

        crossing_check_interval = max(1, round(0.005 / timestep))
        if (
            enforce_leg_crossing_constraint
            and step % crossing_check_interval == 0
            and _has_leg_crossing(
                data,
                front_thigh_body_id=leg_body_ids["front_thigh"],
                front_shank_body_id=leg_body_ids["front_shank"],
                rear_thigh_body_id=leg_body_ids["rear_thigh"],
                rear_shank_body_id=leg_body_ids["rear_shank"],
                front_foot_site_id=front_foot_site_id,
                rear_foot_site_id=rear_foot_site_id,
            )
        ):
            leg_crossing_detected = True
            first_leg_crossing_time = float(data.time)

        foot_gap = float(
            np.linalg.norm(
                np.asarray(data.site_xpos[front_foot_site_id])[[0, 2]]
                - np.asarray(data.site_xpos[rear_foot_site_id])[[0, 2]]
            )
        )
        foot_surface_gap = foot_gap - 2.0 * FIXED_PARAMETERS.foot_radius
        maximum_foot_gap = max(maximum_foot_gap, foot_gap)
        minimum_foot_surface_gap = min(
            minimum_foot_surface_gap, foot_surface_gap
        )
        if foot_surface_gap <= 0.0:
            foot_contact_steps += 1
            consecutive_foot_contact_steps += 1
            longest_foot_contact_steps = max(
                longest_foot_contact_steps, consecutive_foot_contact_steps
            )
        else:
            consecutive_foot_contact_steps = 0
        target_gap_ramp = min(max(float(data.time) / 0.25, 0.0), 1.0)
        target_gap_ramp = target_gap_ramp * target_gap_ramp * (
            3.0 - 2.0 * target_gap_ramp
        )
        gap_deficit = (
            max(
                minimum_foot_surface_gap_m * target_gap_ramp
                - foot_surface_gap,
                0.0,
            )
            if minimum_foot_surface_gap_m > 0.0
            else 0.0
        )
        foot_gap_deficit_integral += gap_deficit * timestep
        maximum_foot_gap_deficit = max(
            maximum_foot_gap_deficit, gap_deficit
        )
        if detailed:
            record(targets)

        if (
            leg_crossing_detected
            or not np.isfinite(data.qpos).all()
            or data.qpos[1] > 1.0
        ):
            break

    final_phase = float(data.qpos[root_pitch_qpos_address])
    final_root_x = float(data.qpos[root_x_qpos_address])
    rolling_radius = FIXED_PARAMETERS.shell_contact_radius
    root_x_displacement = final_root_x - initial_root_x
    translation_phase = root_x_displacement / rolling_radius
    rolling_progress = 0.5 * (final_phase + translation_phase)
    rolling_mismatch = abs(final_phase - translation_phase)
    tail_phase_progress = final_phase - tail_start_phase
    tail_translation_progress = (
        final_root_x - tail_start_root_x
    ) / rolling_radius
    tail_rolling_progress = 0.5 * (
        tail_phase_progress + tail_translation_progress
    )
    conservative_rolling_progress = min(final_phase, translation_phase)
    conservative_tail_progress = min(
        tail_phase_progress, tail_translation_progress
    )
    executed_steps = max(int(round(float(data.time) / timestep)), 1)
    airborne_fraction = airborne_steps / executed_steps
    saturation_fraction = saturated_steps / executed_steps
    self_contact_fraction = self_contact_steps / executed_steps
    forbidden_contact_fraction = forbidden_contact_steps / executed_steps
    forbidden_contact_time = forbidden_contact_steps * timestep
    maximum_allowed_penetration_excess = max(
        maximum_allowed_foot_penetration - ALLOWED_FOOT_PENETRATION_M, 0.0
    )
    foot_contact_time = foot_contact_steps * timestep
    foot_separation_penalty = (
        FOOT_CONTACT_TIME_WEIGHT * foot_contact_time
        + LONGEST_FOOT_CONTACT_WEIGHT
        * longest_foot_contact_steps
        * timestep
        + FOOT_GAP_DEFICIT_INTEGRAL_WEIGHT * foot_gap_deficit_integral
        + MAXIMUM_FOOT_GAP_DEFICIT_WEIGHT * maximum_foot_gap_deficit
        + FOOT_SURFACE_PENETRATION_WEIGHT
        * max(-minimum_foot_surface_gap, 0.0)
        if minimum_foot_surface_gap_m > 0.0
        else 0.0
    )
    foot_gap_safety_penalty = (
        FOOT_GAP_AIRBORNE_EXCESS_WEIGHT
        * max(
            airborne_fraction - FOOT_GAP_MAXIMUM_AIRBORNE_FRACTION,
            0.0,
        )
        + FOOT_GAP_FORBIDDEN_EXCESS_WEIGHT
        * max(
            maximum_forbidden_penetration
            - FOOT_GAP_MAXIMUM_FORBIDDEN_PENETRATION_M,
            0.0,
        )
        if minimum_foot_surface_gap_m > 0.0
        else 0.0
    )
    collision_penalty = (
        FORBIDDEN_CONTACT_TIME_WEIGHT * forbidden_contact_time
        + FORBIDDEN_PENETRATION_INTEGRAL_WEIGHT
        * forbidden_penetration_integral
        + MAXIMUM_FORBIDDEN_PENETRATION_WEIGHT
        * maximum_forbidden_penetration
        + ALLOWED_PENETRATION_EXCESS_INTEGRAL_WEIGHT
        * allowed_penetration_excess_integral
        + MAXIMUM_ALLOWED_PENETRATION_EXCESS_WEIGHT
        * maximum_allowed_penetration_excess
        + LEG_CROSSING_FAILURE_PENALTY * float(leg_crossing_detected)
    )
    constraint_penalty = (
        0.10 * actuator_positive_work
        + 0.80 * rolling_mismatch
        + 80.0 * max(airborne_fraction - 0.08, 0.0)
        + 80.0 * max(maximum_foot_gap - 0.20, 0.0)
        + 30.0 * saturation_fraction
        + collision_penalty
        + foot_separation_penalty
        + foot_gap_safety_penalty
    )
    # Reward net rolling measured by both body rotation and ground translation.
    # The final and last-quarter progress dominate the one-time peak so a
    # candidate that surges forward and then rolls back cannot score well.
    if objective == "barrier":
        # Curriculum stage 1: first discover a sufficiently energetic motion
        # that gets over the initial potential/contact barrier, without
        # mistaking an airborne spin or a fully opened body for rolling.
        score = (
            1.00 * min(maximum_phase, max(translation_phase, 0.0))
            + 0.15 * maximum_phase
            - 0.10 * backward_travel
            - constraint_penalty
        )
    else:
        score = (
            1.30 * conservative_rolling_progress
            + 1.00 * conservative_tail_progress
            + 0.10 * maximum_phase
            - 0.70 * backward_travel
            - constraint_penalty
        )
    summary: dict[str, float | int | bool] = {
        "score": score,
        "duration_s": float(data.time),
        "maximum_phase_rad": maximum_phase,
        "maximum_phase_deg": math.degrees(maximum_phase),
        "maximum_turns": maximum_phase / (2.0 * math.pi),
        "final_phase_rad": final_phase,
        "final_phase_deg": math.degrees(final_phase),
        "net_turns": final_phase / (2.0 * math.pi),
        "root_x_displacement_m": root_x_displacement,
        "translation_equivalent_phase_rad": translation_phase,
        "translation_equivalent_turns": translation_phase / (2.0 * math.pi),
        "rolling_progress_rad": rolling_progress,
        "rolling_progress_turns": rolling_progress / (2.0 * math.pi),
        "conservative_rolling_progress_rad": conservative_rolling_progress,
        "conservative_rolling_turns": (
            conservative_rolling_progress / (2.0 * math.pi)
        ),
        "rolling_mismatch_rad": rolling_mismatch,
        "tail_rolling_progress_rad": tail_rolling_progress,
        "conservative_tail_progress_rad": conservative_tail_progress,
        "forward_travel_rad": forward_travel,
        "backward_travel_rad": backward_travel,
        "actuator_positive_work_J": actuator_positive_work,
        "actuator_absorbed_work_J": actuator_absorbed_work,
        "actuator_net_work_J": actuator_net_work,
        "airborne_fraction": airborne_fraction,
        "airborne_total_s": airborne_steps * timestep,
        "longest_airborne_s": longest_airborne_steps * timestep,
        "maximum_foot_gap_m": maximum_foot_gap,
        "minimum_foot_surface_gap_m": minimum_foot_surface_gap,
        "target_minimum_foot_surface_gap_m": minimum_foot_surface_gap_m,
        "foot_contact_fraction": foot_contact_steps / executed_steps,
        "foot_contact_total_s": foot_contact_time,
        "longest_foot_contact_s": longest_foot_contact_steps * timestep,
        "foot_gap_deficit_integral_m_s": foot_gap_deficit_integral,
        "maximum_foot_gap_deficit_m": maximum_foot_gap_deficit,
        "foot_separation_penalty": foot_separation_penalty,
        "foot_gap_safety_penalty": foot_gap_safety_penalty,
        "foot_gap_maximum_airborne_fraction": (
            FOOT_GAP_MAXIMUM_AIRBORNE_FRACTION
        ),
        "foot_gap_maximum_forbidden_penetration_m": (
            FOOT_GAP_MAXIMUM_FORBIDDEN_PENETRATION_M
        ),
        "maximum_joint_tracking_error_deg": math.degrees(maximum_joint_error),
        "maximum_actuator_torque_Nm": maximum_torque,
        "actuator_saturation_fraction": saturation_fraction,
        "self_contact_fraction": self_contact_fraction,
        "self_contact_total_s": self_contact_steps * timestep,
        "forbidden_contact_fraction": forbidden_contact_fraction,
        "forbidden_contact_total_s": forbidden_contact_time,
        "forbidden_penetration_integral_m_s": (
            forbidden_penetration_integral
        ),
        "maximum_forbidden_penetration_m": maximum_forbidden_penetration,
        "maximum_allowed_foot_penetration_m": (
            maximum_allowed_foot_penetration
        ),
        "allowed_foot_penetration_tolerance_m": (
            ALLOWED_FOOT_PENETRATION_M
        ),
        "allowed_penetration_excess_integral_m_s": (
            allowed_penetration_excess_integral
        ),
        "collision_penalty": collision_penalty,
        "leg_crossing_constraint_enabled": enforce_leg_crossing_constraint,
        "leg_crossing_detected": leg_crossing_detected,
        "first_leg_crossing_time_s": first_leg_crossing_time,
        "reached_one_turn": maximum_phase >= 2.0 * math.pi,
        "completed_one_turn": conservative_rolling_progress >= 2.0 * math.pi,
        "completed_two_turns": conservative_rolling_progress >= 4.0 * math.pi,
    }
    row_array = np.asarray(rows, dtype=float) if detailed else None
    return ControllerRollout(
        score=score,
        summary=summary,
        columns=columns if detailed else None,
        rows=row_array,
    )


def _update_distribution(
    samples: np.ndarray,
    scores: np.ndarray,
    *,
    elite_count: int,
    previous_mean: np.ndarray,
    previous_std: np.ndarray,
    smoothing: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    elite_indices = np.argsort(scores)[-elite_count:]
    elites = samples[elite_indices]
    elite_mean = elites.mean(axis=0)
    elite_std = elites.std(axis=0)
    mean = smoothing * elite_mean + (1.0 - smoothing) * previous_mean
    std = smoothing * elite_std + (1.0 - smoothing) * previous_std
    return mean, np.maximum(std, 0.015)


def optimize_controller(
    *,
    model_path: Path = MODEL_PATH,
    generations: int,
    population: int,
    elite_count: int,
    duration: float,
    seed: int,
    barrier_generations: int = 4,
    initial_parameters: np.ndarray | None = None,
    workers: int = 1,
    minimum_foot_surface_gap_m: float = 0.0,
    foot_gap_tracking_margin_m: float = FOOT_GAP_TRACKING_MARGIN_M,
    enforce_leg_crossing_constraint: bool = True,
) -> tuple[np.ndarray, list[dict[str, float | int | str]], ControllerRollout]:
    if not 1 <= elite_count <= population:
        raise ValueError("elite_count must be between 1 and population")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    rng = np.random.default_rng(seed)
    lower = np.concatenate(
        [
            COEFFICIENT_BOUNDS[:, 0],
            [OSCILLATOR_RATE_BOUNDS[0], OSCILLATOR_COUPLING_BOUNDS[0]],
        ]
    )
    upper = np.concatenate(
        [
            COEFFICIENT_BOUNDS[:, 1],
            [OSCILLATOR_RATE_BOUNDS[1], OSCILLATOR_COUPLING_BOUNDS[1]],
        ]
    )
    mean = 0.5 * (lower + upper)
    mean[:8] = 0.0
    std = (upper - lower) / 5.0
    if initial_parameters is not None:
        if initial_parameters.shape != (10,):
            raise ValueError("initial_parameters must have shape (10,)")
        mean = np.clip(initial_parameters, lower, upper)
    best_parameters = mean.copy()
    best_rollout: ControllerRollout | None = None
    history: list[dict[str, float | int | str]] = []

    model_path = Path(model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path)) if workers == 1 else None
    executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_rollout_worker,
            initargs=(str(model_path),),
        )
        if workers > 1
        else None
    )
    try:
        for generation in range(generations):
            objective = (
                "barrier" if generation < barrier_generations else "sustained"
            )
            rollout_duration = (
                min(duration, 2.5) if objective == "barrier" else duration
            )
            if generation == 0 and initial_parameters is None:
                samples = rng.uniform(lower, upper, size=(population, 10))
            else:
                samples = rng.normal(mean, std, size=(population, 10))
            samples = np.clip(samples, lower, upper)
            samples[0] = best_parameters
            if generation == barrier_generations:
                # Scores from the two curriculum stages are not comparable.
                # Keep the barrier coefficients as a seed, then select the
                # sustained controller afresh.
                best_rollout = None
            if executor is None:
                assert model is not None
                generation_rollouts = [
                    rollout_controller(
                        model,
                        sample[:8],
                        duration=rollout_duration,
                        oscillator_rate=float(sample[8]),
                        oscillator_coupling=float(sample[9]),
                        objective=objective,
                        minimum_foot_surface_gap_m=(
                            minimum_foot_surface_gap_m
                        ),
                        foot_gap_tracking_margin_m=(
                            foot_gap_tracking_margin_m
                        ),
                        enforce_leg_crossing_constraint=(
                            enforce_leg_crossing_constraint
                        ),
                        detailed=False,
                    )
                    for sample in samples
                ]
            else:
                tasks = [
                    (
                        sample,
                        rollout_duration,
                        objective,
                        minimum_foot_surface_gap_m,
                        foot_gap_tracking_margin_m,
                        enforce_leg_crossing_constraint,
                    )
                    for sample in samples
                ]
                generation_rollouts = list(
                    executor.map(
                        _rollout_worker,
                        tasks,
                        chunksize=max(1, population // (4 * workers)),
                    )
                )
            scores = np.asarray(
                [rollout.score for rollout in generation_rollouts], dtype=float
            )
            for sample, rollout in zip(samples, generation_rollouts):
                if best_rollout is None or rollout.score > best_rollout.score:
                    best_rollout = rollout
                    best_parameters = sample.copy()

            mean, std = _update_distribution(
                samples,
                scores,
                elite_count=elite_count,
                previous_mean=mean,
                previous_std=std,
            )
            mean = np.clip(mean, lower, upper)
            generation_best_index = int(np.argmax(scores))
            generation_best = generation_rollouts[generation_best_index]
            history.append(
                {
                    "generation": generation,
                    "objective": objective,
                    "generation_best_score": generation_best.score,
                    "generation_best_phase_deg": float(
                        generation_best.summary["maximum_phase_deg"]
                    ),
                    "generation_best_final_phase_deg": float(
                        generation_best.summary["final_phase_deg"]
                    ),
                    "generation_best_rate_rad_s": float(
                        samples[generation_best_index, 8]
                    ),
                    "generation_best_coupling_per_s": float(
                        samples[generation_best_index, 9]
                    ),
                    "generation_best_collision_penalty": float(
                        generation_best.summary["collision_penalty"]
                    ),
                    "generation_best_forbidden_contact_s": float(
                        generation_best.summary["forbidden_contact_total_s"]
                    ),
                    "generation_best_max_forbidden_penetration_mm": 1000.0
                    * float(
                        generation_best.summary[
                            "maximum_forbidden_penetration_m"
                        ]
                    ),
                    "generation_best_foot_contact_s": float(
                        generation_best.summary["foot_contact_total_s"]
                    ),
                    "generation_best_min_foot_surface_gap_mm": 1000.0
                    * float(
                        generation_best.summary[
                            "minimum_foot_surface_gap_m"
                        ]
                    ),
                    "global_best_score": float(best_rollout.score),
                    "global_best_phase_deg": float(
                        best_rollout.summary["maximum_phase_deg"]
                    ),
                    "global_best_final_phase_deg": float(
                        best_rollout.summary["final_phase_deg"]
                    ),
                    "population_mean_score": float(scores.mean()),
                    "population_std_score": float(scores.std()),
                }
            )
            print(
                f"generation={generation:02d} "
                f"objective={objective} "
                f"best_score={best_rollout.score:.3f} "
                f"max_phase={best_rollout.summary['maximum_phase_deg']:.1f}deg "
                f"final_phase={best_rollout.summary['final_phase_deg']:.1f}deg "
                f"collision_penalty="
                f"{best_rollout.summary['collision_penalty']:.3f} "
                f"foot_contact="
                f"{best_rollout.summary['foot_contact_total_s']:.3f}s",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    assert best_rollout is not None
    return best_parameters, history, best_rollout


def _coefficient_summary(
    parameters: np.ndarray,
    minimum_foot_surface_gap_m: float = 0.0,
    foot_gap_tracking_margin_m: float = FOOT_GAP_TRACKING_MARGIN_M,
) -> dict[str, object]:
    coefficients = parameters[:8]
    raw = {
        name: float(value) for name, value in zip(PARAMETER_NAMES, coefficients)
    }
    joints = {}
    for joint_index, (joint_name, _) in enumerate(JOINT_TARGETS):
        sine = float(coefficients[2 * joint_index])
        cosine = float(coefficients[2 * joint_index + 1])
        joints[joint_name] = {
            "sine_coefficient_rad": sine,
            "cosine_coefficient_rad": cosine,
            "amplitude_rad": math.hypot(sine, cosine),
            "amplitude_deg": math.degrees(math.hypot(sine, cosine)),
            "phase_offset_rad": math.atan2(cosine, sine),
            "phase_offset_deg": math.degrees(math.atan2(cosine, sine)),
        }
    return {
        "controller": "phase_locked_oscillator",
        "oscillator_rate_rad_s": float(parameters[8]),
        "oscillator_period_s": 2.0 * math.pi / float(parameters[8]),
        "oscillator_coupling_per_s": float(parameters[9]),
        "minimum_foot_surface_gap_m": minimum_foot_surface_gap_m,
        "nominal_knee_bias_rad": knee_bias_for_foot_gap(
            minimum_foot_surface_gap_m
        ),
        "foot_gap_tracking_margin_m": foot_gap_tracking_margin_m,
        "raw_coefficients": raw,
        "joint_sinusoid": joints,
        "collision_objective": {
            "allowed_contact": "front_foot_proxy__rear_foot_proxy",
            "allowed_foot_penetration_tolerance_m": (
                ALLOWED_FOOT_PENETRATION_M
            ),
            "forbidden_contact_time_weight": (
                FORBIDDEN_CONTACT_TIME_WEIGHT
            ),
            "forbidden_penetration_integral_weight": (
                FORBIDDEN_PENETRATION_INTEGRAL_WEIGHT
            ),
            "maximum_forbidden_penetration_weight": (
                MAXIMUM_FORBIDDEN_PENETRATION_WEIGHT
            ),
            "allowed_penetration_excess_integral_weight": (
                ALLOWED_PENETRATION_EXCESS_INTEGRAL_WEIGHT
            ),
            "maximum_allowed_penetration_excess_weight": (
                MAXIMUM_ALLOWED_PENETRATION_EXCESS_WEIGHT
            ),
            "leg_crossing_failure_penalty": LEG_CROSSING_FAILURE_PENALTY,
            "foot_contact_time_weight": FOOT_CONTACT_TIME_WEIGHT,
            "longest_foot_contact_weight": LONGEST_FOOT_CONTACT_WEIGHT,
            "foot_gap_deficit_integral_weight": (
                FOOT_GAP_DEFICIT_INTEGRAL_WEIGHT
            ),
            "maximum_foot_gap_deficit_weight": (
                MAXIMUM_FOOT_GAP_DEFICIT_WEIGHT
            ),
            "foot_surface_penetration_weight": (
                FOOT_SURFACE_PENETRATION_WEIGHT
            ),
            "maximum_airborne_fraction": (
                FOOT_GAP_MAXIMUM_AIRBORNE_FRACTION
            ),
            "maximum_forbidden_penetration_m": (
                FOOT_GAP_MAXIMUM_FORBIDDEN_PENETRATION_M
            ),
            "tracking_margin_m": foot_gap_tracking_margin_m,
        },
    }


def _load_controller_parameters(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["raw_coefficients"]
    return np.asarray(
        [
            *(float(raw[name]) for name in PARAMETER_NAMES),
            float(payload["oscillator_rate_rad_s"]),
            float(payload["oscillator_coupling_per_s"]),
        ],
        dtype=float,
    )


def _write_history(
    path: Path, history: list[dict[str, float | int | str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _write_rollout_csv(path: Path, rollout: ControllerRollout) -> None:
    assert rollout.columns is not None and rollout.rows is not None
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(rollout.columns)
        writer.writerows(rollout.rows)


def _write_rollout_plot(
    path: Path,
    baseline: ControllerRollout,
    controlled: ControllerRollout,
) -> None:
    image = Image.new("RGB", (1400, 1360), color=(24, 31, 41))
    draw = ImageDraw.Draw(image)
    draw.text(
        (28, 16),
        "Phase-locked active rolling oscillator",
        fill=(244, 247, 251),
    )
    draw.text(
        (28, 38),
        (
            f"max phase={controlled.summary['maximum_phase_deg']:.1f} deg, "
            f"final={controlled.summary['final_phase_deg']:.1f} deg, "
            f"net actuator work={controlled.summary['actuator_net_work_J']:.2f} J"
        ),
        fill=(180, 191, 205),
    )
    time = controlled.column("time_s")
    controlled_torque = np.max(
        np.abs(
            np.column_stack(
                [
                    controlled.column(f"{joint_name}_torque_Nm")
                    for joint_name, _ in JOINT_TARGETS
                ]
            )
        ),
        axis=1,
    )
    panels = (
        (
            (18, 72, 690, 386),
            (
                ("fixed compact", baseline.column("phase_deg"), (70, 180, 255)),
                ("phase control", controlled.column("phase_deg"), (255, 164, 74)),
            ),
            "Root roll phase (deg)",
            False,
        ),
        (
            (710, 72, 1382, 386),
            (
                (
                    "target",
                    controlled.column("front_hip_target_rad"),
                    (255, 164, 74),
                ),
                (
                    "actual",
                    controlled.column("front_hip_position_rad"),
                    (70, 180, 255),
                ),
            ),
            "Front hip target and position (rad)",
            False,
        ),
        (
            (18, 394, 690, 708),
            (
                (
                    "target",
                    controlled.column("front_knee_target_rad"),
                    (255, 164, 74),
                ),
                (
                    "actual",
                    controlled.column("front_knee_position_rad"),
                    (70, 180, 255),
                ),
            ),
            "Front knee target and position (rad)",
            False,
        ),
        (
            (710, 394, 1382, 708),
            (
                ("max torque", controlled_torque, (208, 124, 255)),
                (
                    "6 Nm limit",
                    np.full_like(controlled_torque, 6.0),
                    (255, 164, 74),
                ),
            ),
            "Maximum absolute actuator torque (N m)",
            False,
        ),
        (
            (18, 716, 690, 1030),
            (
                (
                    "net work",
                    controlled.column("actuator_net_work_J"),
                    (92, 214, 143),
                ),
                (
                    "power",
                    controlled.column("actuator_power_W"),
                    (255, 164, 74),
                ),
            ),
            "Actuator net work (J) and power (W)",
            True,
        ),
        (
            (710, 716, 1382, 1030),
            (
                (
                    "foot gap mm",
                    1000.0 * controlled.column("foot_gap_m"),
                    (255, 164, 74),
                ),
                (
                    "roundness mm",
                    1000.0 * controlled.column("roundness_peak_to_peak_m"),
                    (70, 180, 255),
                ),
            ),
            "Shape deformation (mm)",
            False,
        ),
        (
            (18, 1038, 690, 1352),
            (
                (
                    "all self-contact pairs",
                    controlled.column("self_contact_pair_count"),
                    (92, 214, 143),
                ),
                (
                    "forbidden pairs",
                    controlled.column("forbidden_contact_pair_count"),
                    (255, 97, 104),
                ),
            ),
            "Active self-contact pair count",
            True,
        ),
        (
            (710, 1038, 1382, 1352),
            (
                (
                    "forbidden penetration",
                    1000.0 * controlled.column("forbidden_penetration_m"),
                    (255, 97, 104),
                ),
                (
                    "foot penetration",
                    1000.0
                    * controlled.column("allowed_foot_penetration_m"),
                    (208, 124, 255),
                ),
                (
                    "foot tolerance",
                    np.full_like(
                        time, 1000.0 * ALLOWED_FOOT_PENETRATION_M
                    ),
                    (255, 164, 74),
                ),
            ),
            "Self-contact penetration depth (mm)",
            True,
        ),
    )
    for bounds, series, title, zero_line in panels:
        _draw_panel(
            draw,
            bounds,
            time,
            series,
            title,
            zero_line=zero_line,
        )
    image.save(path)


def write_outputs(
    output_dir: Path,
    parameters: np.ndarray,
    history: list[dict[str, float | int | str]],
    baseline: ControllerRollout,
    controlled: ControllerRollout,
    minimum_foot_surface_gap_m: float = 0.0,
    foot_gap_tracking_margin_m: float = FOOT_GAP_TRACKING_MARGIN_M,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    controller_path = output_dir / "best_phase_controller.json"
    history_path = output_dir / "cem_history.csv"
    rollout_path = output_dir / "best_rollout.csv"
    plot_path = output_dir / "best_rollout.png"
    payload = _coefficient_summary(
        parameters,
        minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m,
    )
    payload["rollout_summary"] = controlled.summary
    controller_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_history(history_path, history)
    _write_rollout_csv(rollout_path, controlled)
    _write_rollout_plot(plot_path, baseline, controlled)
    return controller_path, history_path, rollout_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--final-duration", type=float, default=8.0)
    parser.add_argument("--barrier-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--minimum-foot-gap-mm",
        type=float,
        default=0.0,
        help="Minimum desired foot surface gap after the startup ramp.",
    )
    parser.add_argument(
        "--foot-gap-tracking-margin-mm",
        type=float,
        default=1000.0 * FOOT_GAP_TRACKING_MARGIN_M,
        help="Extra target gap used to cover servo tracking error.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--initial-controller",
        type=Path,
        default=None,
        help="Warm-start CEM from a previously saved controller JSON.",
    )
    parser.add_argument(
        "--allow-leg-crossing",
        action="store_true",
        help="Disable the geometric leg-crossing failure constraint.",
    )
    args = parser.parse_args()
    if args.minimum_foot_gap_mm < 0.0:
        parser.error("--minimum-foot-gap-mm cannot be negative")
    if args.foot_gap_tracking_margin_mm < 0.0:
        parser.error("--foot-gap-tracking-margin-mm cannot be negative")
    minimum_foot_surface_gap_m = args.minimum_foot_gap_mm / 1000.0
    foot_gap_tracking_margin_m = (
        args.foot_gap_tracking_margin_mm / 1000.0
    )

    initial_parameters = (
        _load_controller_parameters(args.initial_controller)
        if args.initial_controller is not None
        else None
    )
    parameters, history, _ = optimize_controller(
        model_path=args.model,
        generations=args.generations,
        population=args.population,
        elite_count=args.elite_count,
        duration=args.duration,
        seed=args.seed,
        barrier_generations=args.barrier_generations,
        initial_parameters=initial_parameters,
        workers=args.workers,
        minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
        enforce_leg_crossing_constraint=not args.allow_leg_crossing,
    )
    baseline_model = mujoco.MjModel.from_xml_path(str(args.model))
    baseline = rollout_controller(
        baseline_model,
        np.zeros(8),
        duration=args.final_duration,
        enforce_leg_crossing_constraint=not args.allow_leg_crossing,
        detailed=True,
    )
    controlled_model = mujoco.MjModel.from_xml_path(str(args.model))
    controlled = rollout_controller(
        controlled_model,
        parameters[:8],
        duration=args.final_duration,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
        enforce_leg_crossing_constraint=not args.allow_leg_crossing,
        detailed=True,
    )
    outputs = write_outputs(
        args.output_dir,
        parameters,
        history,
        baseline,
        controlled,
        minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m,
    )

    print("best_parameters=" + np.array2string(parameters, precision=6))
    for key, value in controlled.summary.items():
        print(f"{key}={value}")
    for output in outputs:
        print(f"output={output}")


if __name__ == "__main__":
    main()
