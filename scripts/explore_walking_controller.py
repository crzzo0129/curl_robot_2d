"""Explore a planar walking baseline with foot trajectories and analytic IK.

The planar model merges each left/right leg pair into one virtual leg.  Its
walking analogue is therefore a two-contact dynamic gait, not a four-beat
crawl.  This script keeps the rolling controller and rolling RL task intact
while testing whether the expanded mechanism can sustain an upright gait.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "walking_exploration"

PARAMETER_NAMES = (
    "body_height_m",
    "step_length_m",
    "foot_lift_m",
    "frequency_hz",
    "duty_factor",
    "pitch_kp_m_per_rad",
    "pitch_kd_m_s_per_rad",
    "velocity_gain_s",
    "foot_center_offset_m",
    "pitch_placement_gain_m_per_rad",
)
PARAMETER_BOUNDS = np.asarray(
    [
        (0.230, 0.285),
        (0.020, 0.120),
        (0.025, 0.075),
        (0.50, 2.40),
        (0.60, 0.90),
        (0.000, 0.080),
        (0.000, 0.035),
        (0.000, 0.100),
        (-0.030, 0.040),
        (-0.120, 0.120),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class WalkingControllerConfig:
    body_height_m: float = 0.250
    step_length_m: float = 0.070
    foot_lift_m: float = 0.035
    frequency_hz: float = 1.40
    duty_factor: float = 0.64
    pitch_kp_m_per_rad: float = 0.035
    pitch_kd_m_s_per_rad: float = 0.008
    velocity_gain_s: float = 0.030
    foot_center_offset_m: float = 0.012
    pitch_placement_gain_m_per_rad: float = 0.040

    @property
    def desired_speed_m_s(self) -> float:
        return self.step_length_m * self.frequency_hz

    @classmethod
    def from_array(cls, values: np.ndarray) -> "WalkingControllerConfig":
        if values.shape != (len(PARAMETER_NAMES),):
            raise ValueError("walking controller parameter count is invalid")
        return cls(**dict(zip(PARAMETER_NAMES, map(float, values))))

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in PARAMETER_NAMES], dtype=float
        )


@dataclass(frozen=True)
class WalkingRollout:
    score: float
    summary: dict[str, float | bool]
    qpos: np.ndarray | None = None
    targets: np.ndarray | None = None
    diagnostics: np.ndarray | None = None


_WORKER_MODEL: mujoco.MjModel | None = None


def _initialize_worker(model_path: str) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(model_path)


def _rollout_worker(
    task: tuple[np.ndarray, float],
) -> WalkingRollout:
    if _WORKER_MODEL is None:
        raise RuntimeError("walking rollout worker is not initialized")
    parameters, duration_s = task
    return rollout_walking_controller(
        _WORKER_MODEL,
        WalkingControllerConfig.from_array(parameters),
        duration_s=duration_s,
    )


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"missing MuJoCo object: {name}")
    return int(value)


def leg_inverse_kinematics(
    outward_x_m: float,
    depth_m: float,
) -> tuple[float, float]:
    """Solve one outward-bending leg in the model's effective coordinates.

    ``outward_x_m`` is positive toward +x for the front leg and toward -x for
    the rear leg.  ``depth_m`` is the downward hip-to-foot displacement.
    """

    length = FIXED_PARAMETERS.edge_length
    radius = math.hypot(outward_x_m, depth_m)
    minimum_radius = 1.0e-6
    maximum_radius = 2.0 * length - 1.0e-6
    radius = float(np.clip(radius, minimum_radius, maximum_radius))
    direction = math.atan2(outward_x_m, depth_m)
    half_bend = math.acos(radius / (2.0 * length))
    hip = direction + half_bend
    knee = 2.0 * half_bend
    return (
        float(np.clip(hip, *FIXED_PARAMETERS.hip.safe_range)),
        float(np.clip(knee, *FIXED_PARAMETERS.knee.safe_range)),
    )


def leg_forward_kinematics(
    hip_angle: float,
    knee_angle: float,
) -> tuple[float, float]:
    """Return outward and downward foot coordinates for one virtual leg."""

    length = FIXED_PARAMETERS.edge_length
    return (
        length * math.sin(hip_angle)
        + length * math.sin(hip_angle - knee_angle),
        length * math.cos(hip_angle)
        + length * math.cos(knee_angle - hip_angle),
    )


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def foot_trajectory(
    leg_phase: float,
    config: WalkingControllerConfig,
) -> tuple[float, float, bool]:
    """Return physical fore-aft offset, downward depth, and stance state."""

    phase = leg_phase % 1.0
    if phase < config.duty_factor:
        stance_phase = phase / config.duty_factor
        fore_aft = config.step_length_m * (0.5 - stance_phase)
        return fore_aft, config.body_height_m, True

    swing_phase = (phase - config.duty_factor) / (
        1.0 - config.duty_factor
    )
    blend = _smoothstep(swing_phase)
    fore_aft = config.step_length_m * (blend - 0.5)
    lift = config.foot_lift_m * math.sin(math.pi * blend)
    return fore_aft, config.body_height_m - lift, False


def walking_targets(
    time_s: float,
    root_pitch_rad: float,
    root_pitch_rate_rad_s: float,
    root_velocity_m_s: float,
    config: WalkingControllerConfig,
    *,
    root_height_m: float | None = None,
) -> tuple[np.ndarray, tuple[bool, bool]]:
    """Compute front/rear joint targets from phase and body feedback."""

    cycle_phase = (time_s * config.frequency_hz) % 1.0
    front_x, front_depth, front_stance = foot_trajectory(
        cycle_phase, config
    )
    rear_x, rear_depth, rear_stance = foot_trajectory(
        cycle_phase + 0.5, config
    )
    front_x += config.foot_center_offset_m
    rear_x += config.foot_center_offset_m
    if root_height_m is not None:
        actual_ground_depth = root_height_m - FIXED_PARAMETERS.foot_radius
        height_error = (
            config.body_height_m
            + FIXED_PARAMETERS.foot_radius
            - root_height_m
        )
        stance_extension = float(np.clip(height_error, 0.0, 0.035))
        if front_stance:
            front_depth += stance_extension
        if rear_stance:
            rear_depth += stance_extension
        sag_compensation = min(
            actual_ground_depth - config.body_height_m,
            0.0,
        )
        if not front_stance:
            front_depth += sag_compensation
        if not rear_stance:
            rear_depth += sag_compensation

    pitch_correction = (
        config.pitch_kp_m_per_rad * root_pitch_rad
        + config.pitch_kd_m_s_per_rad * root_pitch_rate_rad_s
    )
    pitch_correction = float(np.clip(pitch_correction, -0.035, 0.035))
    if front_stance:
        front_depth += pitch_correction
    if rear_stance:
        rear_depth -= pitch_correction

    velocity_correction = config.velocity_gain_s * (
        root_velocity_m_s - config.desired_speed_m_s
    )
    velocity_correction = float(np.clip(velocity_correction, -0.035, 0.035))
    placement_correction = config.pitch_placement_gain_m_per_rad * (
        root_pitch_rad + 0.12 * root_pitch_rate_rad_s
    )
    placement_correction = float(
        np.clip(placement_correction, -0.040, 0.040)
    )
    if not front_stance:
        front_x += velocity_correction + placement_correction
    if not rear_stance:
        rear_x += velocity_correction + placement_correction

    front_hip, front_knee = leg_inverse_kinematics(front_x, front_depth)
    # The rear virtual leg's effective positive x points backward.
    rear_hip, rear_knee = leg_inverse_kinematics(-rear_x, rear_depth)
    return (
        np.asarray(
            [front_hip, front_knee, rear_hip, rear_knee], dtype=float
        ),
        (front_stance, rear_stance),
    )


def _model_setup(model: mujoco.MjModel) -> dict[str, object]:
    joint_names = ("front_hip", "front_knee", "rear_hip", "rear_knee")
    joint_qpos = np.asarray(
        [
            model.jnt_qposadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in joint_names
        ],
        dtype=int,
    )
    joint_dof = np.asarray(
        [
            model.jnt_dofadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in joint_names
        ],
        dtype=int,
    )
    front_foot_geom = _id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "front_foot_proxy",
    )
    rear_foot_geom = _id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "rear_foot_proxy",
    )
    return {
        "root_x_qpos": int(
            model.jnt_qposadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
            ]
        ),
        "root_z_qpos": int(
            model.jnt_qposadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_z")
            ]
        ),
        "root_pitch_qpos": int(
            model.jnt_qposadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")
            ]
        ),
        "root_x_dof": int(
            model.jnt_dofadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
            ]
        ),
        "root_pitch_dof": int(
            model.jnt_dofadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")
            ]
        ),
        "joint_qpos": joint_qpos,
        "joint_dof": joint_dof,
        "front_foot_site": _id(
            model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
        ),
        "rear_foot_site": _id(
            model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
        ),
        "floor_geom": _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        "front_foot_geom": front_foot_geom,
        "rear_foot_geom": rear_foot_geom,
        "foot_geoms": frozenset((front_foot_geom, rear_foot_geom)),
    }


def _contact_state(
    data: mujoco.MjData,
    *,
    floor_geom: int,
    foot_geoms: frozenset[int],
) -> tuple[frozenset[int], int, int]:
    foot_contacts: set[int] = set()
    nonfoot_ground_contacts = 0
    self_contacts = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        pair = frozenset((first, second))
        if floor_geom in pair:
            robot_geom = next(iter(pair - {floor_geom}), floor_geom)
            if robot_geom in foot_geoms:
                foot_contacts.add(robot_geom)
            else:
                nonfoot_ground_contacts += 1
        else:
            self_contacts += 1
    return frozenset(foot_contacts), nonfoot_ground_contacts, self_contacts


def rollout_walking_controller(
    model: mujoco.MjModel,
    config: WalkingControllerConfig,
    *,
    duration_s: float,
    detailed: bool = False,
) -> WalkingRollout:
    """Simulate one walking controller and return physical diagnostics."""

    setup = _model_setup(model)
    data = mujoco.MjData(model)
    root_x_qpos = int(setup["root_x_qpos"])
    root_z_qpos = int(setup["root_z_qpos"])
    root_pitch_qpos = int(setup["root_pitch_qpos"])
    root_x_dof = int(setup["root_x_dof"])
    root_pitch_dof = int(setup["root_pitch_dof"])
    joint_qpos = np.asarray(setup["joint_qpos"])

    initial_targets, _ = walking_targets(0.0, 0.0, 0.0, 0.0, config)
    data.qpos[:] = 0.0
    data.qpos[root_z_qpos] = (
        config.body_height_m + FIXED_PARAMETERS.foot_radius
    )
    data.qpos[joint_qpos] = initial_targets
    data.ctrl[:] = initial_targets
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    timestep = float(model.opt.timestep)
    steps = int(math.ceil(duration_s / timestep))
    initial_x = float(data.qpos[root_x_qpos])
    previous_x = initial_x
    forward_travel = 0.0
    backward_travel = 0.0
    minimum_root_z = float(data.qpos[root_z_qpos])
    maximum_root_z = minimum_root_z
    maximum_abs_pitch = 0.0
    foot_contact_steps = 0
    double_support_steps = 0
    airborne_steps = 0
    planned_swing_slots = 0
    swing_contact_slots = 0
    planned_stance_slots = 0
    missed_stance_slots = 0
    nonfoot_ground_steps = 0
    consecutive_nonfoot_ground_steps = 0
    consecutive_airborne_steps = 0
    self_contact_steps = 0
    saturated_steps = 0
    failed = False
    failure_nonfinite = False
    failure_root_height = False
    failure_pitch = False
    failure_nonfoot_ground = False
    failure_airborne = False

    qpos_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    diagnostic_rows: list[np.ndarray] = []

    for step in range(steps):
        pitch = float(data.qpos[root_pitch_qpos])
        pitch_rate = float(data.qvel[root_pitch_dof])
        velocity = float(data.qvel[root_x_dof])
        targets, planned_stance = walking_targets(
            float(data.time),
            pitch,
            pitch_rate,
            velocity,
            config,
            root_height_m=float(data.qpos[root_z_qpos]),
        )
        data.ctrl[:] = targets
        mujoco.mj_step(model, data)

        root_x = float(data.qpos[root_x_qpos])
        root_z = float(data.qpos[root_z_qpos])
        pitch = float(data.qpos[root_pitch_qpos])
        delta_x = root_x - previous_x
        forward_travel += max(delta_x, 0.0)
        backward_travel += max(-delta_x, 0.0)
        previous_x = root_x
        minimum_root_z = min(minimum_root_z, root_z)
        maximum_root_z = max(maximum_root_z, root_z)
        maximum_abs_pitch = max(maximum_abs_pitch, abs(pitch))

        contacting_feet, nonfoot_ground, self_contacts = _contact_state(
            data,
            floor_geom=int(setup["floor_geom"]),
            foot_geoms=setup["foot_geoms"],
        )
        foot_contacts = len(contacting_feet)
        front_contact = int(setup["front_foot_geom"]) in contacting_feet
        rear_contact = int(setup["rear_foot_geom"]) in contacting_feet
        foot_contact_steps += int(foot_contacts > 0)
        double_support_steps += int(foot_contacts == 2)
        airborne_steps += int(foot_contacts == 0)
        if foot_contacts == 0:
            consecutive_airborne_steps += 1
        else:
            consecutive_airborne_steps = 0
        for planned, actual in zip(
            planned_stance, (front_contact, rear_contact)
        ):
            if planned:
                planned_stance_slots += 1
                missed_stance_slots += int(not actual)
            else:
                planned_swing_slots += 1
                swing_contact_slots += int(actual)
        nonfoot_ground_steps += int(nonfoot_ground > 0)
        if nonfoot_ground > 0:
            consecutive_nonfoot_ground_steps += 1
        else:
            consecutive_nonfoot_ground_steps = 0
        self_contact_steps += int(self_contacts > 0)
        saturated_steps += int(np.any(np.abs(data.actuator_force) >= 5.994))

        if detailed:
            qpos_rows.append(np.asarray(data.qpos).copy())
            target_rows.append(targets.copy())
            diagnostic_rows.append(
                np.asarray(
                    [
                        data.time,
                        root_x,
                        root_z,
                        pitch,
                        data.qvel[root_x_dof],
                        data.qvel[root_pitch_dof],
                        foot_contacts,
                        float(front_contact),
                        float(rear_contact),
                        data.site_xpos[int(setup["front_foot_site"]), 2],
                        data.site_xpos[int(setup["rear_foot_site"]), 2],
                        nonfoot_ground,
                        self_contacts,
                        float(planned_stance[0]),
                        float(planned_stance[1]),
                    ],
                    dtype=float,
                )
            )

        physics_finite = np.isfinite(data.qpos).all() and np.isfinite(
            data.qvel
        ).all()
        failure_nonfinite = not physics_finite
        failure_root_height = root_z < 0.115 or root_z > 0.55
        failure_pitch = abs(pitch) > 0.80
        failure_nonfoot_ground = (
            consecutive_nonfoot_ground_steps
            > int(round(0.050 / timestep))
        )
        failure_airborne = (
            consecutive_airborne_steps > int(round(0.080 / timestep))
        )
        failed = bool(
            failure_nonfinite
            or failure_root_height
            or failure_pitch
            or failure_nonfoot_ground
            or failure_airborne
        )
        if failed:
            break

    executed_steps = step + 1
    elapsed_s = executed_steps * timestep
    survival_fraction = min(elapsed_s / duration_s, 1.0)
    displacement = float(data.qpos[root_x_qpos]) - initial_x
    tail_speed = displacement / max(elapsed_s, timestep)
    foot_contact_fraction = foot_contact_steps / executed_steps
    double_support_fraction = double_support_steps / executed_steps
    airborne_fraction = airborne_steps / executed_steps
    swing_contact_fraction = swing_contact_slots / max(planned_swing_slots, 1)
    missed_stance_fraction = missed_stance_slots / max(
        planned_stance_slots, 1
    )
    nonfoot_ground_fraction = nonfoot_ground_steps / executed_steps
    self_contact_fraction = self_contact_steps / executed_steps
    saturation_fraction = saturated_steps / executed_steps
    expected_double_support = max(2.0 * config.duty_factor - 1.0, 0.0)
    excess_double_support = max(
        double_support_fraction - expected_double_support - 0.10,
        0.0,
    )

    # Survival is dominant: falling and rolling forward cannot beat sustained
    # upright motion.  Progress then separates stable stationary and walking
    # controllers, while contact and torque terms reject fragile solutions.
    score = (
        100.0 * survival_fraction
        + 50.0 * float(not failed)
        + 10.0 * displacement
        + 1.0 * tail_speed
        - 10.0 * backward_travel
        - 6.0 * maximum_abs_pitch
        - 30.0 * nonfoot_ground_fraction
        - 8.0 * airborne_fraction
        - 12.0 * swing_contact_fraction
        - 8.0 * missed_stance_fraction
        - 10.0 * excess_double_support
        - 2.0 * self_contact_fraction
        - 1.0 * saturation_fraction
    )
    summary: dict[str, float | bool] = {
        "duration_s": elapsed_s,
        "requested_duration_s": duration_s,
        "survival_fraction": survival_fraction,
        "failed": failed,
        "failure_nonfinite": failure_nonfinite,
        "failure_root_height": failure_root_height,
        "failure_pitch": failure_pitch,
        "failure_nonfoot_ground": failure_nonfoot_ground,
        "failure_airborne": failure_airborne,
        "root_x_displacement_m": displacement,
        "mean_velocity_m_s": tail_speed,
        "forward_travel_m": forward_travel,
        "backward_travel_m": backward_travel,
        "minimum_root_z_m": minimum_root_z,
        "maximum_root_z_m": maximum_root_z,
        "maximum_abs_pitch_rad": maximum_abs_pitch,
        "foot_contact_fraction": foot_contact_fraction,
        "double_support_fraction": double_support_fraction,
        "airborne_fraction": airborne_fraction,
        "swing_contact_fraction": swing_contact_fraction,
        "missed_stance_fraction": missed_stance_fraction,
        "expected_double_support_fraction": expected_double_support,
        "excess_double_support_fraction": excess_double_support,
        "nonfoot_ground_fraction": nonfoot_ground_fraction,
        "self_contact_fraction": self_contact_fraction,
        "torque_saturation_fraction": saturation_fraction,
        "desired_speed_m_s": config.desired_speed_m_s,
    }
    return WalkingRollout(
        score=float(score),
        summary=summary,
        qpos=(np.asarray(qpos_rows) if detailed else None),
        targets=(np.asarray(target_rows) if detailed else None),
        diagnostics=(
            np.asarray(diagnostic_rows) if detailed else None
        ),
    )


def search_walking_controller(
    model: mujoco.MjModel,
    *,
    generations: int,
    population: int,
    duration_s: float,
    seed: int,
    workers: int,
    initial_config: WalkingControllerConfig | None = None,
) -> tuple[WalkingControllerConfig, list[dict[str, float]]]:
    """Search gait and feedback parameters with a compact CEM loop."""

    if generations < 1 or population < 4:
        raise ValueError("search needs at least one generation and population 4")
    rng = np.random.default_rng(seed)
    lower = PARAMETER_BOUNDS[:, 0]
    upper = PARAMETER_BOUNDS[:, 1]
    best_config = initial_config or WalkingControllerConfig()
    mean = np.clip(best_config.as_array(), lower, upper)
    best_config = WalkingControllerConfig.from_array(mean)
    std = 0.28 * (upper - lower)
    elite_count = max(2, population // 6)
    best_score = -math.inf
    history: list[dict[str, float]] = []
    stage_count = min(3, generations)
    stage_boundaries = np.linspace(0, generations, stage_count + 1).round()
    stage_boundaries = stage_boundaries.astype(int)
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(str(MODEL_PATH),),
        )
    try:
        for generation in range(generations):
            stage = int(
                np.searchsorted(stage_boundaries[1:], generation, side="right")
            )
            stage = min(stage, stage_count - 1)
            stage_duration = duration_s * float(stage + 1) / stage_count
            if generation == stage_boundaries[stage]:
                # Scores from different rollout horizons are not comparable.
                best_score = -math.inf
            samples = rng.normal(mean, std, size=(population, len(mean)))
            samples = np.clip(samples, lower, upper)
            samples[0] = best_config.as_array()
            if executor is None:
                rollouts = [
                    rollout_walking_controller(
                        model,
                        WalkingControllerConfig.from_array(sample),
                        duration_s=stage_duration,
                    )
                    for sample in samples
                ]
            else:
                rollouts = list(
                    executor.map(
                        _rollout_worker,
                        ((sample, stage_duration) for sample in samples),
                        chunksize=max(1, population // (4 * workers)),
                    )
                )
            scores = np.asarray([rollout.score for rollout in rollouts])
            order = np.argsort(scores)[::-1]
            elites = samples[order[:elite_count]]
            elite_mean = elites.mean(axis=0)
            elite_std = elites.std(axis=0)
            mean = 0.65 * mean + 0.35 * elite_mean
            std = np.maximum(
                0.70 * std + 0.30 * elite_std,
                0.015 * (upper - lower),
            )
            generation_best = rollouts[int(order[0])]
            if generation_best.score > best_score:
                best_score = generation_best.score
                best_config = WalkingControllerConfig.from_array(
                    samples[int(order[0])]
                )
            history.append(
                {
                    "generation": float(generation),
                    "stage": float(stage),
                    "rollout_duration_s": float(stage_duration),
                    "generation_best_score": float(generation_best.score),
                    "global_best_score": float(best_score),
                    "generation_best_displacement_m": float(
                        generation_best.summary["root_x_displacement_m"]
                    ),
                    "generation_best_survival_fraction": float(
                        generation_best.summary["survival_fraction"]
                    ),
                    "population_mean_score": float(scores.mean()),
                    "population_std_score": float(scores.std()),
                }
            )
            print(
                f"generation={generation:02d} "
                f"horizon={stage_duration:.1f}s "
                f"best={best_score:+.3f} "
                f"generation_dx="
                f"{generation_best.summary['root_x_displacement_m']:+.3f}m "
                f"survival="
                f"{generation_best.summary['survival_fraction']:.1%}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()
    return best_config, history


def _write_outputs(
    output_dir: Path,
    config: WalkingControllerConfig,
    rollout: WalkingRollout,
    history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "controller": asdict(config),
        "score": rollout.score,
        "summary": rollout.summary,
        "search_history": history,
        "diagnostic_columns": (
            "time_s",
            "root_x_m",
            "root_z_m",
            "root_pitch_rad",
            "root_velocity_m_s",
            "root_pitch_rate_rad_s",
            "foot_contact_count",
            "front_foot_contact",
            "rear_foot_contact",
            "front_foot_z_m",
            "rear_foot_z_m",
            "nonfoot_ground_contact_count",
            "self_contact_count",
            "front_planned_stance",
            "rear_planned_stance",
        ),
    }
    (output_dir / "walking_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    assert rollout.qpos is not None
    assert rollout.targets is not None
    assert rollout.diagnostics is not None
    np.savez_compressed(
        output_dir / "walking_rollout.npz",
        qpos=rollout.qpos,
        targets=rollout.targets,
        diagnostics=rollout.diagnostics,
        reward=np.zeros(len(rollout.qpos), dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--search-duration", type=float, default=4.0)
    parser.add_argument("--evaluation-duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-summary",
        type=Path,
        help="Seed CEM from a previous walking_summary.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.search_duration <= 0.0 or args.evaluation_duration <= 0.0:
        parser.error("durations must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    config = WalkingControllerConfig()
    if args.initial_summary is not None:
        payload = json.loads(
            args.initial_summary.read_text(encoding="utf-8")
        )
        controller_values = asdict(WalkingControllerConfig())
        controller_values.update(payload["controller"])
        config = WalkingControllerConfig(**controller_values)
    history: list[dict[str, float]] = []
    if args.search:
        config, history = search_walking_controller(
            model,
            generations=args.generations,
            population=args.population,
            duration_s=args.search_duration,
            seed=args.seed,
            workers=args.workers,
            initial_config=config,
        )
    rollout = rollout_walking_controller(
        model,
        config,
        duration_s=args.evaluation_duration,
        detailed=True,
    )
    _write_outputs(args.output_dir, config, rollout, history)
    print(json.dumps({"score": rollout.score, **rollout.summary}, indent=2))
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
