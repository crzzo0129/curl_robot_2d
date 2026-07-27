"""Run rigid or servo-held compact release baselines in the existing model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts.analyze_roll_phase import analyze_rigid_phase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"

LOCK_NAMES = (
    "lock_front_hip_compact",
    "lock_front_knee_compact",
    "lock_rear_hip_compact",
    "lock_rear_knee_compact",
)
JOINT_TARGETS = (
    ("front_hip", FIXED_PARAMETERS.compact_hip_angle),
    ("front_knee", FIXED_PARAMETERS.compact_knee_angle),
    ("rear_hip", FIXED_PARAMETERS.compact_hip_angle),
    ("rear_knee", FIXED_PARAMETERS.compact_knee_angle),
)


@dataclass(frozen=True)
class ReleaseResult:
    columns: tuple[str, ...]
    rows: np.ndarray
    summary: dict[str, float | int | str | bool]

    def column(self, name: str) -> np.ndarray:
        return self.rows[:, self.columns.index(name)]


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Missing {object_type.name}: {name}")
    return object_id


def _qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def _kinetic_energy(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass_matrix)
    return float(0.5 * data.qvel @ mass_matrix @ data.qvel)


def _contact_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_geom_id: int,
) -> tuple[int, float, float, float, float, float]:
    normal_force_sum = 0.0
    normal_force_max = 0.0
    tangential_force_sum = 0.0
    weighted_slip_sum = 0.0
    slip_max = 0.0
    ground_contacts = 0

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = (int(contact.geom[0]), int(contact.geom[1]))
        if floor_geom_id not in (geom1, geom2):
            continue
        moving_geom_id = geom2 if geom1 == floor_geom_id else geom1
        moving_body_id = int(model.geom_bodyid[moving_geom_id])

        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, force)
        normal_force = abs(float(force[0]))
        tangential_force = float(np.linalg.norm(force[1:3]))

        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))
        mujoco.mj_jac(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            np.asarray(contact.pos),
            moving_body_id,
        )
        contact_velocity = jacobian_position @ data.qvel
        slip_speed = abs(float(contact_velocity[0]))

        ground_contacts += 1
        normal_force_sum += normal_force
        normal_force_max = max(normal_force_max, normal_force)
        tangential_force_sum += tangential_force
        weighted_slip_sum += normal_force * slip_speed
        slip_max = max(slip_max, slip_speed)

    weighted_slip = (
        weighted_slip_sum / normal_force_sum if normal_force_sum > 0.0 else 0.0
    )
    return (
        ground_contacts,
        normal_force_sum,
        normal_force_max,
        tangential_force_sum,
        weighted_slip,
        slip_max,
    )


def _longest_true_duration(mask: np.ndarray, timestep: float) -> float:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * timestep


def _shell_roundness(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    shell_geom_ids: list[int],
) -> tuple[float, float]:
    """Fit one circle to all shell-capsule endpoints in the side plane."""

    points = []
    for geom_id in shell_geom_ids:
        rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        axis = rotation[:, 2]
        center = np.asarray(data.geom_xpos[geom_id])
        half_length = float(model.geom_size[geom_id, 1])
        points.extend(
            (
                (center - half_length * axis)[[0, 2]],
                (center + half_length * axis)[[0, 2]],
            )
        )
    point_array = np.asarray(points)
    matrix = np.column_stack(
        (2.0 * point_array[:, 0], 2.0 * point_array[:, 1], np.ones(len(points)))
    )
    right_hand_side = np.sum(point_array**2, axis=1)
    solution, *_ = np.linalg.lstsq(matrix, right_hand_side, rcond=None)
    fitted_center = solution[:2]
    radii = np.linalg.norm(point_array - fitted_center, axis=1)
    residuals = radii - radii.mean()
    return float(np.sqrt(np.mean(residuals**2))), float(np.ptp(radii))


def run_release(
    model: mujoco.MjModel,
    *,
    joint_mode: str,
    duration: float = 5.0,
) -> ReleaseResult:
    if joint_mode not in ("rigid", "servo"):
        raise ValueError("joint_mode must be 'rigid' or 'servo'")
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    data = mujoco.MjData(model)
    compact_key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
    torso_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    floor_geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    front_foot_site_id = _id(
        model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
    )
    rear_foot_site_id = _id(
        model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
    )
    shell_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if "_shell_"
        in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
    ]
    root_pitch_qpos_address = _qpos_address(model, "root_pitch")
    root_pitch_joint_id = _id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch"
    )
    root_pitch_dof_address = int(model.jnt_dofadr[root_pitch_joint_id])
    joint_info = []
    for joint_name, target in JOINT_TARGETS:
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_servo"
        )
        force_limit = float(np.max(np.abs(model.actuator_forcerange[actuator_id])))
        joint_info.append(
            (
                joint_name,
                target,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
                actuator_id,
                force_limit,
            )
        )

    phase_reference = analyze_rigid_phase(model, samples=361).summary
    ideal_turning_phase_deg = float(
        phase_reference["ideal_rest_forward_turning_phase_deg"]
    )
    potential_peak_to_peak = float(
        phase_reference["potential_peak_to_peak_J"]
    )

    mujoco.mj_resetDataKeyframe(model, data, compact_key_id)
    data.qvel[:] = 0.0
    root_dof_addresses = [
        int(
            model.jnt_dofadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]
        )
        for joint_name in ("root_x", "root_z", "root_pitch")
    ]
    model.dof_damping[root_dof_addresses] = 0.0

    if joint_mode == "rigid":
        data.ctrl[:] = 0.0
        for lock_name in LOCK_NAMES:
            equality_id = _id(model, mujoco.mjtObj.mjOBJ_EQUALITY, lock_name)
            data.eq_active[equality_id] = 1
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        model.dof_damping[:] = 0.0
    else:
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    mujoco.mj_forward(model, data)

    timestep = float(model.opt.timestep)
    step_count = int(math.ceil(duration / timestep))
    total_mass = float(model.body_mass.sum())
    gravity = float(np.linalg.norm(model.opt.gravity))
    apothem = FIXED_PARAMETERS.regular_pentagon_apothem
    contact_radius = FIXED_PARAMETERS.shell_contact_radius

    columns = (
        "time_s",
        "phase_rad",
        "phase_deg",
        "pitch_rate_rad_s",
        "root_x_m",
        "root_z_m",
        "circle_center_x_m",
        "circle_center_z_m",
        "com_x_m",
        "com_z_m",
        "potential_energy_J",
        "kinetic_energy_J",
        "total_energy_J",
        "energy_change_J",
        "ground_contact_count",
        "normal_force_sum_N",
        "normal_force_max_N",
        "tangential_force_sum_N",
        "contact_slip_weighted_m_s",
        "contact_slip_max_m_s",
        "nominal_circle_slip_m_s",
        "joint_error_max_rad",
        "front_hip_error_rad",
        "front_knee_error_rad",
        "rear_hip_error_rad",
        "rear_knee_error_rad",
        "front_hip_torque_Nm",
        "front_knee_torque_Nm",
        "rear_hip_torque_Nm",
        "rear_knee_torque_Nm",
        "actuator_power_W",
        "saturated_actuator_count",
        "joint_damping_power_W",
        "foot_gap_m",
        "roundness_rms_m",
        "roundness_peak_to_peak_m",
        "airborne",
    )
    rows: list[list[float]] = []
    initial_total_energy: float | None = None

    def record() -> None:
        nonlocal initial_total_energy

        torso_rotation = np.asarray(data.xmat[torso_id]).reshape(3, 3)
        circle_center = np.asarray(data.xpos[torso_id]) + torso_rotation @ np.array(
            [0.0, 0.0, -apothem]
        )
        com = np.asarray(data.subtree_com[torso_id])
        potential = total_mass * gravity * float(com[2])
        kinetic = _kinetic_energy(model, data)
        total_energy = potential + kinetic
        if initial_total_energy is None:
            initial_total_energy = total_energy

        (
            contact_count,
            normal_force_sum,
            normal_force_max,
            tangential_force_sum,
            contact_slip_weighted,
            contact_slip_max,
        ) = _contact_metrics(model, data, floor_geom_id)

        phase = float(data.qpos[root_pitch_qpos_address])
        pitch_rate = float(data.qvel[root_pitch_dof_address])
        center_velocity_x = float(data.qvel[0]) - (
            apothem * pitch_rate * math.cos(phase)
        )
        nominal_circle_slip = center_velocity_x - contact_radius * pitch_rate
        joint_errors = [
            float(data.qpos[qpos_address]) - target
            for _, target, qpos_address, _, _, _ in joint_info
        ]
        actuator_forces = [
            float(data.actuator_force[actuator_id])
            for _, _, _, _, actuator_id, _ in joint_info
        ]
        actuator_power = sum(
            force * float(data.qvel[dof_address])
            for force, (_, _, _, dof_address, _, _) in zip(
                actuator_forces, joint_info
            )
        )
        saturated_count = sum(
            abs(force) >= 0.999 * force_limit
            for force, (_, _, _, _, _, force_limit) in zip(
                actuator_forces, joint_info
            )
        )
        joint_damping_power = -sum(
            float(model.dof_damping[dof_address])
            * float(data.qvel[dof_address]) ** 2
            for _, _, _, dof_address, _, _ in joint_info
        )
        foot_gap = float(
            np.linalg.norm(
                np.asarray(data.site_xpos[front_foot_site_id])[[0, 2]]
                - np.asarray(data.site_xpos[rear_foot_site_id])[[0, 2]]
            )
        )
        roundness_rms, roundness_peak_to_peak = _shell_roundness(
            model, data, shell_geom_ids
        )

        rows.append(
            [
                float(data.time),
                phase,
                math.degrees(phase),
                pitch_rate,
                float(data.qpos[0]),
                float(data.qpos[1]),
                float(circle_center[0]),
                float(circle_center[2]),
                float(com[0]),
                float(com[2]),
                potential,
                kinetic,
                total_energy,
                total_energy - initial_total_energy,
                float(contact_count),
                normal_force_sum,
                normal_force_max,
                tangential_force_sum,
                contact_slip_weighted,
                contact_slip_max,
                nominal_circle_slip,
                max(abs(error) for error in joint_errors),
                *joint_errors,
                *actuator_forces,
                actuator_power,
                float(saturated_count),
                joint_damping_power,
                foot_gap,
                roundness_rms,
                roundness_peak_to_peak,
                float(contact_count == 0),
            ]
        )

    record()
    for _ in range(step_count):
        mujoco.mj_step(model, data)
        record()

    row_array = np.asarray(rows, dtype=float)
    phase_values = row_array[:, columns.index("phase_rad")]
    maximum_phase_index = int(np.argmax(phase_values))
    actual_turning_phase_deg = float(
        row_array[maximum_phase_index, columns.index("phase_deg")]
    )
    total_energy_values = row_array[:, columns.index("total_energy_J")]
    energy_at_turning = float(total_energy_values[maximum_phase_index])
    initial_energy = float(total_energy_values[0])
    airborne = row_array[:, columns.index("airborne")].astype(bool)
    contact_mask = ~airborne
    weighted_slip_values = row_array[
        :, columns.index("contact_slip_weighted_m_s")
    ]
    actuator_power_values = row_array[:, columns.index("actuator_power_W")]
    damping_power_values = row_array[:, columns.index("joint_damping_power_W")]
    actuator_positive_work = float(
        np.maximum(actuator_power_values[:-1], 0.0).sum() * timestep
    )
    actuator_absorbed_work = float(
        -np.minimum(actuator_power_values[:-1], 0.0).sum() * timestep
    )
    actuator_net_work = float(actuator_power_values[:-1].sum() * timestep)
    passive_damping_loss = float(-damping_power_values[:-1].sum() * timestep)
    actuator_net_work_to_max = float(
        actuator_power_values[:maximum_phase_index].sum() * timestep
    )
    passive_damping_loss_to_max = float(
        -damping_power_values[:maximum_phase_index].sum() * timestep
    )
    energy_change_to_max = energy_at_turning - initial_energy
    residual_loss_to_max = (
        actuator_net_work_to_max
        - passive_damping_loss_to_max
        - energy_change_to_max
    )

    summary: dict[str, float | int | str | bool] = {
        "model": str(MODEL_PATH),
        "duration_s": float(row_array[-1, 0]),
        "timestep_s": timestep,
        "steps": step_count,
        "joint_mode": joint_mode,
        "actuation_disabled": joint_mode == "rigid",
        "root_dof_damping_disabled": True,
        "internal_dof_damping_disabled": joint_mode == "rigid",
        "shell_segments_per_edge": FIXED_PARAMETERS.shell_segments_per_edge,
        "nominal_ground_friction": FIXED_PARAMETERS.nominal_ground_friction,
        "ideal_lossless_turning_phase_deg": ideal_turning_phase_deg,
        "actual_max_phase_deg": actual_turning_phase_deg,
        "actual_max_phase_time_s": float(row_array[maximum_phase_index, 0]),
        "turning_phase_shortfall_deg": (
            ideal_turning_phase_deg - actual_turning_phase_deg
        ),
        "initial_total_energy_J": initial_energy,
        "total_energy_at_max_phase_J": energy_at_turning,
        "mechanical_energy_change_to_max_phase_J": energy_change_to_max,
        "actuator_net_work_to_max_phase_J": actuator_net_work_to_max,
        "passive_damping_loss_to_max_phase_J": passive_damping_loss_to_max,
        "residual_contact_constraint_loss_to_max_phase_J": residual_loss_to_max,
        "energy_loss_to_max_phase_J": initial_energy - energy_at_turning,
        "energy_loss_fraction_of_potential_range": (
            (initial_energy - energy_at_turning) / potential_peak_to_peak
        ),
        "final_total_energy_J": float(total_energy_values[-1]),
        "final_energy_change_J": float(total_energy_values[-1] - initial_energy),
        "actuator_positive_work_J": actuator_positive_work,
        "actuator_absorbed_work_J": actuator_absorbed_work,
        "actuator_net_work_J": actuator_net_work,
        "passive_joint_damping_loss_J": passive_damping_loss,
        "max_pitch_rate_rad_s": float(
            row_array[:, columns.index("pitch_rate_rad_s")].max()
        ),
        "min_pitch_rate_rad_s": float(
            row_array[:, columns.index("pitch_rate_rad_s")].min()
        ),
        "airborne_time_s": float(airborne.sum() * timestep),
        "airborne_fraction": float(airborne.mean()),
        "longest_airborne_time_s": _longest_true_duration(airborne, timestep),
        "max_ground_contact_count": int(
            row_array[:, columns.index("ground_contact_count")].max()
        ),
        "max_normal_force_sum_N": float(
            row_array[:, columns.index("normal_force_sum_N")].max()
        ),
        "max_normal_force_sum_over_weight": float(
            row_array[:, columns.index("normal_force_sum_N")].max()
            / (total_mass * gravity)
        ),
        "max_single_contact_normal_force_N": float(
            row_array[:, columns.index("normal_force_max_N")].max()
        ),
        "max_contact_slip_m_s": float(
            row_array[:, columns.index("contact_slip_max_m_s")].max()
        ),
        "mean_force_weighted_slip_while_contact_m_s": float(
            weighted_slip_values[contact_mask].mean()
            if contact_mask.any()
            else 0.0
        ),
        "max_nominal_circle_slip_abs_m_s": float(
            np.abs(row_array[:, columns.index("nominal_circle_slip_m_s")]).max()
        ),
        "max_joint_error_rad": float(
            row_array[:, columns.index("joint_error_max_rad")].max()
        ),
        "max_joint_error_deg": math.degrees(
            float(row_array[:, columns.index("joint_error_max_rad")].max())
        ),
        "max_foot_gap_m": float(
            row_array[:, columns.index("foot_gap_m")].max()
        ),
        "max_roundness_rms_m": float(
            row_array[:, columns.index("roundness_rms_m")].max()
        ),
        "max_roundness_peak_to_peak_m": float(
            row_array[:, columns.index("roundness_peak_to_peak_m")].max()
        ),
        "max_saturated_actuator_count": int(
            row_array[:, columns.index("saturated_actuator_count")].max()
        ),
        "any_actuator_saturation_fraction": float(
            np.mean(
                row_array[:, columns.index("saturated_actuator_count")] > 0.0
            )
        ),
    }
    for joint_name, _, _, _, _, force_limit in joint_info:
        error_values = np.abs(row_array[:, columns.index(f"{joint_name}_error_rad")])
        torque_values = np.abs(row_array[:, columns.index(f"{joint_name}_torque_Nm")])
        summary[f"{joint_name}_max_error_deg"] = math.degrees(
            float(error_values.max())
        )
        summary[f"{joint_name}_max_torque_Nm"] = float(torque_values.max())
        summary[f"{joint_name}_saturation_fraction"] = float(
            np.mean(torque_values >= 0.999 * force_limit)
        )
    return ReleaseResult(columns=columns, rows=row_array, summary=summary)


def _write_csv(path: Path, result: ReleaseResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(result.columns)
        writer.writerows(result.rows)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_values: np.ndarray,
    series: tuple[tuple[str, np.ndarray, tuple[int, int, int]], ...],
    title: str,
    *,
    zero_line: bool = False,
) -> None:
    left, top, right, bottom = bounds
    plot_left = left + 74
    plot_top = top + 38
    plot_right = right - 22
    plot_bottom = bottom - 38
    draw.text((left + 8, top + 8), title, fill=(232, 237, 244))

    all_values = np.concatenate([values for _, values, _ in series])
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    if zero_line:
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5
    y_padding = 0.08 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding
    x_min = float(x_values[0])
    x_max = float(x_values[-1])

    def map_x(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (
            plot_right - plot_left
        )

    def map_y(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (
            plot_bottom - plot_top
        )

    for fraction in np.linspace(0.0, 1.0, 5):
        x_value = x_min + fraction * (x_max - x_min)
        x = map_x(x_value)
        draw.line((x, plot_top, x, plot_bottom), fill=(55, 66, 80), width=1)
        draw.text(
            (x - 9, plot_bottom + 8),
            f"{x_value:.1f}",
            fill=(165, 175, 188),
        )
        y_value = y_min + fraction * (y_max - y_min)
        y = map_y(y_value)
        draw.line((plot_left, y, plot_right, y), fill=(55, 66, 80), width=1)
        draw.text(
            (left + 4, y - 6),
            f"{y_value:.3g}",
            fill=(165, 175, 188),
        )
    if zero_line and y_min < 0.0 < y_max:
        y = map_y(0.0)
        draw.line((plot_left, y, plot_right, y), fill=(190, 195, 202), width=2)

    legend_x = plot_left + 6
    for label, values, color in series:
        points = [
            (map_x(float(x_value)), map_y(float(y_value)))
            for x_value, y_value in zip(x_values, values)
        ]
        draw.line(points, fill=color, width=2, joint="curve")
        draw.line(
            (legend_x, plot_top + 7, legend_x + 18, plot_top + 7),
            fill=color,
            width=3,
        )
        draw.text(
            (legend_x + 23, plot_top + 1),
            label,
            fill=(202, 210, 220),
        )
        legend_x += 115

    draw.rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        outline=(105, 116, 130),
        width=1,
    )
    draw.text(
        ((plot_left + plot_right) / 2 - 24, plot_bottom + 23),
        "time (s)",
        fill=(180, 190, 202),
    )


def _write_plot(path: Path, result: ReleaseResult) -> None:
    image = Image.new("RGB", (1400, 1040), color=(24, 31, 41))
    draw = ImageDraw.Draw(image)
    draw.text(
        (28, 16),
        f"{str(result.summary['joint_mode']).title()} compact zero-speed release baseline",
        fill=(244, 247, 251),
    )
    draw.text(
        (28, 38),
        (
            f"max phase={result.summary['actual_max_phase_deg']:.2f} deg, "
            f"shortfall={result.summary['turning_phase_shortfall_deg']:.2f} deg, "
            "contact/constraint residual="
            f"{result.summary['residual_contact_constraint_loss_to_max_phase_J']:.3f} J"
        ),
        fill=(180, 191, 205),
    )

    time = result.column("time_s")
    initial_energy = result.column("total_energy_J")[0]
    panels = (
        (
            (18, 72, 690, 386),
            (("phase", result.column("phase_deg"), (70, 180, 255)),),
            "Root roll phase (deg)",
            False,
        ),
        (
            (710, 72, 1382, 386),
            (
                (
                    "pitch rate",
                    result.column("pitch_rate_rad_s"),
                    (255, 164, 74),
                ),
            ),
            "Angular velocity (rad/s)",
            True,
        ),
        (
            (18, 394, 690, 708),
            (
                (
                    "potential-U0",
                    result.column("potential_energy_J")
                    - result.column("potential_energy_J")[0],
                    (92, 214, 143),
                ),
                (
                    "kinetic",
                    result.column("kinetic_energy_J"),
                    (70, 180, 255),
                ),
                (
                    "total-E0",
                    result.column("total_energy_J") - initial_energy,
                    (255, 164, 74),
                ),
            ),
            "Mechanical energy (J)",
            True,
        ),
        (
            (710, 394, 1382, 708),
            (
                (
                    "normal sum",
                    result.column("normal_force_sum_N"),
                    (208, 124, 255),
                ),
                (
                    "normal max",
                    result.column("normal_force_max_N"),
                    (255, 164, 74),
                ),
            ),
            "Ground normal force (N)",
            False,
        ),
        (
            (18, 716, 690, 1030),
            (
                (
                    "contact slip",
                    result.column("contact_slip_weighted_m_s"),
                    (255, 164, 74),
                ),
                (
                    "circle slip",
                    np.abs(result.column("nominal_circle_slip_m_s")),
                    (70, 180, 255),
                ),
            ),
            "Slip speed (m/s)",
            False,
        ),
        (
            (710, 716, 1382, 1030),
            (
                (
                    "joint error",
                    np.degrees(result.column("joint_error_max_rad")),
                    (92, 214, 143),
                ),
            ),
            "Maximum compact joint error (deg)",
            False,
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


def write_outputs(output_dir: Path, result: ReleaseResult) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{result.summary['joint_mode']}_release"
    csv_path = output_dir / f"{prefix}.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    plot_path = output_dir / f"{prefix}.png"
    _write_csv(csv_path, result)
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_plot(plot_path, result)
    return csv_path, summary_path, plot_path


def write_comparison(
    output_dir: Path,
    rigid: ReleaseResult,
    servo: ReleaseResult,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "release_comparison_summary.json"
    plot_path = output_dir / "release_comparison.png"

    comparison = {
        "rigid_actual_max_phase_deg": rigid.summary["actual_max_phase_deg"],
        "servo_actual_max_phase_deg": servo.summary["actual_max_phase_deg"],
        "servo_minus_rigid_max_phase_deg": (
            float(servo.summary["actual_max_phase_deg"])
            - float(rigid.summary["actual_max_phase_deg"])
        ),
        "rigid_residual_loss_to_max_phase_J": rigid.summary[
            "residual_contact_constraint_loss_to_max_phase_J"
        ],
        "servo_residual_loss_to_max_phase_J": servo.summary[
            "residual_contact_constraint_loss_to_max_phase_J"
        ],
        "servo_actuator_net_work_to_max_phase_J": servo.summary[
            "actuator_net_work_to_max_phase_J"
        ],
        "servo_passive_damping_loss_to_max_phase_J": servo.summary[
            "passive_damping_loss_to_max_phase_J"
        ],
        "servo_max_joint_error_deg": servo.summary["max_joint_error_deg"],
        "servo_max_foot_gap_m": servo.summary["max_foot_gap_m"],
        "servo_max_roundness_peak_to_peak_m": servo.summary[
            "max_roundness_peak_to_peak_m"
        ],
        "servo_max_saturated_actuator_count": servo.summary[
            "max_saturated_actuator_count"
        ],
        "servo_any_actuator_saturation_fraction": servo.summary[
            "any_actuator_saturation_fraction"
        ],
        "servo_max_joint_torque_Nm": max(
            float(servo.summary[f"{joint_name}_max_torque_Nm"])
            for joint_name, _ in JOINT_TARGETS
        ),
    }
    summary_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    image = Image.new("RGB", (1400, 1040), color=(24, 31, 41))
    draw = ImageDraw.Draw(image)
    draw.text(
        (28, 16),
        "Rigid vs servo compact zero-speed release",
        fill=(244, 247, 251),
    )
    draw.text(
        (28, 38),
        (
            f"phase: {rigid.summary['actual_max_phase_deg']:.2f} vs "
            f"{servo.summary['actual_max_phase_deg']:.2f} deg; "
            f"servo joint error={servo.summary['max_joint_error_deg']:.2f} deg; "
            f"foot gap={1000.0 * float(servo.summary['max_foot_gap_m']):.1f} mm"
        ),
        fill=(180, 191, 205),
    )
    time = rigid.column("time_s")
    servo_torques = np.max(
        np.abs(
            np.column_stack(
                [
                    servo.column(f"{joint_name}_torque_Nm")
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
                ("rigid", rigid.column("phase_deg"), (70, 180, 255)),
                ("servo", servo.column("phase_deg"), (255, 164, 74)),
            ),
            "Root roll phase (deg)",
            False,
        ),
        (
            (710, 72, 1382, 386),
            (
                (
                    "rigid",
                    rigid.column("total_energy_J")
                    - rigid.column("total_energy_J")[0],
                    (70, 180, 255),
                ),
                (
                    "servo",
                    servo.column("total_energy_J")
                    - servo.column("total_energy_J")[0],
                    (255, 164, 74),
                ),
            ),
            "Mechanical energy change (J)",
            True,
        ),
        (
            (18, 394, 690, 708),
            (
                (
                    "rigid",
                    np.degrees(rigid.column("joint_error_max_rad")),
                    (70, 180, 255),
                ),
                (
                    "servo",
                    np.degrees(servo.column("joint_error_max_rad")),
                    (255, 164, 74),
                ),
            ),
            "Maximum compact joint error (deg)",
            False,
        ),
        (
            (710, 394, 1382, 708),
            (
                (
                    "rigid",
                    1000.0 * rigid.column("foot_gap_m"),
                    (70, 180, 255),
                ),
                (
                    "servo",
                    1000.0 * servo.column("foot_gap_m"),
                    (255, 164, 74),
                ),
            ),
            "Front/rear foot gap (mm)",
            False,
        ),
        (
            (18, 716, 690, 1030),
            (
                (
                    "rigid",
                    1000.0 * rigid.column("roundness_peak_to_peak_m"),
                    (70, 180, 255),
                ),
                (
                    "servo",
                    1000.0 * servo.column("roundness_peak_to_peak_m"),
                    (255, 164, 74),
                ),
            ),
            "Shell radial peak-to-peak spread (mm)",
            False,
        ),
        (
            (710, 716, 1382, 1030),
            (
                ("max torque", servo_torques, (208, 124, 255)),
                (
                    "6 Nm limit",
                    np.full_like(servo_torques, 6.0),
                    (255, 164, 74),
                ),
            ),
            "Maximum absolute servo torque (N m)",
            False,
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
    image.save(plot_path)
    return summary_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-mode", choices=("rigid", "servo", "both"), default="both"
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    modes = ("rigid", "servo") if args.joint_mode == "both" else (args.joint_mode,)
    results = {}
    all_outputs: list[Path] = []
    for mode in modes:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        result = run_release(model, joint_mode=mode, duration=args.duration)
        results[mode] = result
        output_dir = (
            args.output_dir / f"{mode}_release"
            if args.output_dir is not None and args.joint_mode == "both"
            else args.output_dir
            if args.output_dir is not None
            else DEFAULT_OUTPUT_ROOT / f"{mode}_release"
        )
        all_outputs.extend(write_outputs(output_dir, result))

        print(f"model={MODEL_PATH}")
        for key, value in result.summary.items():
            if key != "model":
                print(f"{mode}.{key}={value}")

    if args.joint_mode == "both":
        comparison_dir = (
            args.output_dir / "release_comparison"
            if args.output_dir is not None
            else DEFAULT_OUTPUT_ROOT / "release_comparison"
        )
        all_outputs.extend(
            write_comparison(
                comparison_dir, results["rigid"], results["servo"]
            )
        )
    for output in all_outputs:
        print(f"output={output}")


if __name__ == "__main__":
    main()
