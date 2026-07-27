"""Rigid kinematic roll-phase analysis using the existing MuJoCo model.

The four internal joints stay at the ``compact`` keyframe values.  For each
root pitch, the nominal shell circle is placed tangent to the floor with
no-slip center motion.  The script calls ``mj_forward`` only: it does not step
the dynamics and therefore does not introduce servo compliance.
"""

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

from curl_robot_2d.parameters import FIXED_PARAMETERS, FixedParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "rigid_phase"


@dataclass(frozen=True)
class PhaseAnalysis:
    columns: tuple[str, ...]
    rows: np.ndarray
    summary: dict[str, float | int | str]

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


def _planar_inertia_about_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point_xz: np.ndarray,
) -> float:
    """Return physical Iyy of all robot bodies about a world x-z point."""

    inertia = 0.0
    for body_id in range(1, model.nbody):
        mass = float(model.body_mass[body_id])
        inertia_rotation = np.asarray(data.ximat[body_id]).reshape(3, 3)
        inertia_world = (
            inertia_rotation
            @ np.diag(np.asarray(model.body_inertia[body_id]))
            @ inertia_rotation.T
        )
        body_com_xz = np.asarray(data.xipos[body_id])[[0, 2]]
        distance_squared = float(np.sum((body_com_xz - point_xz) ** 2))
        inertia += float(inertia_world[1, 1]) + mass * distance_squared
    return inertia


def analyze_rigid_phase(
    model: mujoco.MjModel,
    *,
    samples: int = 361,
    parameters: FixedParameters = FIXED_PARAMETERS,
) -> PhaseAnalysis:
    if samples < 2:
        raise ValueError("samples must be at least 2")

    data = mujoco.MjData(model)
    compact_key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
    torso_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    root_x_address = _qpos_address(model, "root_x")
    root_z_address = _qpos_address(model, "root_z")
    root_pitch_address = _qpos_address(model, "root_pitch")

    mujoco.mj_resetDataKeyframe(model, data, compact_key_id)
    compact_qpos = data.qpos.copy()
    internal_qpos = compact_qpos[3:].copy()

    contact_radius = parameters.shell_contact_radius
    apothem = parameters.regular_pentagon_apothem
    gravity = float(np.linalg.norm(model.opt.gravity))
    total_mass = float(model.body_mass.sum())

    columns = (
        "phase_rad",
        "phase_deg",
        "root_x_m",
        "root_z_m",
        "circle_center_x_m",
        "circle_center_z_m",
        "contact_x_m",
        "contact_z_m",
        "com_x_m",
        "com_z_m",
        "com_offset_x_m",
        "com_offset_z_m",
        "com_offset_radius_m",
        "potential_energy_J",
        "potential_from_min_J",
        "gravity_torque_Nm",
        "inertia_about_contact_kg_m2",
    )
    rows: list[list[float]] = []

    for phase in np.linspace(0.0, 2.0 * math.pi, samples):
        # Positive rotation about +y rolls the nominal circle toward +x.
        circle_center_x = contact_radius * phase
        root_x = circle_center_x + apothem * math.sin(phase)
        root_z = contact_radius + apothem * math.cos(phase)

        data.qpos[:] = compact_qpos
        data.qpos[root_x_address] = root_x
        data.qpos[root_z_address] = root_z
        data.qpos[root_pitch_address] = phase
        mujoco.mj_forward(model, data)

        if not np.array_equal(data.qpos[3:], internal_qpos):
            raise RuntimeError("Internal compact joint coordinates changed")

        torso_rotation = np.asarray(data.xmat[torso_id]).reshape(3, 3)
        measured_center = np.asarray(data.xpos[torso_id]) + torso_rotation @ np.array(
            [0.0, 0.0, -apothem]
        )
        contact = np.array([measured_center[0], 0.0, 0.0])
        com = np.asarray(data.subtree_com[torso_id]).copy()
        com_offset = com - measured_center
        potential = total_mass * gravity * float(com[2])
        gravity_torque = total_mass * gravity * float(com[0] - contact[0])
        contact_inertia = _planar_inertia_about_point(
            model, data, contact[[0, 2]]
        )

        rows.append(
            [
                phase,
                math.degrees(phase),
                root_x,
                root_z,
                float(measured_center[0]),
                float(measured_center[2]),
                float(contact[0]),
                float(contact[2]),
                float(com[0]),
                float(com[2]),
                float(com_offset[0]),
                float(com_offset[2]),
                float(np.linalg.norm(com_offset[[0, 2]])),
                potential,
                0.0,
                gravity_torque,
                contact_inertia,
            ]
        )

    row_array = np.asarray(rows, dtype=float)
    eccentricity = float(row_array[0, columns.index("com_offset_radius_m")])
    initial_offset_x = float(row_array[0, columns.index("com_offset_x_m")])
    initial_offset_z = float(row_array[0, columns.index("com_offset_z_m")])
    offset_phase = math.atan2(initial_offset_x, initial_offset_z)
    potential_index = columns.index("potential_energy_J")
    relative_potential_index = columns.index("potential_from_min_J")
    potential_min = total_mass * gravity * (contact_radius - eccentricity)
    row_array[:, relative_potential_index] = (
        row_array[:, potential_index] - potential_min
    )

    # These extrema are analytical properties of the rigid eccentric COM,
    # independent of whether the requested phase grid lands exactly on them.
    com_height_min = contact_radius - eccentricity
    com_height_max = contact_radius + eccentricity
    com_height_max_phase = math.degrees((-offset_phase) % (2.0 * math.pi))
    com_height_min_phase = math.degrees(
        (math.pi - offset_phase) % (2.0 * math.pi)
    )
    torque_amplitude = total_mass * gravity * eccentricity
    torque_max = torque_amplitude
    torque_min = -torque_amplitude
    torque_max_phase = math.degrees(
        (math.pi / 2.0 - offset_phase) % (2.0 * math.pi)
    )
    torque_min_phase = math.degrees(
        (3.0 * math.pi / 2.0 - offset_phase) % (2.0 * math.pi)
    )
    compact_potential_above_min = total_mass * gravity * (
        initial_offset_z + eccentricity
    )
    energy_from_compact_to_forward_peak = total_mass * gravity * (
        eccentricity - initial_offset_z
    )
    ideal_rest_forward_turning_phase = math.degrees(
        (2.0 * math.pi - 2.0 * offset_phase) % (2.0 * math.pi)
    )
    center_z_error = float(
        np.max(
            np.abs(
                row_array[:, columns.index("circle_center_z_m")] - contact_radius
            )
        )
    )
    eccentricity_error = float(
        np.ptp(row_array[:, columns.index("com_offset_radius_m")])
    )

    summary: dict[str, float | int | str] = {
        "model": str(MODEL_PATH),
        "samples": samples,
        "total_mass_kg": total_mass,
        "gravity_m_s2": gravity,
        "shell_contact_radius_m": contact_radius,
        "compact_com_offset_x_m": initial_offset_x,
        "compact_com_offset_z_m": initial_offset_z,
        "com_eccentricity_m": eccentricity,
        "com_eccentricity_over_contact_radius": eccentricity / contact_radius,
        "com_height_min_m": com_height_min,
        "com_height_min_phase_deg": com_height_min_phase,
        "com_height_max_m": com_height_max,
        "com_height_max_phase_deg": com_height_max_phase,
        "potential_peak_to_peak_J": 2.0 * total_mass * gravity * eccentricity,
        "compact_potential_above_min_J": compact_potential_above_min,
        "energy_from_compact_to_forward_peak_J": (
            energy_from_compact_to_forward_peak
        ),
        "ideal_rest_forward_turning_phase_deg": ideal_rest_forward_turning_phase,
        "compact_gravity_torque_Nm": total_mass * gravity * initial_offset_x,
        "gravity_torque_min_Nm": torque_min,
        "gravity_torque_min_phase_deg": torque_min_phase,
        "gravity_torque_max_Nm": torque_max,
        "gravity_torque_max_phase_deg": torque_max_phase,
        "inertia_about_circle_center_kg_m2": _planar_inertia_about_point(
            model,
            data,
            np.asarray(data.subtree_com[torso_id])[[0, 2]]
            - row_array[-1, [10, 11]],
        ),
        "inertia_about_contact_min_kg_m2": float(
            row_array[:, columns.index("inertia_about_contact_kg_m2")].min()
        ),
        "inertia_about_contact_max_kg_m2": float(
            row_array[:, columns.index("inertia_about_contact_kg_m2")].max()
        ),
        "circle_center_z_max_error_m": center_z_error,
        "com_eccentricity_peak_to_peak_error_m": eccentricity_error,
    }
    return PhaseAnalysis(columns=columns, rows=row_array, summary=summary)


def _write_csv(path: Path, analysis: PhaseAnalysis) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(analysis.columns)
        writer.writerows(analysis.rows)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    color: tuple[int, int, int],
    *,
    zero_line: bool = False,
) -> None:
    left, top, right, bottom = bounds
    plot_left = left + 75
    plot_top = top + 34
    plot_right = right - 24
    plot_bottom = bottom - 42
    draw.text((left + 8, top + 8), title, fill=(230, 235, 242))

    y_min = float(y_values.min())
    y_max = float(y_values.max())
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5
    padding = 0.08 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    def map_x(value: float) -> float:
        return plot_left + (value - x_values[0]) / (
            x_values[-1] - x_values[0]
        ) * (plot_right - plot_left)

    def map_y(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (
            plot_bottom - plot_top
        )

    for phase in (0.0, 90.0, 180.0, 270.0, 360.0):
        x = map_x(phase)
        draw.line((x, plot_top, x, plot_bottom), fill=(55, 66, 80), width=1)
        draw.text((x - 10, plot_bottom + 8), f"{phase:.0f}", fill=(165, 175, 188))
    for fraction in np.linspace(0.0, 1.0, 5):
        value = y_min + fraction * (y_max - y_min)
        y = map_y(value)
        draw.line((plot_left, y, plot_right, y), fill=(55, 66, 80), width=1)
        draw.text(
            (left + 5, y - 6),
            f"{value:.3g}",
            fill=(165, 175, 188),
        )
    if zero_line and y_min < 0.0 < y_max:
        y = map_y(0.0)
        draw.line((plot_left, y, plot_right, y), fill=(180, 185, 192), width=2)

    points = [
        (map_x(float(x_value)), map_y(float(y_value)))
        for x_value, y_value in zip(x_values, y_values)
    ]
    draw.line(points, fill=color, width=3, joint="curve")
    draw.rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        outline=(105, 116, 130),
        width=1,
    )
    draw.text(
        ((plot_left + plot_right) / 2 - 32, plot_bottom + 24),
        "phase (deg)",
        fill=(180, 190, 202),
    )


def _write_plot(path: Path, analysis: PhaseAnalysis) -> None:
    image = Image.new("RGB", (1280, 900), color=(24, 31, 41))
    draw = ImageDraw.Draw(image)
    draw.text(
        (28, 18),
        "Rigid compact roll-phase analysis",
        fill=(244, 247, 251),
    )
    draw.text(
        (28, 40),
        (
            f"R={analysis.summary['shell_contact_radius_m']:.6f} m, "
            f"COM eccentricity={analysis.summary['com_eccentricity_m']:.6f} m, "
            f"delta U={analysis.summary['potential_peak_to_peak_J']:.3f} J"
        ),
        fill=(180, 191, 205),
    )

    phase = analysis.column("phase_deg")
    panels = (
        (
            (20, 76, 630, 480),
            analysis.column("com_z_m"),
            "COM height (m)",
            (70, 180, 255),
            False,
        ),
        (
            (650, 76, 1260, 480),
            analysis.column("potential_from_min_J"),
            "Potential above minimum (J)",
            (92, 214, 143),
            False,
        ),
        (
            (20, 486, 630, 890),
            analysis.column("gravity_torque_Nm"),
            "Gravity torque about contact (N m)",
            (255, 164, 74),
            True,
        ),
        (
            (650, 486, 1260, 890),
            analysis.column("inertia_about_contact_kg_m2"),
            "Inertia about contact (kg m^2)",
            (208, 124, 255),
            False,
        ),
    )
    for bounds, values, title, color, zero_line in panels:
        _draw_panel(
            draw,
            bounds,
            phase,
            values,
            title,
            color,
            zero_line=zero_line,
        )
    image.save(path)


def write_outputs(output_dir: Path, analysis: PhaseAnalysis) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rigid_phase.csv"
    summary_path = output_dir / "rigid_phase_summary.json"
    plot_path = output_dir / "rigid_phase.png"
    _write_csv(csv_path, analysis)
    summary_path.write_text(
        json.dumps(analysis.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_plot(plot_path, analysis)
    return csv_path, summary_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=361)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    analysis = analyze_rigid_phase(model, samples=args.samples)
    outputs = write_outputs(args.output_dir, analysis)

    print(f"model={MODEL_PATH}")
    for key, value in analysis.summary.items():
        if key != "model":
            print(f"{key}={value}")
    for output in outputs:
        print(f"output={output}")


if __name__ == "__main__":
    main()
