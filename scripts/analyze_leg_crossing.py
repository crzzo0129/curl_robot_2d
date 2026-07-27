"""Diagnose front/rear leg crossings in the saved active-controller rollout."""

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
from curl_robot_2d.planar_geometry import (
    proper_segments_intersect,
    segment_distance,
    trim_segment_distal,
)
from scripts.replay_active_controller import (
    DEFAULT_CONTROLLER_PATH,
    advance_controller,
    configure_tracking_camera,
    initialize_simulation,
    load_controller,
)
from scripts.run_release_baseline import MODEL_PATH, _draw_panel, _id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "leg_crossing_analysis"
PAIR_NAMES = (
    "front_thigh__rear_thigh",
    "front_thigh__rear_shank",
    "front_shank__rear_thigh",
    "front_shank__rear_shank",
)


@dataclass(frozen=True)
class LinkSegment:
    start: np.ndarray
    end: np.ndarray
    radius: float


def _link_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, int],
) -> dict[str, LinkSegment]:
    front_hip = np.asarray(data.xpos[ids["front_thigh"]])[[0, 2]]
    front_knee = np.asarray(data.xpos[ids["front_shank"]])[[0, 2]]
    front_foot = np.asarray(data.site_xpos[ids["front_foot_site"]])[[0, 2]]
    rear_hip = np.asarray(data.xpos[ids["rear_thigh"]])[[0, 2]]
    rear_knee = np.asarray(data.xpos[ids["rear_shank"]])[[0, 2]]
    rear_foot = np.asarray(data.site_xpos[ids["rear_foot_site"]])[[0, 2]]
    return {
        "front_thigh": LinkSegment(
            front_hip, front_knee, FIXED_PARAMETERS.upper_proxy_radius
        ),
        "front_shank": LinkSegment(
            front_knee, front_foot, FIXED_PARAMETERS.lower_proxy_radius
        ),
        "rear_thigh": LinkSegment(
            rear_hip, rear_knee, FIXED_PARAMETERS.upper_proxy_radius
        ),
        "rear_shank": LinkSegment(
            rear_knee, rear_foot, FIXED_PARAMETERS.lower_proxy_radius
        ),
    }


def _pair_metrics(
    links: dict[str, LinkSegment],
    first_name: str,
    second_name: str,
) -> tuple[bool, bool, float, float]:
    first = links[first_name]
    second = links[second_name]
    crossing = proper_segments_intersect(
        first.start, first.end, second.start, second.end
    )

    first_start, first_end = first.start, first.end
    second_start, second_end = second.start, second.end
    if {first_name, second_name} == {"front_shank", "rear_shank"}:
        # The compact construction intentionally shares one zero-thickness
        # distal vertex.  Remove only a short distal portion for clearance;
        # an actual interior crossing remains visible to `crossing` above.
        trim_fraction = FIXED_PARAMETERS.foot_radius / FIXED_PARAMETERS.edge_length
        first_start, first_end = trim_segment_distal(
            first_start, first_end, trim_fraction
        )
        second_start, second_end = trim_segment_distal(
            second_start, second_end, trim_fraction
        )
    core_crossing = proper_segments_intersect(
        first_start, first_end, second_start, second_end
    )
    centerline_distance = segment_distance(
        first_start, first_end, second_start, second_end
    )
    estimated_clearance = centerline_distance - first.radius - second.radius
    return crossing, core_crossing, centerline_distance, estimated_clearance


def _event_count(mask: np.ndarray) -> int:
    previous = False
    events = 0
    for value in mask:
        current = bool(value)
        events += int(current and not previous)
        previous = current
    return events


def _longest_duration(mask: np.ndarray, timestep: float) -> float:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * timestep


def _write_csv(path: Path, columns: tuple[str, ...], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_plot(path: Path, columns: tuple[str, ...], rows: np.ndarray) -> None:
    def column(name: str) -> np.ndarray:
        return rows[:, columns.index(name)]

    image = Image.new("RGB", (1400, 1040), color=(24, 31, 41))
    draw = ImageDraw.Draw(image)
    first_crossing = column("any_crossing") > 0.5
    first_text = (
        f"{column('time_s')[np.argmax(first_crossing)]:.3f} s"
        if np.any(first_crossing)
        else "none"
    )
    draw.text(
        (28, 16),
        "Leg crossing diagnostic: saved active controller",
        fill=(244, 247, 251),
    )
    draw.text(
        (28, 38),
        f"first proper crossing={first_text}",
        fill=(180, 191, 205),
    )
    time = column("time_s")
    panels = (
        (
            (18, 72, 690, 386),
            (("root phase", column("phase_deg"), (255, 164, 74)),),
            "Root roll phase (deg)",
            False,
        ),
        (
            (710, 72, 1382, 386),
            (
                ("crossing pairs", column("crossing_count"), (255, 97, 104)),
                (
                    "core crossing pairs",
                    column("core_crossing_count"),
                    (208, 124, 255),
                ),
                (
                    "self-contact pairs",
                    column("self_contact_pair_count"),
                    (92, 214, 143),
                ),
                ("any crossing", column("any_crossing"), (255, 201, 92)),
            ),
            "Topological and core crossings",
            True,
        ),
        (
            (18, 394, 690, 708),
            tuple(
                (
                    name.replace("__", " / "),
                    column(f"{name}_core_crossing"),
                    color,
                )
                for name, color in zip(
                    PAIR_NAMES,
                    (
                        (255, 97, 104),
                        (255, 164, 74),
                        (208, 124, 255),
                        (70, 180, 255),
                    ),
                )
            ),
            "Core crossing state by pair",
            True,
        ),
        (
            (710, 394, 1382, 708),
            (
                (
                    "minimum clearance",
                    1000.0 * column("minimum_estimated_clearance_m"),
                    (255, 97, 104),
                ),
                ("zero clearance", np.zeros_like(time), (180, 191, 205)),
            ),
            "Minimum estimated structural clearance (mm)",
            True,
        ),
        (
            (18, 716, 690, 1030),
            tuple(
                (
                    name.replace("__", " / "),
                    1000.0 * column(f"{name}_centerline_distance_m"),
                    color,
                )
                for name, color in zip(
                    PAIR_NAMES,
                    (
                        (255, 97, 104),
                        (255, 164, 74),
                        (208, 124, 255),
                        (70, 180, 255),
                    ),
                )
            ),
            "Pair centerline distance (mm)",
            True,
        ),
        (
            (710, 716, 1382, 1030),
            (
                (
                    "minimum active contact distance",
                    1000.0 * column("minimum_self_contact_distance_m"),
                    (255, 97, 104),
                ),
                ("zero penetration", np.zeros_like(time), (180, 191, 205)),
            ),
            "Minimum active self-contact distance (mm)",
            True,
        ),
    )
    for bounds, series, title, zero_line in panels:
        _draw_panel(
            draw, bounds, time, series, title, zero_line=zero_line
        )
    image.save(path)


def _render_snapshot(
    path: Path,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    time_s: float,
    phase_deg: float,
    crossing_pairs: list[str],
    camera_distance: float,
) -> None:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    configure_tracking_camera(model, camera, distance=camera_distance)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    try:
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        image = Image.fromarray(renderer.render())
    finally:
        renderer.close()
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 10, 650, 66), fill=(20, 26, 35))
    draw.text(
        (22, 18),
        f"time={time_s:.3f} s  phase={phase_deg:.1f} deg",
        fill=(244, 247, 251),
    )
    draw.text(
        (22, 41),
        "crossing: " + ", ".join(pair.replace("__", "/") for pair in crossing_pairs),
        fill=(255, 120, 120),
    )
    image.save(path)


def analyze(
    controller_path: Path,
    output_dir: Path,
    *,
    duration: float,
    camera_distance: float,
) -> dict[str, object]:
    coefficients, oscillator_rate, oscillator_coupling = load_controller(
        controller_path
    )
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data, root_pitch_qpos_address, _ = initialize_simulation(model)
    ids = {
        name: _id(
            model,
            mujoco.mjtObj.mjOBJ_SITE if name.endswith("_site") else mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        for name in (
            "front_thigh",
            "front_shank",
            "rear_thigh",
            "rear_shank",
            "front_foot_site",
            "rear_foot_site",
        )
    }
    pair_links = {
        "front_thigh__rear_thigh": ("front_thigh", "rear_thigh"),
        "front_thigh__rear_shank": ("front_thigh", "rear_shank"),
        "front_shank__rear_thigh": ("front_shank", "rear_thigh"),
        "front_shank__rear_shank": ("front_shank", "rear_shank"),
    }
    columns = (
        "time_s",
        "phase_rad",
        "phase_deg",
        "crossing_count",
        "any_crossing",
        "core_crossing_count",
        "any_core_crossing",
        "minimum_estimated_clearance_m",
        "foot_endpoint_distance_m",
        "self_contact_pair_count",
        "any_self_contact",
        "minimum_self_contact_distance_m",
        *tuple(
            item
            for pair_name in PAIR_NAMES
            for item in (
                f"{pair_name}_crossing",
                f"{pair_name}_core_crossing",
                f"{pair_name}_centerline_distance_m",
                f"{pair_name}_estimated_clearance_m",
            )
        ),
    )
    rows: list[list[float]] = []
    contact_frames: list[dict[str, float]] = []
    first_snapshot: tuple[np.ndarray, float, float, list[str]] | None = None
    first_core_snapshot: tuple[np.ndarray, float, float, list[str]] | None = None
    pair_first_core_snapshots: dict[
        str, tuple[np.ndarray, float, float, list[str]]
    ] = {}
    maximum_snapshot: tuple[np.ndarray, float, float, list[str]] | None = None
    maximum_crossing_count = 0
    oscillator_phase = 0.0
    timestep = float(model.opt.timestep)
    steps = int(math.ceil(duration / timestep))

    def record() -> None:
        nonlocal first_snapshot, first_core_snapshot
        nonlocal maximum_snapshot, maximum_crossing_count
        links = _link_segments(model, data, ids)
        pair_values: list[float] = []
        active_pairs = []
        active_core_pairs = []
        clearances = []
        for pair_name in PAIR_NAMES:
            first_name, second_name = pair_links[pair_name]
            crossing, core_crossing, distance, clearance = _pair_metrics(
                links, first_name, second_name
            )
            pair_values.extend(
                (float(crossing), float(core_crossing), distance, clearance)
            )
            clearances.append(clearance)
            if crossing:
                active_pairs.append(pair_name)
            if core_crossing:
                active_core_pairs.append(pair_name)
        crossing_count = len(active_pairs)
        core_crossing_count = len(active_core_pairs)
        phase = float(data.qpos[root_pitch_qpos_address])
        foot_distance = float(
            np.linalg.norm(
                links["front_shank"].end - links["rear_shank"].end
            )
        )
        active_contacts: dict[str, float] = {}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first_geom_id = int(contact.geom1)
            second_geom_id = int(contact.geom2)
            first_geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, first_geom_id
            )
            second_geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, second_geom_id
            )
            if first_geom_name == "floor" or second_geom_name == "floor":
                continue
            pair_name = "__".join(sorted((first_geom_name, second_geom_name)))
            active_contacts[pair_name] = min(
                active_contacts.get(pair_name, math.inf),
                float(contact.dist),
            )
        contact_frames.append(active_contacts)
        minimum_contact_distance = (
            min(active_contacts.values()) if active_contacts else 0.0
        )
        rows.append(
            [
                float(data.time),
                phase,
                math.degrees(phase),
                float(crossing_count),
                float(crossing_count > 0),
                float(core_crossing_count),
                float(core_crossing_count > 0),
                min(clearances),
                foot_distance,
                float(len(active_contacts)),
                float(bool(active_contacts)),
                minimum_contact_distance,
                *pair_values,
            ]
        )
        if crossing_count > 0 and first_snapshot is None:
            first_snapshot = (
                np.asarray(data.qpos).copy(),
                float(data.time),
                math.degrees(phase),
                active_pairs.copy(),
            )
        if core_crossing_count > 0 and first_core_snapshot is None:
            first_core_snapshot = (
                np.asarray(data.qpos).copy(),
                float(data.time),
                math.degrees(phase),
                active_core_pairs.copy(),
            )
        for pair_name in active_core_pairs:
            if pair_name not in pair_first_core_snapshots:
                pair_first_core_snapshots[pair_name] = (
                    np.asarray(data.qpos).copy(),
                    float(data.time),
                    math.degrees(phase),
                    [pair_name],
                )
        if crossing_count > maximum_crossing_count:
            maximum_crossing_count = crossing_count
            maximum_snapshot = (
                np.asarray(data.qpos).copy(),
                float(data.time),
                math.degrees(phase),
                active_pairs.copy(),
            )

    record()
    for _ in range(steps):
        oscillator_phase = advance_controller(
            model,
            data,
            coefficients,
            oscillator_rate,
            oscillator_coupling,
            oscillator_phase,
            root_pitch_qpos_address,
        )
        record()

    row_array = np.asarray(rows, dtype=float)
    any_mask = row_array[:, columns.index("any_crossing")] > 0.5
    any_core_mask = row_array[:, columns.index("any_core_crossing")] > 0.5
    summary: dict[str, object] = {
        "controller_source": str(controller_path),
        "model_source": str(MODEL_PATH),
        "duration_s": float(data.time),
        "timestep_s": timestep,
        "compact_foot_surface_contact_is_allowed": True,
        "any_proper_crossing": bool(np.any(any_mask)),
        "first_crossing_time_s": (
            float(row_array[np.argmax(any_mask), columns.index("time_s")])
            if np.any(any_mask)
            else None
        ),
        "first_crossing_phase_deg": (
            float(row_array[np.argmax(any_mask), columns.index("phase_deg")])
            if np.any(any_mask)
            else None
        ),
        "crossing_total_s": float(np.count_nonzero(any_mask) * timestep),
        "crossing_fraction": float(np.mean(any_mask)),
        "crossing_event_count": _event_count(any_mask),
        "longest_crossing_event_s": _longest_duration(any_mask, timestep),
        "any_core_crossing": bool(np.any(any_core_mask)),
        "first_core_crossing_time_s": (
            float(row_array[np.argmax(any_core_mask), columns.index("time_s")])
            if np.any(any_core_mask)
            else None
        ),
        "first_core_crossing_phase_deg": (
            float(row_array[np.argmax(any_core_mask), columns.index("phase_deg")])
            if np.any(any_core_mask)
            else None
        ),
        "core_crossing_total_s": float(
            np.count_nonzero(any_core_mask) * timestep
        ),
        "core_crossing_fraction": float(np.mean(any_core_mask)),
        "core_crossing_event_count": _event_count(any_core_mask),
        "longest_core_crossing_event_s": _longest_duration(
            any_core_mask, timestep
        ),
        "maximum_simultaneous_crossing_pairs": int(
            np.max(row_array[:, columns.index("crossing_count")])
        ),
        "minimum_estimated_clearance_m": float(
            np.min(row_array[:, columns.index("minimum_estimated_clearance_m")])
        ),
        "self_contact_total_s": float(
            np.count_nonzero(
                row_array[:, columns.index("any_self_contact")] > 0.5
            )
            * timestep
        ),
        "self_contact_fraction": float(
            np.mean(row_array[:, columns.index("any_self_contact")] > 0.5)
        ),
        "maximum_simultaneous_self_contact_pairs": int(
            np.max(row_array[:, columns.index("self_contact_pair_count")])
        ),
        "minimum_self_contact_distance_m": float(
            min(
                (
                    distance
                    for frame in contact_frames
                    for distance in frame.values()
                ),
                default=0.0,
            )
        ),
        "final_roll_phase_deg": float(
            row_array[-1, columns.index("phase_deg")]
        ),
        "final_roll_turns": float(
            row_array[-1, columns.index("phase_rad")] / (2.0 * math.pi)
        ),
    }
    contact_pair_summary: dict[str, dict[str, object]] = {}
    all_contact_pairs = sorted(
        {pair_name for frame in contact_frames for pair_name in frame}
    )
    for pair_name in all_contact_pairs:
        mask = np.asarray(
            [pair_name in frame for frame in contact_frames], dtype=bool
        )
        distances = [
            frame[pair_name] for frame in contact_frames if pair_name in frame
        ]
        contact_pair_summary[pair_name] = {
            "total_contact_s": float(np.count_nonzero(mask) * timestep),
            "contact_fraction": float(np.mean(mask)),
            "event_count": _event_count(mask),
            "longest_event_s": _longest_duration(mask, timestep),
            "minimum_contact_distance_m": float(min(distances)),
        }
    summary["self_contact_pairs"] = contact_pair_summary
    pair_summary = {}
    for pair_name in PAIR_NAMES:
        mask = row_array[:, columns.index(f"{pair_name}_crossing")] > 0.5
        core_mask = (
            row_array[:, columns.index(f"{pair_name}_core_crossing")] > 0.5
        )
        pair_summary[pair_name] = {
            "ever_crossed": bool(np.any(mask)),
            "first_crossing_time_s": (
                float(row_array[np.argmax(mask), columns.index("time_s")])
                if np.any(mask)
                else None
            ),
            "total_crossing_s": float(np.count_nonzero(mask) * timestep),
            "event_count": _event_count(mask),
            "longest_event_s": _longest_duration(mask, timestep),
            "core_ever_crossed": bool(np.any(core_mask)),
            "core_first_crossing_time_s": (
                float(
                    row_array[
                        np.argmax(core_mask), columns.index("time_s")
                    ]
                )
                if np.any(core_mask)
                else None
            ),
            "core_total_crossing_s": float(
                np.count_nonzero(core_mask) * timestep
            ),
            "core_event_count": _event_count(core_mask),
            "core_longest_event_s": _longest_duration(core_mask, timestep),
            "minimum_centerline_distance_m": float(
                np.min(
                    row_array[
                        :, columns.index(f"{pair_name}_centerline_distance_m")
                    ]
                )
            ),
            "minimum_estimated_clearance_m": float(
                np.min(
                    row_array[
                        :, columns.index(f"{pair_name}_estimated_clearance_m")
                    ]
                )
            ),
        }
    summary["pairs"] = pair_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "leg_crossing_timeseries.csv"
    summary_path = output_dir / "leg_crossing_summary.json"
    plot_path = output_dir / "leg_crossing_diagnostic.png"
    _write_csv(csv_path, columns, row_array)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_plot(plot_path, columns, row_array)
    if first_snapshot is not None:
        qpos, time_s, phase_deg, active_pairs = first_snapshot
        _render_snapshot(
            output_dir / "first_crossing.png",
            model,
            qpos,
            time_s=time_s,
            phase_deg=phase_deg,
            crossing_pairs=active_pairs,
            camera_distance=camera_distance,
        )
    if first_core_snapshot is not None:
        qpos, time_s, phase_deg, active_pairs = first_core_snapshot
        _render_snapshot(
            output_dir / "first_core_crossing.png",
            model,
            qpos,
            time_s=time_s,
            phase_deg=phase_deg,
            crossing_pairs=active_pairs,
            camera_distance=camera_distance,
        )
    for pair_name, snapshot in pair_first_core_snapshots.items():
        qpos, time_s, phase_deg, active_pairs = snapshot
        _render_snapshot(
            output_dir / f"first_core_{pair_name}.png",
            model,
            qpos,
            time_s=time_s,
            phase_deg=phase_deg,
            crossing_pairs=active_pairs,
            camera_distance=camera_distance,
        )
    if maximum_snapshot is not None:
        qpos, time_s, phase_deg, active_pairs = maximum_snapshot
        _render_snapshot(
            output_dir / "maximum_crossing.png",
            model,
            qpos,
            time_s=time_s,
            phase_deg=phase_deg,
            crossing_pairs=active_pairs,
            camera_distance=camera_distance,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={csv_path}")
    print(f"output={summary_path}")
    print(f"output={plot_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--camera-distance", type=float, default=0.75)
    args = parser.parse_args()
    analyze(
        args.controller,
        args.output_dir,
        duration=args.duration,
        camera_distance=args.camera_distance,
    )


if __name__ == "__main__":
    main()
