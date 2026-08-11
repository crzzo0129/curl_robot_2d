"""Report exact contact pairs for the 160 mm shell with motor proxies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import re

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d.parameters import FIXED_PARAMETERS, REAL_GEOMETRY_PARAMETERS
from scripts import optimize_phase_controller as phase_controller
from scripts.replay_active_controller import (
    advance_controller,
    configure_tracking_camera,
    initialize_simulation,
    load_controller_options,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT / "results/shell_radius_160mm_motors_54x33/model.xml"
)
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results/shell_radius_160mm_warm_start_cem/best_phase_controller.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/shell_radius_160mm_motors_54x33/contact_analysis"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-count", type=int, default=8)
    parser.add_argument(
        "--geometry",
        choices=("shell160-motors54x33", "real-foot39"),
        default="shell160-motors54x33",
    )
    return parser.parse_args(argv)


def _name(model, kind, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"id_{index}"


def _render_snapshot(path: Path, model, state, event) -> None:
    data = mujoco.MjData(model)
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.ctrl[:] = state["ctrl"]
    data.time = event["maximum_depth_time_s"]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    configure_tracking_camera(model, camera, distance=0.72)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    try:
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = Image.fromarray(renderer.render())
    finally:
        renderer.close()
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 950, 92), fill=(18, 24, 33))
    draw.text(
        (22, 18),
        f"{event['category']}  t={event['maximum_depth_time_s']:.3f}s  "
        f"depth={1000.0 * event['maximum_depth_m']:.3f} mm",
        fill=(255, 230, 150),
    )
    draw.text((22, 44), event["geom_pair"], fill=(255, 145, 145))
    draw.text((22, 68), event["body_pair"], fill=(220, 229, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.duration <= 0.0 or args.snapshot_count < 0:
        raise SystemExit("duration must be positive and snapshot-count nonnegative")

    geometry = (
        replace(REAL_GEOMETRY_PARAMETERS, foot_radius=0.0195)
        if args.geometry == "real-foot39"
        else replace(
            FIXED_PARAMETERS,
            shell_contact_radius_override=0.160,
            shell_arc_coverage_angle_override=(
                FIXED_PARAMETERS.shell_arc_coverage_angle
            ),
            motor_radius=0.027,
            motor_half_thickness_y=0.0165,
        )
    )
    phase_controller._activate_geometry(geometry)
    model_path = args.model.expanduser().resolve()
    controller_path = args.controller.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    (
        coefficients,
        oscillator_rate,
        oscillator_coupling,
        minimum_gap_m,
        tracking_margin_m,
        knee_bias_rad,
    ) = load_controller_options(controller_path)
    data, root_pitch_address, _ = initialize_simulation(
        model, minimum_gap_m, geometry
    )
    start_pitch = float(data.qpos[root_pitch_address])
    start_x = float(data.qpos[0])
    timestep = float(model.opt.timestep)
    floor_id = model.geom("floor").id
    foot_pair = frozenset(
        (model.geom("front_foot_proxy").id, model.geom("rear_foot_proxy").id)
    )

    records = defaultdict(
        lambda: {
            "active_steps": 0,
            "intervals": [],
            "maximum_depth_m": 0.0,
            "maximum_depth_time_s": 0.0,
            "state_at_maximum_depth": None,
            "body_pair": "",
        }
    )
    previous: set[tuple[str, str, str]] = set()
    starts: dict[tuple[str, str, str], float] = {}
    category_steps = defaultdict(int)
    special_union_steps = defaultdict(int)
    oscillator_phase = 0.0

    while data.time < args.duration:
        phase_controller._activate_geometry(geometry)
        oscillator_phase = advance_controller(
            model,
            data,
            coefficients,
            oscillator_rate,
            oscillator_coupling,
            oscillator_phase,
            root_pitch_address,
            minimum_gap_m,
            tracking_margin_m,
            knee_bias_rad,
        )
        current: set[tuple[str, str, str]] = set()
        deepest = {}
        for contact in data.contact:
            geom_ids = (int(contact.geom1), int(contact.geom2))
            geom_set = frozenset(geom_ids)
            if floor_id in geom_set:
                category = "ground"
            elif geom_set == foot_pair:
                category = "allowed_foot"
            else:
                category = "forbidden_self"
            geom_names = tuple(
                sorted(
                    _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id in geom_ids
                )
            )
            key = (category, geom_names[0], geom_names[1])
            body_names = tuple(
                sorted(
                    _name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[geom_id]),
                    )
                    for geom_id in geom_ids
                )
            )
            depth = max(-float(contact.dist), 0.0)
            current.add(key)
            old = deepest.get(key)
            if old is None or depth > old[0]:
                deepest[key] = (depth, " / ".join(body_names))

        for category in {key[0] for key in current}:
            category_steps[category] += 1
        if any(key[0] == "ground" and "_motor" in (key[1] + key[2]) for key in current):
            special_union_steps["any_motor_ground"] += 1
        if any(
            key[0] == "forbidden_self" and "_motor" in (key[1] + key[2])
            for key in current
        ):
            special_union_steps["motor_forbidden_self"] += 1
        if any(
            key[0] == "forbidden_self" and "_motor" not in (key[1] + key[2])
            for key in current
        ):
            special_union_steps["nonmotor_forbidden_self"] += 1
        for key, (depth, body_pair) in deepest.items():
            record = records[key]
            record["active_steps"] += 1
            record["body_pair"] = body_pair
            if depth > record["maximum_depth_m"]:
                record["maximum_depth_m"] = depth
                record["maximum_depth_time_s"] = float(data.time)
                record["state_at_maximum_depth"] = {
                    "qpos": np.asarray(data.qpos).copy(),
                    "qvel": np.asarray(data.qvel).copy(),
                    "ctrl": np.asarray(data.ctrl).copy(),
                }
        for key in current - previous:
            starts[key] = float(data.time)
        for key in previous - current:
            records[key]["intervals"].append(
                (starts.pop(key), float(data.time))
            )
        previous = current

    for key in previous:
        records[key]["intervals"].append((starts[key], float(data.time)))

    events = []
    states = {}
    for key, record in records.items():
        category, geom1, geom2 = key
        intervals = [
            {
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
            }
            for start, end in record["intervals"]
        ]
        event = {
            "category": category,
            "geom_pair": f"{geom1} / {geom2}",
            "body_pair": record["body_pair"],
            "involves_motor": "_motor" in geom1 or "_motor" in geom2,
            "total_contact_s": record["active_steps"] * timestep,
            "event_count": len(intervals),
            "longest_event_s": max(
                (item["duration_s"] for item in intervals), default=0.0
            ),
            "maximum_depth_m": record["maximum_depth_m"],
            "maximum_depth_time_s": record["maximum_depth_time_s"],
            "intervals": intervals,
        }
        events.append(event)
        states[event["geom_pair"]] = record["state_at_maximum_depth"]

    category_rank = {"forbidden_self": 0, "allowed_foot": 1, "ground": 2}
    events.sort(
        key=lambda event: (
            category_rank[event["category"]],
            -event["total_contact_s"],
            -event["maximum_depth_m"],
        )
    )
    snapshot_candidates = [
        event
        for event in events
        if event["category"] == "forbidden_self" or event["involves_motor"]
    ][: args.snapshot_count]
    for index, event in enumerate(snapshot_candidates, start=1):
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", event["geom_pair"])
        path = output_dir / f"{index:02d}_{safe_name}.png"
        _render_snapshot(path, model, states[event["geom_pair"]], event)
        event["snapshot"] = str(path.resolve())

    result = {
        "model": str(model_path),
        "controller": str(controller_path),
        "duration_s": float(data.time),
        "geometry": args.geometry,
        "edge_length_mm": 1000.0 * geometry.edge_length,
        "shell_radius_mm": 1000.0 * geometry.shell_contact_radius,
        "foot_diameter_mm": 2000.0 * geometry.foot_radius,
        "motor_diameter_mm": 2000.0 * geometry.motor_radius,
        "motor_thickness_mm": 2000.0 * geometry.motor_half_thickness_y,
        "net_turns": (
            float(data.qpos[root_pitch_address]) - start_pitch
        ) / (2.0 * math.pi),
        "root_x_displacement_m": float(data.qpos[0]) - start_x,
        "category_union_time_s": {
            category: steps * timestep
            for category, steps in sorted(category_steps.items())
        },
        "special_union_time_s": {
            category: steps * timestep
            for category, steps in sorted(special_union_steps.items())
        },
        "events": events,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "contact_statistics.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for event in events:
        if event["category"] == "ground" and not event["involves_motor"]:
            continue
        print(
            f"{event['category']:14s} {event['geom_pair']}: "
            f"total={event['total_contact_s']:.3f}s "
            f"events={event['event_count']} "
            f"longest={event['longest_event_s']:.3f}s "
            f"max={1000.0 * event['maximum_depth_m']:.3f}mm "
            f"at={event['maximum_depth_time_s']:.3f}s"
        )
    print(f"output={output}")


if __name__ == "__main__":
    main()
