"""Locate and render non-torso self contacts in a 2-D CEM rollout."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS
from scripts.replay_active_controller import (
    _activate_geometry,
    advance_controller,
    configure_tracking_camera,
    initialize_simulation,
    load_controller_options,
)


def _name(model, kind, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"id_{index}"


def _render(path: Path, model, qpos, *, event: dict, oblique: bool) -> None:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    configure_tracking_camera(model, camera, distance=0.75)
    if oblique:
        camera.azimuth = 135.0
        camera.elevation = -18.0
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
        (22, 20),
        f"t={event['maximum_depth_time_s']:.3f}s  depth={1000*event['maximum_depth_m']:.3f} mm",
        fill=(255, 235, 150),
    )
    draw.text((22, 44), event["body_pair"], fill=(255, 135, 135))
    draw.text((22, 68), event["geom_pair_at_maximum_depth"], fill=(225, 231, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _activate_geometry(REAL_GEOMETRY_PARAMETERS)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    (
        coefficients,
        oscillator_rate,
        oscillator_coupling,
        minimum_foot_gap_m,
        tracking_margin_m,
        knee_bias_rad,
    ) = load_controller_options(args.controller)
    data, root_pitch_address, _ = initialize_simulation(
        model, minimum_foot_gap_m, REAL_GEOMETRY_PARAMETERS
    )
    floor_id = model.geom("floor").id
    timestep = float(model.opt.timestep)
    phase = 0.0
    active_previous: set[tuple[str, str]] = set()
    interval_starts: dict[tuple[str, str], float] = {}
    records = defaultdict(
        lambda: {
            "steps": 0,
            "intervals": [],
            "maximum_depth_m": 0.0,
            "maximum_depth_time_s": 0.0,
            "geom_pair_at_maximum_depth": "",
            "qpos_at_maximum_depth": None,
            "geom_pair_steps": defaultdict(int),
        }
    )

    while data.time < args.duration:
        phase = advance_controller(
            model,
            data,
            coefficients,
            oscillator_rate,
            oscillator_coupling,
            phase,
            root_pitch_address,
            minimum_foot_gap_m,
            tracking_margin_m,
            knee_bias_rad,
            1.0,
        )
        current: set[tuple[str, str]] = set()
        deepest_by_body_pair: dict[tuple[str, str], tuple[float, str]] = {}
        for contact in data.contact:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if floor_id in (geom1, geom2):
                continue
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            if body1 == body2:
                continue
            body_pair = tuple(sorted((
                _name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                _name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
            )))
            geom_pair = " / ".join(sorted((
                _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            )))
            depth = max(-float(contact.dist), 0.0)
            current.add(body_pair)
            previous = deepest_by_body_pair.get(body_pair)
            if previous is None or depth > previous[0]:
                deepest_by_body_pair[body_pair] = (depth, geom_pair)

        for pair, (depth, geom_pair) in deepest_by_body_pair.items():
            record = records[pair]
            record["steps"] += 1
            record["geom_pair_steps"][geom_pair] += 1
            if depth > record["maximum_depth_m"]:
                record["maximum_depth_m"] = depth
                record["maximum_depth_time_s"] = float(data.time)
                record["geom_pair_at_maximum_depth"] = geom_pair
                record["qpos_at_maximum_depth"] = np.asarray(data.qpos).copy()
        for pair in current - active_previous:
            interval_starts[pair] = float(data.time)
        for pair in active_previous - current:
            records[pair]["intervals"].append(
                [interval_starts.pop(pair), float(data.time)]
            )
        active_previous = current

    for pair in active_previous:
        records[pair]["intervals"].append(
            [interval_starts[pair], float(data.time)]
        )

    events = []
    ranked = sorted(records.items(), key=lambda item: item[1]["steps"], reverse=True)
    for index, (pair, record) in enumerate(ranked, start=1):
        intervals = [
            {"start_s": start, "end_s": end, "duration_s": end - start}
            for start, end in record["intervals"]
        ]
        event = {
            "rank": index,
            "body_pair": " / ".join(pair),
            "total_contact_s": record["steps"] * timestep,
            "intervals": intervals,
            "maximum_depth_m": record["maximum_depth_m"],
            "maximum_depth_time_s": record["maximum_depth_time_s"],
            "geom_pair_at_maximum_depth": record["geom_pair_at_maximum_depth"],
            "most_frequent_geom_pairs": [
                {"geom_pair": name, "contact_s": steps * timestep}
                for name, steps in sorted(
                    record["geom_pair_steps"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            ],
        }
        if record["qpos_at_maximum_depth"] is not None:
            stem = f"{index:02d}_{pair[0]}__{pair[1]}"
            side = args.output_dir / f"{stem}_side.png"
            oblique = args.output_dir / f"{stem}_oblique.png"
            _render(side, model, record["qpos_at_maximum_depth"], event=event, oblique=False)
            _render(oblique, model, record["qpos_at_maximum_depth"], event=event, oblique=True)
            event["side_snapshot"] = str(side.resolve())
            event["oblique_snapshot"] = str(oblique.resolve())
        events.append(event)

    result = {
        "model": str(args.model.resolve()),
        "controller": str(args.controller.resolve()),
        "duration_s": args.duration,
        "torso_leg_collision_excluded": True,
        "events": events,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "contact_events.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for event in events:
        print(
            f"{event['body_pair']}: {event['total_contact_s']:.3f}s, "
            f"max={1000*event['maximum_depth_m']:.3f}mm "
            f"at t={event['maximum_depth_time_s']:.3f}s"
        )
    print(f"output={output}")


if __name__ == "__main__":
    main()
