"""Analyze exact contacts while lifting a 2-D CEM reference to 3-D."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS
from curl_robot_2d_mjx.cem_reference import advance_oscillator, load_cem_reference
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.environment_3d import apply_physics_options_3d
from scripts.evaluate_3d_symmetric_cem_reference import (
    activate_planar_geometry,
    map_planar_to_curl_3d_targets,
    planar_cem_target,
)
from scripts.view_3d_cem_reference import _reset, _rolling_axis_tilt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "assets/curl_robot_3d_real_geometry.xml"
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results/staged_cem_real_geometry_180_d50_foot60"
    / "03_foot_gap_2mm/best_phase_controller.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results/staged_cem_real_geometry_180_d50_foot60"
    / "contact_analysis_3d_reference"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--snapshot-count", type=int, default=10)
    return parser.parse_args(argv)


def _name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index)) or f"id_{index}"


def _logical_geom(name: str) -> str:
    return re.sub(r"_motor_collision_\d+$", "_motor", name)


def _render(path, model, state, event):
    data = mujoco.MjData(model)
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.ctrl[:] = state["ctrl"]
    data.time = event["maximum_depth_time_s"]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=540, width=720)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("torso").id
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.distance = 0.9
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    try:
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = Image.fromarray(renderer.render())
    finally:
        renderer.close()
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 712, 84), fill=(18, 24, 33))
    draw.text(
        (18, 16),
        f"{event['category']}  t={event['maximum_depth_time_s']:.3f}s  "
        f"depth={1000.0 * event['maximum_depth_m']:.3f} mm",
        fill=(255, 230, 150),
    )
    draw.text((18, 40), event["logical_geom_pair"], fill=(255, 145, 145))
    draw.text((18, 62), event["body_pair"], fill=(220, 229, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main(argv=None):
    args = parse_args(argv)
    activate_planar_geometry(REAL_GEOMETRY_PARAMETERS)
    model_path = args.model.expanduser().resolve()
    controller_path = args.controller.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    apply_physics_options_3d(model, physics_profile_3d("reference", Rolling3DConfig()))
    config = load_cem_reference(controller_path)
    data = mujoco.MjData(model)
    joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES_3D])
    qpos_indices = np.asarray([model.jnt_qposadr[j] for j in joint_ids])
    actuator_ids = np.asarray(
        [model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D]
    )
    ctrl_low = np.asarray(model.actuator_ctrlrange[actuator_ids, 0])
    ctrl_high = np.asarray(model.actuator_ctrlrange[actuator_ids, 1])
    model.actuator_gainprm[actuator_ids, 0] = args.kp
    model.actuator_biasprm[actuator_ids, 1] = -args.kp
    model.actuator_biasprm[actuator_ids, 2] = -args.kd
    model.actuator_forcerange[actuator_ids, 0] = -args.torque_limit
    model.actuator_forcerange[actuator_ids, 1] = args.torque_limit

    phase = 0.0
    rolling_phase = 0.0
    initial = np.clip(
        map_planar_to_curl_3d_targets(planar_cem_target(phase, config)),
        ctrl_low,
        ctrl_high,
    )
    _reset(model, data, qpos_indices, actuator_ids, initial)
    start_x, start_y = float(data.qpos[0]), float(data.qpos[1])
    timestep = float(model.opt.timestep)
    control_repeat = max(1, round(args.control_dt / timestep))
    torso_id = model.body("torso").id
    floor_id = model.geom("floor").id
    allowed_foot_pairs = {
        frozenset((
            model.geom("front_left_foot_proxy").id,
            model.geom("rear_left_foot_proxy").id,
        )),
        frozenset((
            model.geom("front_right_foot_proxy").id,
            model.geom("rear_right_foot_proxy").id,
        )),
    }

    records = defaultdict(lambda: {
        "steps": 0, "intervals": [], "maximum_depth_m": 0.0,
        "maximum_depth_time_s": 0.0, "state": None, "body_pair": "",
        "exact_pairs": defaultdict(int),
    })
    previous = set()
    starts = {}
    category_steps = defaultdict(int)
    tilt = []
    saturation = []

    while data.time < args.duration:
        for _ in range(control_repeat):
            phase = float(
                advance_oscillator(np, rolling_phase, phase, timestep, config)
            )
            ctrl = np.clip(
                map_planar_to_curl_3d_targets(planar_cem_target(phase, config)),
                ctrl_low,
                ctrl_high,
            )
            data.ctrl[actuator_ids] = ctrl
            mujoco.mj_step(model, data)
            rolling_phase += float(data.qvel[4]) * timestep

            current = set()
            deepest = {}
            for contact in data.contact:
                geom_ids = (int(contact.geom1), int(contact.geom2))
                raw_names = tuple(sorted(
                    _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    for gid in geom_ids
                ))
                logical_names = tuple(sorted(_logical_geom(n) for n in raw_names))
                geom_set = frozenset(geom_ids)
                if floor_id in geom_set:
                    category = "ground"
                elif geom_set in allowed_foot_pairs:
                    category = "allowed_foot"
                else:
                    category = "forbidden_self"
                key = (category, logical_names[0], logical_names[1])
                body_names = tuple(sorted(
                    _name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
                    for gid in geom_ids
                ))
                depth = max(-float(contact.dist), 0.0)
                current.add(key)
                old = deepest.get(key)
                if old is None or depth > old[0]:
                    deepest[key] = (depth, " / ".join(body_names), " / ".join(raw_names))
            for category in {k[0] for k in current}:
                category_steps[category] += 1
            for key, (depth, body_pair, exact_pair) in deepest.items():
                rec = records[key]
                rec["steps"] += 1
                rec["body_pair"] = body_pair
                rec["exact_pairs"][exact_pair] += 1
                if depth > rec["maximum_depth_m"]:
                    rec["maximum_depth_m"] = depth
                    rec["maximum_depth_time_s"] = float(data.time)
                    rec["state"] = {
                        "qpos": np.asarray(data.qpos).copy(),
                        "qvel": np.asarray(data.qvel).copy(),
                        "ctrl": np.asarray(data.ctrl).copy(),
                    }
            for key in current - previous:
                starts[key] = float(data.time)
            for key in previous - current:
                records[key]["intervals"].append((starts.pop(key), float(data.time)))
            previous = current
        rotation = data.xmat[torso_id].reshape(3, 3)
        tilt.append(_rolling_axis_tilt(rotation))
        saturation.append(float(np.mean(
            np.abs(data.actuator_force[actuator_ids]) >= 0.99 * args.torque_limit
        )))

    for key in previous:
        records[key]["intervals"].append((starts[key], float(data.time)))

    events = []
    states = {}
    for (category, geom1, geom2), rec in records.items():
        intervals = [
            {"start_s": a, "end_s": b, "duration_s": b - a}
            for a, b in rec["intervals"]
        ]
        event = {
            "category": category,
            "logical_geom_pair": f"{geom1} / {geom2}",
            "body_pair": rec["body_pair"],
            "involves_motor": "_motor" in geom1 or "_motor" in geom2,
            "total_contact_s": rec["steps"] * timestep,
            "event_count": len(intervals),
            "longest_event_s": max((x["duration_s"] for x in intervals), default=0.0),
            "maximum_depth_m": rec["maximum_depth_m"],
            "maximum_depth_time_s": rec["maximum_depth_time_s"],
            "intervals": intervals,
            "most_frequent_exact_pairs": [
                {"geom_pair": name, "contact_s": steps * timestep}
                for name, steps in sorted(
                    rec["exact_pairs"].items(), key=lambda x: x[1], reverse=True
                )[:8]
            ],
        }
        events.append(event)
        states[event["logical_geom_pair"]] = rec["state"]
    category_rank = {"forbidden_self": 0, "allowed_foot": 1, "ground": 2}
    events.sort(key=lambda x: (category_rank[x["category"]], -x["total_contact_s"]))
    for index, event in enumerate(
        [x for x in events if x["category"] == "forbidden_self"][:args.snapshot_count], 1
    ):
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", event["logical_geom_pair"])
        path = output_dir / f"{index:02d}_{name}.png"
        _render(path, model, states[event["logical_geom_pair"]], event)
        event["snapshot"] = str(path.resolve())

    result = {
        "model": str(model_path), "controller": str(controller_path),
        "duration_s": float(data.time), "foot_diameter_mm": 60.0,
        "net_reference_turns": phase / (2.0 * math.pi),
        "rolling_phase_turns": rolling_phase / (2.0 * math.pi),
        "distance_x_m": float(data.qpos[0]) - start_x,
        "distance_y_m": float(data.qpos[1]) - start_y,
        "rolling_axis_tilt_rms_rad": float(np.sqrt(np.mean(np.square(tilt)))),
        "rolling_axis_tilt_max_rad": float(np.max(tilt)),
        "torque_saturation_fraction": float(np.mean(saturation)),
        "category_union_time_s": {
            k: v * timestep for k, v in sorted(category_steps.items())
        },
        "events": events,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "contact_statistics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for event in events:
        if event["category"] != "ground":
            print(
                f"{event['logical_geom_pair']}: total={event['total_contact_s']:.3f}s "
                f"events={event['event_count']} longest={event['longest_event_s']:.3f}s "
                f"max={1000.0*event['maximum_depth_m']:.3f}mm "
                f"at={event['maximum_depth_time_s']:.3f}s"
            )
    print(f"output={output}")


if __name__ == "__main__":
    main()
