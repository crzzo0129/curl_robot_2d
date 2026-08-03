"""Render the zero-residual 3-D walking reference in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d_mjx.config_walking_3d import (
    Walking3DConfig,
    smoothstep_ramp,
    walking_physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import apply_physics_options_3d
from curl_robot_2d_mjx.environment_walking_3d import (
    FOOT_GEOM_NAMES_3D,
    WALKING_MODEL_PATH_3D,
    validate_walking_morphology_3d,
)
from curl_robot_2d_mjx.walking_reference_3d import walking_reference_3d


DEFAULT_OUTPUT = Path("renders/3d_walking_reference.gif")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the 3-D walking reference without a learned residual."
    )
    parser.add_argument("--xml", type=Path, default=WALKING_MODEL_PATH_3D)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=0.82)
    parser.add_argument("--azimuth", type=float, default=125.0)
    parser.add_argument("--elevation", type=float, default=-14.0)
    parser.add_argument("--physics-profile", default="reference")
    parser.add_argument(
        "--hide-contacts",
        action="store_true",
        help="Do not draw MuJoCo contact points.",
    )
    return parser.parse_args(argv)


def _object_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"missing MuJoCo object: {name}")
    return int(object_id)


def _initialize_reference(model, data, task: Walking3DConfig):
    key_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_KEY, task.reset_keyframe_name
    )
    joint_ids = np.asarray(
        [
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES_3D
        ],
        dtype=np.int32,
    )
    qpos_indices = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in joint_ids],
        dtype=np.int32,
    )
    actuator_ids = np.asarray(
        [
            _object_id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_servo"
            )
            for name in JOINT_NAMES_3D
        ],
        dtype=np.int32,
    )
    joint_low = model.jnt_range[joint_ids, 0]
    joint_high = model.jnt_range[joint_ids, 1]
    phase = 2.0 * math.pi * task.reference.initial_phase_fraction
    initial_reference = walking_reference_3d(np, phase, task.reference)
    reset_ctrl = np.asarray(model.key_ctrl[key_id], dtype=np.float64)
    startup_ctrl = np.clip(
        reset_ctrl
        + task.reset_reference_weight
        * (initial_reference["joint_targets"] - reset_ctrl),
        joint_low,
        joint_high,
    )

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[qpos_indices] = startup_ctrl
    data.qpos[0:2] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[actuator_ids] = startup_ctrl
    mujoco.mj_forward(model, data)
    return qpos_indices, actuator_ids, joint_low, joint_high, startup_ctrl, phase


def _configure_camera(model, args):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = _object_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso"
    )
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    camera.distance = args.camera_distance
    return camera


def _contact_counts(model, data, floor_geom_id: int, foot_geom_ids: set[int]):
    foot_contacts = 0
    nonfoot_contacts = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if floor_geom_id not in (geom1, geom2):
            continue
        other = geom2 if geom1 == floor_geom_id else geom1
        if other in foot_geom_ids:
            foot_contacts += 1
        else:
            nonfoot_contacts += 1
    return foot_contacts, nonfoot_contacts


def run(argv=None):
    args = parse_args(argv)
    if args.duration <= 0.0 or args.fps <= 0:
        raise SystemExit("--duration and --fps must be positive")
    if args.width <= 0 or args.height <= 0 or args.camera_distance <= 0.0:
        raise SystemExit("render dimensions and camera distance must be positive")

    task = walking_physics_profile_3d(args.physics_profile)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    validate_walking_morphology_3d(model, task.reference)
    apply_physics_options_3d(model, task)
    data = mujoco.MjData(model)
    (
        _,
        actuator_ids,
        joint_low,
        joint_high,
        startup_ctrl,
        phase,
    ) = _initialize_reference(model, data, task)

    floor_geom_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    foot_geom_ids = {
        _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in FOOT_GEOM_NAMES_3D
    }
    camera = _configure_camera(model, args)
    scene_option = mujoco.MjvOption()
    if not args.hide_contacts:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames: list[Image.Image] = []
    next_frame_time = 0.0
    frame_period = 1.0 / args.fps
    start_x = float(data.qpos[0])
    min_root_z = float(data.qpos[2])
    max_root_z = float(data.qpos[2])
    max_foot_contacts = 0
    max_nonfoot_contacts = 0
    control_step = 0

    try:
        while data.time < args.duration:
            elapsed_s = control_step * task.control_timestep
            reference_blend = float(
                smoothstep_ramp(np, np.asarray(elapsed_s), task.startup_reference_ramp_s)
            )
            reference = walking_reference_3d(np, phase, task.reference)
            target = np.clip(
                startup_ctrl
                + reference_blend
                * (reference["joint_targets"] - startup_ctrl),
                joint_low,
                joint_high,
            )
            data.ctrl[actuator_ids] = target

            for _ in range(task.action_repeat):
                if data.time >= args.duration:
                    break
                mujoco.mj_step(model, data)
                min_root_z = min(min_root_z, float(data.qpos[2]))
                max_root_z = max(max_root_z, float(data.qpos[2]))
                foot_count, nonfoot_count = _contact_counts(
                    model, data, floor_geom_id, foot_geom_ids
                )
                max_foot_contacts = max(max_foot_contacts, foot_count)
                max_nonfoot_contacts = max(max_nonfoot_contacts, nonfoot_count)
                if data.time + 1.0e-12 < next_frame_time:
                    continue
                renderer.update_scene(
                    data, camera=camera, scene_option=scene_option
                )
                frame = Image.fromarray(renderer.render())
                frames.append(
                    frame.quantize(
                        colors=128, method=Image.Quantize.FASTOCTREE
                    )
                )
                next_frame_time += frame_period

            phase = (
                phase
                + 2.0
                * math.pi
                * task.reference.frequency_hz
                * task.control_timestep
            ) % (2.0 * math.pi)
            control_step += 1
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No animation frames were rendered")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000.0 / args.fps),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return {
        "status": "ok",
        "output": str(args.output.resolve()),
        "duration_s": float(data.time),
        "frames": len(frames),
        "frequency_hz": task.reference.frequency_hz,
        "duty_factor": task.reference.duty_factor,
        "step_length_m": task.reference.step_length_m,
        "foot_lift_m": task.reference.foot_lift_m,
        "desired_speed_m_s": task.reference.desired_speed_m_s,
        "distance_x_m": float(data.qpos[0] - start_x),
        "average_speed_m_s": float((data.qpos[0] - start_x) / data.time),
        "root_z_range_m": [min_root_z, max_root_z],
        "max_foot_ground_contacts": max_foot_contacts,
        "max_nonfoot_ground_contacts": max_nonfoot_contacts,
    }


def main(argv=None):
    print(json.dumps(run(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
