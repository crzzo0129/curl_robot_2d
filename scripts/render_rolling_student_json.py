#!/usr/bin/env python3
"""Render the current RTNeural rolling STUDENT in CPU MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_rolling_student_5s import (
    CONTROL_DT,
    JOINT_NAMES,
    MODEL_JOINT_NAMES,
    PHYSICS_DT,
    SUBSTEPS,
    configure_training_physics,
    infer,
    initial_history,
    load_policy,
    policy_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=0.78)
    parser.add_argument("--azimuth", type=float, default=132.0)
    parser.add_argument("--elevation", type=float, default=-16.0)
    parser.add_argument("--show-contacts", action="store_true")
    parser.add_argument("--packages", type=Path)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    choices = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    )
    for choice in choices:
        if Path(choice).is_file():
            return ImageFont.truetype(choice, size=size)
    return ImageFont.load_default()


def overlay(
    pixels: np.ndarray,
    *,
    time_s: float,
    phase_turns: float,
    root_x: float,
    root_y: float,
    peak_torque: float,
    self_contacts: int,
    kp: float,
    kd: float,
) -> np.ndarray:
    frame = Image.fromarray(pixels)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle((18, 18, 430, 177), radius=12, fill=(12, 18, 29, 218))
    draw.text((36, 31), "ROLLING STUDENT", font=font(26, True), fill=(248, 250, 252, 255))
    draw.text(
        (36, 68),
        f"time  {time_s:4.2f} s     phase  {phase_turns:+5.2f} turns",
        font=font(18),
        fill=(225, 231, 239, 255),
    )
    draw.text(
        (36, 96),
        f"x  {root_x:+5.2f} m       lateral  {root_y:+6.3f} m",
        font=font(18),
        fill=(225, 231, 239, 255),
    )
    draw.text(
        (36, 124),
        f"peak |torque| {peak_torque:4.2f} N m   self contacts {self_contacts:d}",
        font=font(18),
        fill=(225, 231, 239, 255),
    )
    draw.text(
        (36, 151),
        f"Kp {kp:g} / Kd {kd:g}  |  50 Hz POLICY  |  FULL FIRST ACTION",
        font=font(13, True),
        fill=(120, 202, 255, 255),
    )
    return np.asarray(frame)


def main() -> None:
    args = parse_args()
    if args.packages is not None:
        sys.path.append(str(args.packages.resolve()))
    import imageio.v2 as imageio
    import mujoco

    policy, layers = load_policy(args.policy)
    kp = float(policy["kp"])
    kd = float(policy["kd"])
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    configure_training_physics(model, mujoco)
    data = mujoco.MjData(model)

    compact_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_servo")
            for name in MODEL_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    actuator_joint_ids = np.asarray(model.actuator_trnid[actuator_ids, 0], dtype=np.int32)
    joint_qpos_ids = np.asarray(model.jnt_qposadr[actuator_joint_ids], dtype=np.int32)
    if min(compact_key, torso_id, int(np.min(actuator_ids))) < 0:
        raise ValueError("model does not match rolling deployment names")

    # The JSON gains are part of the deployed policy contract.  Apply them to
    # the MuJoCo position servos so rendering follows the file being shown.
    model.actuator_gainprm[actuator_ids, 0] = kp
    model.actuator_biasprm[actuator_ids, 1] = -kp
    model.actuator_biasprm[actuator_ids, 2] = -kd

    mujoco.mj_resetDataKeyframe(model, data, compact_key)
    compact = np.asarray(policy["default_joint_pos"], dtype=np.float64)
    action_scale = np.asarray(policy["action_scale"], dtype=np.float64)
    joint_low = np.asarray(policy["joint_lower_limits"], dtype=np.float64)
    joint_high = np.asarray(policy["joint_upper_limits"], dtype=np.float64)
    data.ctrl[actuator_ids] = compact
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = torso_id
    camera.distance = args.camera_distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    scene_option = mujoco.MjvOption()
    if args.show_contacts:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    thumbnail = args.output.with_name(args.output.stem + "_preview.png")
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_params=["-movflags", "+faststart"],
    )

    history = initial_history()
    previous_action = np.zeros(12, dtype=np.float32)
    action = previous_action.copy()
    target = compact.copy()
    rolling_phase = 0.0
    next_frame_time = 0.0
    frame_interval = 1.0 / args.fps
    rendered_frames = 0
    peak_torque = 0.0
    thumbnail_saved = False

    def self_contact_count() -> int:
        count = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if floor_geom_id not in (int(contact.geom1), int(contact.geom2)):
                count += 1
        return count

    def render_frame() -> None:
        nonlocal rendered_frames, thumbnail_saved
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        pixels = overlay(
            renderer.render(),
            time_s=float(data.time),
            phase_turns=rolling_phase / (2.0 * math.pi),
            root_x=float(data.qpos[0]),
            root_y=float(data.qpos[1]),
            peak_torque=peak_torque,
            self_contacts=self_contact_count(),
            kp=kp,
            kd=kd,
        )
        writer.append_data(pixels)
        rendered_frames += 1
        if not thumbnail_saved and data.time >= 3.0:
            Image.fromarray(pixels).save(thumbnail)
            thumbnail_saved = True

    try:
        render_frame()
        next_frame_time += frame_interval
        policy_steps = int(math.ceil(args.duration / CONTROL_DT))
        for _ in range(policy_steps):
            frame = policy_frame(data, torso_id, joint_qpos_ids, compact, previous_action)
            history = np.concatenate((frame, history[:-36])).astype(np.float32)
            action = np.clip(infer(layers, history), -1.0, 1.0)
            target = np.clip(compact + action_scale * action, joint_low, joint_high)
            data.ctrl[actuator_ids] = target
            for _ in range(SUBSTEPS):
                if data.time + 0.5 * PHYSICS_DT >= args.duration:
                    break
                mujoco.mj_step(model, data)
                rolling_phase += PHYSICS_DT * float(data.qvel[4])
                peak_torque = max(
                    peak_torque,
                    float(np.max(np.abs(data.actuator_force[actuator_ids]))),
                )
                while data.time + 1e-12 >= next_frame_time:
                    render_frame()
                    next_frame_time += frame_interval
            previous_action = action
            if data.time + 0.5 * PHYSICS_DT >= args.duration:
                break
    finally:
        writer.close()
        renderer.close()

    metadata = {
        "output": str(args.output.resolve()),
        "thumbnail": str(thumbnail.resolve()),
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "kp": kp,
        "kd": kd,
        "duration_s": float(data.time),
        "fps": args.fps,
        "frames": rendered_frames,
        "phase_turns": rolling_phase / (2.0 * math.pi),
        "root_x_m": float(data.qpos[0]),
        "lateral_m": float(data.qpos[1]),
        "peak_abs_torque_Nm": peak_torque,
        "reset": "nominal compact; no noise",
        "fade_in": False,
        "contact_visualization": bool(args.show_contacts),
        "policy_frequency_Hz": 1.0 / CONTROL_DT,
        "physics_frequency_Hz": 1.0 / PHYSICS_DT,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
