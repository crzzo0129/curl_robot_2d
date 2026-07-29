"""Replay the saved active rolling controller in a viewer or animated GIF."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scripts.optimize_phase_controller import (
    FOOT_GAP_TRACKING_MARGIN_M,
    PARAMETER_NAMES,
    controller_targets,
)
from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts.run_release_baseline import JOINT_TARGETS, MODEL_PATH, _id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase_controller"
    / "best_phase_controller.json"
)
DEFAULT_GIF_PATH = (
    PROJECT_ROOT / "results" / "phase_controller" / "active_roll.gif"
)


def load_controller(path: Path) -> tuple[np.ndarray, float, float]:
    return load_controller_options(path)[:3]


def load_controller_options(
    path: Path,
) -> tuple[np.ndarray, float, float, float, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["raw_coefficients"]
    coefficients = np.asarray([raw[name] for name in PARAMETER_NAMES], dtype=float)
    return (
        coefficients,
        float(payload["oscillator_rate_rad_s"]),
        float(payload["oscillator_coupling_per_s"]),
        float(payload.get("minimum_foot_surface_gap_m", 0.0)),
        float(
            payload.get(
                "foot_gap_tracking_margin_m", FOOT_GAP_TRACKING_MARGIN_M
            )
        ),
        float(payload.get("nominal_knee_bias_rad", 0.0)),
    )


def initialize_simulation(
    model: mujoco.MjModel,
    minimum_foot_surface_gap_m: float = 0.0,
) -> tuple[mujoco.MjData, int, list[int]]:
    data = mujoco.MjData(model)
    compact_key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
    root_pitch_joint_id = _id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch"
    )
    root_pitch_qpos_address = int(model.jnt_qposadr[root_pitch_joint_id])
    actuator_ids = [
        _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_servo")
        for joint_name, _ in JOINT_TARGETS
    ]

    mujoco.mj_resetDataKeyframe(model, data, compact_key_id)
    if minimum_foot_surface_gap_m > 0.0:
        separated = replace(
            FIXED_PARAMETERS,
            compact_foot_surface_gap=minimum_foot_surface_gap_m,
        )
        root_z_joint = _id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "root_z"
        )
        data.qpos[int(model.jnt_qposadr[root_z_joint])] = (
            separated.compact_root_height
        )
        for joint_name in ("front_knee", "rear_knee"):
            joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[int(model.jnt_qposadr[joint_id])] = (
                separated.compact_knee_angle
            )
    data.qvel[:] = 0.0
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    for root_joint_name in ("root_x", "root_z", "root_pitch"):
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
        model.dof_damping[int(model.jnt_dofadr[joint_id])] = 0.0
    mujoco.mj_forward(model, data)
    return data, root_pitch_qpos_address, actuator_ids


def advance_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    coefficients: np.ndarray,
    oscillator_rate: float,
    oscillator_coupling: float,
    oscillator_phase: float,
    root_pitch_qpos_address: int,
    minimum_foot_surface_gap_m: float = 0.0,
    foot_gap_tracking_margin_m: float = FOOT_GAP_TRACKING_MARGIN_M,
    knee_bias_rad: float = 0.0,
) -> float:
    phase = float(data.qpos[root_pitch_qpos_address])
    oscillator_phase_rate = oscillator_rate + oscillator_coupling * math.sin(
        phase - oscillator_phase
    )
    oscillator_phase += float(model.opt.timestep) * max(
        0.1, oscillator_phase_rate
    )
    data.ctrl[:] = controller_targets(
        phase,
        float(data.time),
        coefficients,
        oscillator_rate=oscillator_rate,
        control_phase=oscillator_phase,
        knee_bias_rad=knee_bias_rad,
        minimum_foot_surface_gap_m=minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=foot_gap_tracking_margin_m,
    )
    mujoco.mj_step(model, data)
    return oscillator_phase


def configure_tracking_camera(
    model: mujoco.MjModel,
    camera: mujoco.MjvCamera,
    *,
    distance: float,
) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    camera.azimuth = 90.0
    camera.elevation = 0.0
    camera.distance = distance


def render_gif(
    controller_path: Path,
    output_path: Path,
    *,
    duration: float,
    fps: int,
    width: int,
    height: int,
    camera_distance: float,
    diagnostics: bool,
) -> None:
    (
        coefficients,
        oscillator_rate,
        oscillator_coupling,
        minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m,
        knee_bias_rad,
    ) = load_controller_options(controller_path)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data, root_pitch_qpos_address, _ = initialize_simulation(
        model, minimum_foot_surface_gap_m
    )
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    configure_tracking_camera(model, camera, distance=camera_distance)
    scene_option = mujoco.MjvOption()
    if diagnostics:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    frames: list[Image.Image] = []
    oscillator_phase = 0.0
    next_frame_time = 0.0
    frame_period = 1.0 / fps
    try:
        while data.time < duration:
            oscillator_phase = advance_controller(
                model,
                data,
                coefficients,
                oscillator_rate,
                oscillator_coupling,
                oscillator_phase,
                root_pitch_qpos_address,
                minimum_foot_surface_gap_m,
                foot_gap_tracking_margin_m,
                knee_bias_rad,
            )
            if data.time + 1e-12 < next_frame_time:
                continue
            renderer.update_scene(
                data, camera=camera, scene_option=scene_option
            )
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            turns = float(data.qpos[root_pitch_qpos_address]) / (2.0 * math.pi)
            draw.rectangle((12, 10, 245, 58), fill=(20, 26, 35))
            draw.text(
                (22, 18),
                f"time {data.time:4.1f} s   roll {turns:5.2f} turns",
                fill=(244, 247, 251),
            )
            draw.text(
                (22, 37),
                f"x = {float(data.qpos[0]):5.2f} m",
                fill=(190, 201, 214),
            )
            frames.append(
                frame.quantize(colors=128, method=Image.Quantize.FASTOCTREE)
            )
            next_frame_time += frame_period
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No animation frames were rendered")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000.0 / fps),
        loop=0,
        optimize=False,
        disposal=2,
    )


def launch_viewer(
    controller_path: Path,
    *,
    duration: float,
    camera_distance: float,
) -> None:
    import mujoco.viewer

    (
        coefficients,
        oscillator_rate,
        oscillator_coupling,
        minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m,
        knee_bias_rad,
    ) = load_controller_options(controller_path)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data, root_pitch_qpos_address, _ = initialize_simulation(
        model, minimum_foot_surface_gap_m
    )
    oscillator_phase = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_tracking_camera(
            model, viewer.cam, distance=camera_distance
        )
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running() and data.time < duration:
            start = time.perf_counter()
            oscillator_phase = advance_controller(
                model,
                data,
                coefficients,
                oscillator_rate,
                oscillator_coupling,
                oscillator_phase,
                root_pitch_qpos_address,
                minimum_foot_surface_gap_m,
                foot_gap_tracking_margin_m,
                knee_bias_rad,
            )
            viewer.sync()
            remaining = float(model.opt.timestep) - (
                time.perf_counter() - start
            )
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER_PATH)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--camera-distance", type=float, default=0.75)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_GIF_PATH)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    if args.viewer:
        launch_viewer(
            args.controller,
            duration=args.duration,
            camera_distance=args.camera_distance,
        )
    else:
        render_gif(
            args.controller,
            args.output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera_distance=args.camera_distance,
            diagnostics=args.diagnostics,
        )
        print(args.output)


if __name__ == "__main__":
    main()
