"""Render a saved 3-D MJX policy rollout without loading JAX."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.config_3d import (
    PHYSICS_PROFILE_NAMES_3D,
    Rolling3DConfig,
    physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    MODEL_PATH_3D,
    apply_physics_options_3d,
)
from curl_robot_2d_mjx.runtime import select_mujoco_gl_backend


def _load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as rollout:
        if "qpos" not in rollout:
            raise ValueError(f"{path} does not contain a qpos array")
        qpos = np.asarray(rollout["qpos"], dtype=np.float64)
        reward = np.asarray(
            rollout["reward"]
            if "reward" in rollout
            else np.zeros(qpos.shape[0]),
            dtype=np.float64,
        )
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError("qpos must be a non-empty [time, nq] array")
    if reward.shape != (qpos.shape[0],):
        raise ValueError("reward must have one value per qpos sample")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or infinity")
    if not np.isfinite(reward).all():
        raise ValueError("reward contains NaN or infinity")
    return qpos, reward


def _frame_indices(
    sample_count: int, *, control_dt: float, fps: float
) -> np.ndarray:
    if sample_count <= 0 or control_dt <= 0.0 or fps <= 0.0:
        raise ValueError("sample_count, control_dt and fps must be positive")
    duration = (sample_count - 1) * control_dt
    frame_times = np.arange(0.0, duration + 1e-12, 1.0 / fps)
    indices = np.unique(
        np.minimum(np.rint(frame_times / control_dt), sample_count - 1).astype(
            int
        )
    )
    if indices[-1] != sample_count - 1:
        indices = np.append(indices, sample_count - 1)
    return indices


def _rolling_axis_tilt(rotation: np.ndarray) -> float:
    body_y_axis = rotation[:, 1]
    alignment = float(np.clip(abs(body_y_axis[1]), 0.0, 1.0))
    return math.acos(alignment)


def _pitch_angle(rotation: np.ndarray) -> float:
    return math.atan2(float(-rotation[2, 0]), float(rotation[0, 0]))


def _rollout_orientation_diagnostics(
    model, data, torso_body: int, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    pitch = []
    tilt = []
    for row in qpos:
        data.qpos[:] = row
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        rotation = np.asarray(data.xmat[torso_body]).reshape(3, 3)
        pitch.append(_pitch_angle(rotation))
        tilt.append(_rolling_axis_tilt(rotation))
    return np.unwrap(np.asarray(pitch)), np.asarray(tilt)


def render_rollout(
    rollout_path: Path,
    output_path: Path,
    *,
    model_path: Path = MODEL_PATH_3D,
    physics_profile: str,
    control_dt: float,
    fps: float,
    width: int,
    height: int,
    camera_distance: float,
    azimuth: float,
    elevation: float,
    diagnostics: bool,
) -> dict[str, float | int | str]:
    import mujoco

    qpos, reward = _load_rollout(rollout_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    task = physics_profile_3d(physics_profile, Rolling3DConfig())
    apply_physics_options_3d(model, task)
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"rollout nq={qpos.shape[1]} does not match model nq={model.nq}"
        )

    torso_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso"
    )
    if torso_body < 0:
        raise ValueError("missing MuJoCo body: torso")

    data = mujoco.MjData(model)
    pitch, axis_tilt = _rollout_orientation_diagnostics(
        model, data, torso_body, qpos
    )
    indices = _frame_indices(qpos.shape[0], control_dt=control_dt, fps=fps)
    initial_x = float(qpos[0, 0])
    initial_y = float(qpos[0, 1])
    turn_radius = 2.0 * math.pi * FIXED_PARAMETERS.shell_contact_radius
    cumulative_reward = np.cumsum(reward)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = torso_body
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = camera_distance
    scene_option = mujoco.MjvOption()
    if diagnostics:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    frames: list[Image.Image] = []
    try:
        for index in indices:
            data.qpos[:] = qpos[index]
            data.qvel[:] = 0.0
            data.time = index * control_dt
            mujoco.mj_forward(model, data)
            renderer.update_scene(
                data, camera=camera, scene_option=scene_option
            )
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            x_displacement = float(qpos[index, 0] - initial_x)
            lateral = float(qpos[index, 1] - initial_y)
            translation_turns = x_displacement / max(turn_radius, 1e-9)
            pitch_turns = float((pitch[index] - pitch[0]) / (2.0 * math.pi))
            draw.rectangle((12, 10, 430, 104), fill=(18, 22, 30))
            draw.text(
                (22, 18),
                f"time {data.time:4.2f}s  shell {translation_turns:+6.2f} turns",
                fill=(244, 247, 251),
            )
            draw.text(
                (22, 40),
                f"pitch {pitch_turns:+6.2f} turns  x {x_displacement:+6.3f} m",
                fill=(206, 216, 228),
            )
            draw.text(
                (22, 62),
                f"y {lateral:+6.3f} m  tilt {axis_tilt[index]:.4f} rad",
                fill=(206, 216, 228),
            )
            draw.text(
                (22, 84),
                f"reward {cumulative_reward[index]:+.1f}",
                fill=(206, 216, 228),
            )
            frames.append(
                frame.quantize(
                    colors=128, method=Image.Quantize.FASTOCTREE
                )
            )
    finally:
        renderer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000.0 / fps)),
        loop=0,
        optimize=False,
        disposal=2,
    )

    final_translation_turns = float((qpos[-1, 0] - initial_x) / turn_radius)
    final_pitch_turns = float((pitch[-1] - pitch[0]) / (2.0 * math.pi))
    return {
        "output": str(output_path),
        "samples": int(qpos.shape[0]),
        "frames": len(frames),
        "duration_s": float((qpos.shape[0] - 1) * control_dt),
        "physics_profile": task.physics_profile,
        "root_x_displacement_m": float(qpos[-1, 0] - initial_x),
        "final_lateral_drift_m": float(qpos[-1, 1] - initial_y),
        "translation_equivalent_turns": final_translation_turns,
        "pitch_turns": final_pitch_turns,
        "conservative_turns": float(
            min(final_translation_turns, abs(final_pitch_turns))
        ),
        "axis_tilt_rms_rad": float(np.sqrt(np.mean(np.square(axis_tilt)))),
        "axis_tilt_max_rad": float(np.max(axis_tilt)),
        "total_reward": float(np.sum(reward)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--model-xml", type=Path, default=MODEL_PATH_3D)
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILE_NAMES_3D,
        default="cg20",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=0.9)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa"),
        default="auto",
    )
    args = parser.parse_args(argv)
    if args.control_dt <= 0.0 or args.fps <= 0.0:
        parser.error("--control-dt and --fps must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mujoco_gl == "auto":
        os.environ["MUJOCO_GL"] = select_mujoco_gl_backend()
    else:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    if os.environ["MUJOCO_GL"] not in ("", "disable"):
        os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
    output = args.output or args.rollout.with_name("policy_rollout.gif")
    summary = render_rollout(
        args.rollout,
        output,
        model_path=args.model_xml,
        physics_profile=args.physics_profile,
        control_dt=args.control_dt,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        azimuth=args.azimuth,
        elevation=args.elevation,
        diagnostics=args.diagnostics,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
