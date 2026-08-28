"""Render a saved 3-D MJX policy rollout without loading JAX."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d_mjx.config_3d import (
    GEOMETRY_NAMES_3D,
    PHYSICS_PROFILE_NAMES_3D,
    Rolling3DConfig,
    physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d,
    geometry_parameters_3d,
    model_path_3d,
)
from curl_robot_2d_mjx.runtime import select_mujoco_gl_backend


FAILURE_METRICS = (
    "failure_nonfinite",
    "failure_nonfinite_action",
    "failure_nonfinite_physics",
    "failure_root_low",
    "failure_root_high",
    "failure_lateral_drift",
    "failure_axis_tilt",
    "failure_forbidden_depth",
    "failure_forbidden_contact",
)


def _load_rollout(path: Path) -> dict[str, np.ndarray]:
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
        values = {name: np.asarray(rollout[name]) for name in rollout.files}
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError("qpos must be a non-empty [time, nq] array")
    if reward.shape != (qpos.shape[0],):
        raise ValueError("reward must have one value per qpos sample")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or infinity")
    if not np.isfinite(reward).all():
        raise ValueError("reward contains NaN or infinity")
    values["qpos"] = qpos
    values["reward"] = reward
    return values


def _optional_series(
    rollout: dict[str, np.ndarray], name: str, sample_count: int
) -> np.ndarray | None:
    if name not in rollout:
        return None
    values = np.asarray(rollout[name], dtype=np.float64)
    if values.shape != (sample_count,):
        raise ValueError(f"{name} must have one value per qpos sample")
    return values


def _scalar_text(rollout: dict[str, np.ndarray], name: str) -> str | None:
    if name not in rollout:
        return None
    value = np.asarray(rollout[name])
    if value.shape != ():
        return None
    return str(value.item())


def _failure_at(rollout: dict[str, np.ndarray], index: int) -> str | None:
    for metric in FAILURE_METRICS:
        if metric in rollout and float(np.asarray(rollout[metric])[index]) > 0.5:
            return metric.removeprefix("failure_")
    return None


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
    model_path: Path | None = None,
    geometry: str = "rollingquad_2",
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

    rollout = _load_rollout(rollout_path)
    qpos = rollout["qpos"]
    reward = rollout["reward"]
    selected_model_path = model_path or model_path_3d(geometry)
    model = mujoco.MjModel.from_xml_path(str(selected_model_path))
    task = physics_profile_3d(
        physics_profile, Rolling3DConfig(geometry=geometry)
    )
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
    turn_radius = (
        2.0
        * math.pi
        * geometry_parameters_3d(geometry).shell_contact_radius
    )
    cumulative_reward = np.cumsum(reward)
    lateral_series = _optional_series(
        rollout, "lateral_drift_m", qpos.shape[0]
    )
    if lateral_series is None:
        lateral_series = qpos[:, 1] - initial_y
    residual_rms = _optional_series(
        rollout, "residual_action_rms", qpos.shape[0]
    )
    differential_rms = _optional_series(
        rollout, "differential_residual_rms", qpos.shape[0]
    )
    mode = _scalar_text(rollout, "mode")
    seed_index = _scalar_text(rollout, "seed_index")
    title_parts = [
        part
        for part in (mode, f"seed {seed_index}" if seed_index else None)
        if part
    ]
    title = " | ".join(title_parts)

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
            lateral = float(lateral_series[index])
            max_abs_lateral = float(
                np.max(np.abs(lateral_series[: index + 1]))
            )
            translation_turns = x_displacement / max(turn_radius, 1e-9)
            pitch_turns = float((pitch[index] - pitch[0]) / (2.0 * math.pi))
            failure = _failure_at(rollout, index)
            panel_bottom = 148 if residual_rms is not None else 126
            draw.rectangle((12, 10, 470, panel_bottom), fill=(18, 22, 30))
            if title:
                draw.text((22, 16), title, fill=(244, 247, 251))
                first_row = 38
            else:
                first_row = 18
            draw.text(
                (22, first_row),
                f"time {data.time:4.2f}s  shell {translation_turns:+6.2f} turns",
                fill=(244, 247, 251),
            )
            draw.text(
                (22, first_row + 22),
                f"pitch {pitch_turns:+6.2f} turns  x {x_displacement:+6.3f} m",
                fill=(206, 216, 228),
            )
            draw.text(
                (22, first_row + 44),
                f"y {lateral:+6.3f} m  max |y| {max_abs_lateral:.3f} m",
                fill=(206, 216, 228),
            )
            draw.text(
                (22, first_row + 66),
                f"tilt {axis_tilt[index]:.4f} rad  "
                f"reward {cumulative_reward[index]:+.1f}",
                fill=(206, 216, 228),
            )
            if residual_rms is not None:
                differential = (
                    float(differential_rms[index])
                    if differential_rms is not None
                    else 0.0
                )
                draw.text(
                    (22, first_row + 88),
                    f"residual {residual_rms[index]:.4f}  "
                    f"differential {differential:.4f}",
                    fill=(206, 216, 228),
                )
            if failure:
                draw.text(
                    (300, first_row),
                    f"FAIL {failure}",
                    fill=(255, 116, 104),
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
        "final_lateral_drift_m": float(lateral_series[-1]),
        "max_abs_lateral_drift_m": float(np.max(np.abs(lateral_series))),
        "lateral_path_m": float(
            np.sum(np.abs(np.diff(np.r_[0.0, lateral_series])))
        ),
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
    parser.add_argument("--model-xml", type=Path)
    parser.add_argument(
        "--geometry", choices=GEOMETRY_NAMES_3D, default="rollingquad_2"
    )
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
    if args.rollout.is_dir():
        rollout_paths = sorted(args.rollout.glob("*.npz"))
        if not rollout_paths:
            raise SystemExit(f"No .npz rollouts found in {args.rollout}")
        output_dir = args.output or (args.rollout / "rendered")
        if output_dir.suffix:
            raise SystemExit("--output must be a directory for directory input")
        summaries = []
        for rollout_path in rollout_paths:
            summaries.append(
                render_rollout(
                    rollout_path,
                    output_dir / f"{rollout_path.stem}.gif",
                    model_path=args.model_xml,
                    geometry=args.geometry,
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
            )
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return

    output = args.output or args.rollout.with_name("policy_rollout.gif")
    summary = render_rollout(
        args.rollout,
        output,
        model_path=args.model_xml,
        geometry=args.geometry,
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
