"""Render a saved deterministic MJX policy rollout without loading JAX."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from curl_robot_2d_mjx.environment import MODEL_PATH
from curl_robot_2d_mjx.runtime import select_mujoco_gl_backend


def _load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as rollout:
        if "qpos" not in rollout:
            raise ValueError(f"{path} does not contain a qpos array")
        qpos = np.asarray(rollout["qpos"])
        reward = np.asarray(
            rollout["reward"]
            if "reward" in rollout
            else np.zeros(qpos.shape[0])
        )
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError("qpos must be a non-empty [time, nq] array")
    if reward.shape != (qpos.shape[0],):
        raise ValueError("reward must have one value per qpos sample")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or infinity")
    return qpos, reward


def _frame_indices(
    sample_count: int, *, control_dt: float, fps: int
) -> np.ndarray:
    if sample_count <= 0 or control_dt <= 0.0 or fps <= 0:
        raise ValueError("sample_count, control_dt and fps must be positive")
    duration = (sample_count - 1) * control_dt
    frame_times = np.arange(0.0, duration + 1e-12, 1.0 / fps)
    indices = np.unique(
        np.minimum(np.rint(frame_times / control_dt), sample_count - 1)
        .astype(int)
    )
    if indices[-1] != sample_count - 1:
        indices = np.append(indices, sample_count - 1)
    return indices


def render_rollout(
    rollout_path: Path,
    output_path: Path,
    *,
    control_dt: float,
    fps: int,
    width: int,
    height: int,
    camera_distance: float,
    diagnostics: bool,
) -> dict[str, float | int | str]:
    import mujoco

    qpos, reward = _load_rollout(rollout_path)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"rollout nq={qpos.shape[1]} does not match model nq={model.nq}"
        )

    root_x_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_x"
    )
    root_pitch_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch"
    )
    torso_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso"
    )
    root_x_qpos = int(model.jnt_qposadr[root_x_joint])
    root_pitch_qpos = int(model.jnt_qposadr[root_pitch_joint])

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = torso_body
    camera.azimuth = 90.0
    camera.elevation = 0.0
    camera.distance = camera_distance
    scene_option = mujoco.MjvOption()
    if diagnostics:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    indices = _frame_indices(
        qpos.shape[0], control_dt=control_dt, fps=fps
    )
    initial_phase = float(qpos[0, root_pitch_qpos])
    initial_x = float(qpos[0, root_x_qpos])
    cumulative_reward = np.cumsum(reward)
    frames: list[Image.Image] = []
    try:
        for index in indices:
            data.qpos[:] = qpos[index]
            data.time = index * control_dt
            mujoco.mj_forward(model, data)
            renderer.update_scene(
                data, camera=camera, scene_option=scene_option
            )
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            turns = (
                float(data.qpos[root_pitch_qpos]) - initial_phase
            ) / (2.0 * math.pi)
            displacement = float(data.qpos[root_x_qpos]) - initial_x
            draw.rectangle((12, 10, 300, 78), fill=(20, 26, 35))
            draw.text(
                (22, 18),
                f"time {data.time:4.2f} s   roll {turns:6.2f} turns",
                fill=(244, 247, 251),
            )
            draw.text(
                (22, 39),
                f"x {displacement:+6.2f} m   reward {cumulative_reward[index]:+.1f}",
                fill=(190, 201, 214),
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
        duration=round(1000.0 / fps),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return {
        "output": str(output_path),
        "samples": int(qpos.shape[0]),
        "frames": len(frames),
        "duration_s": float((qpos.shape[0] - 1) * control_dt),
        "net_turns": float(
            (qpos[-1, root_pitch_qpos] - initial_phase)
            / (2.0 * math.pi)
        ),
        "root_x_displacement_m": float(
            qpos[-1, root_x_qpos] - initial_x
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=0.75)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa"),
        default="auto",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mujoco_gl == "auto":
        os.environ["MUJOCO_GL"] = select_mujoco_gl_backend()
    else:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    output = args.output or args.rollout.with_name("policy_rollout.gif")
    summary = render_rollout(
        args.rollout,
        output,
        control_dt=args.control_dt,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        diagnostics=args.diagnostics,
    )
    for name, value in summary.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
