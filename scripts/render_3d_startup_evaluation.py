"""Render one saved best-startup evaluation trajectory without loading JAX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.render_mjx_3d_policy import render_rollout


def _episode_trace(arrays_path: Path, episode: int) -> dict[str, np.ndarray]:
    terminal_failure = None
    with np.load(arrays_path) as source:
        key = "qpos_first_episodes"
        if key not in source:
            raise ValueError(f"{arrays_path} does not contain {key}")
        qpos = np.asarray(source[key])
        if qpos.ndim != 3 or not 0 <= episode < qpos.shape[1]:
            raise ValueError(f"episode must be in [0,{qpos.shape[1] - 1}]")
        trace = {"qpos": qpos[:, episode]}
        for source_name, output_name in (
            ("gate_error_first_episodes", "gate_error"),
            ("teacher_active_next_first_episodes", "teacher_active"),
        ):
            if source_name in source:
                value = np.asarray(source[source_name])
                trace[output_name] = value[:, episode]
        for name in ("nonfinite", "root_low", "root_high", "lateral_drift",
                     "axis_tilt", "forbidden_depth", "forbidden_contact"):
            key = f"failure_{name}"
            if key in source and float(np.asarray(source[key])[episode]) > .5:
                terminal_failure = key
                break

    # evaluate_startup freezes a terminal state for the rest of its fixed loop.
    # Keep the real terminal transition, discard the repeated frozen tail.
    identical = np.all(trace["qpos"][1:] == trace["qpos"][:-1], axis=1)
    repeated = np.flatnonzero(identical)
    if repeated.size:
        length = int(repeated[0] + 1)
        trace = {name: value[:length] for name, value in trace.items()}
    trace["reward"] = np.zeros(len(trace["qpos"]), dtype=np.float32)
    if terminal_failure:
        failure = np.zeros(len(trace["qpos"]), dtype=np.float32)
        failure[-1] = 1.
        trace[terminal_failure] = failure
    trace["mode"] = np.asarray("startup evaluation_best")
    trace["seed_index"] = np.asarray(episode)
    return trace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="startup PPO result directory")
    parser.add_argument("--episode", type=int, default=0,
                        help="saved evaluation episode, normally 0..3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=25.)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--mujoco-gl", choices=("auto", "egl", "glfw", "osmesa"), default="egl")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = args.result.resolve()
    arrays_path = result / "evaluation_best_arrays.npz"
    config_path = result / "training_config.json"
    if not arrays_path.is_file() or not config_path.is_file():
        raise SystemExit("result must contain evaluation_best_arrays.npz and training_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = config["task"]
    output = args.output or result / f"evaluation_best_episode{args.episode}.gif"
    rollout_path = result / f"evaluation_best_episode{args.episode}_render_trace.npz"
    np.savez_compressed(rollout_path, **_episode_trace(arrays_path, args.episode))

    import os
    from curl_robot_2d_mjx.runtime import select_mujoco_gl_backend
    os.environ["MUJOCO_GL"] = (select_mujoco_gl_backend()
                               if args.mujoco_gl == "auto" else args.mujoco_gl)
    if os.environ["MUJOCO_GL"]:
        os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
    summary = render_rollout(rollout_path, output,
        geometry=task.get("geometry", "rollingquad_2"),
        physics_profile=task.get("physics_profile", "cg20"),
        control_dt=float(task.get("control_timestep", .02)), fps=args.fps,
        width=960, height=720, camera_distance=0.8, azimuth=135., elevation=-18.,
        diagnostics=args.diagnostics)
    summary.update(source=str(arrays_path), episode=args.episode,
                   note="terminal inferred from the first exactly repeated frozen state")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
