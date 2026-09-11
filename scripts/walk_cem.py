"""Run a stand-start sinusoidal walker, or optimize it with offline CEM."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from curl_robot_2d.walking_cem import BOUNDS, MODEL_PATH, GaitParameters, SineWalkingController, rollout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "assets/controllers/walk_cem.json"
_model = None


def initialize_worker(path):
    global _model
    _model = mujoco.MjModel.from_xml_path(path)


def evaluate_worker(task):
    vector, duration, speed, hold, ramp = task
    controller = SineWalkingController(_model, GaitParameters.from_vector(vector), hold_s=hold, ramp_s=ramp,
                                      heading_feedback=True)
    return rollout(_model, controller, duration_s=duration, target_speed=speed)[0]


def load_policy(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "rollingquad-sine-cem-v1":
        raise ValueError("Unsupported controller format")
    p = GaitParameters(**payload["parameters"])
    GaitParameters.from_vector(p.vector())
    return p, payload


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_gif(model, trajectory, path):
    from PIL import Image
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 0.9, 135, -20
    frames = []
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        for frame_time in np.arange(0, trajectory["time"][-1], 0.05):
            i = min(np.searchsorted(trajectory["time"], frame_time), len(trajectory["time"]) - 1)
            data.qpos[:] = trajectory["qpos"][i]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.Palette.ADAPTIVE))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)


def optimize(args, parameters):
    rng = np.random.default_rng(args.seed)
    lower, upper = BOUNDS.T
    mean = (parameters.vector() - lower) / (upper - lower)
    std = np.full(len(mean), 0.18)
    best = parameters.vector()
    best_summary = None
    history = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(args.model.resolve()),)) as executor:
        for iteration in range(args.iterations):
            samples = np.clip(rng.normal(mean, std, (args.population, len(mean))), 0, 1)
            samples[0] = (best - lower) / (upper - lower)
            samples[1] = mean
            vectors = lower + samples * (upper - lower)
            summaries = list(executor.map(evaluate_worker,
                [(v, args.duration, args.speed, args.hold, args.ramp) for v in vectors]))
            scores = np.array([s["score"] for s in summaries])
            ranking = np.argsort(scores)[::-1]
            elites = samples[ranking[:max(2, round(args.population * args.elite_fraction))]]
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            std = np.maximum(0.035, 0.25 * std + 0.75 * elites.std(axis=0))
            winner = ranking[0]
            if best_summary is None or scores[winner] > best_summary["score"]:
                best, best_summary = vectors[winner].copy(), summaries[winner]
            entry = dict(iteration=iteration + 1, **best_summary)
            history.append(entry)
            print(f"CEM {iteration + 1}/{args.iterations}: score={best_summary['score']:.3f} "
                  f"x={best_summary['distance_x_m']:.3f} m tilt={best_summary['max_tilt_deg']:.1f} "
                  f"completed={best_summary['completed']}", flush=True)
            save_json(args.out / "best_controller.json", dict(
                format="rollingquad-sine-cem-v1", parameters=asdict(GaitParameters.from_vector(best)),
                hold_s=args.hold, ramp_s=args.ramp, target_speed_m_s=args.speed,
                heading_feedback=True,
                model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                seed=args.seed, population=args.population, iterations=iteration + 1,
                training=best_summary))
            save_json(args.out / "history.json", history)
    return GaitParameters.from_vector(best)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("run", "optimize"), default="run")
    parser.add_argument("--controller", choices=("cem", "sine"), default="cem")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--hold", type=float, default=None)
    parser.add_argument("--ramp", type=float, default=None)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/walk_cem")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--gif", type=Path, help="Render a 20 fps GIF of the evaluated rollout")
    args = parser.parse_args(argv)
    parameters, metadata = GaitParameters(), {}
    policy = args.policy
    if policy is None and args.mode == "run" and args.controller == "cem":
        policy = DEFAULT_POLICY
    if policy is not None:
        parameters, metadata = load_policy(policy)
        expected_hash = metadata.get("model_sha256")
        if expected_hash and expected_hash != hashlib.sha256(args.model.read_bytes()).hexdigest():
            parser.error("Policy model hash differs from --model; optimize a policy for this model")
    args.speed = args.speed if args.speed is not None else metadata.get("target_speed_m_s", 0.15)
    args.hold = args.hold if args.hold is not None else metadata.get("hold_s", 0.5)
    args.ramp = args.ramp if args.ramp is not None else metadata.get("ramp_s", 1.0)
    if (not np.isfinite([args.duration, args.speed, args.hold, args.ramp]).all()
            or args.hold < 0 or args.ramp <= 0 or args.duration <= args.hold + args.ramp or args.speed <= 0):
        parser.error("Require speed/ramp > 0, hold >= 0, duration > hold + ramp; all finite")
    if args.population < 4 or args.iterations < 1 or args.workers < 1 or not 0 < args.elite_fraction <= 0.5:
        parser.error("Require population >= 4, iterations/workers >= 1, 0 < elite-fraction <= 0.5")
    if args.mode == "optimize":
        parameters = optimize(args, parameters)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    controller = SineWalkingController(model, parameters, hold_s=args.hold, ramp_s=args.ramp,
        heading_feedback=args.mode == "optimize" or metadata.get("heading_feedback", False))
    kwargs = dict(duration_s=args.duration, target_speed=args.speed, record=True)
    if args.view:
        from mujoco import viewer as mj_viewer
        view_data = mujoco.MjData(model)
        controller.reset(view_data)
        with mj_viewer.launch_passive(model, view_data) as viewer:
            viewer.cam.distance = 1.0
            viewer.cam.elevation = -20
            wall_start = time.perf_counter()
            def update_view(m, d):
                if not viewer.is_running():
                    return False
                view_data.qpos[:] = d.qpos
                view_data.qvel[:] = d.qvel
                view_data.ctrl[:] = d.ctrl
                view_data.time = d.time
                mujoco.mj_forward(m, view_data)
                viewer.cam.lookat[:] = d.qpos[:3]
                viewer.sync()
                time.sleep(max(0, min(0.02, d.time - (time.perf_counter() - wall_start))))
                return True
            summary, trajectory = rollout(model, controller, callback=update_view, **kwargs)
    else:
        summary, trajectory = rollout(model, controller, **kwargs)
    save_json(args.out / "evaluation.json", dict(parameters=asdict(parameters), **summary))
    np.savez_compressed(args.out / "rollout.npz", **trajectory)
    if args.gif:
        render_gif(model, trajectory, args.gif)
    print(json.dumps(summary, indent=2))
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
