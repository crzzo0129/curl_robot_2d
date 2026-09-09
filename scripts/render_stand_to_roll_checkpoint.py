"""Replay a stand-to-roll PPO checkpoint in MJX and render the actual trajectory."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bc-params", type=Path, required=True)
    parser.add_argument("--cem-data", type=Path, default=Path("results/cem_cycle_data/cem_cycles.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/full_stand_video"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-eval", action="store_true", help="Evaluate loads without video or parameter updates")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--limit-torque", action="store_true", help="Override checkpoint task with a 3 Nm cap and 2 Nm penalty")
    parser.add_argument("--mujoco-gl", default="egl", choices=("egl", "osmesa", "glfw"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    checkpoint = args.checkpoint.resolve()
    config_path = next((p / "training_config.json" for p in (checkpoint, *checkpoint.parents)
                        if (p / "training_config.json").is_file()), None)
    if config_path is None:
        parser.error("Cannot find training_config.json above checkpoint")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("pipeline") != "one_policy_bc_then_snapshot_reset_curriculum_v2":
        parser.error("Expected a stand-to-roll v2 checkpoint")
    if config.get("bc_sha256") != hashlib.sha256(args.bc_params.read_bytes()).hexdigest():
        parser.error("BC file must match the training normalizer")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty; choose a new --out")

    from curl_robot_2d_mjx.runtime import configure_cloud_runtime
    configure_cloud_runtime(preallocate=False, mujoco_gl=args.mujoco_gl)
    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.io import model as model_io
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks
    from curl_robot_2d_mjx.config_stand_to_roll import StandToRollConfig
    from curl_robot_2d_mjx.environment_stand_to_roll_3d import make_stand_to_roll_env_3d
    from curl_robot_2d_mjx.stand_to_roll_training import preprocess_observation

    normalizer, _ = model_io.load_params(args.bc_params)
    def preprocess(obs, unused):
        return preprocess_observation(jp, obs, normalizer)

    hidden = tuple(config["arguments"]["hidden_layers"])
    networks = ppo_networks.make_ppo_networks(
        720, 12, preprocess_observations_fn=preprocess,
        policy_hidden_layer_sizes=hidden, value_hidden_layer_sizes=hidden,
        activation=jax.nn.elu, distribution_type="tanh_normal")
    params = ppo_checkpoint.load(str(checkpoint))
    policy = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
    task = replace(StandToRollConfig(**config["task"]), observation_noise_enabled=False)
    if args.limit_torque:
        task = replace(task, torque_hard_limit_nm=3.0, torque_soft_limit_nm=2.0,
                       reward_torque_excess=0.1, reward_lateral_before_capture=0.0,
                       reward_handoff_y=0.1, reward_handoff_vy=0.1, reward_handoff_axis=0.1)
    if args.load_eval or args.limit_torque:
        task = replace(task, load_diagnostics=True)
    env = make_stand_to_roll_env_3d(task, matcher_npz=args.cem_data, seed=args.seed)
    if args.load_eval:
        from dataclasses import asdict
        from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D
        print(f"Evaluating {args.episodes} episodes; torque cap={task.torque_hard_limit_nm} Nm "
              "(0 means original model limits). Compiling...", flush=True)
        state = jax.jit(jax.vmap(env.reset))(
            jax.random.split(jax.random.PRNGKey(args.seed), args.episodes))
        step_batch = jax.jit(jax.vmap(env.step))
        alive = jp.ones(args.episodes, dtype=bool)
        totals = {name: jp.zeros(args.episodes) for name in state.metrics}
        for i in range(task.episode_length):
            action, _ = policy(state.obs, jax.random.PRNGKey(args.seed + i + 1))
            state = step_batch(state, action)
            totals = {name: total + jp.where(alive, state.metrics[name], 0.0)
                      for name, total in totals.items()}
            alive = alive & (~state.done.astype(bool))
            if not bool(jp.any(alive)):
                break
        values = {name: np.asarray(value) for name, value in totals.items()}
        duration = max(float(values["load_duration_s"].sum()), 1e-12)
        joints = {}
        for i, name in enumerate(CONTROLLER_JOINT_NAMES_3D):
            joints[name] = {
                "peak_nm": float(values[f"torque_{i}_peak_nm"].max()),
                "rms_nm": float(np.sqrt(values[f"torque_{i}_square_integral"].sum() / duration)),
                "over_2nm_fraction": float(values[f"torque_{i}_over2_s"].sum() / duration),
                "at_3nm_fraction": float(values[f"torque_{i}_at3_s"].sum() / duration),
            }
        report = {
            "checkpoint": str(checkpoint), "seed": args.seed, "episodes": args.episodes,
            "task": asdict(task), "capture_rate": float(values["captured"].mean()),
            "insurance_rate": float(values["insurance_success"].mean()),
            "failure_rate": float(values["failed"].mean()), "joints": joints,
            "single_contact_peak_n": float(values["contact_peak_n"].max()),
            "total_ground_normal_peak_n": float(values["contact_total_peak_n"].max()),
            "mean_episode_ground_normal_impulse_ns": float(values["contact_normal_impulse_ns"].mean()),
            "finite_loads": all(bool(np.isfinite(value).all()) for value in values.values()),
            "contact_note": "Solver normal forces sampled each physics step; impulse includes body support, not just impacts. No contact-force safety threshold applied.",
        }
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out / "episodes.npz", **values)
        (args.out / "load_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Capture {report['capture_rate']:.1%} | Insurance {report['insurance_rate']:.1%} "
              f"| Failed {report['failure_rate']:.1%}")
        print(f"{'Joint':28s} {'Peak Nm':>8s} {'RMS Nm':>8s} {'>2 Nm':>8s} {'>=2.99':>8s}")
        for name, row in joints.items():
            print(f"{name:28s} {row['peak_nm']:8.3f} {row['rms_nm']:8.3f} "
                  f"{row['over_2nm_fraction']:8.1%} {row['at_3nm_fraction']:8.1%}")
        print(f"Ground peak: single contact {report['single_contact_peak_n']:.1f} N | "
              f"total {report['total_ground_normal_peak_n']:.1f} N")
        print(f"Report: {args.out / 'load_report.json'}", flush=True)
        return
    print("Replaying checkpoint in MJX (first compilation may take several minutes)...", flush=True)
    state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    step = jax.jit(env.step)
    rows = [np.asarray(state.pipeline_state.qpos)]
    rewards = [0.0]
    metrics = {name: [float(value)] for name, value in state.metrics.items()}
    for i in range(task.episode_length):
        action, _ = policy(state.obs, jax.random.PRNGKey(args.seed + i + 1))
        state = step(state, action)
        rows.append(np.asarray(state.pipeline_state.qpos))
        rewards.append(float(state.reward))
        for name, value in state.metrics.items():
            metrics[name].append(float(value))
        if bool(state.done):
            break
    args.out.mkdir(parents=True, exist_ok=True)
    rollout = args.out / "rollout.npz"
    arrays = {name: np.asarray(values) for name, values in metrics.items()}
    arrays.update(qpos=np.asarray(rows), reward=np.asarray(rewards))
    np.savez_compressed(rollout, **arrays)
    report = {name: float(np.sum(values[1:])) for name, values in metrics.items()}
    report.update(checkpoint=str(checkpoint), seed=args.seed, duration_s=(len(rows)-1)*task.control_timestep)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Rendering rollout.gif...", flush=True)
    from scripts.render_mjx_3d_policy import render_rollout
    render_rollout(rollout, args.out / "rollout.gif", geometry=task.geometry,
                   physics_profile=task.physics_profile, control_dt=task.control_timestep,
                   fps=25, width=960, height=640, camera_distance=1.2,
                   azimuth=135, elevation=-18, diagnostics=True)
    # GIF is always available; MP4 additionally needs imageio and its FFmpeg backend.
    try:
        import imageio.v2 as imageio
        from PIL import Image, ImageSequence
        with Image.open(args.out / "rollout.gif") as gif:
            with imageio.get_writer(str(args.out / "rollout.mp4"), fps=25) as writer:
                for frame in ImageSequence.Iterator(gif):
                    writer.append_data(np.asarray(frame.convert("RGB")))
        print(f"Video: {args.out / 'rollout.mp4'}", flush=True)
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        print(f"MP4 unavailable ({error}); animated GIF: {args.out / 'rollout.gif'}", flush=True)
    print(f"Capture={report['captured']:.0f} Insurance={report['insurance_success']:.0f} "
          f"Failed={report['failed']:.0f} Duration={report['duration_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
