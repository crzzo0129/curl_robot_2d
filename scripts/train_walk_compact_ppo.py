"""Train the walking-start -> compact transition actor (stage one, no rolling).

Episode = real 0.4 m/s walking snapshot (deploy-interface walking policy) and
the 12-DoF actor must curl into the compact pose while decelerating.  Pose-only
terminal gate; no rolling teacher; obs = 36x20 deploy history.

Run (cloud MJX env):

    python -m scripts.train_walk_compact_ppo \
        --snapshots results/walk_start_snapshots_0p4 \
        --preset h200 --max-devices 1 \
        --out results/walk_compact_stage1_seed0

Local contract checks without training:
    python -m scripts.train_walk_compact_ppo --dry-run --out <new dir>
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.walk_compact_3d import (
    ACTION_SIZE,
    CONTROL_TIMESTEP_S,
    GEOMETRY,
    MESH_XML_REL,
    OBSERVATION_SIZE,
    PROJECT_ROOT,
    WALK_COMPACT_CONTRACT,
    WalkCompactConfig,
    prepare_runtime_xml,
    xml_fingerprint,
)
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime

PRESETS = {
    "smoke": dict(steps=4096, envs=4, eval_envs=4, num_evals=2, batch_size=4, num_minibatches=1),
    "4090": dict(steps=10_000_000, envs=256, eval_envs=32, num_evals=20, batch_size=128, num_minibatches=4),
    "h200": dict(steps=20_000_000, envs=1024, eval_envs=128, num_evals=30, batch_size=256, num_minibatches=8),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots", type=Path, default=None,
                   help="directory written by scripts.collect_walking_start_snapshots "
                        "(default: PROJECT_ROOT/results/walk_start_snapshots_0p4)")
    p.add_argument("--xml", type=Path,
                   default=PROJECT_ROOT / MESH_XML_REL,
                   help="source mesh XML (compact keyframe must be the target)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preset", choices=PRESETS, default="smoke")
    for key in PRESETS["smoke"]:
        p.add_argument("--" + key.replace("_", "-"), type=int)
    p.add_argument("--budget-s", type=float, default=WalkCompactConfig.budget_s)
    p.add_argument("--confirmation-steps", type=int)
    compact_defaults = WalkCompactConfig()
    for field in ("joint_position_rad", "root_z_m", "orientation_rad", "lateral_m",
                  "pose_reward_weight", "success_bonus", "time_cost",
                  "action_change_cost", "torque_cost",
                  "upward_velocity_weight", "upward_velocity_sigma_m_s",
                  "excess_height_weight", "excess_height_margin_m",
                  "excess_height_sigma_m", "angular_velocity_weight",
                  "angular_velocity_sigma_rad_s"):
        p.add_argument("--" + field.replace("_", "-"), type=float,
                       default=getattr(compact_defaults, field))
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--entropy-cost", type=float, default=0.001)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--updates-per-batch", type=int, default=4)
    p.add_argument("--hidden-layers", nargs="+", type=int, default=[512, 256, 128])
    p.add_argument("--initial-policy-std", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restore", type=Path, help="actor params .bin; restores weights/normalizer")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="env interface smoke only; no PPO or success claim")
    p.add_argument("--max-devices", type=int, default=1)
    p.add_argument("--memory-fraction", type=float, default=0.80)
    p.add_argument("--mujoco-gl", default="disable")
    args = p.parse_args(argv)

    def _resolve(path):
        path = Path(path)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    if args.snapshots is None:
        args.snapshots = PROJECT_ROOT / "results" / "walk_start_snapshots_0p4"
    args.snapshots = _resolve(args.snapshots)
    args.xml = _resolve(args.xml)
    args.out = _resolve(args.out)
    if args.restore is not None:
        args.restore = _resolve(args.restore)
    if not args.snapshots.is_dir():
        p.error(f"missing snapshot directory: {args.snapshots} "
                f"(relative paths resolve against {PROJECT_ROOT})")
    if args.confirmation_steps is None:
        args.confirmation_steps = compact_defaults.confirmation_steps
    for key, default in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.out.exists() and any(args.out.iterdir()):
        p.error("output directory is not empty; use a new directory, including for resuming")
    if args.smoke_steps < 0 or args.smoke_steps > 400:
        p.error("smoke-steps must be in [0, 400]")
    if args.confirmation_steps < 1:
        p.error("confirmation-steps must be positive")
    if not math.isfinite(args.budget_s) or args.budget_s <= 0:
        p.error("budget-s must be finite and positive")
    if any(x < 1 for x in args.hidden_layers):
        p.error("hidden layers must be positive")
    return args


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def build_config(args):
    return WalkCompactConfig(
        budget_s=args.budget_s,
        confirmation_steps=args.confirmation_steps,
        joint_position_rad=args.joint_position_rad,
        root_z_m=args.root_z_m,
        orientation_rad=args.orientation_rad,
        lateral_m=args.lateral_m,
        pose_reward_weight=args.pose_reward_weight,
        success_bonus=args.success_bonus,
        time_cost=args.time_cost,
        action_change_cost=args.action_change_cost,
        torque_cost=args.torque_cost,
        upward_velocity_weight=args.upward_velocity_weight,
        upward_velocity_sigma_m_s=args.upward_velocity_sigma_m_s,
        excess_height_weight=args.excess_height_weight,
        excess_height_margin_m=args.excess_height_margin_m,
        excess_height_sigma_m=args.excess_height_sigma_m,
        angular_velocity_weight=args.angular_velocity_weight,
        angular_velocity_sigma_rad_s=args.angular_velocity_sigma_rad_s)


def snapshot_paths(args):
    npz = args.snapshots / "walk_start_snapshots.npz"
    meta = args.snapshots / "walk_start_snapshots_meta.json"
    if not npz.is_file() or not meta.is_file():
        raise ValueError(f"snapshot bank incomplete in {args.snapshots}")
    return npz, meta


def build_payload(args, meta):
    return {
        "contract": WALK_COMPACT_CONTRACT,
        "geometry": GEOMETRY,
        "xml_basename": Path(args.xml).name,
        **xml_fingerprint(Path(args.xml)),
        "snapshot_count": meta["count"],
        "snapshot_command_m_s": meta["command_m_s"],
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "control_dt_s": CONTROL_TIMESTEP_S,
        "startup": asdict(build_config(args)),
        "training": {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
        "model_randomization": False,
        "deployable_actor": False,
        "rolling_teacher": False,
        "terminal_gate": "pose only: joints/root-z/orientation/lateral; velocities ungated",
        "reset": "0.4 m/s walking snapshots from the deploy-interface policy",
        "initial_policy_mean": "zero residual (default pose targets)",
    }


def startup_network_factory(hidden_layers, initial_std):
    from scripts.train_mjx_3d_residual_ppo import _zero_centered_residual_network_factory
    return _zero_centered_residual_network_factory(hidden_layers, "elu", initial_std)


def evaluate_walk_compact(env, policy, *, count, seed):
    """Independent fixed episodes; success/timeout/failure are pulses."""
    import jax
    import jax.numpy as jp
    import numpy as np
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(seed), count))

    def one(s):
        action = policy(s.obs, jax.random.PRNGKey(0))[0]
        return env.step(s, action)

    step = jax.jit(jax.vmap(one))
    totals = {key: np.zeros(count) for key in env._zero_metrics()}
    traces = {key: [] for key in ("qpos", "time", "gate_error", "pose_quality",
                                  "terminal")}
    for index in range(env.episode_length):
        state = step(state)
        metrics = jax.device_get(state.metrics)
        for key in totals:
            totals[key] += np.asarray(metrics[key])
        trace = dict(qpos=np.asarray(state.pipeline_state.q)[:, :3],
                     time=np.asarray(state.pipeline_state.time),
                     gate_error=np.asarray(state.metrics["gate_eligible"]),
                     pose_quality=np.asarray(state.metrics["pose_quality"]),
                     terminal=np.asarray(state.info["terminal"]))
        for key, value in jax.device_get(trace).items():
            traces[key].append(np.asarray(value))
        if index % 50 == 49:
            print(f"[walk-compact eval] step={index + 1}/{env.episode_length}", flush=True)
    success = totals["success"] > 0
    failed = totals["failed"] > 0
    report = {
        "episodes": count, "seed": seed,
        "success_rate": float(success.mean()),
        "timeout_rate": float((totals["timeout"] > 0).mean()),
        "failed_rate": float(failed.mean()),
        "mean_gate_eligible_fraction": float(totals["gate_eligible"].sum() /
                                             (totals["gate_eligible"].shape[0] * env.episode_length)),
        "mean_terminal_pose_quality": float(totals["terminal_pose_quality"].mean()),
        "mean_terminal_gate_error": float(totals["terminal_gate_error"].mean()),
        "scope": "0.4 m/s walking snapshot to compact pose; rolling continuation NOT evaluated",
        "rolling_continuation_evaluated": False,
    }
    return report, {key: np.stack(value) for key, value in traces.items()}


def main(argv=None):
    args = parse_args(argv)
    npz, meta_path = snapshot_paths(args)
    from curl_robot_2d_mjx.walk_compact_3d import validate_snapshot_bank
    _, meta = validate_snapshot_bank(npz, meta_path)
    payload = build_payload(args, meta)
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload
    configure_cloud_runtime(memory_fraction=args.memory_fraction, preallocate=False,
                            xla_triton=False, mujoco_gl=args.mujoco_gl, verbose=True)
    args.out.mkdir(parents=True, exist_ok=True)
    runtime_xml = prepare_runtime_xml(args.xml, args.out / "walk_compact_runtime.xml")
    payload["runtime_xml_lf_sha256"] = xml_fingerprint(runtime_xml)["xml_lf_sha256"]
    payload["runtime"] = describe_runtime()
    write_json(args.out / "training_config.json", payload)
    from curl_robot_2d_mjx.environment_walk_compact_3d import (
        make_walk_compact_env, wrap_walk_compact,
    )
    from curl_robot_2d_mjx.walk_compact_3d import bank_action_arrays
    action = bank_action_arrays(meta)
    env = make_walk_compact_env(runtime_xml, npz, meta_path,
                                config=build_config(args), seed=args.seed)
    if args.smoke_steps:
        import jax
        import jax.numpy as jp
        state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
        step_fn = jax.jit(env.step)
        print("[smoke] compiling walk->compact env", flush=True)
        zero = jp.zeros(ACTION_SIZE)
        for _ in range(args.smoke_steps):
            state = step_fn(state, zero)
        summary = {"mode": "interface_smoke_only", "steps_requested": args.smoke_steps,
                   "physics_time": float(state.pipeline_state.time),
                   "done": float(state.done),
                   "finite_obs": bool(jp.all(jp.isfinite(state.obs))),
                   "finite_qpos": bool(jp.all(jp.isfinite(state.pipeline_state.q)))}
        write_json(args.out / "smoke.json", summary)
        print(summary, flush=True)
        return summary
    import jax
    from brax.io import model as model_io
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    cfg = build_config(args)
    factory = startup_network_factory(args.hidden_layers, args.initial_policy_std)
    net = factory(env.observation_size, env.action_size, running_statistics.normalize)
    inference = ppo_networks.make_inference_fn(net)
    restored = model_io.load_params(args.restore) if args.restore else None
    if args.eval_only:
        report, arrays = evaluate_walk_compact(env, inference(restored, deterministic=True),
                                               count=args.eval_envs, seed=args.seed + 20000)
        write_json(args.out / "evaluation.json", report)
        np.savez_compressed(args.out / "evaluation_arrays.npz", **arrays)
        print(json.dumps(report, indent=2), flush=True)
        return report
    devices = min(args.max_devices, jax.local_device_count())
    history, snapshots, pending = [], {}, {}
    best = {"score": None, "step": None}

    def try_save(step):
        if step not in pending or step not in snapshots:
            return
        score = pending[step]
        if best["score"] is None or score > best["score"]:
            model_io.save_params(args.out / "params_best", snapshots[step])
            best.update(score=score, step=step)
            write_json(args.out / "best_selection.json",
                       {**best, "ranking": "success, negative timeout/failure, reward (lexicographic)",
                        "passed": score[0] >= .95})
        snapshots.pop(step, None)
        pending.pop(step, None)

    def params_callback(step, make_policy, params):
        del make_policy
        snapshots[int(step)] = jax.tree_util.tree_map(
            lambda x: np.asarray(x).copy(), params)
        try_save(int(step))

    def progress(step, metrics):
        clean = {k: float(v) for k, v in metrics.items()}
        history.append({"step": int(step), **clean})
        write_json(args.out / "metrics_history.json", history)
        success = clean.get("eval/episode_success", 0.)
        failed = clean.get("eval/episode_failed", 0.)
        timeout = clean.get("eval/episode_timeout", 0.)
        pending[int(step)] = (success, -failed, -timeout,
                              clean.get("eval/episode_reward", -1e30))
        try_save(int(step))
        print(f"[walk-compact PPO] step={step} success={success:.1%} "
              f"failed={failed:.1%} timeout={timeout:.1%} "
              f"pose={clean.get('eval/episode_pose_quality', 0.):.3f} "
              f"gate={clean.get('eval/episode_gate_eligible', 0.):.3f}", flush=True)

    print(f"[walk-compact PPO] {args.envs} envs, budget={cfg.budget_s}s "
          f"confirm={cfg.confirmation_steps} steps, snapshots={meta['count']}",
          flush=True)
    started = time.perf_counter()
    _, params, _ = ppo.train(environment=env, eval_env=env,
        num_timesteps=args.steps, episode_length=env.episode_length,
        action_repeat=1, num_envs=args.envs, num_eval_envs=args.eval_envs,
        num_evals=args.num_evals, learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost, discounting=cfg.discounting,
        reward_scaling=1., unroll_length=args.unroll_length, batch_size=args.batch_size,
        num_minibatches=args.num_minibatches, num_updates_per_batch=args.updates_per_batch,
        normalize_observations=True, deterministic_eval=True, network_factory=factory,
        seed=args.seed, progress_fn=progress, policy_params_fn=params_callback,
        wrap_env_fn=wrap_walk_compact,
        max_devices_per_host=devices, restore_params=restored)
    model_io.save_params(args.out / "params_final", params)
    if best["step"] is None:
        raise RuntimeError("No checkpoint matched an evaluation; refusing to label final as best")
    best_params = model_io.load_params(args.out / "params_best")
    report, arrays = evaluate_walk_compact(env, inference(best_params, deterministic=True),
                                           count=args.eval_envs, seed=args.seed + 20000)
    write_json(args.out / "evaluation_best.json", report)
    np.savez_compressed(args.out / "evaluation_best_arrays.npz", **arrays)
    summary = {"elapsed_wall_s": time.perf_counter() - started, "best_step": best["step"],
               "evaluation": report, "walk_compact_stage1_complete": True,
               "passes_nominal_acceptance": report["success_rate"] >= .95,
               "deployment_approved": False}
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

