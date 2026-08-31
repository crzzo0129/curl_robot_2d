"""Train an autonomous stand-to-roll skill with a frozen rolling-teacher tail.

This is a privileged-observation startup teacher, not a deployable rolling
student. No fixed stand-to-compact interpolation, state snap, or timed handoff.
"""

import argparse
from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.autonomous_startup_3d import (
    CONTRACT, AUTONOMOUS_STARTUP_OBSERVATION_SIZE, AutonomousStartupConfig, load_candidate_bank, sha256,
)
from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig
from curl_robot_2d_mjx.config_3d import Rolling3DConfig
from curl_robot_2d_mjx.environment_3d import model_path_3d
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime

DEFAULT_BANK = Path(__file__).resolve().parents[1] / "assets/startup_handoff_gain_teacher_t1.json"
PRESETS = {
    "smoke": dict(steps=4096, envs=4, eval_envs=4, num_evals=2, batch_size=4, num_minibatches=1),
    "4090": dict(steps=10_000_000, envs=256, eval_envs=32, num_evals=20, batch_size=128, num_minibatches=4),
    "h200": dict(steps=20_000_000, envs=1024, eval_envs=128, num_evals=30, batch_size=256, num_minibatches=8),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--teacher-config", type=Path)
    p.add_argument("--candidate-bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preset", choices=PRESETS, default="smoke")
    for key in PRESETS["smoke"]:
        p.add_argument("--" + key.replace("_", "-"), type=int)
    p.add_argument("--startup-budget-s", type=float, default=3)
    p.add_argument("--continuation-s", type=float, default=3)
    p.add_argument("--minimum-turns", type=float, default=1.5)
    p.add_argument("--confirmation-steps", type=int, default=3)
    p.add_argument("--gate-scale", type=float, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--entropy-cost", type=float, default=0.001)
    p.add_argument("--discounting", type=float, default=0.995)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--updates-per-batch", type=int, default=4)
    p.add_argument("--hidden-layers", nargs="+", type=int, default=[256, 256, 128])
    p.add_argument("--initial-policy-std", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restore-startup", type=Path,
                   help="startup params only; restores weights/normalizer, not optimizer")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="real-teacher env interface smoke only; no PPO or success claim")
    p.add_argument("--max-devices", type=int, default=1)
    p.add_argument("--memory-fraction", type=float, default=0.80)
    p.add_argument("--mujoco-gl", default="disable")
    args = p.parse_args(argv)
    for key, default in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.teacher_config is None:
        args.teacher_config = args.teacher.parent / "training_config.json"
    for key in ("teacher", "teacher_config", "candidate_bank"):
        if not getattr(args, key).is_file():
            p.error(f"missing {key}: {getattr(args, key)}")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("output directory is not empty; use a new directory, including for resuming")
    for key in (*PRESETS["smoke"], "unroll_length", "updates_per_batch", "max_devices"):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.smoke_steps < 0 or args.smoke_steps > 300:
        p.error("smoke-steps must be in [0,300]")
    if any(x < 1 for x in args.hidden_layers):
        p.error("hidden layers must be positive")
    for key in ("learning_rate", "initial_policy_std", "memory_fraction"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} must be finite and positive")
    if args.initial_policy_std <= 0.001 or args.memory_fraction > 1:
        p.error("invalid standard deviation or memory fraction")
    if not math.isfinite(args.entropy_cost) or args.entropy_cost < 0:
        p.error("entropy-cost must be nonnegative")
    if args.eval_only and not args.restore_startup:
        p.error("eval-only requires restore-startup")
    if args.restore_startup and not args.restore_startup.is_file():
        p.error("startup checkpoint missing")
    return args


def build_inputs(args):
    teacher = json.loads(args.teacher_config.read_text(encoding="utf-8"))
    task = Rolling3DConfig(**teacher["task"])
    ref = CEMReferenceConfig(**teacher["reference"])
    reward = Rolling3DRewardConfig(**teacher["reward"])
    cfg = AutonomousStartupConfig(startup_budget_s=args.startup_budget_s,
        continuation_s=args.continuation_s, minimum_turns=args.minimum_turns,
        confirmation_steps=args.confirmation_steps, gate_scale=args.gate_scale,
        discounting=args.discounting)
    cfg.validate(task.control_timestep)
    bank, bank_payload = load_candidate_bank(args.candidate_bank, teacher_path=args.teacher,
        teacher_payload=teacher, model_path=model_path_3d(task.geometry))
    return task, ref, reward, cfg, bank, bank_payload, teacher


def startup_network_factory(stand_action, hidden_layers, initial_std):
    from brax.training.networks import FeedForwardNetwork
    from scripts.train_mjx_3d_residual_ppo import _zero_centered_residual_network_factory
    import jax.numpy as jp
    from flax.core import unfreeze, freeze, FrozenDict
    base_factory = _zero_centered_residual_network_factory(hidden_layers, "elu", initial_std)

    def factory(observation_size, action_size, preprocess_observations_fn):
        networks = base_factory(observation_size, action_size, preprocess_observations_fn)
        original = networks.policy_network
        def init(key):
            params = original.init(key)
            mutable = unfreeze(params)
            # Initial mean holds stand, not compact. This is only initialization;
            # actions remain full-range, independent, learned position targets.
            mutable["params"]["location"]["bias"] = jp.arctanh(jp.clip(jp.asarray(stand_action), -.999, .999))
            return freeze(mutable) if isinstance(params, FrozenDict) else mutable
        return replace(networks, policy_network=FeedForwardNetwork(init=init, apply=original.apply))
    return factory


def evaluate_startup(env, policy, *, count, seed):
    """Independent fixed episodes, no auto-reset; terminal metrics are pulses."""
    import jax
    import jax.numpy as jp
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(seed), count))
    def one(s):
        action = policy(s.obs, jax.random.PRNGKey(0))[0]
        return env.step(s, action)
    step = jax.jit(jax.vmap(one))
    totals = {key: np.zeros(count) for key in env._zero_metrics()}
    traces = {key: [] for key in ("qpos", "qvel", "ctrl", "time", "teacher_active_next",
                                  "candidate_id", "gate_error", "effective_action", "rolling_phase",
                                  "oscillator_phase", "reference_time_offset")}
    for index in range(env.episode_length):
        state = step(state)
        metrics = jax.device_get(state.metrics)
        for key in totals:
            totals[key] += np.asarray(metrics[key])
        trace = {key: getattr(state.pipeline_state, key) for key in ("qpos", "qvel", "ctrl", "time")}
        trace.update(teacher_active_next=state.info["teacher_active"],
                     candidate_id=state.info["candidate_id"], gate_error=state.metrics["gate_error"],
                     effective_action=state.info["base_info"]["last_action"])
        trace.update({key: state.info["base_info"][key] for key in (
            "rolling_phase", "oscillator_phase", "reference_time_offset")})
        for key, value in jax.device_get(trace).items():
            traces[key].append(np.asarray(value[:min(count, 4)]))
        if index % 50 == 49:
            print(f"[startup eval] step={index + 1}/{env.episode_length}", flush=True)
    handoff = totals["handoff"] > 0
    success = totals["startup_success"] > 0
    report = {
        "episodes": count, "seed": seed, "handoff_rate": float(handoff.mean()),
        "success_rate": float(success.mean()),
        "conditional_success_after_handoff": float(success[handoff].mean()) if handoff.any() else None,
        "startup_timeout_rate": float(totals["startup_timeout"].mean()),
        "tail_insufficient_progress_rate": float(totals["tail_insufficient_progress"].mean()),
        "mean_handoff_time_s": float(totals["handoff_time_s"][handoff].mean()) if handoff.any() else None,
        "mean_tail_turns": float(totals["terminal_tail_turns"].mean()),
        "maximum_abs_y_m": float(totals["terminal_max_abs_y_m"].max()),
        "failure_rates": {name: float(totals[f"failure_{name}"].mean())
                          for name in ("nonfinite", "root_low", "root_high", "lateral_drift",
                                       "axis_tilt", "forbidden_depth", "forbidden_contact")},
        "scope": "startup actor plus frozen rolling teacher; NOT an independent distilled student",
    }
    return report, {**totals, **{f"{key}_first_episodes": np.stack(value) for key, value in traces.items()}}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    task, ref, reward, cfg, bank, bank_payload, teacher = build_inputs(args)
    payload = {"contract": CONTRACT, "task": asdict(task), "startup": asdict(cfg),
        "teacher_config_payload": teacher, "teacher_sha256": sha256(args.teacher),
        "model_sha256": sha256(model_path_3d(task.geometry)),
        "candidate_bank_sha256": sha256(args.candidate_bank),
        "candidate_count": len(bank["time"]), "observation_size": AUTONOMOUS_STARTUP_OBSERVATION_SIZE, "action_size": 8,
        "episode_length": cfg.episode_steps(task.control_timestep),
        "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_randomization": False, "gate_is_provisional": True,
        "reset_pose": "stand", "reference_interpolation_during_startup": False,
        "task_field_describes": "base rolling teacher; autonomous wrapper owns stand reset and episode termination",
        "deployable_student": False, "physics_snap_at_handoff": False,
        "initial_policy_mean": "stand; no folding interpolation",
        "teacher_tail_actions": "ignored startup actor actions; same-episode rewards/values carry downstream credit",
    }
    if args.restore_startup:
        source = json.loads((args.restore_startup.parent / "training_config.json").read_text(encoding="utf-8"))
        for key in ("contract", "observation_size", "action_size", "teacher_sha256",
                    "candidate_bank_sha256", "model_sha256"):
            if source.get(key) != payload[key]:
                raise ValueError(f"incompatible startup restore: {key}")
        if source["training"]["hidden_layers"] != args.hidden_layers:
            raise ValueError("restore hidden layers mismatch")
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload
    configure_cloud_runtime(memory_fraction=args.memory_fraction, preallocate=False,
                            xla_triton=False, mujoco_gl=args.mujoco_gl, verbose=True)
    import jax
    from brax.io import model as model_io
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from curl_robot_2d_mjx.environment_autonomous_startup_3d import (
        make_autonomous_startup_env, wrap_autonomous_startup,
    )
    payload["runtime"] = describe_runtime()
    env = make_autonomous_startup_env(task, ref, reward, bank, args.teacher, teacher, cfg, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "training_config.json", payload)
    write_json(args.out / "candidate_bank.json", bank_payload)
    if args.smoke_steps:
        import jax.numpy as jp
        state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
        step_fn = jax.jit(env.step)
        print("[smoke] compiling real-teacher startup env", flush=True)
        for _ in range(args.smoke_steps):
            state = step_fn(state, env.stand_action)
        summary = {"mode": "interface_smoke_only", "steps_requested": args.smoke_steps,
                   "physics_time": float(state.pipeline_state.time), "done": float(state.done),
                   "teacher_active": bool(state.info["teacher_active"]),
                   "finite": bool(jp.all(jp.isfinite(state.obs)))}
        write_json(args.out / "smoke.json", summary)
        print(summary, flush=True)
        return summary
    factory = startup_network_factory(env.stand_action, args.hidden_layers, args.initial_policy_std)
    net = factory(env.observation_size, env.action_size, running_statistics.normalize)
    inference = ppo_networks.make_inference_fn(net)
    restored = model_io.load_params(args.restore_startup) if args.restore_startup else None
    if args.eval_only:
        report, arrays = evaluate_startup(env, inference(restored, deterministic=True),
                                          count=args.eval_envs, seed=args.seed + 20000)
        write_json(args.out / "evaluation.json", report)
        np.savez_compressed(args.out / "evaluation_arrays.npz", **arrays)
        print(json.dumps(report, indent=2), flush=True)
        return report
    from brax.training.agents.ppo import train as ppo
    signature = inspect.signature(ppo.train).parameters
    required = {"wrap_env_fn", "policy_params_fn", "max_devices_per_host", "restore_params"}
    if not required.issubset(signature):
        raise RuntimeError(f"Brax PPO missing required parameters: {required - signature.keys()}")
    devices = min(args.max_devices, jax.local_device_count())
    if args.envs % devices or (args.batch_size * args.num_minibatches) % args.envs:
        raise ValueError("envs must divide batch_size*num_minibatches and be divisible by active devices")
    history, snapshots, pending = [], {}, {}
    best = {"score": None, "step": None}
    def try_save(step):
        if step not in pending or step not in snapshots:
            return
        score = pending[step]
        if best["score"] is None or score > best["score"]:
            model_io.save_params(args.out / "params_best", snapshots[step])
            best.update(score=score, step=step)
            write_json(args.out / "best_selection.json", {**best,
                "ranking": "success, handoff, negative failure, reward (lexicographic)",
                "passed": score[0] >= .95})
        snapshots.pop(step, None)
        pending.pop(step, None)
    def params_callback(step, make_policy, params):
        del make_policy
        snapshots[int(step)] = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), params)
        try_save(int(step))
    def progress(step, metrics):
        clean = {k: float(v) for k, v in metrics.items()}
        history.append({"step": int(step), **clean})
        write_json(args.out / "metrics_history.json", history)
        success, handoff = (clean.get(f"eval/episode_{key}", 0.) for key in ("startup_success", "handoff"))
        failed = clean.get("eval/episode_failed", 0.)
        pending[int(step)] = (success, handoff, -failed, clean.get("eval/episode_reward", -1e30))
        try_save(int(step))
        print(f"[startup PPO] step={step} handoff={handoff:.1%} success={success:.1%} "
              f"failed={failed:.1%} timeout={clean.get('eval/episode_startup_timeout', 0.):.1%}", flush=True)
    print(f"[startup PPO] {args.envs} envs, budget={cfg.startup_budget_s}s, "
          f"teacher tail={cfg.continuation_s}s, candidates={len(bank['time'])}", flush=True)
    started = time.perf_counter()
    # No large nested rollout at the gate: each env advances one physical
    # control period, including in the frozen-teacher tail.
    _, params, _ = ppo.train(environment=env, eval_env=env,
        num_timesteps=args.steps, episode_length=env.episode_length, action_repeat=1,
        num_envs=args.envs, num_eval_envs=args.eval_envs, num_evals=args.num_evals,
        learning_rate=args.learning_rate, entropy_cost=args.entropy_cost, discounting=cfg.discounting,
        reward_scaling=1., unroll_length=args.unroll_length, batch_size=args.batch_size,
        num_minibatches=args.num_minibatches, num_updates_per_batch=args.updates_per_batch,
        normalize_observations=True, deterministic_eval=True, network_factory=factory,
        seed=args.seed, progress_fn=progress, policy_params_fn=params_callback,
        wrap_env_fn=wrap_autonomous_startup, max_devices_per_host=devices, restore_params=restored)
    model_io.save_params(args.out / "params_final", params)
    if best["step"] is None:
        raise RuntimeError("No checkpoint matched an evaluation; refusing to label final as best")
    best_params = model_io.load_params(args.out / "params_best")
    report, arrays = evaluate_startup(env, inference(best_params, deterministic=True),
                                      count=args.eval_envs, seed=args.seed + 20000)
    write_json(args.out / "evaluation_best.json", report)
    np.savez_compressed(args.out / "evaluation_best_arrays.npz", **arrays)
    summary = {"elapsed_wall_s": time.perf_counter() - started, "best_step": best["step"],
               "evaluation": report, "startup_training_complete": True,
               "passes_nominal_acceptance": report["success_rate"] >= .95,
               "deployment_approved": False}
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
