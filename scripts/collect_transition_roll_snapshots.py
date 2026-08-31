"""Load a frozen residual ROLL checkpoint/config and collect Transition v2 states.

No training, external braking, environment auto-reset, or guessed policy ABI.
CPU --dry-run checks paths/config without importing JAX or producing a bank.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig, residual_gain
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, validate_3d_config
from curl_robot_2d_mjx.config_transition_3d import transition_curriculum_config_3d
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
from curl_robot_2d_mjx.runtime import configure_cloud_runtime


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll-checkpoint", type=Path, required=True,
                        help="ROLL params_best/params_final file, not Transition or student weights")
    parser.add_argument("--roll-config", type=Path,
                        help="matching training_config.json; default: beside ROLL checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="new v2 .npz bank path")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps-per-episode", type=int,
                        help="default: saved ROLL episode_length; never overrides its timeout")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roll-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--require-stage", choices=("brake_early", "brake_later", "brake_full"),
                        default="brake_early", help="exit nonzero if this course lacks cycle coverage")
    parser.add_argument("--memory-fraction", type=float, default=.80)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.roll_config is None:
        args.roll_config = args.roll_checkpoint.parent / "training_config.json"
    for name in ("roll_checkpoint", "roll_config"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} file not found: {getattr(args, name)}")
    if args.out.suffix.lower() != ".npz":
        parser.error("--out must be a new .npz file")
    for path in (args.out, args.out.with_suffix(".summary.json")):
        if path.exists():
            parser.error(f"refusing to overwrite existing artifact: {path}")
    if args.episodes < 1 or args.sample_every < 1 or (
            args.steps_per_episode is not None and args.steps_per_episode < 1):
        parser.error("episode/step/sample counts must be positive")
    if not math.isfinite(args.memory_fraction) or not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0,1]")
    return args


def load_roll_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("task", "reference", "reward", "hidden_layers", "activation", "zero_residual_policy_init")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"ROLL residual PPO config missing {missing}; provide its original training_config.json")
    task = Rolling3DConfig(**payload["task"])
    validate_3d_config(task)
    if task.geometry != "rollingquad_2" or task.direct_effective_action:
        raise ValueError("this loader requires rollingquad_2 residual ROLL, not a direct-action student")
    reference = CEMReferenceConfig(**payload["reference"])
    if len(reference.coefficients) != 8 or not all(math.isfinite(x) for x in reference.coefficients):
        raise ValueError("saved reference must contain eight finite coefficients")
    residual_gain(reference.reference_weight, reference.minimum_residual_gain)
    reward = Rolling3DRewardConfig(**payload["reward"])
    hidden = payload["hidden_layers"]
    if not isinstance(hidden, list) or not hidden or any(type(v) is not int or v < 1 for v in hidden):
        raise ValueError("hidden_layers must contain positive integers")
    if payload["activation"] not in ("elu", "relu", "swish", "tanh"):
        raise ValueError("unsupported saved ROLL activation")
    if type(payload["zero_residual_policy_init"]) is not bool:
        raise ValueError("zero_residual_policy_init must be boolean")
    equivariant = payload.get("reflection_equivariant_policy", False)
    if type(equivariant) is not bool or (equivariant and not payload["zero_residual_policy_init"]):
        raise ValueError("reflection-equivariant ROLL requires the custom residual network")
    if payload["zero_residual_policy_init"] and (
            "initial_policy_std" not in payload or not math.isfinite(payload["initial_policy_std"])
            or payload["initial_policy_std"] <= .001):
        raise ValueError("custom residual network requires saved initial_policy_std > .001")
    return task, reference, reward, payload


def make_frozen_roll_policy(env, payload, params):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    if payload["zero_residual_policy_init"]:
        from scripts.train_mjx_3d_residual_ppo import _zero_centered_residual_network_factory
        factory = _zero_centered_residual_network_factory(
            payload["hidden_layers"], payload["activation"], payload["initial_policy_std"],
            reflection_equivariant=payload.get("reflection_equivariant_policy", False))
    else:
        from scripts.train_mjx_ppo import _network_factory
        factory = _network_factory(payload["hidden_layers"], payload["activation"])
    network = factory(env.observation_size, env.action_size,
                      preprocess_observations_fn=running_statistics.normalize)
    # Only load learned parameters; never call init or substitute a zero actor.
    return networks.make_inference_fn(network)(params, deterministic=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def course_reports(path, model, direction):
    from curl_robot_2d_mjx.transition_initialization_3d import load_roll_snapshots_3d
    reports = {}
    for stage in ("brake_early", "brake_later", "brake_full"):
        task = replace(transition_curriculum_config_3d(stage), snapshot_roll_direction=direction)
        try:
            _, report = load_roll_snapshots_3d(path, model, task, return_report=True,
                                              require_coverage=False)
            reports[stage] = report
        except ValueError as error:
            reports[stage] = {"coverage_complete": False, "error": str(error)}
    return reports


def main(argv=None):
    args = parse_args(argv)
    task, reference, reward, payload = load_roll_config(args.roll_config)
    steps = args.steps_per_episode or task.episode_length
    if steps > task.episode_length:
        raise ValueError("--steps-per-episode exceeds saved ROLL episode_length; "
                         "collector does not silently extend timeouts or change the source policy environment")
    report = {
        "status": "dry_run" if args.dry_run else "collecting",
        "roll_checkpoint": str(args.roll_checkpoint.resolve()),
        "roll_config": str(args.roll_config.resolve()),
        "bank": str(args.out.resolve()), "schema_version": 2,
        "checkpoint_sha256": _sha256(args.roll_checkpoint),
        "config_sha256": _sha256(args.roll_config),
        "task": asdict(task), "reference": asdict(reference), "reward": asdict(reward),
        "hidden_layers": payload["hidden_layers"], "activation": payload["activation"],
        "zero_residual_policy_init": payload["zero_residual_policy_init"],
        "reflection_equivariant_policy": payload.get("reflection_equivariant_policy", False),
        "episodes": args.episodes, "steps_per_episode": steps,
        "sample_every": args.sample_every, "warmup_steps": 0,
        "seed": args.seed, "roll_direction": args.roll_direction,
        "external_braking": False, "source_task_overrides": {},
        "domain_randomization": "saved nominal task; no new training-time randomization wrapper",
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    configure_cloud_runtime(memory_fraction=args.memory_fraction, preallocate=False,
                            xla_triton=False, mujoco_gl=args.mujoco_gl, verbose=True)
    from brax.io import model as model_io
    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d
    from curl_robot_2d_mjx.transition_initialization_3d import collect_roll_snapshots_3d
    env = make_brax_env_3d(task, cem_reference=reference, reward_config=reward, seed=args.seed)
    policy = make_frozen_roll_policy(env, payload, model_io.load_params(args.roll_checkpoint))
    print(f"[ROLL collection] checkpoint={args.roll_checkpoint} obs={env.observation_size} "
          f"action={env.action_size}; compiling reset/inference/step, no training", flush=True)
    started = time.perf_counter()
    result = collect_roll_snapshots_3d(env, policy, args.out,
        source_policy=f"{args.roll_checkpoint.resolve()}#sha256={report['checkpoint_sha256']}",
        seed=args.seed, episodes=args.episodes, steps_per_episode=steps,
        warmup_steps=0, sample_every=args.sample_every,
        progress_fn=lambda row: print("[ROLL episode] " + json.dumps(row), flush=True))
    report.update(collection=result, elapsed_s=time.perf_counter() - started,
                  stage_reports=course_reports(args.out, env.mj_model, args.roll_direction))
    usable = report["stage_reports"][args.require_stage]["coverage_complete"]
    report.update(status="ok" if usable else "insufficient_coverage", required_stage=args.require_stage)
    with args.out.with_suffix(".summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for stage, selected in report["stage_reports"].items():
        print(f"[coverage] {stage}: complete={selected['coverage_complete']} "
              f"samples={selected.get('selected_samples', 0)}", flush=True)
    print(f"[saved] {args.out}\n[report] {args.out.with_suffix('.summary.json')}", flush=True)
    if not usable:
        raise SystemExit(f"bank saved but {args.require_stage} coverage is incomplete; "
                         "inspect the summary before BRAKE training (do not reuse --out for recollection)")


if __name__ == "__main__":
    main()
