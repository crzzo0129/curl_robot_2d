"""Collect primitive CEM reference states; no learned ROLL weights or mesh."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from curl_robot_2d_mjx.config_3d import Rolling3DConfig
from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig, transition_physics_profile_3d
from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.environment_3d import ROLLINGQUAD_2_PRIMITIVE_CEM_CONTROLLER


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=ROLLINGQUAD_2_PRIMITIVE_CEM_CONTROLLER)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--physics-profile", choices=("newton4", "accurate", "cg12"), default="cg12")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.episodes < 1 or args.steps < 2:
        p.error("episodes must be positive and steps >= 2")
    if args.out.suffix != ".npz" or args.out.exists() or args.out.with_suffix(".summary.json").exists():
        p.error("--out must be a new .npz bank and summary path")
    config = transition_physics_profile_3d(args.physics_profile,
        Transition3DConfig(geometry="rollingquad_2_primitive", dynamic_roll_to_stand=True,
                           physics_timestep=0.001))
    # Identical nominal dynamics/control period to the recovery environment.
    common = {name: value for name, value in asdict(config).items()
              if name in Rolling3DConfig.__dataclass_fields__}
    task = replace(Rolling3DConfig(**common), episode_length=args.steps + 1)
    reference = load_cem_reference(args.reference, reference_weight=1.0, minimum_residual_gain=0.0)
    digest = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    report = dict(source_kind="cem_reference_zero_residual", reference=asdict(reference),
                  reference_sha256=digest, task=asdict(task), episodes=args.episodes,
                  steps=args.steps, seed=args.seed, mesh_used=False,
                  status="dry_run" if args.dry_run else "collecting")
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    from curl_robot_2d_mjx.runtime import configure_cloud_runtime
    configure_cloud_runtime(preallocate=False, mujoco_gl="disable", verbose=True)
    import jax.numpy as jp
    from curl_robot_2d_mjx.environment_3d import make_brax_env_3d
    from curl_robot_2d_mjx.transition_initialization_3d import collect_roll_snapshots_3d
    from scripts.inspect_transition_roll_snapshots import inspect_bank
    env = make_brax_env_3d(task, cem_reference=reference, seed=args.seed)
    def policy(obs, key):
        del obs, key
        return jp.zeros((env.action_size,)), {}
    result = collect_roll_snapshots_3d(env, policy, args.out, config=config,
        source_policy=f"cem_reference:{args.reference.resolve()}#sha256={digest};residual=0",
        seed=args.seed, episodes=args.episodes, steps_per_episode=args.steps,
        progress_fn=lambda row: print(json.dumps(row), flush=True))
    report["collection"] = result
    try:
        report["coverage"] = inspect_bank(args.out, replace(config, curriculum_stage="brake_full"))
        report["status"] = "ok"
    except ValueError as error:
        report.update(status="insufficient_coverage", error=str(error))
    args.out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit("reference bank has insufficient rolling coverage; inspect summary before training")


if __name__ == "__main__":
    main()
