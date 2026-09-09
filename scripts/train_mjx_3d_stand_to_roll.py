#!/usr/bin/env python3
"""BC warm-start and staged PPO for one-policy stand-to-roll training.

Examples:
  python -m scripts.train_mjx_3d_stand_to_roll --stage bc --out results/stand_to_roll
  python -m scripts.train_mjx_3d_stand_to_roll --stage rolling_orbit \
      --bc-params results/stand_to_roll/bc/bc_params --out results/stand_to_roll
  python -m scripts.train_mjx_3d_stand_to_roll --stage mixed_75 \
      --bc-params results/stand_to_roll/bc/bc_params \
      --restore-checkpoint results/stand_to_roll/rolling_orbit/ppo_checkpoint \
      --out results/stand_to_roll

Every PPO stage uses the same actor, the same fixed BC observation normalizer,
the same reward weights, and pure policy actions. Snapshot resets precede
the static compact-to-stand curriculum. Version 2 requires retraining BC.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import inspect
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.config_stand_to_roll import (
    STAND_TO_ROLL_CURRICULUM_STAGES,
    StandToRollConfig,
    stand_to_roll_curriculum_config,
)
from curl_robot_2d_mjx.deployment_rolling_3d import (
    CONTROLLER_JOINT_NAMES_3D,
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
)
from curl_robot_2d_mjx.environment_3d import model_path_3d
from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime
from curl_robot_2d_mjx.stand_to_roll_training import (
    STAND_TO_ROLL_ACTION_SIZE,
    STAND_TO_ROLL_HIDDEN_LAYERS,
    action_center_and_scale,
    build_cem_bc_dataset,
    initialize_ppo_actor_from_bc,
    observation_normalizer,
    preprocess_observation,
    BC_CONTRACT_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CEM_DATA = PROJECT_ROOT / "results" / "cem_cycle_data" / "cem_cycles.npz"
PRESETS = {
    "cpu_smoke": {
        "steps": 2048,
        "envs": 4,
        "eval_envs": 4,
        "num_evals": 2,
        "batch_size": 8,
        "num_minibatches": 2,
    },
    "smoke": {
        "steps": 131_072,
        "envs": 64,
        "eval_envs": 16,
        "num_evals": 4,
        "batch_size": 64,
        "num_minibatches": 4,
    },
    "4090": {
        "steps": 10_000_000,
        "envs": 512,
        "eval_envs": 64,
        "num_evals": 10,
        "batch_size": 256,
        "num_minibatches": 8,
    },
    "h200": {
        "steps": 20_000_000,
        "envs": 2048,
        "eval_envs": 256,
        "num_evals": 10,
        "batch_size": 512,
        "num_minibatches": 16,
    },
}


def _print_eval(title, metrics, *, training=False):
    """Compact, log-file-friendly output; full metrics remain in JSON."""
    def value(name, fmt=".3f", suffix=""):
        key = f"eval/episode_{name}" if training else name
        number = metrics.get(key)
        return "--" if number is None else f"{number:{fmt}}{suffix}"

    length = metrics.get("eval/avg_episode_length" if training else "avg_episode_length")
    length_text = "--" if length is None else f"{length:.1f} steps"
    print("\n" + "-" * 68)
    print(title)
    print(f"  Sustained {value('sustained_success', '.1%'):>7}   "
          f"Capture {value('captured', '.1%'):>7}   Failed {value('failed', '.1%'):>7}")
    print(f"  Roll      {value('roll_progress'):>7} rad   "
          f"Sustain {value('sustain_seconds', '.2f'):>7} s   Episode {length_text}")
    print(f"  Failures  lateral {value('failure_lateral', '.1%')} | "
          f"tilt {value('failure_axis_tilt', '.1%')} | height {value('failure_height', '.1%')}")
    print(f"            nonfinite {value('failure_nonfinite', '.1%')} | "
          f"contact {value('forbidden_contact', '.3f')} (episode sum)")
    print(f"  Reward    total {value('reward')} | lateral {value('reward_lateral')} | "
          f"sustain {value('reward_sustain')}")
    if training:
        def stat(key, fmt):
            number = metrics.get(key)
            return "--" if number is None else format(number, fmt)
        print(f"  PPO       KL {stat('training/kl_mean', '.4f')} | "
              f"LR {stat('training/learning_rate', '.1e')} | "
              f"train SPS {stat('training/sps', ',.0f')}")
    print("-" * 68, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("bc",) + STAND_TO_ROLL_CURRICULUM_STAGES,
                        required=True)
    parser.add_argument("--out", type=Path, default=Path("results/mjx_3d_stand_to_roll"))
    parser.add_argument("--cem-data", type=Path, default=DEFAULT_CEM_DATA)
    parser.add_argument("--startup-data", type=Path,
                        help="Pre-action startup BC dataset; never used as matcher data")
    parser.add_argument("--static-eval", action="store_true",
                        help="BC-only evaluation with exactly zero reset velocity and 0 snapshot probability")
    parser.add_argument("--static-curriculum", action="store_true",
                        help="Start slightly_open from BC, then restore subsequent posture stages; zero reset velocity, no snapshots or observation noise")
    parser.add_argument("--bc-params", type=Path)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--steps", type=int, help="Override PPO total environment steps only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-layers", type=int, nargs="+",
                        default=STAND_TO_ROLL_HIDDEN_LAYERS)
    parser.add_argument("--initial-policy-std", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--entropy-cost", type=float, default=0.0)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--updates-per-batch", type=int, default=1)
    parser.add_argument("--max-kl", type=float, default=1.0,
                        help="Abort at evaluation callbacks when reported KL exceeds this limit.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate the BC actor without PPO updates; no restore checkpoint.")
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.steps is not None and (args.steps < 1 or args.stage == "bc" or args.eval_only):
        parser.error("--steps must be positive and applies only to PPO training")
    if args.static_curriculum and args.stage not in (
            "slightly_open", "crouch", "semi_stand", "full_stand"):
        parser.error("--static-curriculum applies to slightly_open through full_stand")
    if args.startup_data is not None and (args.stage != "bc" or not args.startup_data.is_file()):
        parser.error("--startup-data requires --stage bc and an existing dataset")
    if args.static_eval and (not args.eval_only or args.stage != "compact"):
        parser.error("--static-eval requires --stage compact --eval-only")
    if not args.cem_data.is_file() and not args.dry_run:
        parser.error(f"CEM data does not exist: {args.cem_data}")
    if args.stage != "bc" and args.bc_params is None and not args.dry_run:
        parser.error("PPO stages require --bc-params for the fixed normalizer")
    if args.eval_only and (args.stage == "bc" or args.restore_checkpoint is not None):
        parser.error("--eval-only evaluates BC at a PPO stage without --restore-checkpoint")
    direct_start = args.static_curriculum and args.stage == "slightly_open"
    if args.stage not in ("bc", "rolling_orbit") and args.restore_checkpoint is None and not args.eval_only and not direct_start:
        parser.error("stages after rolling_orbit must restore the preceding PPO checkpoint")
    if args.stage == "rolling_orbit" and args.restore_checkpoint is not None:
        parser.error("rolling_orbit starts from BC; do not pass --restore-checkpoint")
    if args.initial_policy_std <= 0.001 or args.updates_per_batch < 1:
        parser.error("initial policy std must exceed 0.001; updates must be positive")
    if args.bc_steps < 1 or args.bc_batch_size < 1:
        parser.error("BC step and batch counts must be positive")
    for value, name in (
        (args.learning_rate, "learning rate"),
        (args.bc_learning_rate, "BC learning rate"),
        (args.initial_policy_std, "initial policy std"),
        (args.max_kl, "max KL"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    return args


def _controller_qpos_indices(model):
    import mujoco

    result = []
    for name in CONTROLLER_JOINT_NAMES_3D:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing controller joint: {name}")
        result.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(result)


def _train_bc(args, stage_out):
    import flax.linen as linen
    import jax
    import jax.numpy as jp
    import mujoco
    import optax
    from brax.io import model as model_io

    task = StandToRollConfig()
    center, scale = action_center_and_scale(task)
    model = mujoco.MjModel.from_xml_path(str(model_path_3d(task.geometry)))
    observations, targets = build_cem_bc_dataset(
        args.cem_data,
        controller_qpos_indices=_controller_qpos_indices(model),
        controller_actuator_indices=np.asarray([
            model.actuator(f"{name}_servo").id for name in CONTROLLER_JOINT_NAMES_3D
        ]),
        action_center=center,
        action_scale=scale,
    )
    # A contiguous holdout plus a history-sized gap avoids shared frames.
    split = int(len(observations) * 0.8)
    validation_start = split + 20
    if split < 1 or validation_start >= len(observations):
        raise ValueError("CEM dataset too short for a temporal holdout with a 20-frame gap")
    normalizer = observation_normalizer(observations[:split])
    train_ids = np.arange(split)
    validation_ids = np.arange(validation_start, len(observations))
    startup_ids = None
    startup_validation_ids = np.asarray([], dtype=int)
    if args.startup_data is not None:
        with np.load(args.startup_data) as bank:
            if int(bank["schema_version"]) != 1:
                raise ValueError("Unsupported startup dataset schema")
            startup_obs = np.asarray(bank["observations"], dtype=np.float32)
            startup_actions = np.asarray(bank["actions"], dtype=np.float32)
            episodes = np.asarray(bank["episode_id"])
            startup_mask = np.asarray(bank["startup"], dtype=bool)
            if (not np.allclose(bank["action_center"], center)
                    or not np.allclose(bank["action_scale"], scale)):
                raise ValueError("Startup action mapping differs from actor")
        n = len(startup_obs)
        if (startup_obs.shape != (n, 720) or startup_actions.shape != (n, 12)
                or episodes.shape != (n,) or startup_mask.shape != (n,)
                or not np.isfinite(startup_obs).all() or not np.isfinite(startup_actions).all()
                or np.max(np.abs(startup_actions)) > 1.0):
            raise ValueError("Invalid startup observation/action arrays")
        unique = np.unique(episodes)
        if len(unique) < 5:
            raise ValueError("At least five successful startup episodes required")
        train_episodes = unique[:max(1, int(len(unique) * 0.8))]
        training = np.isin(episodes, train_episodes)
        offset = len(observations)
        startup_ids = offset + np.flatnonzero(training & startup_mask)
        steady_ids = offset + np.flatnonzero(training & ~startup_mask)
        startup_validation_ids = offset + np.flatnonzero(~training & startup_mask)
        if not len(startup_ids) or not len(startup_validation_ids):
            raise ValueError("Startup windows missing in train/held-out episodes")
        train_ids = np.concatenate((train_ids, steady_ids))
        validation_ids = np.concatenate((validation_ids, offset + np.flatnonzero(~training)))
        observations = np.concatenate((observations, startup_obs))
        targets = np.concatenate((targets, startup_actions))
        normalizer = observation_normalizer(observations[np.concatenate((train_ids, startup_ids))])
    initial_params = None
    if args.bc_params is not None:
        normalizer, initial_params = model_io.load_params(args.bc_params)
        if int(normalizer.get("contract_version", 0)) != BC_CONTRACT_VERSION:
            raise ValueError("BC fine-tuning requires v2 parameters")
    normalized = preprocess_observation(np, observations, normalizer)

    class BCPolicy(linen.Module):
        hidden_layers: tuple[int, ...]

        @linen.compact
        def __call__(self, observation):
            value = observation
            for index, width in enumerate(self.hidden_layers):
                value = linen.Dense(width, name=f"hidden_{index}")(value)
                value = linen.elu(value)
            return jp.tanh(linen.Dense(12, name="location")(value))

    policy = BCPolicy(tuple(args.hidden_layers))
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    params = policy.init(init_key, jp.zeros((1, 720), dtype=jp.float32))
    if initial_params is not None:
        if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(initial_params):
            raise ValueError("BC fine-tuning network structure mismatch")
        if any(a.shape != np.shape(b) for a, b in zip(
                jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(initial_params))):
            raise ValueError("BC fine-tuning network widths mismatch")
        params = jax.tree_util.tree_map(jp.asarray, initial_params)
    optimizer = optax.adam(args.bc_learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(current, opt_state, obs, target):
        def loss_fn(p):
            prediction = policy.apply(p, obs)
            error = prediction - target
            return jp.mean(jp.square(error)), (
                jp.sqrt(jp.mean(jp.square(error))), jp.max(jp.abs(error))
            )
        (loss, diagnostics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(current)
        updates, next_opt_state = optimizer.update(gradients, opt_state, current)
        return optax.apply_updates(current, updates), next_opt_state, loss, diagnostics

    history = []
    for step in range(args.bc_steps):
        rng, batch_key = jax.random.split(rng)
        if startup_ids is None:
            indices = jp.asarray(train_ids)[jax.random.randint(
                batch_key, (args.bc_batch_size,), 0, len(train_ids))]
        else:
            startup_key, steady_key = jax.random.split(batch_key)
            count = max(1, args.bc_batch_size // 2)
            indices = jp.concatenate((
                jp.asarray(startup_ids)[jax.random.randint(startup_key, (count,), 0, len(startup_ids))],
                jp.asarray(train_ids)[jax.random.randint(steady_key, (args.bc_batch_size - count,), 0, len(train_ids))],
            ))
        params, optimizer_state, loss, diagnostics = update(
            params,
            optimizer_state,
            jp.asarray(normalized)[indices],
            jp.asarray(targets)[indices],
        )
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.bc_steps:
            row = {
                "step": step + 1,
                "loss": float(loss),
                "rmse": float(diagnostics[0]),
                "max_abs_error": float(diagnostics[1]),
            }
            history.append(row)
            if step == 0:
                print("\n  BC update       Loss          RMSE       Max error", flush=True)
            print(f"  {step + 1:>6,}/{args.bc_steps:<6,}  {row['loss']:>11.5g}  "
                  f"{row['rmse']:>11.5f}  {row['max_abs_error']:>11.5f}", flush=True)

    checkpoint = (
        {name: np.asarray(value) for name, value in normalizer.items()},
        jax.tree_util.tree_map(np.asarray, params),
    )
    model_io.save_params(stage_out / "bc_params", checkpoint)
    report = {
        "contract_version": BC_CONTRACT_VERSION,
        "training_samples": len(train_ids) + (len(startup_ids) if startup_ids is not None else 0),
        "startup_training_samples": len(startup_ids) if startup_ids is not None else 0,
        "startup_data": str(args.startup_data) if args.startup_data else None,
        "initialized_from": str(args.bc_params) if args.bc_params else None,
        "validation_samples": len(validation_ids),
        "startup_validation_rmse": (float(jp.sqrt(jp.mean(jp.square(
            policy.apply(params, jp.asarray(normalized[startup_validation_ids]))
            - jp.asarray(targets[startup_validation_ids]))))) if len(startup_validation_ids) else None),
        "validation_rmse": float(jp.sqrt(jp.mean(jp.square(
            policy.apply(params, jp.asarray(normalized[validation_ids]))
            - jp.asarray(targets[validation_ids]))))),
        "samples": int(observations.shape[0]),
        "observation_size": int(observations.shape[1]),
        "action_size": int(targets.shape[1]),
        "cem_data": str(args.cem_data.resolve()),
        "action_center": center.tolist(),
        "action_scale": scale.tolist(),
        "history": history,
    }
    (stage_out / "bc_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _train_ppo(args, stage_out):
    import jax
    import jax.numpy as jp
    import jax.nn as jnn
    from brax.io import model as model_io
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from curl_robot_2d_mjx.environment_stand_to_roll_3d import (
        make_stand_to_roll_env_3d,
    )

    bc_normalizer, bc_params = model_io.load_params(args.bc_params)
    if args.restore_checkpoint is not None:
        checkpoint = args.restore_checkpoint.resolve()
        config_path = next((parent / "training_config.json"
                            for parent in (checkpoint, *checkpoint.parents)
                            if (parent / "training_config.json").is_file()), None)
        if config_path is None:
            raise ValueError("Restore requires the preceding stage's training_config.json")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        expected = STAND_TO_ROLL_CURRICULUM_STAGES[
            STAND_TO_ROLL_CURRICULUM_STAGES.index(args.stage) - 1]
        if (previous.get("pipeline") != "one_policy_bc_then_snapshot_reset_curriculum_v2"
                or previous.get("stage") != expected
                or previous.get("bc_sha256") != hashlib.sha256(args.bc_params.read_bytes()).hexdigest()):
            raise ValueError("Restore must use the preceding v2 stage and identical BC normalization/weights")
    if int(bc_normalizer.get("contract_version", 0)) != BC_CONTRACT_VERSION:
        raise ValueError("Legacy BC checkpoint: retrain --stage bc into a new output directory (contract v2)")
    mean = jp.asarray(bc_normalizer["mean"])
    std = jp.asarray(bc_normalizer["std"])
    if (mean.shape != (720,) or std.shape != (720,) or bool(jp.any(std <= 0.0))
            or not bool(jp.all(jp.isfinite(mean))) or not bool(jp.all(jp.isfinite(std)))):
        raise ValueError("BC normalizer must contain positive 720-value mean/std")
    bc_params = jax.tree_util.tree_map(jp.asarray, bc_params)
    task = stand_to_roll_curriculum_config(args.stage)
    if args.static_eval:
        task = replace(task, reset_velocity_noise_rad_s=0.0, snapshot_reset_probability=0.0)
    if args.static_curriculum:
        task = replace(task, reset_velocity_noise_rad_s=0.0,
                       snapshot_reset_probability=0.0, observation_noise_enabled=False)
    train_env = make_stand_to_roll_env_3d(task, matcher_npz=args.cem_data, seed=args.seed)
    eval_env = make_stand_to_roll_env_3d(
        StandToRollConfig(**{**asdict(task), "observation_noise_enabled": False}),
        matcher_npz=args.cem_data,
        seed=args.seed + 10_000,
    )
    if train_env.observation_size != 720 or train_env.action_size != 12:
        raise RuntimeError("stand-to-roll actor contract must be obs=720, action=12")

    def fixed_preprocess(observation, unused_statistics):
        del unused_statistics
        return preprocess_observation(jp, observation, {
            "mean": mean, "std": std, "clip": float(bc_normalizer["clip"]),
        })

    def network_factory(observation_size, action_size, preprocess_observations_fn):
        del preprocess_observations_fn
        networks = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=fixed_preprocess,
            policy_hidden_layer_sizes=tuple(args.hidden_layers),
            value_hidden_layer_sizes=tuple(args.hidden_layers),
            activation=jnn.elu,
            distribution_type="tanh_normal",
        )
        if args.restore_checkpoint is None:
            original_init = networks.policy_network.init

            def init(key):
                return initialize_ppo_actor_from_bc(
                    jp,
                    original_init(key),
                    bc_params,
                    hidden_layers=tuple(args.hidden_layers),
                    initial_std=args.initial_policy_std,
                )

            networks = replace(
                networks,
                policy_network=replace(networks.policy_network, init=init),
            )
        return networks

    preset = dict(PRESETS[args.preset])
    if args.steps is not None:
        preset["steps"] = args.steps
    if args.eval_only:
        networks = network_factory(720, 12, fixed_preprocess)
        actor = networks.policy_network.init(jax.random.PRNGKey(args.seed))
        def policy(obs):
            logits = networks.policy_network.apply(None, actor, obs)
            return networks.parametric_action_distribution.mode(logits)
        # Check the actual Brax distribution output against the BC forward pass.
        reset = jax.jit(jax.vmap(eval_env.reset))
        step_env = jax.jit(jax.vmap(eval_env.step))
        act = jax.jit(policy)
        state = reset(jax.random.split(jax.random.PRNGKey(args.seed), preset["eval_envs"]))
        value = fixed_preprocess(state.obs, None)
        for i in range(len(args.hidden_layers)):
            layer = bc_params["params"][f"hidden_{i}"]
            value = jnn.elu(value @ layer["kernel"] + layer["bias"])
        layer = bc_params["params"]["location"]
        expected = jp.tanh(value @ layer["kernel"] + layer["bias"])
        error = float(jp.max(jp.abs(act(state.obs) - expected)))
        if not math.isfinite(error) or error > 1e-5:
            raise RuntimeError(f"BC/PPO deterministic output mismatch: {error}")
        alive = jp.ones((preset["eval_envs"],), dtype=bool)
        totals = {name: jp.zeros_like(alive, dtype=jp.float32) for name in state.metrics}
        lengths = jp.zeros_like(alive, dtype=jp.float32)
        for _ in range(task.episode_length):
            state = step_env(state, act(state.obs))
            totals = {name: value + jp.where(alive, state.metrics[name], 0.0)
                      for name, value in totals.items()}
            lengths = lengths + alive
            alive = alive & (~state.done.astype(bool))
        result = {name: float(jp.mean(value)) for name, value in totals.items()}
        result["avg_episode_length"] = float(jp.mean(lengths))
        result["bc_ppo_max_action_error"] = error
        result["reset_velocity_noise_rad_s"] = task.reset_velocity_noise_rad_s
        result["snapshot_reset_probability"] = task.snapshot_reset_probability
        (stage_out / "bc_closed_loop_eval.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    progress_history = []

    def progress(step, metrics):
        row = {"step": int(step)}
        for name, value in metrics.items():
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                row[name] = float(value.item())
        progress_history.append(row)
        (stage_out / "metrics_history.json").write_text(
            json.dumps(progress_history, indent=2) + "\n", encoding="utf-8")
        kl = row.get("training/kl_mean", 0.0)
        if any(not math.isfinite(v) for v in row.values()) or kl > args.max_kl:
            (stage_out / "training_aborted.json").write_text(
                json.dumps({"reason": "nonfinite metrics or excessive KL", "metrics": row}, indent=2)
                + "\n", encoding="utf-8")
            _print_eval(f"PPO {args.stage} | ABORTED at {int(step):,} steps", row, training=True)
            raise RuntimeError(f"PPO instability at step {step}: KL={kl}; see training_aborted.json")
        percent = min(100.0, 100.0 * int(step) / preset["steps"])
        elapsed = time.perf_counter() - started
        _print_eval(
            f"PPO {args.stage} | {int(step):,}/{preset['steps']:,} ({percent:.1f}%)"
            f" | {elapsed / 60:.1f} min", row, training=True)

    train_parameters = inspect.signature(ppo.train).parameters
    kwargs = {}
    if "save_checkpoint_path" in train_parameters:
        kwargs["save_checkpoint_path"] = str((stage_out / "ppo_checkpoint").resolve())
    else:
        raise RuntimeError("Brax PPO must support save_checkpoint_path")
    if "max_grad_norm" in train_parameters:
        kwargs["max_grad_norm"] = 0.5
    if args.restore_checkpoint is not None:
        if "restore_checkpoint_path" not in train_parameters:
            raise RuntimeError("installed Brax PPO cannot restore curriculum checkpoints")
        kwargs["restore_checkpoint_path"] = str(args.restore_checkpoint.resolve())

    started = time.perf_counter()
    _, params, final_metrics = ppo.train(
        environment=train_env,
        eval_env=eval_env,
        num_timesteps=preset["steps"],
        episode_length=task.episode_length,
        action_repeat=1,
        num_envs=preset["envs"],
        num_eval_envs=preset["eval_envs"],
        num_evals=preset["num_evals"],
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=1.0,
        unroll_length=args.unroll_length,
        batch_size=preset["batch_size"],
        num_minibatches=preset["num_minibatches"],
        num_updates_per_batch=args.updates_per_batch,
        normalize_observations=False,
        deterministic_eval=True,
        network_factory=network_factory,
        seed=args.seed,
        progress_fn=progress,
        **kwargs,
    )
    model_io.save_params(stage_out / "params_final", params)
    clean_metrics = {name: float(value) for name, value in (final_metrics or {}).items()}
    capture_rate = clean_metrics.get("eval/episode_captured", 0.0)
    failure_rate = clean_metrics.get("eval/episode_failed", 1.0)
    summary = {
        "stage": args.stage,
        "task": asdict(task),
        "elapsed_s": time.perf_counter() - started,
        "final_metrics": clean_metrics,
        "capture_rate": capture_rate,
        "failure_rate": failure_rate,
        "stage_passed": (capture_rate >= 0.80 and failure_rate <= 0.20
                         and clean_metrics.get("eval/episode_sustained_success", 0.0) >= 0.80),
        "next_stage": (
            STAND_TO_ROLL_CURRICULUM_STAGES[
                STAND_TO_ROLL_CURRICULUM_STAGES.index(args.stage) + 1
            ]
            if args.stage != STAND_TO_ROLL_CURRICULUM_STAGES[-1]
            else None
        ),
    }
    (stage_out / "metrics_history.json").write_text(
        json.dumps(progress_history, indent=2) + "\n", encoding="utf-8"
    )
    (stage_out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    task = None if args.stage == "bc" else stand_to_roll_curriculum_config(args.stage)
    if args.static_eval:
        task = replace(task, reset_velocity_noise_rad_s=0.0, snapshot_reset_probability=0.0)
    if args.static_curriculum:
        task = replace(task, reset_velocity_noise_rad_s=0.0,
                       snapshot_reset_probability=0.0, observation_noise_enabled=False)
    payload = {
        "pipeline": "one_policy_bc_then_snapshot_reset_curriculum_v2",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "stage": args.stage,
        "actor_observation": "train_ppo_deploy 36x20 newest-first",
        "actor_observation_size": ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
        "action_size": STAND_TO_ROLL_ACTION_SIZE,
        "cem_online_control": False,
        "teacher_shaping_annealed": False,
        "task": asdict(task) if task else None,
        "preset": {**PRESETS[args.preset], **({"steps": args.steps} if args.steps is not None else {})},
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return payload
    stage_out = args.out / (f"eval_bc_{args.stage}" if args.eval_only else args.stage)
    if stage_out.exists() and any(stage_out.iterdir()):
        raise SystemExit(f"output directory is not empty: {stage_out}")
    configure_cloud_runtime(
        memory_fraction=args.memory_fraction,
        preallocate=False,
        xla_triton=False,
        mujoco_gl=args.mujoco_gl,
        verbose=True,
    )
    stage_out.mkdir(parents=True, exist_ok=True)
    payload["runtime"] = describe_runtime()
    if args.bc_params is not None:
        payload["bc_sha256"] = hashlib.sha256(args.bc_params.read_bytes()).hexdigest()
    (stage_out / "training_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    mode = "BC evaluation" if args.eval_only else ("BC training" if args.stage == "bc" else "PPO training")
    print("\n" + "=" * 68)
    print(f"{mode} | {args.stage}")
    if args.stage != "bc":
        p = payload["preset"]
        print(f"  Budget    {p['steps']:,} steps | envs {p['envs']} | eval envs {p['eval_envs']}")
        print(f"  Reset     alpha [{task.reset_alpha_min:.2f}, {task.reset_alpha_max:.2f}]"
              f" | snapshots {task.snapshot_reset_probability:.0%}")
        print(f"  Noise     velocity +/-{task.reset_velocity_noise_rad_s:g}"
              f" | observation {'on' if task.observation_noise_enabled else 'off'}")
    print(f"  Output    {stage_out}")
    print("=" * 68, flush=True)
    result = _train_bc(args, stage_out) if args.stage == "bc" else _train_ppo(args, stage_out)
    if args.stage == "bc":
        print(f"\nBC complete | samples {result['samples']:,}")
        print(f"  Validation RMSE: {result['validation_rmse']:.5f}")
        if result.get("startup_validation_rmse") is not None:
            print(f"  Startup validation RMSE: {result['startup_validation_rmse']:.5f}")
        report_name = "bc_summary.json"
    elif args.eval_only:
        _print_eval(f"BC evaluation complete | {args.stage}", result)
        print(f"  BC/PPO action mismatch: {result['bc_ppo_max_action_error']:.3g}")
        report_name = "bc_closed_loop_eval.json"
    else:
        status = "PASS" if result["stage_passed"] else "NOT PASSED"
        print(f"\nStage {args.stage}: {status} | elapsed {result['elapsed_s'] / 60:.1f} min")
        print("  Required: sustained >=80%, capture >=80%, failed <=20%")
        if result["stage_passed"] and result["next_stage"]:
            print(f"  Next stage: {result['next_stage']}")
        report_name = "summary.json"
    print(f"  Full report: {stage_out / report_name}", flush=True)
    return result


if __name__ == "__main__":
    main()
