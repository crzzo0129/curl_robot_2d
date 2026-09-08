"""Small, gated DAgger experiment for the v2 stand-to-roll BC actor.

The teacher is a phase-matched lookup of recorded CEM next actions, not a
proven recovery controller. Its closed-loop gate must pass before labeling.
No local validation has been performed; run this on the cloud MJX runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.runtime import configure_cloud_runtime
from curl_robot_2d_mjx.stand_to_roll_training import (
    BC_CONTRACT_VERSION, build_cem_bc_dataset, preprocess_observation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-params", type=Path, required=True)
    parser.add_argument("--cem-data", type=Path, default=Path("results/cem_cycle_data/cem_cycles.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/stand_to_roll_dagger"))
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.envs < 1 or args.updates < 1:
        parser.error("envs and updates must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    configure_cloud_runtime(memory_fraction=0.85, preallocate=False,
                            xla_triton=False, mujoco_gl="disable", verbose=True)
    import jax
    import jax.numpy as jp
    import optax
    from brax.io import model as model_io
    from curl_robot_2d_mjx.config_stand_to_roll import stand_to_roll_curriculum_config
    from curl_robot_2d_mjx.environment_stand_to_roll_3d import make_stand_to_roll_env_3d
    from mujoco import mjx

    norm, params = model_io.load_params(args.bc_params)
    if int(norm.get("contract_version", 0)) != BC_CONTRACT_VERSION:
        raise ValueError("DAgger requires v2 BC parameters")
    params = jax.tree_util.tree_map(jp.asarray, params)
    frozen_norm = jax.tree_util.tree_map(jp.asarray, norm)
    task = replace(stand_to_roll_curriculum_config("rolling_orbit"),
                   observation_noise_enabled=False)
    env = make_stand_to_roll_env_3d(task, matcher_npz=args.cem_data, seed=args.seed)
    offline_obs, offline_actions = build_cem_bc_dataset(
        args.cem_data, controller_qpos_indices=np.asarray(env.controller_qpos_indices),
        controller_actuator_indices=np.asarray(env.controller_actuator_indices),
        action_center=np.asarray(env.action_center), action_scale=np.asarray(env.action_scale))
    with np.load(args.cem_data) as bank:
        phases = np.asarray(bank["cem_phase_wrapped"])[19:-1]
    # Precompute a compact phase table. Each label is the NEXT recorded action.
    grid = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
    delta = phases[:, None] - grid[None, :]
    nearest = np.argmin(np.abs(np.arctan2(np.sin(delta), np.cos(delta))), axis=0)
    teacher_table = jp.asarray(offline_actions[nearest])
    hidden_count = len(params["params"]) - 1

    def actor(p, obs):
        value = preprocess_observation(jp, obs, frozen_norm)
        for i in range(hidden_count):
            layer = p["params"][f"hidden_{i}"]
            value = jax.nn.elu(value @ layer["kernel"] + layer["bias"])
        layer = p["params"]["location"]
        return jp.tanh(value @ layer["kernel"] + layer["bias"])

    def teacher(state):
        phase, distance = env._match(state.pipeline_state)
        index = jp.mod(jp.rint(phase * (200 / (2 * jp.pi))).astype(jp.int32), 200)
        return teacher_table[index], distance

    def reset(key, perturb):
        state = env.reset(key)
        joint_key, velocity_key, obs_key = jax.random.split(jax.random.fold_in(key, 73), 3)
        data = state.pipeline_state
        joints = jp.clip(data.qpos[env.controller_qpos_indices]
                         + perturb * 0.01 * jax.random.uniform(joint_key, (12,), minval=-1, maxval=1),
                         env.joint_low, env.joint_high)
        data = data.replace(qpos=data.qpos.at[env.controller_qpos_indices].set(joints),
                            qvel=data.qvel + perturb * 0.05 * jax.random.uniform(
                                velocity_key, data.qvel.shape, minval=-1, maxval=1))
        data = mjx.forward(env.mjx_model, data)
        history = state.obs.at[:36].set(env._frame(data, state.info["last_action"], obs_key))
        _, distance = env._match(data)
        info = {**state.info, "history": history, "previous_cem_distance": distance,
                "previous_compact_distance": jp.sqrt(jp.mean(jp.square(joints - env.compact_joint_position)))}
        return state.replace(pipeline_state=data, obs=history, info=info)

    @jax.jit
    def rollout(p, seed, beta, perturb):
        rng = jax.random.PRNGKey(seed)
        states = jax.vmap(reset, in_axes=(0, None))(jax.random.split(rng, args.envs), perturb)
        alive = jp.ones(args.envs, dtype=bool)

        def step(carry, _):
            states, alive, rng = carry
            rng, key = jax.random.split(rng)
            labels, distance = jax.vmap(teacher)(states)
            policy_action = actor(p, states.obs)
            intervene = jax.random.uniform(key, (args.envs,)) < beta
            action = jp.where(intervene[:, None], labels, policy_action)
            next_states = jax.vmap(env.step)(states, action)
            output = (states.obs, labels, alive & (distance < 3.0),
                      {name: jp.where(alive, value, 0.0)
                       for name, value in next_states.metrics.items()}, alive)
            next_states = jax.tree_util.tree_map(
                lambda new, old: jp.where(alive.reshape((args.envs,) + (1,) * (new.ndim - 1)), new, old),
                next_states, states)
            return (next_states, alive & (~next_states.done.astype(bool)), rng), output

        _, output = jax.lax.scan(step, (states, alive, rng), None, length=task.episode_length)
        return output

    def summarize(output):
        metrics, alive = output[3], output[4]
        report = {name: float(jp.mean(jp.sum(value, axis=0))) for name, value in metrics.items()}
        report["avg_episode_length"] = float(jp.mean(jp.sum(alive, axis=0)))
        return report

    def passed(report):
        return (all(np.isfinite(v) for v in report.values())
                and report["sustained_success"] >= 0.8 and report["failed"] <= 0.2)

    def score(report):
        return (report["sustained_success"], -report["failed"], report["roll_progress"])

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"teacher": "phase_matched_recorded_next_action", "rounds": [],
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}

    def save_report():
        (args.out / "dagger_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("[DAgger] Evaluating teacher recovery gate", flush=True)
    for name, perturb in (("teacher_nominal", 0.0), ("teacher_perturbed", 1.0)):
        report[name] = summarize(rollout(params, args.seed + 10000, 1.0, perturb))
        save_report()
        print(f"[DAgger] {name}: {report[name]}", flush=True)
        if not passed(report[name]):
            report["status"] = "teacher_gate_failed"
            save_report()
            print("Teacher cannot recover reliably. No DAgger updates or replacement checkpoint produced.", flush=True)
            return

    selection_seed = args.seed + 20000
    best_report = summarize(rollout(params, selection_seed, 0.0, 1.0))
    report["baseline_selection"] = best_report
    baseline_params = params
    best_params = params
    collected_obs, collected_actions = [], []
    offline_obs, offline_actions = jp.asarray(offline_obs), jp.asarray(offline_actions)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-5))

    @jax.jit
    def update(p, opt_state, obs, target):
        def loss_fn(weights):
            return jp.mean(jp.square(actor(weights, obs) - target))
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    for iteration, beta in enumerate((0.5, 0.25, 0.0), 1):
        print(f"[DAgger] round={iteration} teacher_probability={beta}", flush=True)
        output = rollout(best_params, args.seed + iteration, beta, 1.0)
        valid = np.asarray(output[2]).reshape(-1)
        collected_obs.append(np.asarray(output[0]).reshape(-1, 720)[valid])
        collected_actions.append(np.asarray(output[1]).reshape(-1, 12)[valid])
        obs_bank = jp.asarray(np.concatenate(collected_obs))
        action_bank = jp.asarray(np.concatenate(collected_actions))
        if not len(obs_bank):
            raise RuntimeError("No valid near-orbit DAgger samples")
        params = best_params
        opt_state = optimizer.init(params)
        rng = jax.random.PRNGKey(args.seed + iteration)
        for _ in range(args.updates):
            rng, a, b = jax.random.split(rng, 3)
            original = jax.random.randint(a, (128,), 0, len(offline_obs))
            extra = jax.random.randint(b, (128,), 0, len(obs_bank))
            obs = jp.concatenate((offline_obs[original], obs_bank[extra]))
            targets = jp.concatenate((offline_actions[original], action_bank[extra]))
            params, opt_state, loss = update(params, opt_state, obs, targets)
        candidate = summarize(rollout(params, selection_seed, 0.0, 1.0))
        finite = np.isfinite(float(loss)) and all(np.isfinite(v) for v in candidate.values())
        accepted = finite and score(candidate) > score(best_report)
        if accepted:
            best_params, best_report = params, candidate
        report["rounds"].append({"round": iteration, "samples": len(obs_bank),
                                 "loss": float(loss), "accepted": accepted, "evaluation": candidate})
        save_report()
        print(f"[DAgger] accepted={accepted} success={candidate['sustained_success']:.3f}", flush=True)

    # Final acceptance uses previously unseen reset seeds, with identical resets
    # for baseline and candidate. Never export a failed candidate as improved BC.
    report["baseline_holdout"] = summarize(rollout(baseline_params, args.seed + 30000, 0.0, 1.0))
    report["candidate_holdout"] = summarize(rollout(best_params, args.seed + 30000, 0.0, 1.0))
    accepted = (passed(report["candidate_holdout"])
                and score(report["candidate_holdout"]) > score(report["baseline_holdout"]))
    report["status"] = "passed" if accepted else "student_gate_failed"
    if accepted:
        model_io.save_params(args.out / "bc_params", (norm, jax.tree_util.tree_map(np.asarray, best_params)))
    save_report()
    print(f"[DAgger] {report['status']}; see {args.out / 'dagger_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
