"""Paired, deterministic policy evaluations from one fixed set of reset keys."""

import hashlib
import json

import numpy as np


def summarize_command_groups(totals, forward_commands, yaw_commands, *,
                             speed_min, speed_max, episode_length, minimum_turns):
    """Describe tracking and survival separately, without extra simulator work."""
    steps = np.asarray(totals["steps"])
    failed = np.asarray(totals["failed"]) > 0.5
    success = (~failed) & (np.asarray(totals["turns"]) >= minimum_turns)
    fixed = np.asarray(totals["command_changed"]) < 0.5
    full = steps == episode_length
    edges = np.linspace(speed_min, speed_max, 4)
    speed_bin = (np.zeros(len(steps), dtype=int) if speed_min == speed_max else
                 np.searchsorted(edges[1:-1], forward_commands, side="right"))
    speeds = {name: fixed & (speed_bin == i)
              for i, name in enumerate(("low", "medium", "high"))}
    turns = {"straight": fixed & (np.abs(yaw_commands) <= 1e-3),
             "left_positive_yaw": fixed & (yaw_commands > 1e-3),
             "right_negative_yaw": fixed & (yaw_commands < -1e-3)}

    def summarize(mask):
        count = int(np.sum(mask))
        if not count:
            return {"episodes": 0, "success_rate": None,
                    "forward_mae_m_s": None, "yaw_mae_rad_s": None}
        samples = max(float(np.sum(steps[mask])), 1.0)
        survivors = mask & full & ~failed
        survivor_samples = float(np.sum(steps[survivors]))
        return {
            "episodes": count, "success_count": int(np.sum(success & mask)),
            "success_rate": float(np.mean(success[mask])),
            "failure_rate": float(np.mean(failed[mask])),
            "full_horizon_rate": float(np.mean(full[mask])),
            "mean_steps": float(np.mean(steps[mask])),
            "insufficient_turns_without_failure": int(np.sum(mask & ~failed & ~success)),
            "forward_mae_m_s": float(np.sum(totals["vx_abs"][mask]) / samples),
            "forward_bias_m_s": float(np.sum(totals["vx_signed"][mask]) / samples),
            "yaw_mae_rad_s": float(np.sum(totals["yaw_abs"][mask]) / samples),
            "yaw_bias_rad_s": float(np.sum(totals["yaw_signed"][mask]) / samples),
            "full_horizon_failure_free_episodes": int(np.sum(survivors)),
            "full_horizon_failure_free_forward_mae_m_s": (
                float(np.sum(totals["vx_abs"][survivors]) / survivor_samples)
                if survivor_samples else None),
            "failure_counts": {name: int(np.sum(np.asarray(value)[mask] > 0.5))
                               for name, value in totals.items() if name.startswith("failure_")},
        }

    return {
        "speed_bin_edges_m_s": edges.tolist(),
        "aggregation": "Rates by episode, errors by active transition; survivor metrics have selection bias.",
        "overall": summarize(np.ones(len(steps), dtype=bool)),
        "by_speed": {name: summarize(mask) for name, mask in speeds.items()},
        "by_turn": {name: summarize(mask) for name, mask in turns.items()},
        "by_speed_and_turn": {f"{speed}/{turn}": summarize(sm & tm)
                              for speed, sm in speeds.items() for turn, tm in turns.items()},
        "variable_command_episodes": summarize(~fixed),
    }


def make_fixed_evaluator(env, networks, anchor_policy, *, count, seed,
                         episode_length, minimum_turns, speed_bounds=None,
                         observation_noise_scale=0.0):
    import jax
    import jax.numpy as jp
    from curl_robot_2d_mjx.distillation_execution import timed_stage

    # This uses the unwrapped env: a finished episode is frozen, never reset.
    reset_keys = jax.random.split(jax.random.PRNGKey(seed), count)
    with timed_stage("fixed PPO evaluation initial states"):
        initial = jax.jit(jax.vmap(env.reset))(reset_keys)
        jax.block_until_ready(initial)
    step_batch = jax.jit(jax.vmap(env.step))
    failures = tuple(name for name in initial.metrics if name.startswith("failure_"))
    rewards = tuple(name for name in initial.metrics if name.startswith("reward_"))
    fingerprint = hashlib.sha256()
    for value in jax.tree_util.tree_leaves(initial):
        array = np.asarray(jax.device_get(value))
        fingerprint.update(str((array.shape, array.dtype)).encode())
        fingerprint.update(array.tobytes())
    manifest = {
        "seed": seed, "episodes": count, "episode_length": episode_length,
        "initial_state_sha256": fingerprint.hexdigest(),
        "reset_keys": np.asarray(jax.device_get(reset_keys)).tolist(),
        "forward_commands_m_s": np.asarray(jax.device_get(initial.info["forward_velocity_command"])).tolist(),
        "yaw_commands_rad_s": np.asarray(jax.device_get(initial.info["yaw_rate_command"])).tolist(),
        "observation_noise_scale": observation_noise_scale,
        "description": "Identical initial state, history, commands and RNG for every policy; failed episodes freeze.",
    }

    @jax.jit
    def rollout(params):
        z = jp.zeros((count,))
        totals = {
            "steps": z, "return": z, "turns": z,
            "vx_abs": z, "vx_signed": z, "yaw_abs": z, "yaw_signed": z,
            "action_error_sq": z, "std_sum": z, "saturation": z,
            **{name: z for name in failures},
            **{name: z for name in rewards},
            "failed": z, "non_lateral_failed": z, "command_changed": z,
        }

        def advance(carry, _):
            state, active, totals = carry
            logits = networks.policy_network.apply(params[0], params[1], state.obs)
            dist = networks.parametric_action_distribution
            action = dist.mode(logits)
            reference = jax.vmap(anchor_policy)(state.obs["state"])
            candidate = step_batch(state, action)
            increment = {
                "steps": jp.ones_like(z), "return": candidate.reward,
                "turns": candidate.metrics["roll_progress_rad"] / (2 * jp.pi),
                "vx_abs": candidate.metrics["forward_velocity_error_abs_m_s"],
                "vx_signed": candidate.metrics["forward_velocity_error_m_s"],
                "yaw_abs": candidate.metrics["yaw_rate_error_abs_rad_s"],
                "yaw_signed": (candidate.metrics["rolling_axis_heading_rate_rad_s"]
                               - candidate.metrics["yaw_rate_command_rad_s"]),
                "action_error_sq": jp.mean(jp.square(action - reference), axis=-1),
                "std_sum": jp.mean(dist.create_dist(logits).scale, axis=-1),
                "saturation": jp.mean((jp.abs(action) > 0.95).astype(jp.float32), axis=-1),
                **{name: candidate.metrics[name] for name in rewards},
            }
            next_totals = dict(totals)
            for name, value in increment.items():
                next_totals[name] = totals[name] + jp.where(active, value, 0.0)
            for name in (*failures, "failed"):
                next_totals[name] = jp.maximum(totals[name], jp.where(active, candidate.metrics[name], 0.0))
            next_totals["non_lateral_failed"] = jp.maximum(
                totals["non_lateral_failed"], jp.where(active, candidate.metrics["failed_non_lateral"], 0.0))
            changed = ((jp.abs(candidate.info["forward_velocity_command"] - initial.info["forward_velocity_command"]) > 1e-6)
                       | (jp.abs(candidate.info["yaw_rate_command"] - initial.info["yaw_rate_command"]) > 1e-6))
            next_totals["command_changed"] = jp.maximum(
                totals["command_changed"], (active & changed).astype(jp.float32))

            def keep(new, old):
                mask = active.reshape(active.shape + (1,) * (new.ndim - active.ndim))
                return jp.where(mask, new, old)

            next_state = jax.tree_util.tree_map(keep, candidate, state)
            return (next_state, active & (candidate.done < 0.5), next_totals), None

        (_, _, totals), _ = jax.lax.scan(
            advance, (initial, jp.ones((count,), dtype=jp.bool_), totals),
            xs=None, length=episode_length,
        )
        return totals

    def evaluate(params):
        # Brax callbacks give unreplicated params; keep the diagnostic on one
        # device, like Brax's own evaluator, rather than changing PPO sharding.
        params = jax.device_put(jax.tree_util.tree_map(np.asarray, params), jax.local_devices()[0])
        with timed_stage("fixed PPO policy evaluation"):
            totals = jax.device_get(rollout(params))
        samples = max(float(np.sum(totals["steps"])), 1.0)
        success = (totals["failed"] < 0.5) & (totals["turns"] >= minimum_turns)
        return {
            "episodes": count,
            "diagnostics_finite": bool(all(np.all(np.isfinite(value)) for value in totals.values())),
            "initial_state_sha256": manifest["initial_state_sha256"],
            "success_rate": float(np.mean(success)),
            "failure_rate": float(np.mean(totals["failed"] > 0.5)),
            "non_lateral_failure_rate": float(np.mean(totals["non_lateral_failed"] > 0.5)),
            "full_horizon_rate": float(np.mean(totals["steps"] == episode_length)),
            "mean_steps": float(np.mean(totals["steps"])),
            "mean_turns": float(np.mean(totals["turns"])),
            "mean_return": float(np.mean(totals["return"])),
            "forward_mae_m_s": float(np.sum(totals["vx_abs"]) / samples),
            "forward_bias_m_s": float(np.sum(totals["vx_signed"]) / samples),
            "yaw_mae_rad_s": float(np.sum(totals["yaw_abs"]) / samples),
            "yaw_bias_rad_s": float(np.sum(totals["yaw_signed"]) / samples),
            "same_state_student_action_rmse": float(np.sqrt(np.sum(totals["action_error_sq"]) / samples)),
            "mean_pre_tanh_policy_std": float(np.sum(totals["std_sum"]) / samples),
            "action_saturation_fraction": float(np.sum(totals["saturation"]) / samples),
            "failure_counts": {name: int(np.sum(totals[name] > 0.5)) for name in failures},
            "reward_mean_per_active_step": {name: float(np.sum(totals[name]) / samples) for name in rewards},
            "command_evaluation": (summarize_command_groups(
                totals, np.asarray(manifest["forward_commands_m_s"]),
                np.asarray(manifest["yaw_commands_rad_s"]), speed_min=speed_bounds[0],
                speed_max=speed_bounds[1], episode_length=episode_length,
                minimum_turns=minimum_turns) if speed_bounds is not None else None),
            "per_episode": {name: np.asarray(value).tolist() for name, value in totals.items()},
        }

    return evaluate, manifest


def write_json(path, payload):
    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [finite_json(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(finite_json(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
