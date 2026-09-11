"""Generate reusable rolling reset states before PPO, never during resets."""

import numpy as np


def build_cem_snapshot_pool(teacher_env, observation_env, *, count, seed,
                            min_steps, max_steps, num_devices=1):
    import jax
    import jax.numpy as jp
    from curl_robot_2d_mjx.deployment_rolling_3d import (
        initial_rolling_deploy_history_3d,
        effective_action_to_controller_action_3d,
    )
    from curl_robot_2d_mjx.distillation_execution import BatchExecution, timed_stage

    execution = BatchExecution(num_devices)

    def generate_one(key):
        reset_key, warmup_key = jax.random.split(key)
        state = teacher_env.reset(reset_key)
        history = initial_rolling_deploy_history_3d(jp)
        previous = jp.zeros((12,))
        warmup_steps = jax.random.randint(warmup_key, (), min_steps, max_steps + 1)

        def advance(carry, index):
            current, old_history, old_previous = carry
            new_history = observation_env._actor_observation(
                current, old_history, old_previous, jp.zeros((12,)),
                jax.random.fold_in(key, index),
            )
            candidate = teacher_env.step(current, jp.zeros((8,)))
            new_previous = effective_action_to_controller_action_3d(
                jp, candidate.info["last_action"]
            )
            take = (index < warmup_steps) & (candidate.done < 0.5)
            return jax.tree_util.tree_map(
                lambda new, old: jp.where(take, new, old),
                (candidate, new_history, new_previous), carry,
            ), None

        result, _ = jax.lax.scan(advance, (state, history, previous), jp.arange(max_steps))
        return result

    generate = execution.batch_jit(jax.vmap(generate_one))
    with timed_stage(f"PPO CEM snapshot pool seed={seed} candidates={count}"):
        pool = jax.device_get(generate(jax.random.split(jax.random.PRNGKey(seed), count)))
    state, _, _ = pool
    # A warmup timer alone does not prove the robot is rolling. Reject stalled,
    # nonfinite, terminal and very early states instead of silently using compact.
    valid = ((np.asarray(state.info["step_count"]) >= min_steps)
             & (np.asarray(state.done) < 0.5)
             & (np.asarray(state.metrics["failed"]) < 0.5)
             & (np.asarray(state.metrics["forward_velocity_m_s"]) > 0.05)
             & (np.abs(np.asarray(state.pipeline_state.qvel)[:, 4]) > 0.5)
             & np.all(np.isfinite(np.asarray(state.pipeline_state.qpos)), axis=-1)
             & np.all(np.isfinite(np.asarray(state.pipeline_state.qvel)), axis=-1))
    indices = np.flatnonzero(valid)
    if len(indices) < max(4, count // 8):
        raise RuntimeError(
            f"Only {len(indices)}/{count} valid rolling snapshots. "
            "Check the CEM reference or increase warmup; refusing compact reset fallback."
        )
    pool = jax.tree_util.tree_map(lambda value: np.asarray(value)[indices], pool)
    selected = pool[0]
    summary = {
        "source": "cem_rolling_proxy_not_actual_stand_to_roll_handoff",
        "seed": seed, "candidate_count": count, "accepted_count": len(indices),
        "warmup_min_steps": min_steps, "warmup_max_steps": max_steps,
        "forward_commands_m_s": np.asarray(selected.info["forward_velocity_command"]).tolist(),
        "yaw_commands_rad_s": np.asarray(selected.info["yaw_rate_command"]).tolist(),
        "actual_warmup_steps": np.asarray(selected.info["step_count"]).tolist(),
        "minimum_forward_speed_m_s": 0.05, "minimum_abs_roll_rate_rad_s": 0.5,
    }
    print(f"[PPO snapshots] retained {len(indices)}/{count}; "
          "resets sample this pool without teacher warmup", flush=True)
    return pool, summary
