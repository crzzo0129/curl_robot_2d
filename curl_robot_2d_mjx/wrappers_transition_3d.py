"""Complete PPO episode resets, without changing live ROLL handoff semantics."""


def select_reset_lanes(xp, done, fresh, current):
    """Select entire per-environment leaves, including histories and RNG keys."""
    mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
    return xp.where(mask, fresh, current)


def wrap_transition_3d(env, episode_length, action_repeat=1, randomization_fn=None):
    """Brax wrap_env_fn: vmap + episode accounting + full-state auto-reset.

    Return terminal reward/done/metrics/truncation, but fresh physics, actor
    input and task memory for ended lanes. Clear episode accounting only on
    the NEXT step, after PPO/EvalWrapper have consumed terminal statistics.
    Fresh reset RNG resamples curriculum perturbations/snapshot indices.
    """
    import jax
    import jax.numpy as jp
    from brax.envs.base import Wrapper
    from brax.envs.wrappers import training

    if action_repeat != 1 or randomization_fn is not None:
        raise ValueError("Transition owns action repeat; wrapper randomization is unsupported")

    class FullResetWrapper(Wrapper):
        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(info={
                **state.info, "transition_reset_rng": rng,
                "transition_needs_reset": jp.zeros_like(state.done, dtype=bool),
            })

        def step(self, state, action):
            info = dict(state.info)
            pending = info["transition_needs_reset"]
            choose_pending = lambda fresh, old: select_reset_lanes(jp, pending, fresh, old)
            for name in ("steps", "truncation", "episode_done", "episode_metrics", "time_out"):
                info[name] = jax.tree_util.tree_map(
                    lambda old: choose_pending(jp.zeros_like(old), old), info[name])
            state = state.replace(info=info, done=jp.zeros_like(state.done))
            terminal = self.env.step(state, action)
            done = terminal.done > 0

            def reset_ended(current):
                keys = jax.vmap(lambda key: jax.random.split(key, 2))(
                    current.info["transition_reset_rng"])
                fresh = self.env.reset(keys[:, 0])
                choose = lambda new, old: select_reset_lanes(jp, done, new, old)
                next_info = dict(current.info)
                # Preserve terminal accounting, including time_out for PPO
                # bootstrap. Everything owned by the task is reset immediately.
                terminal_fields = {"steps", "truncation", "episode_done",
                                   "episode_metrics", "time_out"}
                for name in fresh.info:
                    if name not in terminal_fields:
                        next_info[name] = jax.tree_util.tree_map(
                            choose, fresh.info[name], current.info[name])
                next_info["transition_reset_rng"] = choose(
                    keys[:, 1], current.info["transition_reset_rng"])
                return current.replace(
                    pipeline_state=jax.tree_util.tree_map(
                        choose, fresh.pipeline_state, current.pipeline_state),
                    obs=jax.tree_util.tree_map(choose, fresh.obs, current.obs),
                    info=next_info)

            # Avoid constructing fresh MJX data on steps where no lane ends.
            result = jax.lax.cond(jp.any(done), reset_ended, lambda current: current, terminal)
            return result.replace(info={**result.info, "transition_needs_reset": done})

    return FullResetWrapper(training.EpisodeWrapper(
        training.VmapWrapper(env), episode_length, action_repeat=1))
