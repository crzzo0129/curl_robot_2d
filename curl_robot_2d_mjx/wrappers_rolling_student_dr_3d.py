"""Full-reset PPO wrapper for rolling Student deploy randomization."""


def select_reset_lanes_rolling_3d(xp, done, fresh, current):
    mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
    return xp.where(mask, fresh, current)


def wrap_rolling_student_dr_3d(
    env,
    episode_length,
    action_repeat=1,
    randomization_fn=None,
):
    """Vectorize, account episodes and fully reset ended deploy-DR lanes."""

    import jax
    import jax.numpy as jp
    from brax.envs.base import Wrapper
    from brax.envs.wrappers import training

    if action_repeat != 1:
        raise ValueError("rolling Student env owns the physics action repeat")
    vector_env = (
        training.VmapWrapper(env)
        if randomization_fn is None
        else training.DomainRandomizationVmapWrapper(env, randomization_fn)
    )

    class FullResetWrapper(Wrapper):
        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(
                info={
                    **state.info,
                    "student_dr_reset_rng": rng,
                    "student_dr_needs_reset": jp.zeros_like(
                        state.done, dtype=bool
                    ),
                }
            )

        def step(self, state, action):
            info = dict(state.info)
            pending = info["student_dr_needs_reset"]

            def choose_pending(fresh, old):
                return select_reset_lanes_rolling_3d(jp, pending, fresh, old)

            for name in (
                "steps",
                "truncation",
                "episode_done",
                "episode_metrics",
                "time_out",
            ):
                info[name] = jax.tree_util.tree_map(
                    lambda old: choose_pending(jp.zeros_like(old), old),
                    info[name],
                )
            state = state.replace(info=info, done=jp.zeros_like(state.done))
            terminal = self.env.step(state, action)
            done = terminal.done > 0.0

            def reset_ended(current):
                keys = jax.vmap(lambda key: jax.random.split(key, 2))(
                    current.info["student_dr_reset_rng"]
                )
                fresh = self.env.reset(keys[:, 0])

                def choose(new, old):
                    return select_reset_lanes_rolling_3d(jp, done, new, old)

                terminal_fields = {
                    "steps",
                    "truncation",
                    "episode_done",
                    "episode_metrics",
                    "time_out",
                }
                next_info = dict(current.info)
                for name in fresh.info:
                    if name not in terminal_fields:
                        next_info[name] = jax.tree_util.tree_map(
                            choose, fresh.info[name], current.info[name]
                        )
                next_info["student_dr_reset_rng"] = choose(
                    keys[:, 1], current.info["student_dr_reset_rng"]
                )
                return current.replace(
                    pipeline_state=jax.tree_util.tree_map(
                        choose, fresh.pipeline_state, current.pipeline_state
                    ),
                    obs=jax.tree_util.tree_map(
                        choose, fresh.obs, current.obs
                    ),
                    info=next_info,
                )

            result = jax.lax.cond(
                jp.any(done), reset_ended, lambda current: current, terminal
            )
            return result.replace(
                info={**result.info, "student_dr_needs_reset": done}
            )

    return FullResetWrapper(
        training.EpisodeWrapper(
            vector_env, episode_length, action_repeat=1
        )
    )
