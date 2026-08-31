"""Exercise the actual Brax wrapper with a tiny fake task, without MJX compile."""

import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("jax", "brax")),
                     "JAX/Brax required, but no GPU/MJX needed")
class TransitionWrapperTests(unittest.TestCase):
    def make_env(self, length=3):
        import jax
        import jax.numpy as jp
        from brax.envs.base import Env, State
        from curl_robot_2d_mjx.wrappers_transition_3d import wrap_transition_3d

        class Toy(Env):
            observation_size = {"state": 3}
            action_size = 1
            backend = "test"

            def reset(self, rng):
                token = jax.random.uniform(rng, (1,))
                obs = jp.concatenate((token, jp.zeros(2)))
                zero = jp.asarray(0., dtype=jp.float32)
                return State({"token": token}, {"state": obs}, zero, zero,
                             metrics={"reward": zero, "success": zero, "failure": zero,
                                      "timeout": zero},
                             info={"step_count": jp.asarray(0), "mode": jp.asarray(0),
                                   "actor_history": obs, "last_action": jp.zeros(1),
                                   "rng": rng, "time_out": zero})

            def step(self, state, action):
                count = state.info["step_count"] + 1
                success = action[0] == 1
                failure = action[0] == 2
                timeout = (count >= length) & (~success) & (~failure)
                obs = state.obs["state"] + 1
                return state.replace(obs={"state": obs}, reward=jp.asarray(1.),
                                     done=(success | failure | timeout).astype(jp.float32),
                                     metrics={"reward": jp.asarray(1.), "success": success.astype(jp.float32),
                                              "failure": failure.astype(jp.float32),
                                              "timeout": timeout.astype(jp.float32)},
                                     info={**state.info, "step_count": count,
                                           "mode": state.info["mode"] + 1, "actor_history": obs,
                                           "last_action": action, "time_out": timeout.astype(jp.float32)})

        return wrap_transition_3d(Toy(), episode_length=length)

    def test_partial_lane_success_failure_reset_preserves_terminal_statistics(self):
        import jax
        import jax.numpy as jp
        env = self.make_env()
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 3))
        initial_token = np.asarray(state.pipeline_state["token"]).copy()
        initial_rng = np.asarray(state.info["transition_reset_rng"]).copy()
        step = jax.jit(env.step)
        state = step(state, jp.array([[1.], [2.], [0.]]))
        np.testing.assert_array_equal(state.done, [1, 1, 0])
        np.testing.assert_array_equal(state.info["step_count"], [0, 0, 1])
        np.testing.assert_array_equal(state.info["mode"], [0, 0, 1])
        np.testing.assert_array_equal(state.info["steps"], [1, 1, 1])
        np.testing.assert_array_equal(state.metrics["success"], [1, 0, 0])
        np.testing.assert_array_equal(state.metrics["failure"], [0, 1, 0])
        np.testing.assert_array_equal(state.info["actor_history"], state.obs["state"])
        np.testing.assert_array_equal(state.pipeline_state["token"][2], initial_token[2])
        self.assertFalse(np.array_equal(state.pipeline_state["token"][:2], initial_token[:2]))
        np.testing.assert_array_equal(state.info["transition_reset_rng"][2], initial_rng[2])
        self.assertFalse(np.array_equal(state.info["transition_reset_rng"][:2], initial_rng[:2]))
        state = step(state, jp.zeros((3, 1)))
        np.testing.assert_array_equal(state.done, [0, 0, 0])
        np.testing.assert_array_equal(state.info["step_count"], [1, 1, 2])
        np.testing.assert_array_equal(state.info["steps"], [1, 1, 2])
        np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], [1, 1, 2])

    def test_repeated_timeouts_do_not_create_one_step_episodes(self):
        import jax
        import jax.numpy as jp
        env = self.make_env()
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
        step = jax.jit(env.step)
        for tick in range(9):
            state = step(state, jp.zeros((2, 1)))
            ended = (tick + 1) % 3 == 0
            np.testing.assert_array_equal(state.done, np.full(2, ended))
            np.testing.assert_array_equal(state.info["time_out"], np.full(2, ended))
            np.testing.assert_array_equal(state.info["step_count"], np.full(2, (tick + 1) % 3))
            np.testing.assert_array_equal(state.info["steps"], np.full(2, tick % 3 + 1))
            np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"],
                                          np.full(2, tick % 3 + 1))

    def test_eval_wrapper_reads_terminal_length_not_reset_length(self):
        import jax
        import jax.numpy as jp
        from brax.envs.wrappers.training import EvalWrapper
        env = EvalWrapper(self.make_env())
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
        step = jax.jit(env.step)
        for _ in range(5):
            state = step(state, jp.zeros((2, 1)))
        metrics = state.info["eval_metrics"]
        np.testing.assert_array_equal(metrics.episode_steps, [3, 3])
        np.testing.assert_array_equal(metrics.episode_metrics["timeout"], [1, 1])
        np.testing.assert_array_equal(metrics.episode_metrics["reward"], [3, 3])


if __name__ == "__main__":
    unittest.main()
