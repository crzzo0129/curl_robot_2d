"""Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 on CPU."""
import unittest
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from curl_robot_2d_mjx.distillation_execution import (
    BatchExecution, make_reset_finished_rollouts,
)


class State(NamedTuple):
    done: object
    observation: object


class DistillationExecutionTest(unittest.TestCase):
    def test_reset_keeps_state_history_and_previous_action_together(self):
        reset = make_reset_finished_rollouts()
        state = State(jp.array([0., 1., 0., 1.]), jp.arange(8).reshape(4, 2))
        pool = State(jp.zeros(4), state.observation + 100)
        history = jp.arange(12).reshape(4, 3)
        previous = jp.arange(4).reshape(4, 1)
        expected = np.array([False, True, False, True])
        new_state, new_history, new_previous, rate = reset(
            state, pool, history, previous, history + 200, previous + 300,
        )
        np.testing.assert_array_equal(new_state.observation,
                                      np.where(expected[:, None], pool.observation, state.observation))
        np.testing.assert_array_equal(new_history,
                                      np.where(expected[:, None], history + 200, history))
        np.testing.assert_array_equal(new_previous,
                                      np.where(expected[:, None], previous + 300, previous))
        self.assertEqual(float(rate), 0.5)
        # No finished environments: cached snapshots must not perturb anything.
        same = reset(new_state, pool, new_history, new_previous, history, previous)
        np.testing.assert_array_equal(same[0].observation, new_state.observation)
        np.testing.assert_array_equal(same[1], new_history)
        np.testing.assert_array_equal(same[2], new_previous)

    @unittest.skipUnless(jax.local_device_count() >= 4, "requires four CPU or GPU devices")
    def test_global_gradient_and_optimizer_match_single_device(self):
        import optax
        execution = BatchExecution(4)
        optimizer = optax.adam(0.01)

        def update(params, opt_state, observation, target, velocity_target):
            def loss_fn(p):
                error = observation @ p - target
                loss = jp.mean(error ** 2) + 0.2 * jp.mean((observation @ p - velocity_target) ** 2)
                return loss, (jp.sqrt(jp.mean(error ** 2)), jp.max(jp.abs(error)))
            (loss, diagnostics), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, next_opt_state = optimizer.update(grad, opt_state, params)
            return optax.apply_updates(params, updates), next_opt_state, loss, diagnostics

        rng = np.random.default_rng(123)
        batch = [rng.normal(size=shape).astype(np.float32)
                 for shape in ((16, 3), (16, 2), (16, 2))]
        params = jp.ones((3, 2))
        reference = (params, optimizer.init(params))
        parallel = jax.device_put(reference, execution.replicated)
        train = execution.train_jit(update)
        single = jax.jit(update)
        for _ in range(3):
            expected = single(*reference, *batch)
            actual = train(*parallel, *batch)
            for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
                np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
            self.assertEqual(len(actual[0].addressable_shards), 4)
            self.assertTrue(actual[0].is_fully_replicated)
            reference, parallel = expected[:2], actual[:2]

        # Reset/simulation output is partitioned, not replicated in full.
        reset = execution.batch_jit(lambda x: State(jp.zeros(x.shape[0]), x * 2))
        state = reset(batch[0])
        self.assertEqual(len(state.observation.addressable_shards), 4)
        self.assertEqual(state.observation.addressable_shards[0].data.shape, (4, 3))
        np.testing.assert_allclose(state.observation, batch[0] * 2)


if __name__ == "__main__":
    unittest.main()
