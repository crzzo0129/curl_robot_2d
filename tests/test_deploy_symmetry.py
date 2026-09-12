"""Pure NumPy checks of deploy reflection and loss wiring; no JAX execution."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.deploy_symmetry import (
    bind_ppo_loss,
    consistency_statistics,
    make_symmetry_loss,
    mirror_action,
    mirror_observation,
    with_front_back_symmetry,
)


class FrontBackReflectionTest(unittest.TestCase):
    def test_known_frame_vectors_and_legs(self):
        frame = np.arange(1, 37, dtype=float)
        result = mirror_observation(np, np.tile(frame, 20)).reshape(20, 36)
        expected = np.array([
            1, -2, -3, -4, 5, 6, -7, 8, -9, -10, 11, 12,
            19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18,
            31, 32, 33, 34, 35, 36, 25, 26, 27, 28, 29, 30,
        ])
        np.testing.assert_array_equal(result, np.tile(expected, (20, 1)))

    def test_all_history_frames_batches_and_involution(self):
        obs = np.arange(2 * 3 * 720, dtype=np.float32).reshape(2, 3, 720)
        original = obs.copy()
        reflected = mirror_observation(np, obs)
        self.assertEqual(reflected.dtype, obs.dtype)
        np.testing.assert_array_equal(mirror_observation(np, reflected), original)
        np.testing.assert_array_equal(obs, original)
        for t in range(20):
            np.testing.assert_array_equal(reflected[..., t*36 + 6], -obs[..., t*36 + 6])
            np.testing.assert_array_equal(reflected[..., t*36 + 14], obs[..., t*36 + 20])
        action = np.arange(24).reshape(2, 12)
        np.testing.assert_array_equal(mirror_action(np, mirror_action(np, action)), action)

    def test_reject_wrong_controller_layout(self):
        with self.assertRaises(ValueError):
            mirror_action(np, np.zeros(8))
        with self.assertRaises(ValueError):
            mirror_observation(np, np.zeros(36))

    def test_straight_gate_ignores_turning_stand_and_history_commands(self):
        obs = np.zeros((2, 2, 720))
        obs[..., 42] = 0.5  # old straight commands must not enable the gate
        obs[0, 0, 6:9] = [.3, 0, 0]
        obs[0, 1, 6:9] = [-.3, 0, 0]
        obs[1, 0, 6:9] = [.3, 0, 1]
        action = np.zeros((2, 2, 12))
        reflected_action = np.ones_like(action)
        reflected_action[1] = 1000  # excluded samples cannot dominate the loss
        mse, fraction = consistency_statistics(np, obs, action, reflected_action)
        self.assertEqual(mse, 1)
        self.assertEqual(fraction, .5)
        obs[..., 6:9] = 0
        mse, fraction = consistency_statistics(np, obs, action, reflected_action)
        self.assertEqual(mse, 0)
        self.assertEqual(fraction, 0)

    def test_equivariant_policy_zero_cost_and_single_leg_mismatch_detected(self):
        rng = np.random.default_rng(2)
        obs = rng.normal(size=(2, 4, 720))
        obs[..., 6:9] = [.4, 0, 0]
        policy = lambda x: np.tanh(x[..., 12:24])
        a, reflected_a = policy(obs), policy(mirror_observation(np, obs))
        self.assertEqual(consistency_statistics(np, obs, a, reflected_a)[0], 0)
        reflected_a[..., 0] += .3
        self.assertGreater(consistency_statistics(np, obs, a, reflected_a)[0], 0)


class LossAdapterTest(unittest.TestCase):
    def test_left_right_loss_covers_turning_without_front_back_term(self):
        obs = np.zeros((2, 720), dtype=np.float32)
        obs[:, 6:9] = [[0, 0, 1], [0, 0, -1]]
        captured = []
        def apply(norm, params, value):
            captured.append(value.copy())
            return np.broadcast_to(np.arange(12) / 10, (2, 12))
        network = SimpleNamespace(policy_network=SimpleNamespace(apply=apply),
                                  parametric_action_distribution=SimpleNamespace(mode=np.tanh))
        base = lambda *args, **kwargs: (2., {"v_loss": 1.})
        loss = make_symmetry_loss(base, 0., left_right_weight=.01, array_module=np)
        total, metrics = loss(SimpleNamespace(policy=None), None,
                              SimpleNamespace(observation=obs), None, network)
        self.assertGreater(total, 2.)
        self.assertEqual(metrics['fb_symmetry_loss'], 0)
        self.assertEqual(metrics['lr_symmetry_fraction'], 1)
        self.assertEqual(len(captured), 2)
        np.testing.assert_array_equal(captured[1][:, 8], -obs[:, 8])
        np.testing.assert_array_equal(captured[0], obs)

    def test_preserve_base_ppo_inputs_and_normalize_after_mirroring(self):
        obs = np.zeros((2, 3, 720)); obs[..., 6:9] = [.4, 0, 0]
        obs[..., 12:24] = np.arange(12) / 20
        data = SimpleNamespace(observation=obs, sentinel=object())
        params = SimpleNamespace(policy=np.arange(12) / 10, value=object())
        normalizer = object()
        policy_inputs = []
        def apply(norm, weights, raw_obs):
            self.assertIs(norm, normalizer)
            policy_inputs.append(raw_obs.copy())
            # Deliberately asymmetric normalizer: it cannot be reflected in
            # place of raw physical observations.
            return (raw_obs[..., 12:24] - np.arange(12)/30) / 2 + weights
        networks = SimpleNamespace(
            policy_network=SimpleNamespace(apply=apply),
            parametric_action_distribution=SimpleNamespace(mode=np.tanh))
        def base(p, norm, transitions, rng, ppo_network, entropy_cost):
            self.assertIs(p, params); self.assertIs(norm, normalizer)
            self.assertIs(transitions, data); self.assertIs(ppo_network, networks)
            self.assertEqual(rng, 7); self.assertEqual(entropy_cost, .01)
            return 2., {"total_loss": 2., "policy_loss": .3, "v_loss": 1.7}
        loss = make_symmetry_loss(base, .1, array_module=np)
        total, metrics = loss(params, normalizer, data, 7, networks, entropy_cost=.01)
        self.assertGreater(total, 2.)
        self.assertAlmostEqual(total, 2 + metrics['fb_symmetry_loss'])
        self.assertEqual(metrics['total_loss'], total)
        self.assertEqual(metrics['policy_loss'], .3)
        self.assertEqual(metrics['v_loss'], 1.7)
        np.testing.assert_array_equal(policy_inputs[0], obs)
        np.testing.assert_array_equal(policy_inputs[1], mirror_observation(np, obs))
        obs[..., 6:9] = [0, 0, 1]
        self.assertEqual(loss(params, normalizer, data, 7, networks, entropy_cost=.01)[0], 2.)

    def test_trainer_binding_is_local_and_preserves_other_library_members(self):
        original_loss = lambda x: x + 1
        loss_module = SimpleNamespace(compute_ppo_loss=original_loss, PPONetworkParams=object())
        namespace = {"ppo_losses": loss_module}
        exec("def train(x=2, *, extra=3):\n"
             "    return ppo_losses.compute_ppo_loss(x) + extra, ppo_losses.PPONetworkParams\n", namespace)
        original = namespace['train']
        bound = bind_ppo_loss(original, lambda x: x + 10)
        self.assertEqual(original()[0], 6)
        self.assertEqual(bound()[0], 15)
        self.assertEqual(bound(extra=4)[0], 16)
        self.assertIs(bound()[1], loss_module.PPONetworkParams)
        self.assertIs(loss_module.compute_ppo_loss, original_loss)
        self.assertIs(with_front_back_symmetry(original, 0), original)

    def test_fail_explicitly_for_incompatible_trainer_and_bad_weight(self):
        with self.assertRaises(RuntimeError):
            bind_ppo_loss(lambda: None, lambda: None)
        for weight in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                with_front_back_symmetry(lambda: None, weight)
            with self.assertRaises(ValueError):
                with_front_back_symmetry(lambda: None, 0., weight)


class LeftRightReflectionTest(unittest.TestCase):
    def test_known_reflection_and_time_order(self):
        obs = np.tile(np.arange(1, 37, dtype=np.float32), 20)
        out = mirror_observation(np, obs, 'left_right').reshape(20, 36)
        expected = [-1, 2, -3, 4, -5, 6, 7, -8, -9, 10, -11, 12,
                    16, 17, 18, 13, 14, 15, 22, 23, 24, 19, 20, 21,
                    28, 29, 30, 25, 26, 27, 34, 35, 36, 31, 32, 33]
        np.testing.assert_array_equal(out, np.tile(expected, (20, 1)))
        varying = np.random.default_rng(1).normal(size=(2, 3, 720)).astype(np.float32)
        reflected = mirror_observation(np, varying, 'left_right')
        np.testing.assert_array_equal(mirror_observation(np, reflected, 'left_right'), varying)
        # Orthogonal spatial reflections commute, without changing history order.
        np.testing.assert_array_equal(mirror_observation(np, reflected),
                                      mirror_observation(np, mirror_observation(np, varying), 'left_right'))

    def test_turn_gate_and_equivariant_actor(self):
        obs = np.zeros((3, 720)); obs[:, 6:9] = [[0, 0, 1], [0, 0, -1], [0, 0, 0]]
        obs[:, 12:24] = np.arange(12) / 10
        action = np.tanh(obs[:, 12:24])
        mirrored = np.tanh(mirror_observation(np, obs, 'left_right')[:, 12:24])
        mirrored[2] = 100  # standing sample is excluded
        mse, fraction = consistency_statistics(np, obs, action, mirrored, 'left_right')
        self.assertEqual(mse, 0.)
        self.assertAlmostEqual(fraction, 2 / 3)


if __name__ == '__main__':
    unittest.main()
