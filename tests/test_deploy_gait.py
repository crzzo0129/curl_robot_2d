"""Behavioral reward checks on measured trajectories, including JIT/vmap."""

import functools
import unittest

import jax
import jax.numpy as jp
import numpy as np

from scripts.deploy_gait import init_hip_rom, sample_command, update_hip_rom


UPDATE = functools.partial(
    update_hip_rom, dt=0.02, weight=0.08, target_rad=0.35,
    target_min_rad=0.10, reference_speed=0.45, warmup_s=0.10,
    min_cycle_s=0.20, max_cycle_s=0.80, min_swing_s=0.06,
    min_clearance=0.008)


@jax.jit
def rollout(amplitudes, command, kind=0, height=0.02):
    # Two diagonal pairs alternate; rear hip signs oppose front hip signs.
    phase = jp.arange(200)[:, None] / 30.0 + jp.array([0., .5, .5, 0.])
    hip = (0.9 + amplitudes[None, :] / 2.0
           * jp.sin(2.0 * jp.pi * phase) * jp.array([1., 1., -1., -1.]))
    contact = (phase % 1.0) < 0.5
    contact = jp.where(kind == 1, jp.ones_like(contact), contact)
    # All-leg jumps: no supporting leg while airborne.
    contact = jp.where(kind == 2, jp.broadcast_to(contact[:, :1], contact.shape),
                       contact)
    clearance = jp.where(contact, 0.0, height)

    def step(state, inputs):
        h, c, z = inputs
        return UPDATE(state, h, command, c, z)

    return jax.lax.scan(step, init_hip_rom(hip[0], command),
                        (hip, contact, clearance))


class DeployGaitTest(unittest.TestCase):
    def test_command_buckets_and_matched_speed_magnitudes(self):
        sampler = functools.partial(
            sample_command, vx_range=(-.6, .6), vy_range=(-.15, .15),
            wz_range=(-1.2, 1.2), straight_prob=.6, stand_prob=.1, min_speed=.1)
        commands = np.asarray(jax.jit(jax.vmap(sampler))(
            jax.random.split(jax.random.PRNGKey(17), 20000)))
        straight = np.all(commands[:, 1:] == 0., axis=1)
        forward = straight & (commands[:, 0] > 0.)
        backward = straight & (commands[:, 0] < 0.)
        stand = np.all(commands == 0., axis=1)
        for mask, expected in ((forward, .3), (backward, .3),
                               (~straight, .3), (stand, .1)):
            self.assertAlmostEqual(float(mask.mean()), expected, delta=.015)
        pos = commands[forward, 0]
        neg = -commands[backward, 0]
        self.assertTrue(np.all((pos >= .1) & (pos <= .6)))
        self.assertTrue(np.all((neg >= .1) & (neg <= .6)))
        self.assertAlmostEqual(float(pos.mean()), float(neg.mean()), delta=.015)

    def test_larger_measured_excursion_reduces_deficit_and_saturates(self):
        values = []
        for amplitude in (.0, .12, .25, .40, .70):
            _, m = rollout(jp.full(4, amplitude), jp.array([.45, 0., 0.]))
            values.append(float(jp.mean(m["hip_rom_penalty"][-30:])))
        self.assertTrue(all(a > b for a, b in zip(values[:3], values[1:4])))
        self.assertAlmostEqual(values[3], 0.)
        self.assertAlmostEqual(values[4], 0.)

    def test_forward_backward_and_front_rear_permutation_are_symmetric(self):
        amplitudes = jp.array([.12, .24, .30, .40])
        _, forward = rollout(amplitudes, jp.array([.45, 0., 0.]))
        _, backward = rollout(amplitudes, jp.array([-.45, 0., 0.]))
        _, swapped = rollout(amplitudes[jp.array([2, 3, 0, 1])],
                             jp.array([-.45, 0., 0.]))
        np.testing.assert_allclose(forward["hip_rom_penalty"],
                                   backward["hip_rom_penalty"])
        np.testing.assert_allclose(forward["hip_rom_penalty"][-30:],
                                   swapped["hip_rom_penalty"][-30:], atol=1e-7)

    def test_weak_front_legs_cannot_be_hidden_by_large_rear_amplitude(self):
        _, m = rollout(jp.array([.0, .0, .7, .7]), jp.array([.45, 0., 0.]))
        self.assertAlmostEqual(float(m["hip_rom_penalty"][-1]), .04, places=6)

    def test_ground_shuffling_and_four_leg_jumps_do_not_get_rom_credit(self):
        for kind, height in ((1, .02), (2, .02), (0, .001)):
            _, m = rollout(jp.full(4, .7), jp.array([.45, 0., 0.]), kind, height)
            self.assertAlmostEqual(float(m["hip_rom_penalty"][-1]), .08, places=6)
            self.assertEqual(float(m["hip_rom_valid_fraction"][-1]), 0.)

    def test_stand_lateral_and_turn_commands_disable_cost(self):
        commands = jp.array([[0., 0., 0.], [0., .1, 0.], [.45, .1, 0.],
                             [.45, 0., .5], [0., 0., 1.]])
        _, m = jax.vmap(lambda cmd: rollout(jp.zeros(4), cmd))(commands)
        np.testing.assert_array_equal(m["hip_rom_penalty"], 0.)

    def test_command_change_clears_credit_and_excludes_transition(self):
        state, _ = rollout(jp.full(4, .4), jp.array([.45, 0., 0.]))
        for command in (jp.array([-.45, 0., 0.]), jp.array([.20, 0., 0.])):
            new, m = jax.jit(UPDATE)(state, jp.full(4, 1.4), command,
                                     jp.ones(4, dtype=bool), jp.zeros(4))
            self.assertEqual(float(m["hip_rom_penalty"]), 0.)
            np.testing.assert_array_equal(new["hip_rom_valid"], False)
            np.testing.assert_array_equal(new["hip_rom_last"], 0.)
            np.testing.assert_allclose(new["hip_rom_min"], 1.4)
            self.assertEqual(set(new), set(state))

    def test_stopping_stepping_expires_previous_good_cycle(self):
        command = jp.array([.45, 0., 0.])
        state, _ = rollout(jp.full(4, .4), command)

        def step(s, _):
            return UPDATE(s, jp.full(4, .9), command,
                          jp.ones(4, dtype=bool), jp.zeros(4))

        _, m = jax.jit(lambda s: jax.lax.scan(step, s, None, length=100))(state)
        self.assertAlmostEqual(float(m["hip_rom_penalty"][-1]), .08, places=6)
        self.assertEqual(float(m["hip_rom_valid_fraction"][-1]), 0.)

    def test_target_scales_with_speed_and_stays_bounded(self):
        for speed, target in ((.1, .1), (.225, .175), (.45, .35), (.6, .35)):
            for sign in (-1., 1.):
                _, m = rollout(jp.zeros(4), jp.array([sign * speed, 0., 0.]))
                self.assertAlmostEqual(float(m["hip_rom_target"][-1]), target,
                                       places=6)


if __name__ == "__main__":
    unittest.main()
