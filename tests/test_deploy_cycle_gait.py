"""Synthetic contact/kinematics sequences, NumPy only; no simulation or JAX."""
import unittest
import numpy as np
from scripts.deploy_cycle_gait import init_cycle_gait, update_cycle_gait, METRICS


def sequence(phases=(0., .5, .5, 0.), *, hip_offset=None, hip_amplitude=None,
             foot_amplitude=None, direction=1., speed=.3, period=.6, shift=0., height=.04):
    cmd = np.array([direction * speed, 0., 0.], dtype=np.float32)
    hip = np.ones(4, dtype=np.float32) * .9
    state = init_cycle_gait(np, hip, np.zeros(4), cmd)
    records = []
    for i in range(350):
        phase = (i * .02 / period + np.asarray(phases) + shift) % 1.
        contact = phase < .6  # permits double support between diagonal swings
        hip = .9 + np.asarray(.15 if hip_amplitude is None else hip_amplitude) * np.sin(2*np.pi*phase)
        hip += np.asarray(0. if hip_offset is None else hip_offset)
        foot = np.asarray(.04 if foot_amplitude is None else foot_amplitude) * np.sin(2*np.pi*phase)
        state, metrics = update_cycle_gait(np, state, hip, foot, cmd, contact,
                                          np.where(contact, 0., height), direction*speed, .02,
                                          phase_weight=.05, balance_weight=.02)
        if i > 250:
            records.append(metrics)
    return state, {key: np.mean([r[key] for r in records]) for key in METRICS}


class CycleGaitTest(unittest.TestCase):
    def test_trot_scores_above_pace_and_crawl(self):
        _, trot = sequence()
        _, pace = sequence((0., .5, 0., .5))
        _, crawl = sequence((0., .25, .5, .75))
        self.assertGreater(trot['trot_phase_quality'], .95)
        self.assertLess(pace['trot_phase_quality'], .05)
        self.assertLess(crawl['trot_phase_quality'], .3)
        self.assertLess(trot['cycle_balance_penalty'], 1e-6)

    def test_no_fixed_frequency_or_absolute_phase(self):
        for period in (.4, .8, 1.):
            _, metrics = sequence(period=period, shift=.31)
            self.assertGreater(metrics['trot_phase_quality'], .9)
        _, backward = sequence(direction=-1.)
        self.assertGreater(backward['trot_phase_reward'], .045)

    def test_jump_and_scuff_do_not_get_phase_reward(self):
        for kwargs in ({'phases': (0., 0., 0., 0.)}, {'height': .002}):
            _, metrics = sequence(**kwargs)
            self.assertEqual(metrics['trot_phase_reward'], 0.)
            self.assertEqual(metrics['cycle_valid_fraction'], 0.)

    def test_slow_command_gets_full_credit_without_overspeed(self):
        _, metrics = sequence(speed=.1)
        self.assertAlmostEqual(metrics['trot_phase_reward'], .05, places=6)

    def test_each_cycle_balance_component_detects_asymmetry(self):
        for kwargs, metric in (({'hip_offset': [.1, 0, 0, 0]}, 'cycle_hip_mean_error'),
                               ({'hip_amplitude': [.3, .15, .15, .15]}, 'cycle_hip_rom_error'),
                               ({'foot_amplitude': [.08, .04, .04, .04]}, 'cycle_foot_span_error')):
            _, metrics = sequence(**kwargs)
            self.assertGreater(metrics[metric], .01)
            self.assertGreater(metrics['cycle_balance_penalty'], 0.)
            self.assertLessEqual(metrics['cycle_balance_penalty'], .02)

    def test_stopping_and_wrong_direction_cannot_use_old_phase_credit(self):
        state, _ = sequence()
        hip = np.ones(4) * .9; feet = np.zeros(4); cmd = np.array([.3, 0., 0.])
        for vx in (0., -.3):
            _, metrics = update_cycle_gait(np, state, hip, feet, cmd, np.ones(4, dtype=bool),
                                           np.zeros(4), vx, .02, phase_weight=.05, balance_weight=.02)
            self.assertEqual(metrics['trot_phase_reward'], 0.)
        for _ in range(70):
            state, metrics = update_cycle_gait(np, state, hip, feet, cmd, np.ones(4, dtype=bool),
                                               np.zeros(4), .3, .02, phase_weight=.05, balance_weight=.02)
        self.assertEqual(metrics['cycle_valid_fraction'], 0.)
        self.assertEqual(metrics['trot_phase_reward'], 0.)

    def test_command_change_turn_and_stand_reset_cycles(self):
        state, _ = sequence()
        for cmd in ([0., 0., 1.], [0., 0., 0.], [-.3, 0., 0.]):
            new, metrics = update_cycle_gait(np, state, np.ones(4)*.9, np.zeros(4), np.array(cmd),
                                            np.ones(4, dtype=bool), np.zeros(4), .3, .02,
                                            phase_weight=.05, balance_weight=.02)
            self.assertEqual(set(new), set(state))
            self.assertEqual(set(metrics), set(METRICS))
            self.assertEqual(metrics['trot_phase_reward'], 0.)
            self.assertFalse(np.any(new['cg_valid']))
            for key in state:
                self.assertEqual(np.shape(new[key]), np.shape(state[key]))


if __name__ == '__main__':
    unittest.main()
