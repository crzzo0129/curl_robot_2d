"""Ensure low-work failures cannot win the energy-aware CEM search."""
import copy
import unittest

from scripts.optimize_energy_cem_3d import objective


def baseline():
    window = dict(actual_speed_m_s=.6, lateral_displacement_m=.1,
        max_axis_tilt_deg=1., max_self_penetration_m=.001,
        self_contact_fraction=.05, saturation_fraction=.002,
        positive_j_per_m=10., cot_absolute=.6)
    return dict(rolling_turns=6., windows={'full':dict(window), 'post_startup':dict(window)})


class EnergyObjectiveTest(unittest.TestCase):
    def test_same_speed_energy_improvement_wins(self):
        base = baseline()
        improved = copy.deepcopy(base)
        improved['windows']['post_startup']['cot_absolute'] = .48
        score, feasible, _ = objective(improved, base)
        self.assertTrue(feasible)
        self.assertLess(score, objective(base, base)[0])

    def test_stopping_cannot_win(self):
        base = baseline()
        stopped = copy.deepcopy(base)
        stopped['rolling_turns'] = 0.
        stopped['windows']['post_startup'].update(actual_speed_m_s=0.,
            cot_absolute=None, positive_j_per_m=None)
        score, feasible, violations = objective(stopped, base)
        self.assertFalse(feasible)
        self.assertGreater(violations['speed'], 0.)
        self.assertGreater(score, objective(base, base)[0])

    def test_extra_penetration_rejected_even_with_low_energy(self):
        base = baseline()
        collision = copy.deepcopy(base)
        collision['windows']['post_startup']['cot_absolute'] = .1
        collision['windows']['full']['max_self_penetration_m'] = .004
        score, feasible, _ = objective(collision, base)
        self.assertFalse(feasible)
        self.assertGreater(score, objective(base, base)[0])


if __name__ == '__main__':
    unittest.main()
