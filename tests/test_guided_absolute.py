import unittest
import numpy as np
from scripts.train_mjx_3d_transition_ppo import parse_args, build_task
from curl_robot_2d_mjx.transition_control_3d import interpolated_stand_target, limit_transition_target
from curl_robot_2d_mjx.reward_transition_3d import guided_absolute_reward_config_3d
from curl_robot_2d_mjx.reward_transition_3d import guided_landing_reward_config_3d, touchdown_downward_speed_squared
from tests.test_smooth_stand_reward import SmoothStandRewardTests


class GuidedAbsoluteTests(unittest.TestCase):
    def test_landing_proxy_keeps_preimpact_speed_only_at_contact_onset(self):
        previous = np.zeros((4, 3)); previous[0, 2] = -2.
        stopped = np.zeros((4, 3))
        contact = np.array([1., 0., 0., 0.])
        self.assertEqual(touchdown_downward_speed_squared(np, np.zeros(4), contact, previous, stopped), 1.)
        self.assertEqual(touchdown_downward_speed_squared(np, contact, contact, previous, stopped), 0.)
        self.assertEqual(touchdown_downward_speed_squared(np, np.zeros(4), contact, -previous, stopped), 0.)

    def test_landing_cost_remains_after_deploy_window(self):
        helper = SmoothStandRewardTests()
        terms = helper.terms(guided_landing_reward_config_3d(),
                             deploy_window_fraction=0., touchdown_downward_speed_squared=1.)
        self.assertEqual(terms['touchdown_speed'], -2.)
        old = helper.terms(guided_absolute_reward_config_3d(), touchdown_downward_speed_squared=1.)
        self.assertEqual(old['touchdown_speed'], 0.)

    def test_reference_endpoints_and_no_overshoot(self):
        start = np.array([1., -2., 3.])
        end = np.zeros(3)
        np.testing.assert_array_equal(interpolated_stand_target(np, start, end, 0, .3), start)
        np.testing.assert_array_equal(interpolated_stand_target(np, start, end, 2, .3), end)
        np.testing.assert_allclose(interpolated_stand_target(np, start, end, .15, .3), start/2)

    def test_first_tick_and_reversal_respect_rate_limits(self):
        previous = np.array([1., -1., .2])
        rates = np.array([6., 16., 16.])
        for request in (np.array([-10., 10., -10.]), np.array([10., -10., 10.])):
            result = limit_transition_target(np, request, previous, rates, .02,
                                             np.full(3, -2.), np.full(3, 2.))
            self.assertTrue(np.all(np.abs(result-previous) <= rates*.02+1e-12))
            self.assertTrue(np.all(np.abs(result) <= 2.))
            previous = result

    def test_tracking_penalty_prefers_reference(self):
        helper = SmoothStandRewardTests()
        config = guided_absolute_reward_config_3d()
        on = helper.terms(config, executed_reference_error_squared=0.)
        off = helper.terms(config, executed_reference_error_squared=.09)
        self.assertEqual(on['reference_tracking'], 0.)
        self.assertAlmostEqual(off['reference_tracking'], -2.)

    def test_profile_enables_limiter_without_changing_action_scale(self):
        base = ['--geometry', 'rollingquad_2_abd10_no_self_collision', '--dynamic-roll-to-stand']
        old = build_task(parse_args(base))
        new = build_task(parse_args(base + ['--reward-profile', 'guided_absolute']))
        self.assertEqual(old.target_rate_limits_rad_s, ())
        self.assertEqual(new.target_rate_limits_rad_s, (6.,16.,16.)*4)
        self.assertEqual(old.action_range_fraction, new.action_range_fraction)
        self.assertEqual(new.reference_deploy_duration_s, .3)
        with self.assertRaises(ValueError):
            build_task(parse_args(base + ['--target-rate-limits','0','16','16']))


if __name__ == '__main__':
    unittest.main()
