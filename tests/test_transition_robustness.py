import unittest
from dataclasses import replace
import numpy as np
from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig, validate_transition_config_3d
from curl_robot_2d_mjx.transition_robustness_3d import horizontal_push_force
from curl_robot_2d_mjx.reward_transition_3d import guided_hold_reward_config_3d, guided_hold_robust_reward_config_3d
from scripts.train_mjx_3d_transition_ppo import parse_args, build_task
from tests import test_smooth_stand_reward as reward_tests


class RobustnessTests(unittest.TestCase):
    def test_pulse_is_bounded_finite_and_clears(self):
        cfg = replace(Transition3DConfig(), push_acceleration_m_s2=.4)
        samples = np.array([.1, 0., .25, .9])
        forces = np.array([horizontal_push_force(np, samples, i, .02, 5., cfg)
                           for i in range(150)])
        self.assertEqual(np.count_nonzero(np.linalg.norm(forces, axis=1)), 6)
        np.testing.assert_array_equal(forces[:40], 0.)
        np.testing.assert_array_equal(forces[46:], 0.)
        np.testing.assert_array_equal(forces[:, 2], 0.)
        self.assertLessEqual(np.max(np.linalg.norm(forces, axis=1)), 2.)
        samples[0] = .9
        np.testing.assert_array_equal(horizontal_push_force(np, samples, 42, .02, 5., cfg), 0.)

    def test_stronger_hold_preserves_deploy_and_quiet_optimum(self):
        helper = reward_tests.SmoothStandRewardTests()
        old, new = guided_hold_reward_config_3d(), guided_hold_robust_reward_config_3d()
        motion = dict(target_acceleration_squared=250000., joint_velocity_squared=64.,
                      target_rate_squared=100., combined_speed=.2, foot_slip_velocity_squared=.0625)
        self.assertEqual(helper.terms(old, deploy_window_fraction=1., **motion),
                         helper.terms(new, deploy_window_fraction=1., **motion))
        for name in ('hold_joint_motion', 'hold_body_motion', 'hold_foot_slip'):
            self.assertLess(helper.terms(new, deploy_window_fraction=0., **motion)[name],
                            helper.terms(old, deploy_window_fraction=0., **motion)[name])
            self.assertEqual(helper.terms(new, deploy_window_fraction=0.)[name], 0.)

    def test_opt_in_and_invalid_values(self):
        base = ['--geometry', 'rollingquad_2_abd10_no_self_collision', '--dynamic-roll-to-stand',
                '--reward-profile', 'guided_hold_robust']
        self.assertEqual(build_task(parse_args(base)).push_acceleration_m_s2, 0.)
        task = build_task(parse_args(base + ['--robustness', 'mild']))
        validate_transition_config_3d(task)
        self.assertEqual(task.target_rate_limits_rad_s, (6.,)*12)
        self.assertEqual(task.push_acceleration_m_s2, .4)
        for change in (dict(push_acceleration_m_s2=-1), dict(push_duration_s=0),
                       dict(push_probability=2), dict(push_start_range_s=(2., 1.))):
            with self.assertRaises(ValueError):
                validate_transition_config_3d(replace(task, **change))


if __name__ == '__main__':
    unittest.main()
