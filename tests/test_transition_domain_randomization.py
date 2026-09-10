import unittest
from types import SimpleNamespace
import numpy as np
from curl_robot_2d_mjx.transition_domain_randomization_3d import apply_model_samples
from scripts.train_mjx_3d_transition_ppo import build_task, parse_args


class Model(SimpleNamespace):
    def replace(self, **kwargs):
        return Model(**{**vars(self), **kwargs})


class DomainRandomizationTests(unittest.TestCase):
    def test_bounds_pd_consistency_and_nominal_unchanged(self):
        model = Model(nbody=4, nu=2, geom_friction=np.ones((5, 3)),
                      body_mass=np.array([0., 1., 2., 1.]), body_inertia=np.ones((4, 3)),
                      body_ipos=np.zeros((4, 3)), actuator_gainprm=np.array([[5., 0., 0.]]*2),
                      actuator_biasprm=np.array([[0., -5., -.1]]*2),
                      actuator_forcerange=np.array([[-3., 3.]]*2))
        for endpoint in (0., 1.):
            samples = dict(friction=endpoint, torso=endpoint,
                           mass=np.full(4, endpoint), inertia=np.full(4, endpoint),
                           com=np.full(3, endpoint), kp=np.full(2, endpoint),
                           kd=np.full(2, endpoint), torque=np.full(2, endpoint))
            result = apply_model_samples(np, model, 2, samples)
            self.assertEqual(result.body_mass[0], 0.)
            np.testing.assert_array_equal(result.body_ipos[[0, 1, 3]], 0.)
            self.assertLessEqual(np.max(np.abs(result.body_ipos)), .003)
            np.testing.assert_allclose(result.actuator_gainprm[:, 0], -result.actuator_biasprm[:, 1])
            self.assertTrue(np.all(np.abs(result.actuator_forcerange) <= 3.))
            np.testing.assert_array_equal(result.geom_friction[:, 1:], 1.)
        np.testing.assert_array_equal(model.body_ipos, 0.)
        np.testing.assert_array_equal(model.actuator_forcerange, [[-3., 3.]]*2)

    def test_dr_explicitly_disables_pushes(self):
        args = parse_args(['--geometry', 'rollingquad_2_abd10_no_self_collision',
                           '--dynamic-roll-to-stand', '--reward-profile', 'guided_hold_robust',
                           '--robustness', 'deploy_dr'])
        task = build_task(args)
        self.assertTrue(task.domain_randomization)
        self.assertEqual(task.push_acceleration_m_s2, 0.)
        self.assertEqual(task.target_rate_limits_rad_s, (6.,)*12)


if __name__ == '__main__':
    unittest.main()
