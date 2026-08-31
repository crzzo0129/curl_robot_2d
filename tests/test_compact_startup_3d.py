"""Fast NumPy contract tests; no JAX compilation or training."""
from contextlib import redirect_stderr
import io
import unittest

import numpy as np

from curl_robot_2d_mjx.compact_startup_3d import (
    CompactStartupConfig, compact_potential, compact_target, contact_slip,
)
from curl_robot_2d_mjx.autonomous_startup_3d import gate_errors


class CompactStartupTest(unittest.TestCase):
    def slip(self, velocity=(1., 0., 0.), angular=(0., 0., 0.),
             position=(0., 0., 0.), distance=0., pair=(1, 0), repeats=1):
        cvel = np.zeros((3, 6))
        cvel[1, :3], cvel[1, 3:] = angular, velocity
        return contact_slip(np, np.full(repeats, pair[0]), np.full(repeats, pair[1]),
            np.full(repeats, distance), np.tile(position, (repeats, 1)),
            np.tile((0., 0., 1.), (repeats, 1)), geom_bodyid=np.array([0, 1, 2]),
            body_rootid=np.array([0, 1, 2]), subtree_com=np.zeros((3, 3)), cvel=cvel,
            foot_geom_ids=np.array([1]), floor_geom_id=0)

    def test_contact_tangent_not_normal(self):
        np.testing.assert_allclose(self.slip(), (1., 1., 1.))
        np.testing.assert_allclose(self.slip(velocity=(0., 0., 2.)), 0.)
        np.testing.assert_allclose(self.slip(velocity=(1., 0., 2.)), (1., 1., 1.))

    def test_airborne_self_contact_and_empty_capacity(self):
        for kwargs in ({"distance": .001}, {"pair": (1, 2)},
                       {"pair": (-1, 0)}, {"repeats": 0}):
            with self.subTest(kwargs=kwargs):
                np.testing.assert_allclose(self.slip(**kwargs), 0.)

    def test_contact_order_and_duplicate_count(self):
        np.testing.assert_allclose(self.slip(pair=(0, 1)), self.slip())
        np.testing.assert_allclose(self.slip(repeats=12), self.slip())

    def test_rolling_without_slip_is_not_penalized(self):
        # COM moves +x, but rotation cancels the ground point's velocity.
        np.testing.assert_allclose(self.slip(angular=(0., 1., 0.), position=(0., 0., -1.)), 0.)
        np.testing.assert_allclose(self.slip(velocity=(0., 0., 0.),
            angular=(0., 1., 0.), position=(0., 0., -1.)), (1., 1., 1.))

    def test_model_target_zero_velocity_and_cold_clock(self):
        import mujoco
        from curl_robot_2d_mjx.environment_3d import model_path_3d
        model = mujoco.MjModel.from_xml_path(str(model_path_3d("rollingquad_2")))
        target = compact_target(model)
        np.testing.assert_allclose(target["qpos"][0], model.key("compact").qpos)
        np.testing.assert_allclose(target["ctrl"][0], model.key("compact").ctrl)
        for name in ("qvel", "time", "rolling_phase", "oscillator_phase"):
            np.testing.assert_array_equal(target[name], 0.)
        cfg = CompactStartupConfig()
        cfg.validate(.02)
        self.assertEqual(cfg.episode_steps(.02), 650)
        q, v = target["qpos"][0], target["qvel"][0].copy()
        self.assertLess(gate_errors(np, q, v, 0., target, cfg).max(), .01)
        v[0] = -.03
        self.assertGreater(gate_errors(np, q, v, 0., target, cfg).max(), 1.)
        v[0] = .03
        self.assertGreater(gate_errors(np, q, v, 0., target, cfg).max(), 1.)
        zero_pose, zero_cost = compact_potential(np, q, v * 0, target, cfg)
        pose, cost = compact_potential(np, q, v, target, cfg)
        self.assertEqual(pose, zero_pose)
        self.assertEqual(zero_cost, 0.)
        self.assertGreater(cost, 0.)
        stand = np.array(model.key("stand").qpos)
        far_pose, _ = compact_potential(np, stand, v, target, cfg)
        self.assertGreater(far_pose, 0.)
        self.assertLess(far_pose, pose)

    def test_dynamic_bank_cli_is_rejected(self):
        from scripts.train_mjx_3d_startup_ppo import parse_args
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_args(["--teacher", "unused", "--out", "unused",
                        "--candidate-bank", "old-bank.json"])
        self.assertIn("no longer uses --candidate-bank", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
