"""Reward regressions independent of JAX/MuJoCo and physical reachability."""
from dataclasses import replace
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from curl_robot_2d_mjx.compact_startup_3d import (
    CompactReachConfig, CompactStartupConfig, compact_potential,
    compact_reach_pose_reward, COMPACT_REACH_CONTRACT,
)


class DenseReachRewardTest(unittest.TestCase):
    def setUp(self):
        root = ET.parse(Path(__file__).resolve().parents[1] /
            'assets/rollingquad_description_2/mjcf/rollingquad_primitive.xml').getroot()
        self.stand = np.fromstring(root.find('./keyframe/key[@name="stand"]').get('qpos'), sep=' ')
        self.compact = np.fromstring(root.find('./keyframe/key[@name="compact"]').get('qpos'), sep=' ')
        self.target = {'qpos': self.compact[None]}
        self.cfg = CompactReachConfig()

    def quality(self, q):
        return compact_potential(np, q, np.zeros(18), self.target, self.cfg)[0]

    def test_partial_fold_receives_better_reward_before_gate_success(self):
        qualities = [self.quality((1-u)*self.stand + u*self.compact)
                     for u in (0., .25, .5, .75, 1.)]
        rewards = [compact_reach_pose_reward(np, p, self.cfg) for p in qualities]
        self.assertTrue(np.all(np.diff(rewards) > 0))
        self.assertAlmostEqual(rewards[-1], 0.)
        self.assertTrue(np.all(np.asarray(rewards) <= 0.))

    def test_timeout_does_not_cancel_intermediate_pose_reward(self):
        # Same start/end and failure outcome; one hypothetical path approaches
        # during the middle. This checks incentives, not a simulated trajectory.
        p0 = self.quality(self.stand)
        standing = np.full(150, p0)
        approaching = standing.copy()
        approaching[30:120] = self.quality(.25*self.stand + .75*self.compact)
        discount = self.cfg.discounting ** np.arange(150)
        returns = [np.sum(discount * compact_reach_pose_reward(np, p, self.cfg))
                   for p in (standing, approaching)]
        self.assertGreater(returns[1], returns[0] + 3.)

    def test_reward_does_not_require_low_velocity_to_encourage_folding(self):
        q = .5*self.stand + .5*self.compact
        moving = np.full(18, .1)
        p, settling = compact_potential(np, q, moving, self.target, self.cfg)
        self.assertGreater(settling, 0.)
        self.assertEqual(p, self.quality(q))

    def test_version_isolated_and_success_gate_unchanged(self):
        self.assertIn('v2_dense_pose', COMPACT_REACH_CONTRACT)
        self.assertFalse(hasattr(CompactStartupConfig(), 'pose_reward_weight'))
        self.cfg.validate(.02)
        self.assertEqual(self.cfg.confirmation_steps, 5)
        self.assertEqual(self.cfg.joint_position_rad, .02)
        self.assertEqual(self.cfg.joint_velocity_rad_s, .05)
        for invalid in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                replace(self.cfg, pose_reward_weight=invalid).validate(.02)


if __name__ == '__main__':
    unittest.main()
