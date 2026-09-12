"""CPU-only checks; run from the curl_robot_2d repository root."""
import math
from pathlib import Path
import unittest
import mujoco
import numpy as np
from scripts.rolling_handoff import RollingHandoff, effective_action
from scripts.simulate_rolling_policy_sequence import torso_gyro


class HandoffTests(unittest.TestCase):
    def test_body_gyro_ignores_inertia_frame(self):
        m = mujoco.MjModel.from_xml_path('assets/rollingquad_description_2/mjcf/rollingquad_abd10_no_self_collision.xml')
        d = mujoco.MjData(m)
        torso = m.body('torso').id
        for rotation in ([1, 0, 0, 0], [.5, .5, .5, .5]):
            mujoco.mj_resetDataKeyframe(m, d, m.key('stand').id)
            d.qpos[3:7] = rotation
            d.qvel[3:6] = [.2, 4., -.3]
            mujoco.mj_forward(m, d)
            np.testing.assert_allclose(torso_gyro(m, d, torso), [.2, 4., -.3], atol=1e-12)

    def test_schedule_and_history(self):
        h = RollingHandoff()
        for _ in range(29):
            h.tick(True, .75, .07)
            self.assertEqual(h.yaw, 0)
        for _ in range(50):
            prev = h.yaw
            h.tick(True, .75, .07)
            self.assertLessEqual(abs(h.yaw-prev), .0014+1e-12)
        self.assertEqual(h.stage, 'rolling')
        h = RollingHandoff()
        for _ in range(115):
            h.tick(False, .75, .07)
        self.assertTrue(h.stop_required)
        self.assertEqual(h.yaw, 0)
        self.assertFalse(h.window(0, 0, math.nan, .01))
        class Policy:
            scale = np.array([0, .8, 1.2]*4)
            center = np.array([-.1, .1, .9]*4)
        target = np.array([-.2, .3, 1.2]*4)
        np.testing.assert_allclose(effective_action(Policy, target), [0, .25, .25]*4, atol=1e-7)

    def test_native_schedule_parity(self):
        path = Path('results/rolling_handoff_native_trace.txt')
        if not path.exists():
            self.skipTest('Run the native C++ trace check first')
        h = RollingHandoff()
        rows = []
        stages = {'blending': 6, 'settling': 7, 'command_ramp': 8, 'rolling': 3}
        for i in range(200):
            h.tick(True, .75 if i < 90 else .45, .07 if i < 90 else -.07)
            rows.append([h.ticks, stages[h.stage], h.alpha, h.vx, h.yaw])
        np.testing.assert_allclose(np.loadtxt(path), rows, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
