import unittest
from pathlib import Path
import numpy as np
import mujoco
from curl_robot_2d_mjx.single_foot_compact_probe import (
    SingleFootCompactProbe, triangle_margin, triangle_incenter,
)


class SingleFootProbeTest(unittest.TestCase):
    def test_verified_foot_order_moves_front_pair_first(self):
        self.assertEqual(SingleFootCompactProbe.order, (0, 1, 3, 2))

    def test_support_margin_is_winding_independent_and_signed(self):
        triangle = np.array([[0., 0.], [1., 0.], [0., 1.]])
        center = triangle_incenter(triangle)
        self.assertGreater(triangle_margin(center, triangle), 0.)
        self.assertAlmostEqual(triangle_margin(center, triangle), triangle_margin(center, triangle[::-1]))
        self.assertLess(triangle_margin([1., 1.], triangle), 0.)

    def test_motor_command_does_not_overwrite_physical_state(self):
        source = Path(__file__).resolve().parents[1] / 'assets/rollingquad_description_2/mjcf/rollingquad_primitive.xml'
        model = mujoco.MjModel.from_xml_path(str(source))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
        mujoco.mj_forward(model, data)
        probe = SingleFootCompactProbe(model)
        probe.reset(data)
        expected_first_height = probe.nominal_root[2] + .25*(probe.ik.final_root[2]-probe.nominal_root[2])
        self.assertAlmostEqual(probe.root_goal[2], expected_first_height)
        qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        command = probe.step(data)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        np.testing.assert_array_equal(data.ctrl, ctrl)
        self.assertTrue(np.isfinite(command).all())
        self.assertLessEqual(np.max(np.abs(command-ctrl)), .02000001)
        # No force/pose confirmation means no completed step.
        self.assertEqual(probe.completed, 0)
        probe.done, probe.failed = True, True
        probe.reset(data)
        self.assertFalse(probe.done or probe.failed)
        self.assertEqual(len(probe.events), 1)


if __name__ == '__main__':
    unittest.main()
