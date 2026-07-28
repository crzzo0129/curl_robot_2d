import math
import unittest

import numpy as np

from scripts.explore_walking_controller import (
    WalkingControllerConfig,
    foot_trajectory,
    leg_forward_kinematics,
    leg_inverse_kinematics,
    walking_targets,
)


class WalkingControllerTest(unittest.TestCase):
    def test_inverse_kinematics_round_trip(self) -> None:
        for outward, depth in (
            (0.0, 0.25),
            (0.04, 0.24),
            (-0.03, 0.22),
        ):
            hip, knee = leg_inverse_kinematics(outward, depth)
            actual_outward, actual_depth = leg_forward_kinematics(hip, knee)
            self.assertAlmostEqual(actual_outward, outward, places=9)
            self.assertAlmostEqual(actual_depth, depth, places=9)

    def test_swing_trajectory_is_lifted_and_continuous(self) -> None:
        config = WalkingControllerConfig()
        before = foot_trajectory(config.duty_factor - 1.0e-8, config)
        after = foot_trajectory(config.duty_factor + 1.0e-8, config)
        middle = foot_trajectory(
            0.5 * (1.0 + config.duty_factor), config
        )

        np.testing.assert_allclose(before[:2], after[:2], atol=1.0e-7)
        self.assertFalse(middle[2])
        self.assertAlmostEqual(
            middle[1],
            config.body_height_m - config.foot_lift_m,
            places=9,
        )

    def test_half_cycle_offsets_virtual_legs(self) -> None:
        config = WalkingControllerConfig(
            pitch_kp_m_per_rad=0.0,
            pitch_kd_m_s_per_rad=0.0,
            velocity_gain_s=0.0,
        )
        targets, stance = walking_targets(0.0, 0.0, 0.0, 0.0, config)

        self.assertEqual(targets.shape, (4,))
        self.assertTrue(all(math.isfinite(value) for value in targets))
        self.assertEqual(stance, (True, True))
        self.assertNotAlmostEqual(targets[0], targets[2])


if __name__ == "__main__":
    unittest.main()
