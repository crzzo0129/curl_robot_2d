import math
import unittest

from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS
from scripts.demo_handcrafted_3d_walk import foot_cycle, sagittal_leg_ik


class Handcrafted3DWalkTest(unittest.TestCase):
    def test_sagittal_ik_reconstructs_target(self) -> None:
        geometry = PUPPER_ORIGINAL_SHELL_60_PARAMETERS
        for outward, depth in ((0.01, 0.16), (0.035, 0.145), (-0.02, 0.15)):
            hip, knee = sagittal_leg_ik(outward, depth)
            reconstructed_x = (
                geometry.upper_length * math.sin(hip)
                + geometry.lower_length * math.sin(hip - knee)
            )
            reconstructed_depth = (
                geometry.upper_length * math.cos(hip)
                + geometry.lower_length * math.cos(hip - knee)
            )
            self.assertAlmostEqual(reconstructed_x, outward, places=8)
            self.assertAlmostEqual(reconstructed_depth, depth, places=8)

    def test_trot_foot_cycle_has_continuous_lifted_swing(self) -> None:
        duty = 0.68
        before = foot_cycle(
            duty - 1.0e-8,
            step_length_m=0.045,
            lift_m=0.025,
            duty_factor=duty,
        )
        after = foot_cycle(
            duty + 1.0e-8,
            step_length_m=0.045,
            lift_m=0.025,
            duty_factor=duty,
        )
        middle = foot_cycle(
            0.5 * (1.0 + duty),
            step_length_m=0.045,
            lift_m=0.025,
            duty_factor=duty,
        )

        self.assertAlmostEqual(before[0], after[0], places=6)
        self.assertAlmostEqual(before[1], after[1], places=6)
        self.assertFalse(middle[2])
        self.assertAlmostEqual(middle[1], 0.025, places=8)


if __name__ == "__main__":
    unittest.main()
