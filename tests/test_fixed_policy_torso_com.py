import math
import unittest

from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts import evaluate_fixed_policy_torso_com as sweep


class FixedPolicyTorsoCOMTest(unittest.TestCase):
    def test_circle_center_converts_to_root_coordinates(self) -> None:
        x_root, z_root = sweep.root_coordinates_from_circle(0.0, 0.0)

        self.assertEqual(x_root, 0.0)
        self.assertAlmostEqual(
            z_root,
            -FIXED_PARAMETERS.regular_pentagon_apothem,
            places=12,
        )

    def test_original_com_is_recovered_from_circle_coordinates(self) -> None:
        z_center = (
            FIXED_PARAMETERS.torso_com_z
            + FIXED_PARAMETERS.regular_pentagon_apothem
        )
        x_root, z_root = sweep.root_coordinates_from_circle(
            FIXED_PARAMETERS.torso_com_x, z_center
        )

        self.assertAlmostEqual(x_root, FIXED_PARAMETERS.torso_com_x)
        self.assertAlmostEqual(z_root, FIXED_PARAMETERS.torso_com_z)

    def test_upper_circle_filter_removes_rectangular_corner(self) -> None:
        self.assertTrue(sweep.is_inside_upper_circle(0.050, 0.118228644035, 0.140))
        self.assertFalse(sweep.is_inside_upper_circle(0.075, 0.140, 0.140))
        self.assertFalse(sweep.is_inside_upper_circle(0.0, -0.001, 0.140))
        self.assertTrue(sweep.is_inside_upper_circle(0.0, 0.140, 0.140))
        self.assertTrue(math.hypot(0.050, 0.118228644035) < 0.140)


if __name__ == "__main__":
    unittest.main()
