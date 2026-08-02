import inspect
import unittest

from curl_robot_2d.model import build_mjcf
from curl_robot_2d.parameters import FIXED_PARAMETERS
from dataclasses import replace
from scripts import sweep_torso_com_cem
from scripts.optimize_phase_controller import optimize_controller


class TorsoCOMSweepTest(unittest.TestCase):
    def test_optimizer_accepts_explicit_model_path(self) -> None:
        self.assertIn("model_path", inspect.signature(optimize_controller).parameters)

    def test_variant_name_is_filesystem_friendly(self) -> None:
        self.assertEqual(
            sweep_torso_com_cem.variant_name(-0.01, 0.025),
            "torso_com_x_m0.010_z_p0.025",
        )

    def test_generated_variant_changes_only_torso_com_fields(self) -> None:
        baseline = build_mjcf()
        variant = build_mjcf(
            replace(FIXED_PARAMETERS, torso_com_x=-0.01, torso_com_z=0.025)
        )

        self.assertIn('pos="-0.01 0 0.025"', variant)
        self.assertIn('mass="0.732"', variant)
        self.assertIn('mass="0.1"', variant)
        self.assertNotEqual(baseline, variant)


if __name__ == "__main__":
    unittest.main()
