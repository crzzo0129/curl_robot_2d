from pathlib import Path
import unittest

import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.cem_reference import load_cem_reference
from scripts import evaluate_3d_symmetric_cem_reference as bridge


class SymmetricCEM3DBridgeTest(unittest.TestCase):
    def test_defaults_are_curl_native_not_disk_robot(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            bridge.DEFAULT_CONTROLLER_PATH,
            project_root
            / "results"
            / "collision_constrained_cem_foot_gap_2mm_short_contact"
            / "best_phase_controller.json",
        )
        self.assertEqual(
            bridge.DEFAULT_XML_PATH,
            project_root / "assets" / "curl_robot_3d.xml",
        )
        self.assertTrue(bridge.DEFAULT_CONTROLLER_PATH.exists())
        self.assertTrue(bridge.DEFAULT_XML_PATH.exists())
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        self.assertNotIn("disk_robot", source)

    def test_planar_target_is_finite_and_within_2d_shell_ranges(self) -> None:
        config = load_cem_reference(bridge.DEFAULT_CONTROLLER_PATH)
        target = bridge.planar_cem_target(0.0, config)
        self.assertEqual(target.shape, (4,))
        self.assertTrue(np.isfinite(target).all())
        self.assertGreaterEqual(
            float(np.min(target[[0, 2]])),
            FIXED_PARAMETERS.hip.shell_compatible_range[0],
        )
        self.assertLessEqual(
            float(np.max(target[[0, 2]])),
            FIXED_PARAMETERS.hip.shell_compatible_range[1],
        )
        self.assertGreaterEqual(
            float(np.min(target[[1, 3]])),
            FIXED_PARAMETERS.knee.shell_compatible_range[0],
        )
        self.assertLessEqual(
            float(np.max(target[[1, 3]])),
            FIXED_PARAMETERS.knee.shell_compatible_range[1],
        )

    def test_target_scale_zero_returns_compact_pose(self) -> None:
        target = bridge.PLANAR_COMPACT + np.asarray((0.1, -0.2, 0.3, -0.4))
        np.testing.assert_allclose(
            bridge.scaled_planar_target(target, 0.0),
            bridge.PLANAR_COMPACT,
        )

    def test_phase_rate_scale_is_exposed_for_direction_smokes(self) -> None:
        args = bridge.parse_args(["--phase-rate-scale", "-1.0"])
        self.assertEqual(args.phase_rate_scale, -1.0)

    def test_phase_lock_is_default_and_linear_mode_is_explicit(self) -> None:
        feedback = bridge.parse_args([])
        linear = bridge.parse_args(["--linear-phase"])

        self.assertFalse(feedback.linear_phase)
        self.assertTrue(linear.linear_phase)

    def test_planar_to_curl_3d_mapping_duplicates_left_and_right(self) -> None:
        mapped = bridge.map_planar_to_curl_3d_targets(
            np.asarray((0.3, 0.8, 0.4, 1.0)),
        )
        np.testing.assert_allclose(
            mapped,
            np.asarray((0.3, 0.8, 0.3, 0.8, 0.4, 1.0, 0.4, 1.0)),
        )
        self.assertEqual(len(mapped), 8)


if __name__ == "__main__":
    unittest.main()
