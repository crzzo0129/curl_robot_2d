from pathlib import Path
import unittest

import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d_mjx.cem_reference import load_cem_reference
from scripts import evaluate_3d_symmetric_cem_reference as bridge
from scripts import view_3d_cem_reference as viewer


class SymmetricCEM3DBridgeTest(unittest.TestCase):
    def test_headless_gif_uses_explicit_torso_tracking_camera(self) -> None:
        source = Path(viewer.__file__).read_text(encoding="utf-8")
        self.assertIn("render_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING", source)
        self.assertIn("render_camera.trackbodyid = torso_id", source)
        self.assertIn("camera=render_camera", source)

    def test_defaults_are_curl_native_not_disk_robot(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            bridge.DEFAULT_CONTROLLER_PATH,
            project_root
            / "results"
            / "pupper_r127p5_open60_shell150_45_three_stage_cem"
            / "03_strict_forbidden_collision"
            / "best_phase_controller.json",
        )
        self.assertEqual(
            bridge.DEFAULT_XML_PATH,
            project_root
            / "assets"
            / "rollingquad_description_2"
            / "mjcf"
            / "rollingquad.xml",
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

    def test_startup_target_boost_decays_to_target_scale(self) -> None:
        self.assertAlmostEqual(
            bridge.startup_target_scale(
                0.0,
                target_scale=1.0,
                startup_scale=None,
                ramp_duration_s=0.5,
                startup_boost=0.20,
                startup_boost_duration_s=0.5,
            ),
            1.20,
        )
        self.assertAlmostEqual(
            bridge.startup_target_scale(
                0.5,
                target_scale=1.0,
                startup_scale=None,
                ramp_duration_s=0.5,
                startup_boost=0.20,
                startup_boost_duration_s=0.5,
            ),
            1.0,
        )

    def test_startup_target_scale_can_ramp_from_safe_scale(self) -> None:
        self.assertAlmostEqual(
            bridge.startup_target_scale(
                0.0,
                target_scale=1.0,
                startup_scale=0.25,
                ramp_duration_s=0.5,
                startup_boost=0.0,
                startup_boost_duration_s=0.5,
            ),
            0.25,
        )
        self.assertAlmostEqual(
            bridge.startup_target_scale(
                0.5,
                target_scale=1.0,
                startup_scale=0.25,
                ramp_duration_s=0.5,
                startup_boost=0.0,
                startup_boost_duration_s=0.5,
            ),
            1.0,
        )

    def test_phase_rate_scale_is_exposed_for_direction_smokes(self) -> None:
        args = bridge.parse_args(["--phase-rate-scale", "-1.0"])
        self.assertEqual(args.phase_rate_scale, -1.0)

    def test_startup_reference_boost_is_exposed_for_transfer_smokes(self) -> None:
        args = bridge.parse_args(
            [
                "--target-scale",
                "1.1",
                "--startup-target-scale",
                "0.4",
                "--target-ramp-duration-s",
                "0.1",
                "--startup-target-boost",
                "0.2",
                "--startup-target-boost-duration-s",
                "0.4",
            ]
        )

        self.assertEqual(args.target_scale, 1.1)
        self.assertEqual(args.startup_target_scale, 0.4)
        self.assertEqual(args.target_ramp_duration_s, 0.1)
        self.assertEqual(args.startup_target_boost, 0.2)
        self.assertEqual(args.startup_target_boost_duration_s, 0.4)

    def test_phase_lock_is_default_and_linear_mode_is_explicit(self) -> None:
        feedback = bridge.parse_args([])
        linear = bridge.parse_args(["--linear-phase"])

        self.assertFalse(feedback.linear_phase)
        self.assertTrue(linear.linear_phase)

    def test_default_startup_matches_2d_compact_ramp(self) -> None:
        self.assertEqual(bridge.parse_args([]).startup_target_scale, 0.0)
        self.assertEqual(viewer.parse_args([]).startup_target_scale, 0.0)

    def test_reference_physics_is_default_and_cg12_is_selectable(self) -> None:
        reference = bridge.parse_args([])
        cg12 = bridge.parse_args(["--physics-profile", "cg12"])

        self.assertEqual(reference.physics_profile, "reference")
        self.assertEqual(cg12.physics_profile, "cg12")

    def test_viewer_can_render_the_same_physics_profiles(self) -> None:
        reference = viewer.parse_args([])
        cg12 = viewer.parse_args(["--physics-profile", "cg12"])

        self.assertEqual(reference.physics_profile, "reference")
        self.assertEqual(cg12.physics_profile, "cg12")

    def test_viewer_exposes_reference_startup_boost(self) -> None:
        args = viewer.parse_args(
            [
                "--target-scale",
                "1.1",
                "--startup-target-scale",
                "0.4",
                "--target-ramp-duration-s",
                "0.1",
                "--startup-target-boost",
                "0.2",
                "--startup-target-boost-duration-s",
                "0.4",
            ]
        )

        self.assertEqual(args.target_scale, 1.1)
        self.assertEqual(args.startup_target_scale, 0.4)
        self.assertEqual(args.target_ramp_duration_s, 0.1)
        self.assertEqual(args.startup_target_boost, 0.2)
        self.assertEqual(args.startup_target_boost_duration_s, 0.4)

    def test_planar_to_curl_3d_mapping_duplicates_left_and_right(self) -> None:
        mapped = bridge.map_planar_to_curl_3d_targets(
            np.asarray((0.3, 0.8, 0.4, 1.0)),
        )
        np.testing.assert_allclose(
            mapped,
            np.asarray((0.3, 0.8, 0.3, 0.8, 0.4, 1.0, 0.4, 1.0)),
        )
        self.assertEqual(len(mapped), 8)

    def test_pupper_cpu_reference_enables_rolling_shell_collisions(self) -> None:
        source = Path(bridge.__file__).read_text(encoding="utf-8")

        self.assertIn('if args.geometry == "pupper60":', source)
        self.assertIn(
            "configure_pupper_shell_collisions_3d(model, enabled=True)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
