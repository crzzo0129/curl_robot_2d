from pathlib import Path
import unittest

import numpy as np

from curl_robot_2d_mjx.config_walking_3d import (
    Walking3DConfig,
    validate_walking_3d_config,
)
from curl_robot_2d_mjx.environment_walking_3d import (
    WALKING_ACTION_SIZE_3D,
    WALKING_ASYMMETRIC_ACTOR_OBSERVATION_SIZE_3D,
    WALKING_ASYMMETRIC_CRITIC_OBSERVATION_SIZE_3D,
    WALKING_LEFT_RIGHT_LEG_PERMUTATION_3D,
    gait_phase_features_3d,
    mirror_walking_action_3d,
    mirror_walking_angular_velocity_3d,
    mirror_walking_command_3d,
    mirror_walking_leg_values_3d,
    mirror_walking_linear_velocity_3d,
    mirror_walking_phase_features_3d,
    mirror_walking_projected_gravity_3d,
)
from scripts import train_mjx_3d_walking_ppo


class MJX3DWalkingSymmetryTest(unittest.TestCase):
    def test_leg_permutation_swaps_left_and_right(self) -> None:
        self.assertEqual(
            WALKING_LEFT_RIGHT_LEG_PERMUTATION_3D,
            (1, 0, 3, 2),
        )
        values = np.arange(12)

        mirrored = mirror_walking_leg_values_3d(np, values)

        np.testing.assert_array_equal(
            mirrored,
            np.asarray((3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)),
        )
        np.testing.assert_array_equal(
            mirror_walking_leg_values_3d(np, mirrored), values
        )

    def test_action_mirror_is_an_involution_without_sign_changes(self) -> None:
        action = np.linspace(-1.0, 1.0, WALKING_ACTION_SIZE_3D)

        mirrored = mirror_walking_action_3d(np, action)

        np.testing.assert_allclose(
            mirror_walking_action_3d(np, mirrored), action
        )
        np.testing.assert_allclose(mirrored[:3], action[3:6])
        np.testing.assert_allclose(mirrored[6:9], action[9:12])

    def test_command_keeps_vx_and_reverses_vy_and_yaw(self) -> None:
        command = np.asarray((0.10, -0.03, 0.40))

        mirrored = mirror_walking_command_3d(np, command)

        np.testing.assert_allclose(mirrored, (0.10, 0.03, -0.40))
        np.testing.assert_allclose(
            mirror_walking_command_3d(np, mirrored), command
        )

    def test_vector_components_follow_sagittal_reflection(self) -> None:
        vector = np.asarray((0.2, -0.3, 0.4))

        np.testing.assert_allclose(
            mirror_walking_linear_velocity_3d(np, vector),
            (0.2, 0.3, 0.4),
        )
        np.testing.assert_allclose(
            mirror_walking_angular_velocity_3d(np, vector),
            (-0.2, -0.3, -0.4),
        )
        np.testing.assert_allclose(
            mirror_walking_projected_gravity_3d(np, vector),
            (0.2, 0.3, 0.4),
        )

    def test_phase_mirror_is_a_half_cycle_shift(self) -> None:
        for phase in (0.0, 0.13, 0.5, 0.91):
            features = gait_phase_features_3d(np, phase)
            shifted = gait_phase_features_3d(np, (phase + 0.5) % 1.0)

            np.testing.assert_allclose(
                mirror_walking_phase_features_3d(np, features),
                shifted,
                atol=1.0e-7,
            )

    def test_config_and_cli_preserve_disabled_default(self) -> None:
        config = Walking3DConfig()
        args = train_mjx_3d_walking_ppo.parse_args([])

        self.assertFalse(config.symmetry_augmentation_enabled)
        self.assertEqual(config.symmetry_mirror_probability, 0.5)
        self.assertFalse(args.symmetry_augmentation_enabled)
        self.assertEqual(args.symmetry_mirror_probability, 0.5)

        enabled = train_mjx_3d_walking_ppo.parse_args(
            [
                "--symmetry-augmentation",
                "--symmetry-mirror-probability",
                "0.25",
            ]
        )
        self.assertTrue(enabled.symmetry_augmentation_enabled)
        self.assertEqual(enabled.symmetry_mirror_probability, 0.25)

    def test_config_rejects_invalid_mirror_probability(self) -> None:
        for probability in (-0.01, 1.01, float("nan")):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError, "symmetry_mirror_probability"
                ):
                    validate_walking_3d_config(
                        Walking3DConfig(
                            symmetry_mirror_probability=probability
                        )
                    )

    def test_training_unmirrors_action_and_evaluation_disables_mirroring(
        self,
    ) -> None:
        environment_source = Path(
            train_mjx_3d_walking_ppo.__file__
        ).parents[1] / "curl_robot_2d_mjx" / "environment_walking_3d.py"
        environment_source = environment_source.read_text(encoding="utf-8")
        step_source = environment_source[
            environment_source.index("        def step(") :
        ]
        training_source = Path(
            train_mjx_3d_walking_ppo.__file__
        ).read_text(encoding="utf-8")
        eval_source = training_source[
            training_source.index("    eval_task = replace(") :
        ]

        self.assertLess(
            step_source.index(
                "mirror_walking_action_3d(jp, actor_policy_action)"
            ),
            step_source.index("self.nominal_ctrl\n                + policy_action"),
        )
        self.assertIn("symmetry_augmentation_enabled=False", eval_source)
        self.assertIn('"symmetry_mirrored": jp.asarray(False)', environment_source)

    def test_dimensions_are_unchanged(self) -> None:
        self.assertEqual(WALKING_ACTION_SIZE_3D, 12)
        self.assertEqual(WALKING_ASYMMETRIC_ACTOR_OBSERVATION_SIZE_3D, 47)
        self.assertEqual(WALKING_ASYMMETRIC_CRITIC_OBSERVATION_SIZE_3D, 74)


if __name__ == "__main__":
    unittest.main()
