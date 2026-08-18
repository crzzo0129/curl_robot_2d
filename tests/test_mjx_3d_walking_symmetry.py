import io
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
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
    WALKING_HEADING_OBSERVATION_SIZE_3D,
    WALKING_LEFT_RIGHT_LEG_PERMUTATION_3D,
    gait_phase_features_3d,
    mirror_walking_action_3d,
    mirror_walking_actor_observation_3d,
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

        batched_values = np.arange(2 * 3 * 12).reshape(2, 3, 12)
        batched_mirrored = mirror_walking_leg_values_3d(np, batched_values)
        np.testing.assert_array_equal(
            mirror_walking_leg_values_3d(np, batched_mirrored),
            batched_values,
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

    def test_actor_observation_mirror_is_a_batched_involution(self) -> None:
        cases = (
            (True, True, False, WALKING_ASYMMETRIC_ACTOR_OBSERVATION_SIZE_3D),
            (
                True,
                True,
                True,
                WALKING_ASYMMETRIC_ACTOR_OBSERVATION_SIZE_3D
                + WALKING_HEADING_OBSERVATION_SIZE_3D,
            ),
            (False, False, False, 48),
            (False, True, False, 50),
        )
        for asymmetric, gait_phase, heading, size in cases:
            with self.subTest(
                asymmetric=asymmetric,
                gait_phase=gait_phase,
                heading=heading,
            ):
                observation = np.arange(2 * 3 * size, dtype=np.float32).reshape(
                    2, 3, size
                )
                mirrored = mirror_walking_actor_observation_3d(
                    np,
                    observation,
                    asymmetric_observation_enabled=asymmetric,
                    gait_phase_enabled=gait_phase,
                    heading_observation_enabled=heading,
                )
                np.testing.assert_array_equal(
                    mirror_walking_actor_observation_3d(
                        np,
                        mirrored,
                        asymmetric_observation_enabled=asymmetric,
                        gait_phase_enabled=gait_phase,
                        heading_observation_enabled=heading,
                    ),
                    observation,
                )

    def test_actor_mirror_consistency_scope_adds_expected_mse(self) -> None:
        base_loss = np.asarray(2.0)

        def original_compute_ppo_loss(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network,
            **kwargs,
        ):
            del params, normalizer_params, data, rng, ppo_network, kwargs
            return base_loss, {"total_loss": base_loss, "policy_loss": -0.1}

        class LossModule:
            compute_ppo_loss = staticmethod(original_compute_ppo_loss)

        class PolicyNetwork:
            @staticmethod
            def apply(normalizer_params, policy_params, observation):
                del normalizer_params, policy_params
                shape = observation["state"].shape[:-1] + (12,)
                mean = np.broadcast_to(np.arange(12, dtype=np.float32), shape)
                return mean, np.ones(shape, dtype=np.float32)

        class ActionDistribution:
            @staticmethod
            def mode(logits):
                return logits[0]

        module = LossModule()
        original = module.compute_ppo_loss
        observation = {
            "state": np.zeros((2, 3, 47), dtype=np.float32),
            "privileged_state": np.zeros((2, 3, 74), dtype=np.float32),
        }
        data = SimpleNamespace(observation=observation)
        params = SimpleNamespace(policy="policy")
        ppo_network = SimpleNamespace(
            policy_network=PolicyNetwork(),
            parametric_action_distribution=ActionDistribution(),
        )

        with train_mjx_3d_walking_ppo._actor_mirror_consistency_loss_scope(
            module,
            weight=0.25,
            anchor="canonical_stop_gradient",
            array_module=np,
            stop_gradient=lambda value: value,
            asymmetric_observations=True,
            gait_phase_enabled=True,
        ):
            loss, metrics = module.compute_ppo_loss(
                params, "normalizer", data, "rng", ppo_network
            )
            self.assertAlmostEqual(float(loss), 4.25)
            self.assertAlmostEqual(
                float(metrics["actor_mirror_consistency_loss"]), 9.0
            )
            self.assertAlmostEqual(
                float(metrics["actor_mirror_consistency_rms"]), 3.0
            )
            self.assertAlmostEqual(
                float(metrics["actor_mirror_consistency_weighted_loss"]),
                2.25,
            )
            self.assertAlmostEqual(float(metrics["ppo_total_loss"]), 2.0)

        self.assertIs(module.compute_ppo_loss, original)

        with train_mjx_3d_walking_ppo._actor_mirror_consistency_loss_scope(
            module,
            weight=0.25,
            anchor="symmetric",
            array_module=np,
            stop_gradient=lambda value: self.fail(
                "symmetric consistency must not stop either Actor branch"
            ),
            asymmetric_observations=True,
            gait_phase_enabled=True,
        ):
            loss, metrics = module.compute_ppo_loss(
                params, "normalizer", data, "rng", ppo_network
            )
            self.assertAlmostEqual(float(loss), 4.25)
            self.assertAlmostEqual(
                float(metrics["actor_mirror_consistency_loss"]), 9.0
            )

        self.assertIs(module.compute_ppo_loss, original)

    def test_config_and_cli_preserve_disabled_default(self) -> None:
        config = Walking3DConfig()
        args = train_mjx_3d_walking_ppo.parse_args([])

        self.assertFalse(config.symmetry_augmentation_enabled)
        self.assertEqual(config.symmetry_mirror_probability, 0.5)
        self.assertFalse(args.symmetry_augmentation_enabled)
        self.assertEqual(args.symmetry_mirror_probability, 0.5)
        self.assertEqual(args.actor_mirror_consistency_weight, 0.0)
        self.assertEqual(
            args.actor_mirror_consistency_anchor,
            "canonical_stop_gradient",
        )

        enabled = train_mjx_3d_walking_ppo.parse_args(
            [
                "--symmetry-augmentation",
                "--symmetry-mirror-probability",
                "0.25",
            ]
        )
        self.assertTrue(enabled.symmetry_augmentation_enabled)
        self.assertEqual(enabled.symmetry_mirror_probability, 0.25)

        consistency = train_mjx_3d_walking_ppo.parse_args(
            [
                "--actor-mirror-consistency-weight",
                "0.01",
                "--actor-mirror-consistency-anchor",
                "symmetric",
            ]
        )
        self.assertFalse(consistency.symmetry_augmentation_enabled)
        self.assertEqual(consistency.actor_mirror_consistency_weight, 0.01)
        self.assertEqual(consistency.actor_mirror_consistency_anchor, "symmetric")

    def test_actor_consistency_rejects_invalid_or_episode_mirror_use(self) -> None:
        invalid_argv = (
            ["--actor-mirror-consistency-weight", "-0.01"],
            ["--actor-mirror-consistency-weight", "nan"],
            [
                "--actor-mirror-consistency-weight",
                "0.01",
                "--symmetry-augmentation",
            ],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train_mjx_3d_walking_ppo.parse_args(argv)

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
        self.assertIn("symmetry_mirrored=False", environment_source)
        self.assertIn(
            '"symmetry_mirrored": symmetry_mirrored', environment_source
        )

    def test_dimensions_are_unchanged(self) -> None:
        self.assertEqual(WALKING_ACTION_SIZE_3D, 12)
        self.assertEqual(WALKING_ASYMMETRIC_ACTOR_OBSERVATION_SIZE_3D, 47)
        self.assertEqual(WALKING_ASYMMETRIC_CRITIC_OBSERVATION_SIZE_3D, 74)
        self.assertEqual(WALKING_HEADING_OBSERVATION_SIZE_3D, 2)


if __name__ == "__main__":
    unittest.main()
