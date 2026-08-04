import io
import math
from contextlib import redirect_stderr
from pathlib import Path
import unittest

from curl_robot_2d_mjx.environment_3d import DEFAULT_3D_CEM_CONTROLLER
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
from scripts import train_mjx_3d_residual_ppo


class MJX3DTrainingEntrypointTest(unittest.TestCase):
    def test_training_entry_imports_without_jax(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args([])

        self.assertTrue(callable(train_mjx_3d_residual_ppo.main))
        self.assertEqual(args.preset, "smoke")
        self.assertEqual(args.recipe, "anchored_v1")
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.controller, DEFAULT_3D_CEM_CONTROLLER)
        self.assertEqual(args.reference_weight, 1.0)
        self.assertEqual(args.minimum_residual_gain, 0.05)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.entropy_cost, 1e-2)
        self.assertEqual(args.selection_target_turns, 1.0)
        self.assertFalse(args.zero_residual_policy_init)
        self.assertEqual(args.initial_policy_std, 1.0)
        self.assertTrue(args.deterministic_eval)
        self.assertFalse(args.save_ppo_checkpoints)
        self.assertIsNone(args.ppo_checkpoint_dir)

    def test_push_v2_recipe_applies_training_and_reward_defaults(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(["--recipe", "push_v2"])

        reward = train_mjx_3d_residual_ppo._reward_config_from_args(args)

        self.assertEqual(args.reference_weight, 1.0)
        self.assertEqual(args.minimum_residual_gain, 0.30)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.entropy_cost, 3e-3)
        self.assertEqual(reward.roll_progress, 12.0)
        self.assertEqual(reward.roll_mismatch, 0.25)
        self.assertEqual(reward.backward, 0.4)
        self.assertEqual(reward.residual_action, 0.003)

    def test_phase_locked_v3_uses_bounded_independent_residual_recipe(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--recipe", "phase_locked_v3"]
        )

        reward = train_mjx_3d_residual_ppo._reward_config_from_args(args)

        self.assertEqual(args.reference_weight, 1.0)
        self.assertEqual(args.minimum_residual_gain, 0.15)
        self.assertEqual(args.phase_rate_scale, 1.0)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.entropy_cost, 1e-3)
        self.assertEqual(args.selection_target_turns, 10.0)
        self.assertTrue(args.zero_residual_policy_init)
        self.assertEqual(args.initial_policy_std, 0.20)
        self.assertEqual(reward.roll_progress, 8.0)
        self.assertEqual(reward.roll_mismatch, 0.8)
        self.assertEqual(reward.backward, 1.0)
        self.assertEqual(reward.axis_tilt, 10.0)
        self.assertEqual(reward.residual_action, 0.01)

    def test_tanh_normal_scale_logit_recovers_requested_std(self) -> None:
        scale_logit = (
            train_mjx_3d_residual_ppo._tanh_normal_scale_logit(0.20)
        )
        softplus = math.log1p(math.exp(scale_logit))

        self.assertAlmostEqual(
            softplus + train_mjx_3d_residual_ppo.TANH_NORMAL_MIN_STD,
            0.20,
        )

    def test_observation_width_accepts_brax_shape_tuple(self) -> None:
        self.assertEqual(
            train_mjx_3d_residual_ppo._observation_width(59), 59
        )
        self.assertEqual(
            train_mjx_3d_residual_ppo._observation_width((59,)), 59
        )

    def test_stochastic_eval_requires_explicit_opt_in(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--no-deterministic-eval"]
        )

        self.assertFalse(args.deterministic_eval)

    def test_initial_policy_std_must_exceed_distribution_floor(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mjx_3d_residual_ppo.parse_args(
                [
                    "--recipe",
                    "phase_locked_v3",
                    "--initial-policy-std",
                    "0.001",
                ]
            )

    def test_reward_overrides_use_3d_reward_config(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            [
                "--recipe",
                "push_v2",
                "--reward-roll-progress",
                "7.5",
                "--reward-axis-tilt",
                "9.0",
                "--reward-cross-side-foot-contact",
                "40.0",
            ]
        )

        reward = train_mjx_3d_residual_ppo._reward_config_from_args(args)

        self.assertIsInstance(reward, Rolling3DRewardConfig)
        self.assertEqual(reward.roll_progress, 7.5)
        self.assertEqual(reward.axis_tilt, 9.0)
        self.assertEqual(reward.cross_side_foot_contact, 40.0)
        self.assertEqual(reward.roll_mismatch, 0.25)

    def test_checkpoint_selection_prefers_stable_forward_roll(self) -> None:
        base = {
            "eval/avg_episode_length": 500.0,
            "eval/episode_failed": 0.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_roll_progress_rad": 2.0 * math.pi,
            "eval/avg_lateral_drift_m": 0.01,
            "eval/avg_axis_tilt_rad": 0.05,
            "eval/avg_forbidden_penetration_m": 0.0,
            "eval/avg_forbidden_contact_count": 0.0,
        }
        stable = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            base, episode_length=500
        )
        drifted = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            {**base, "eval/avg_lateral_drift_m": 0.20},
            episode_length=500,
        )
        tilted = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            {**base, "eval/avg_axis_tilt_rad": 0.50},
            episode_length=500,
        )

        self.assertFalse(stable["rejected"])
        self.assertGreater(stable["score"], drifted["score"])
        self.assertGreater(stable["score"], tilted["score"])

    def test_checkpoint_selection_rejects_nonfinite_eval(self) -> None:
        rejected = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            {
                "eval/avg_episode_length": 500.0,
                "eval/episode_failed": 0.0,
                "eval/episode_failure_nonfinite": 1.0,
                "eval/episode_roll_progress_rad": 2.0 * math.pi,
                "eval/avg_lateral_drift_m": 0.0,
                "eval/avg_axis_tilt_rad": 0.0,
                "eval/avg_forbidden_penetration_m": 0.0,
                "eval/avg_forbidden_contact_count": 0.0,
            },
            episode_length=500,
        )

        self.assertTrue(rejected["rejected"])
        self.assertLess(rejected["score"], -999_999.0)

    def test_periodic_checkpoint_directory_requires_checkpoint_flag(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mjx_3d_residual_ppo.parse_args(
                ["--ppo-checkpoint-dir", "/tmp/mjx_3d_ckpt"]
            )

        args = train_mjx_3d_residual_ppo.parse_args(
            [
                "--save-ppo-checkpoints",
                "--ppo-checkpoint-dir",
                "mjx_3d_ckpt",
            ]
        )

        self.assertTrue(args.save_ppo_checkpoints)
        self.assertEqual(args.ppo_checkpoint_dir, Path("mjx_3d_ckpt"))


if __name__ == "__main__":
    unittest.main()
