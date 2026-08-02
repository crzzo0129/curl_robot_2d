import math
import unittest

from curl_robot_2d_mjx.environment_3d import DEFAULT_3D_CEM_CONTROLLER
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
from scripts import train_mjx_3d_residual_ppo


class MJX3DTrainingEntrypointTest(unittest.TestCase):
    def test_training_entry_imports_without_jax(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args([])

        self.assertTrue(callable(train_mjx_3d_residual_ppo.main))
        self.assertEqual(args.preset, "smoke")
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.controller, DEFAULT_3D_CEM_CONTROLLER)
        self.assertEqual(args.reference_weight, 1.0)
        self.assertEqual(args.minimum_residual_gain, 0.05)

    def test_reward_overrides_use_3d_reward_config(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            [
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


if __name__ == "__main__":
    unittest.main()
