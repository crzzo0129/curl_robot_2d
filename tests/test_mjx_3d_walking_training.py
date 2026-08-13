import io
from contextlib import redirect_stderr
from pathlib import Path
import unittest

from curl_robot_2d_mjx.reward_walking_3d import Walking3DRewardConfig
from scripts import train_mjx_3d_walking_ppo
from scripts import train_mjx_3d_real_geometry_walking


class MJX3DWalkingTrainingEntrypointTest(unittest.TestCase):
    def test_training_entry_imports_without_jax(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args([])

        self.assertTrue(callable(train_mjx_3d_walking_ppo.main))
        self.assertEqual(args.preset, "smoke")
        self.assertEqual(args.recipe, "anymal_v1")
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.geometry, "pupper_open60")
        self.assertEqual(args.reset_keyframe, "stand")
        self.assertEqual(args.desired_speed_m_s, 0.20)
        self.assertEqual(args.action_scale_abduction, 0.10)
        self.assertEqual(args.command_forward_min, -0.10)
        self.assertEqual(args.command_forward_max, 0.35)
        self.assertEqual(args.command_lateral_max, 0.15)
        self.assertEqual(args.command_yaw_rate_max, 0.60)
        self.assertFalse(args.no_domain_randomization)
        self.assertEqual(args.action_scale_hip, 0.40)
        self.assertEqual(args.action_scale_knee, 0.55)
        self.assertEqual(args.startup_action_ramp_s, 0.50)
        self.assertEqual(args.terminate_airborne_duration, 0.25)
        self.assertEqual(args.terminate_nonfoot_contact_duration, 0.12)
        self.assertEqual(args.terminate_self_contact_duration, 0.10)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.entropy_cost, 1e-2)
        self.assertFalse(args.save_ppo_checkpoints)
        self.assertIsNone(args.ppo_checkpoint_dir)

    def test_real_geometry_overnight_entry_uses_h200_preset(self) -> None:
        args = train_mjx_3d_real_geometry_walking.parse_args([])
        values = train_mjx_3d_real_geometry_walking.training_argv(args)

        self.assertEqual(args.preset, "h200")
        self.assertIn("--geometry", values)
        self.assertEqual(values[values.index("--geometry") + 1], "real")
        self.assertIn("--save-ppo-checkpoints", values)

    def test_reward_overrides_use_walking_reward_config(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(
            [
                "--reward-velocity-tracking",
                "2.0",
                "--reward-upright",
                "0.8",
                "--reward-nonfoot-contact",
                "3.0",
            ]
        )

        reward = train_mjx_3d_walking_ppo._reward_config_from_args(args)

        self.assertIsInstance(reward, Walking3DRewardConfig)
        self.assertEqual(reward.velocity_tracking, 2.0)
        self.assertEqual(reward.upright, 0.8)
        self.assertEqual(reward.nonfoot_contact, 3.0)

    def test_checkpoint_selection_prefers_stable_commanded_walk(self) -> None:
        base = {
            "eval/avg_episode_length": 500.0,
            "eval/episode_failed": 0.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_forward_progress_m": 0.28,
            "eval/avg_forward_velocity_m_s": 0.028,
            "eval/avg_upright_tilt_rad": 0.05,
            "eval/avg_lateral_drift_m": 0.005,
            "eval/avg_nonfoot_ground_contact_count": 0.0,
            "eval/avg_self_contact_count": 0.0,
        }
        stable = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            base,
            500,
            target_distance_m=0.28,
            desired_speed_m_s=0.028,
        )
        drifted = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {**base, "eval/avg_lateral_drift_m": 0.20},
            500,
            target_distance_m=0.28,
            desired_speed_m_s=0.028,
        )
        tilted = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {**base, "eval/avg_upright_tilt_rad": 0.50},
            500,
            target_distance_m=0.28,
            desired_speed_m_s=0.028,
        )
        stopped = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {
                **base,
                "eval/episode_forward_progress_m": 0.0,
                "eval/avg_forward_velocity_m_s": 0.0,
            },
            500,
            target_distance_m=0.28,
            desired_speed_m_s=0.028,
        )

        self.assertFalse(stable["rejected"])
        self.assertGreater(stable["score"], drifted["score"])
        self.assertGreater(stable["score"], tilted["score"])
        self.assertGreater(stable["score"], stopped["score"])

    def test_checkpoint_progress_is_gated_by_survival(self) -> None:
        crash = {
            "eval/avg_episode_length": 50.0,
            "eval/episode_failed": 1.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_forward_progress_m": 0.28,
            "eval/avg_forward_velocity_m_s": 0.28,
            "eval/avg_upright_tilt_rad": 0.05,
            "eval/avg_lateral_drift_m": 0.0,
            "eval/avg_nonfoot_ground_contact_count": 0.1,
            "eval/avg_self_contact_count": 0.0,
        }

        selection = (
            train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
                crash,
                500,
                target_distance_m=0.28,
                desired_speed_m_s=0.028,
            )
        )

        self.assertEqual(selection["raw_progress_quality"], 1.0)
        self.assertAlmostEqual(selection["progress_quality"], 0.1)

    def test_checkpoint_selection_rejects_nonfinite_eval(self) -> None:
        rejected = (
            train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
                {
                    "eval/avg_episode_length": 500.0,
                    "eval/episode_failed": 0.0,
                    "eval/episode_failure_nonfinite": 1.0,
                    "eval/episode_forward_progress_m": 0.28,
                    "eval/avg_forward_velocity_m_s": 0.028,
                    "eval/avg_upright_tilt_rad": 0.05,
                    "eval/avg_lateral_drift_m": 0.0,
                    "eval/avg_nonfoot_ground_contact_count": 0.0,
                    "eval/avg_self_contact_count": 0.0,
                },
                500,
                target_distance_m=0.28,
                desired_speed_m_s=0.028,
            )
        )

        self.assertTrue(rejected["rejected"])
        self.assertLess(rejected["score"], -999_999.0)

    def test_periodic_checkpoint_directory_requires_checkpoint_flag(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mjx_3d_walking_ppo.parse_args(
                ["--ppo-checkpoint-dir", "/tmp/mjx_3d_walking_ckpt"]
            )

        args = train_mjx_3d_walking_ppo.parse_args(
            [
                "--save-ppo-checkpoints",
                "--ppo-checkpoint-dir",
                "mjx_3d_walking_ckpt",
            ]
        )

        self.assertTrue(args.save_ppo_checkpoints)
        self.assertEqual(
            args.ppo_checkpoint_dir, Path("mjx_3d_walking_ckpt")
        )


if __name__ == "__main__":
    unittest.main()
