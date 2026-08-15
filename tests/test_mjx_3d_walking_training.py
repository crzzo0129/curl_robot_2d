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
        self.assertFalse(hasattr(args, "startup_action_ramp_s"))
        self.assertEqual(args.terminate_airborne_duration, 0.25)
        self.assertEqual(args.terminate_nonfoot_contact_duration, 0.12)
        self.assertEqual(args.terminate_self_contact_duration, 0.10)
        self.assertEqual(args.diagnostic_lateral_drift, 1.50)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.entropy_cost, 1e-2)
        self.assertEqual(args.reset_joint_noise, 0.015)
        self.assertEqual(args.reset_velocity_noise, 0.05)
        self.assertEqual(args.reset_root_xy_velocity_noise, 0.15)
        self.assertEqual(args.reset_root_yaw_rate_noise, 0.20)
        self.assertEqual(args.init_noise_std, 0.30)
        self.assertEqual(args.clipping_epsilon, 0.20)
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertEqual(args.desired_kl, 0.01)
        self.assertEqual(args.learning_rate_schedule, "ADAPTIVE_KL")
        self.assertEqual(args.adaptive_kl_min_lr, 3e-5)
        self.assertEqual(args.adaptive_kl_max_lr, 3e-4)
        self.assertTrue(args.deterministic_eval)
        self.assertFalse(args.save_ppo_checkpoints)
        self.assertIsNone(args.ppo_checkpoint_dir)

    def test_forward_stage1_recipe_is_an_exact_stand_curriculum(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(
            ["--recipe", "forward_stage1_v1"]
        )
        reward = train_mjx_3d_walking_ppo._reward_config_from_args(args)

        self.assertEqual(args.desired_speed_m_s, 0.10)
        self.assertEqual(args.command_forward_min, 0.10)
        self.assertEqual(args.command_forward_max, 0.10)
        self.assertEqual(args.command_lateral_max, 0.0)
        self.assertEqual(args.command_yaw_rate_max, 0.0)
        self.assertEqual(args.command_stop_probability, 0.0)
        self.assertTrue(args.no_observation_noise)
        self.assertTrue(args.no_domain_randomization)
        self.assertEqual(args.reset_joint_noise, 0.0)
        self.assertEqual(args.reset_velocity_noise, 0.0)
        self.assertEqual(args.reset_root_xy_velocity_noise, 0.0)
        self.assertEqual(args.reset_root_yaw_rate_noise, 0.0)
        self.assertEqual(args.updates_per_batch, 1)
        self.assertEqual(args.unroll_length, 40)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.adaptive_kl_min_lr, 2e-6)
        self.assertEqual(args.adaptive_kl_max_lr, 2e-5)
        self.assertEqual(args.entropy_cost, 0.01)
        self.assertEqual(args.reward_scaling, 0.05)
        self.assertEqual(args.init_noise_std, 0.10)
        self.assertEqual(args.action_scale_hip, 0.50)
        self.assertEqual(args.action_scale_knee, 0.65)
        self.assertEqual(args.desired_kl, 0.003)
        self.assertEqual(reward.velocity_tracking, 4.0)
        self.assertEqual(reward.velocity_tracking_sigma_m_s, 0.05)
        self.assertEqual(reward.overspeed, 1.0)
        self.assertEqual(reward.yaw_rate_tracking, 0.25)
        self.assertEqual(reward.forward_progress, 0.0)
        self.assertEqual(reward.upright, 0.2)
        self.assertEqual(reward.upright_sigma_rad, 0.20)
        self.assertEqual(reward.foot_air_time, 0.8)
        self.assertEqual(reward.swing_clearance, 0.15)
        self.assertEqual(reward.swing_clearance_m, 0.025)
        self.assertEqual(reward.termination, 20.0)

    def test_noise_flags_can_override_recipe_defaults(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(
            [
                "--recipe",
                "forward_stage1_v1",
                "--observation-noise",
                "--domain-randomization",
                "--reset-root-xy-velocity-noise",
                "0.02",
            ]
        )

        self.assertFalse(args.no_observation_noise)
        self.assertFalse(args.no_domain_randomization)
        self.assertEqual(args.reset_root_xy_velocity_noise, 0.02)

    def test_stochastic_eval_is_an_explicit_opt_in(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(["--stochastic-eval"])

        self.assertFalse(args.deterministic_eval)

    def test_walking_ppo_uses_stable_distribution_contract(self) -> None:
        source = Path(train_mjx_3d_walking_ppo.__file__).read_text(
            encoding="utf-8"
        )

        for token in (
            'distribution_type="normal"',
            'noise_std_type="log"',
            'state_dependent_std=False',
            "mean_kernel_init_fn=jnn.initializers.uniform",
            'mean_kernel_init_kwargs={"scale": WALKING_ACTOR_MEAN_INIT_SCALE}',
            "mean_clip_scale=WALKING_ACTOR_MEAN_CLIP_SCALE",
            "deterministic_eval=args.deterministic_eval",
            "max_grad_norm=args.max_grad_norm",
            "learning_rate_schedule=args.learning_rate_schedule",
            "learning_rate_schedule_min_lr=args.adaptive_kl_min_lr",
            "learning_rate_schedule_max_lr=args.adaptive_kl_max_lr",
            "normalize_observations=False",
            "bootstrap_on_timeout=True",
            "reset_root_xy_velocity_noise_m_s=0.0",
        ):
            self.assertIn(token, source)

    def test_h200_uses_many_smaller_ppo_updates(self) -> None:
        preset = train_mjx_3d_walking_ppo.PRESETS_WALKING_3D["h200"]
        schedule = train_mjx_3d_walking_ppo._training_step_schedule(
            requested_steps=3_000_000,
            num_evals=6,
            batch_size=preset["batch_size"],
            unroll_length=20,
            num_minibatches=preset["num_minibatches"],
        )

        self.assertEqual(preset["envs"], 2048)
        self.assertEqual(preset["batch_size"], 256)
        self.assertEqual(preset["num_minibatches"], 8)
        self.assertEqual(schedule["rollout_quantum"], 40_960)
        self.assertEqual(schedule["effective_steps"], 3_072_000)
        self.assertEqual(
            schedule["eval_intervals"] * schedule["updates_per_interval"],
            75,
        )

    def test_lateral_drift_threshold_is_diagnostic_with_legacy_alias(self) -> None:
        current = train_mjx_3d_walking_ppo.parse_args(
            ["--diagnostic-lateral-drift", "0.75"]
        )
        legacy = train_mjx_3d_walking_ppo.parse_args(
            ["--terminate-lateral-drift", "0.90"]
        )

        self.assertEqual(current.diagnostic_lateral_drift, 0.75)
        self.assertEqual(legacy.diagnostic_lateral_drift, 0.90)
        self.assertNotIn(
            "failure_lateral_drift",
            train_mjx_3d_walking_ppo.WALKING_FAILURE_METRICS_3D,
        )
        self.assertIn(
            "lateral_drift_exceeded",
            train_mjx_3d_walking_ppo.PER_STEP_WALKING_METRICS_3D,
        )

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
            "eval/avg_planar_velocity_error_m_s": 0.002,
            "eval/avg_yaw_rate_error_rad_s": 0.01,
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
        poor_tracking = (
            train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
                {
                    **base,
                    "eval/avg_planar_velocity_error_m_s": 0.20,
                    "eval/avg_yaw_rate_error_rad_s": 0.60,
                },
                500,
                target_distance_m=0.28,
                desired_speed_m_s=0.028,
            )
        )

        self.assertFalse(stable["rejected"])
        self.assertGreater(stable["score"], drifted["score"])
        self.assertGreater(stable["score"], tilted["score"])
        self.assertGreater(stable["score"], stopped["score"])
        self.assertGreater(stable["score"], poor_tracking["score"])

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

    def test_checkpoint_rank_never_trades_survival_for_contact_quality(self) -> None:
        stable = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {
                "eval/avg_episode_length": 450.0,
                "eval/episode_failed": 0.10,
                "eval/episode_failure_upright_tilt": 0.05,
                "eval/episode_failure_nonfinite": 0.0,
                "eval/episode_forward_progress_m": 0.70,
                "eval/avg_forward_velocity_m_s": 0.08,
                "eval/avg_planar_velocity_error_m_s": 0.03,
                "eval/avg_yaw_rate_error_rad_s": 0.02,
                "eval/avg_upright_tilt_rad": 0.10,
                "eval/avg_lateral_drift_m": 0.02,
                "eval/avg_nonfoot_ground_contact_count": 0.20,
                "eval/avg_self_contact_count": 0.10,
            },
            500,
            target_distance_m=1.0,
            desired_speed_m_s=0.10,
        )
        crashed = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {
                "eval/avg_episode_length": 60.0,
                "eval/episode_failed": 1.0,
                "eval/episode_failure_upright_tilt": 1.0,
                "eval/episode_failure_nonfinite": 0.0,
                "eval/episode_forward_progress_m": 0.12,
                "eval/avg_forward_velocity_m_s": 0.10,
                "eval/avg_planar_velocity_error_m_s": 0.01,
                "eval/avg_yaw_rate_error_rad_s": 0.01,
                "eval/avg_upright_tilt_rad": 0.20,
                "eval/avg_lateral_drift_m": 0.01,
                "eval/avg_nonfoot_ground_contact_count": 0.0,
                "eval/avg_self_contact_count": 0.0,
            },
            500,
            target_distance_m=1.0,
            desired_speed_m_s=0.10,
        )

        self.assertGreater(stable["rank"], crashed["rank"])

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
