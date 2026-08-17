import io
from contextlib import redirect_stderr
from pathlib import Path
import unittest

from curl_robot_2d_mjx.config_walking_3d import Walking3DConfig
from curl_robot_2d_mjx.reward_walking_3d import Walking3DRewardConfig
from scripts import evaluate_mjx_3d_walking_policy
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
        self.assertEqual(args.eval_forward_speeds, (-0.10, 0.20, 0.35))
        self.assertEqual(args.eval_gait_phases, (0.0,))

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
        self.assertEqual(reward.stagnation, 0.2)
        self.assertEqual(reward.stagnation_window_s, 1.0)
        self.assertEqual(reward.stagnation_min_progress_m, 0.05)
        self.assertEqual(reward.upright_stagnation_gate, 1.0)
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

    def test_phase_bootstrap_recipe_enables_stage_a_scaffolding(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(
            ["--recipe", "forward_phase_bootstrap_v1"]
        )
        reward = train_mjx_3d_walking_ppo._reward_config_from_args(args)

        self.assertTrue(args.gait_phase_enabled)
        self.assertEqual(args.gait_cycle_time, 0.625)
        self.assertEqual(args.gait_duty_factor, 0.68)
        self.assertEqual(args.reset_joint_noise, 0.01)
        self.assertEqual(args.reset_velocity_noise, 0.02)
        self.assertEqual(args.reset_root_xy_velocity_noise, 0.03)
        self.assertEqual(args.updates_per_batch, 4)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.desired_kl, 0.01)
        self.assertEqual(args.init_noise_std, 0.30)
        self.assertEqual(reward.velocity_tracking_upright_gate, 0.0)
        self.assertEqual(reward.gait_contact, 1.0)
        self.assertEqual(reward.swing_clearance_m, 0.025)
        self.assertEqual(reward.swing_clearance_target_tracking, 1.0)

    def test_stochastic_eval_is_an_explicit_opt_in(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(["--stochastic-eval"])

        self.assertFalse(args.deterministic_eval)

    def test_training_and_eval_low_progress_termination_are_independent(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(
            [
                "--recipe",
                "unitree_mjlab_velocity_v1",
                "--terminate-low-progress",
                "--no-eval-terminate-low-progress",
            ]
        )

        self.assertTrue(args.terminate_low_progress_enabled)
        self.assertFalse(args.eval_terminate_low_progress_enabled)

    def test_walking_ppo_uses_stable_distribution_contract(self) -> None:
        source = Path(train_mjx_3d_walking_ppo.__file__).read_text(
            encoding="utf-8"
        )

        for token in (
            'distribution_type="normal"',
            'noise_std_type="log"',
            'state_dependent_std=False',
            "mean_kernel_init_fn=jnn.initializers.uniform",
            '"scale": WALKING_ACTOR_MEAN_INIT_SCALE',
            "mean_clip_scale=WALKING_ACTOR_MEAN_CLIP_SCALE",
            'value_obs_key="privileged_state"',
            "deterministic_eval=args.deterministic_eval",
            "max_grad_norm=args.max_grad_norm",
            "learning_rate_schedule=args.learning_rate_schedule",
            "learning_rate_schedule_min_lr=args.adaptive_kl_min_lr",
            "learning_rate_schedule_max_lr=args.adaptive_kl_max_lr",
            "normalize_observations=args.normalize_observations",
            "bootstrap_on_timeout=True",
            "reset_root_xy_velocity_noise_m_s=0.0",
            "args.eval_terminate_low_progress_enabled",
            'f"avg_ppo/step="',
            'f"reward_scale={args.reward_scaling:g}',
        ):
            self.assertIn(token, source)

    def test_unitree_route_b_is_asymmetric_and_pose_reference_free(self) -> None:
        args = train_mjx_3d_walking_ppo.parse_args(
            ["--recipe", "unitree_mjlab_velocity_v1"]
        )
        reward = train_mjx_3d_walking_ppo._reward_config_from_args(args)

        self.assertTrue(args.gait_phase_enabled)
        self.assertTrue(args.asymmetric_observations)
        self.assertTrue(args.normalize_observations)
        self.assertFalse(args.small_actor_mean_init)
        self.assertEqual(args.gait_cycle_time, 0.60)
        self.assertEqual(args.gait_duty_factor, 0.56)
        self.assertEqual(args.desired_speed_m_s, 0.20)
        self.assertEqual(args.command_forward_min, 0.10)
        self.assertEqual(args.command_forward_max, 0.30)
        self.assertEqual(args.command_lateral_max, 0.0)
        self.assertEqual(args.command_yaw_rate_max, 0.0)
        self.assertEqual(args.command_stop_probability, 0.0)
        self.assertEqual(args.hidden_layers, [512, 256, 128])
        self.assertEqual(args.critic_hidden_layers, [512, 256, 128])
        self.assertEqual(args.action_scale_abduction, 0.08)
        self.assertEqual(args.action_scale_hip, 0.25)
        self.assertEqual(args.action_scale_knee, 0.25)
        self.assertEqual(args.terminate_upright_tilt, 1.22)
        self.assertEqual(args.terminate_nonfoot_force_min, 1.0)
        self.assertFalse(args.terminate_low_progress_enabled)
        self.assertFalse(args.eval_terminate_low_progress_enabled)
        self.assertEqual(args.terminate_low_progress_window, 0.50)
        self.assertEqual(args.terminate_low_progress_duration, 2.0)
        self.assertEqual(args.terminate_low_progress_command_ratio, 0.50)
        self.assertEqual(args.terminate_low_progress_cap, 0.05)
        self.assertEqual(args.unroll_length, 24)
        self.assertEqual(args.updates_per_batch, 5)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.adaptive_kl_max_lr, 3e-4)
        self.assertEqual(args.reward_scaling, 0.02)
        self.assertEqual(args.init_noise_std, 0.5)
        self.assertEqual(reward.velocity_tracking_sigma_m_s, 0.10)
        self.assertEqual(reward.yaw_rate_tracking, 0.75)
        self.assertEqual(reward.yaw_rate_tracking_progress_gate, 1.0)
        self.assertEqual(reward.gait_contact, 0.5)
        self.assertEqual(reward.orientation, 1.0)
        self.assertEqual(reward.foot_clearance, 1.0)
        self.assertEqual(reward.joint_acceleration, 2.5e-7)
        self.assertEqual(reward.termination, 200.0)
        self.assertEqual(reward.upright, 0.0)
        self.assertEqual(reward.swing_clearance, 0.0)

    def test_unitree_discovery_is_nominal_and_robust_compatible(self) -> None:
        discovery = train_mjx_3d_walking_ppo.parse_args(
            ["--recipe", "unitree_mjlab_velocity_discovery_v1"]
        )
        robust = train_mjx_3d_walking_ppo.parse_args(
            ["--recipe", "unitree_mjlab_velocity_v1"]
        )
        reward = train_mjx_3d_walking_ppo._reward_config_from_args(
            discovery
        )

        self.assertEqual(discovery.desired_speed_m_s, 0.10)
        self.assertEqual(discovery.command_forward_min, 0.10)
        self.assertEqual(discovery.command_forward_max, 0.10)
        self.assertEqual(discovery.command_lateral_max, 0.0)
        self.assertEqual(discovery.command_yaw_rate_max, 0.0)
        self.assertEqual(discovery.eval_forward_speeds, (0.10,))
        self.assertEqual(discovery.eval_gait_phases, (0.0, 0.5))
        self.assertTrue(discovery.no_observation_noise)
        self.assertTrue(discovery.no_domain_randomization)
        self.assertEqual(discovery.reset_joint_noise, 0.0)
        self.assertEqual(discovery.reset_velocity_noise, 0.0)
        self.assertEqual(discovery.reset_root_xy_velocity_noise, 0.0)
        self.assertEqual(discovery.reset_root_yaw_rate_noise, 0.0)
        self.assertFalse(discovery.terminate_low_progress_enabled)
        self.assertFalse(discovery.eval_terminate_low_progress_enabled)
        self.assertEqual(discovery.reward_scaling, 0.02)
        self.assertEqual(reward.velocity_tracking_sigma_m_s, 0.10)
        self.assertEqual(reward.yaw_rate_tracking, 0.25)
        self.assertEqual(reward.yaw_rate_tracking_progress_gate, 1.0)
        self.assertEqual(reward.gait_contact, 0.5)
        self.assertEqual(reward.foot_clearance, 0.0)
        self.assertEqual(reward.termination, 200.0)

        # The observation and policy contracts must remain identical so a
        # full PPO checkpoint can continue into the robust recipe.
        for name in (
            "gait_phase_enabled",
            "gait_cycle_time",
            "gait_duty_factor",
            "asymmetric_observations",
            "normalize_observations",
            "small_actor_mean_init",
            "hidden_layers",
            "critic_hidden_layers",
            "action_scale_abduction",
            "action_scale_hip",
            "action_scale_knee",
        ):
            self.assertEqual(getattr(discovery, name), getattr(robust, name))

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

    def test_contact_force_peak_is_not_divided_by_episode_length(self) -> None:
        metrics = {
            "eval/avg_episode_length": 100.0,
            "eval/episode_nonfoot_ground_max_force_n": 50.0,
            "eval/episode_nonfoot_ground_peak_force_n": 80.0,
            "eval/episode_heading_change_rad": -0.30,
            "eval/episode_lateral_progress_m": 0.08,
            "eval/episode_foot_contact_fl": 60.0,
        }

        train_mjx_3d_walking_ppo._add_per_step_walking_metrics_3d(metrics)

        self.assertEqual(metrics["eval/avg_nonfoot_ground_max_force_n"], 0.5)
        self.assertEqual(
            metrics["eval/episode_nonfoot_ground_peak_force_n"], 80.0
        )
        self.assertNotIn(
            "eval/avg_nonfoot_ground_peak_force_n", metrics
        )
        self.assertNotIn("eval/avg_heading_change_rad", metrics)
        self.assertNotIn("eval/avg_lateral_progress_m", metrics)
        self.assertEqual(metrics["eval/avg_foot_contact_fl"], 0.60)

    def test_checkpoint_selection_prefers_straighter_equal_tracking(self) -> None:
        base = {
            "eval/avg_episode_length": 500.0,
            "eval/episode_failed": 0.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_failure_upright_tilt": 0.0,
            "eval/episode_forward_progress_m": 1.0,
            "eval/avg_forward_velocity_m_s": 0.10,
            "eval/avg_planar_velocity_error_m_s": 0.02,
            "eval/avg_yaw_rate_error_rad_s": 0.10,
            "eval/avg_upright_tilt_rad": 0.02,
            "eval/avg_lateral_drift_m": 0.02,
            "eval/avg_nonfoot_ground_contact_count": 0.0,
            "eval/avg_self_contact_count": 0.0,
        }
        straight = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {
                **base,
                "eval/episode_heading_change_rad": 0.05,
                "eval/episode_lateral_progress_m": 0.01,
            },
            500,
            target_distance_m=1.0,
            desired_speed_m_s=0.10,
        )
        curved = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {
                **base,
                "eval/episode_heading_change_rad": -0.60,
                "eval/episode_lateral_progress_m": 0.12,
            },
            500,
            target_distance_m=1.0,
            desired_speed_m_s=0.10,
        )

        self.assertGreater(straight["direction_quality"], curved["direction_quality"])
        self.assertGreater(straight["rank"], curved["rank"])

    def test_evaluation_grid_summary_exposes_worst_case_and_phase_span(self) -> None:
        def case(speed, phase, achieved, heading, lateral, failed=False):
            return {
                "command_forward_velocity_m_s": speed,
                "initial_gait_phase": phase,
                "forward_velocity_error_m_s": achieved - speed,
                "average_forward_velocity_m_s": achieved,
                "unwrapped_heading_change_rad": heading,
                "final_lateral_drift_m": lateral,
                "average_signed_yaw_rate_rad_s": heading / 10.0,
                "average_abs_yaw_rate_error_rad_s": abs(heading) / 10.0,
                "failed": failed,
                "timed_out": not failed,
            }

        grid = train_mjx_3d_walking_ppo._summarize_evaluation_grid_walking_3d(
            (
                case(0.08, 0.0, 0.07, -0.10, 0.01),
                case(0.08, 0.5, 0.09, 0.20, -0.02),
                case(0.15, 0.0, 0.12, -0.40, 0.05),
                case(0.15, 0.5, 0.14, -0.20, 0.03),
            )
        )

        self.assertEqual(grid["case_count"], 4)
        self.assertEqual(grid["failed_case_count"], 0)
        self.assertAlmostEqual(grid["maximum_absolute_velocity_error_m_s"], 0.03)
        self.assertAlmostEqual(grid["maximum_absolute_heading_change_rad"], 0.40)
        self.assertEqual(len(grid["phase_sensitivity"]), 2)
        self.assertIn(
            "initial_phase_sensitive",
            grid["diagnosis"]["observed_pattern_flags"],
        )

    def test_grid_diagnosis_separates_systematic_bias_from_leg_asymmetry(self) -> None:
        def case(phase, heading, contact_delta, action_delta):
            left_contact = 0.55 + 0.5 * contact_delta
            right_contact = 0.55 - 0.5 * contact_delta
            return {
                "command_forward_velocity_m_s": 0.10,
                "initial_gait_phase": phase,
                "forward_velocity_error_m_s": 0.0,
                "average_forward_velocity_m_s": 0.10,
                "unwrapped_heading_change_rad": heading,
                "final_lateral_drift_m": 0.02,
                "average_signed_yaw_rate_rad_s": heading / 10.0,
                "average_abs_yaw_rate_error_rad_s": abs(heading) / 10.0,
                "failed": False,
                "timed_out": True,
                "feet": {
                    "fl": {"contact_fraction": left_contact},
                    "fr": {"contact_fraction": right_contact},
                    "rl": {"contact_fraction": left_contact},
                    "rr": {"contact_fraction": right_contact},
                },
                "control": {
                    "average_action_rms_left_right_delta": action_delta,
                },
            }

        grid = train_mjx_3d_walking_ppo._summarize_evaluation_grid_walking_3d(
            (
                case(0.0, -0.40, 0.12, -0.11),
                case(0.5, -0.35, 0.11, -0.12),
            )
        )

        flags = grid["diagnosis"]["observed_pattern_flags"]
        self.assertIn("systematic_direction_bias", flags)
        self.assertIn("left_right_contact_imbalance", flags)
        self.assertIn("left_right_action_imbalance", flags)
        self.assertNotIn("initial_phase_sensitive", flags)

    def test_grid_direction_diagnosis_ignores_stagnant_speed_cases(self) -> None:
        def case(speed, phase, achieved, heading):
            return {
                "command_forward_velocity_m_s": speed,
                "initial_gait_phase": phase,
                "forward_velocity_error_m_s": achieved - speed,
                "average_forward_velocity_m_s": achieved,
                "unwrapped_heading_change_rad": heading,
                "final_lateral_drift_m": 0.0,
                "average_signed_yaw_rate_rad_s": heading / 10.0,
                "average_abs_yaw_rate_error_rad_s": abs(heading) / 10.0,
                "failed": False,
                "timed_out": True,
            }

        grid = train_mjx_3d_walking_ppo._summarize_evaluation_grid_walking_3d(
            (
                case(0.08, 0.0, 0.015, 0.52),
                case(0.08, 0.5, 0.011, 0.45),
                case(0.10, 0.0, 0.104, -0.63),
                case(0.10, 0.5, 0.105, -0.59),
                case(0.15, 0.0, 0.014, 0.18),
                case(0.15, 0.5, 0.008, -0.02),
            )
        )

        diagnosis = grid["diagnosis"]
        self.assertEqual(diagnosis["locomoting_case_count"], 2)
        self.assertEqual(
            diagnosis["heading_change_bias"]["direction"], "negative"
        )
        self.assertEqual(
            diagnosis["heading_change_bias"]["consistency"], 1.0
        )
        self.assertIn(
            "systematic_direction_bias",
            diagnosis["observed_pattern_flags"],
        )
        self.assertNotIn(
            "initial_phase_sensitive",
            diagnosis["observed_pattern_flags"],
        )

    def test_standalone_walking_evaluator_is_import_safe(self) -> None:
        args = evaluate_mjx_3d_walking_policy.parse_args(
            [
                "params_best",
                "--speeds",
                "0.08",
                "0.10",
                "0.15",
                "--gait-phases",
                "0",
                "0.5",
            ]
        )

        self.assertEqual(args.speeds, [0.08, 0.10, 0.15])
        self.assertEqual(args.gait_phases, [0.0, 0.5])

    def test_standalone_old_config_grid_uses_training_command_range(self) -> None:
        evaluation_task = Walking3DConfig(
            desired_speed_m_s=0.10,
            command_forward_velocity_range_m_s=(0.10, 0.10),
            gait_phase_enabled=True,
        )
        config = {
            "task": {
                "desired_speed_m_s": 0.10,
                "command_forward_velocity_range_m_s": [0.08, 0.15],
                "gait_phase_enabled": True,
            }
        }

        speeds, phases = evaluate_mjx_3d_walking_policy._default_grid(
            config, evaluation_task
        )

        self.assertEqual(speeds, (0.08, 0.10, 0.15))
        self.assertEqual(phases, (0.0, 0.5))

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

    def test_checkpoint_rank_does_not_select_full_episode_standing(self) -> None:
        common = {
            "eval/episode_failure_upright_tilt": 0.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/avg_yaw_rate_error_rad_s": 0.0,
            "eval/avg_upright_tilt_rad": 0.0,
            "eval/avg_lateral_drift_m": 0.0,
            "eval/avg_nonfoot_ground_contact_count": 0.0,
            "eval/avg_self_contact_count": 0.0,
        }
        standing = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {
                **common,
                "eval/avg_episode_length": 500.0,
                "eval/episode_failed": 0.0,
                "eval/episode_forward_progress_m": 0.0,
                "eval/avg_forward_velocity_m_s": 0.0,
                "eval/avg_planar_velocity_error_m_s": 0.20,
            },
            500,
            target_distance_m=2.0,
            desired_speed_m_s=0.20,
        )
        walking = train_mjx_3d_walking_ppo._checkpoint_selection_walking_3d(
            {
                **common,
                "eval/avg_episode_length": 499.0,
                "eval/episode_failed": 1.0,
                "eval/episode_forward_progress_m": 1.996,
                "eval/avg_forward_velocity_m_s": 0.20,
                "eval/avg_planar_velocity_error_m_s": 0.0,
            },
            500,
            target_distance_m=2.0,
            desired_speed_m_s=0.20,
        )

        self.assertEqual(standing["completed"], 0.0)
        self.assertEqual(standing["meaningful_progress"], 0.0)
        self.assertGreater(walking["rank"], standing["rank"])
        self.assertFalse(
            train_mjx_3d_walking_ppo._checkpoint_is_selectable_walking_3d(
                standing, None
            )
        )
        self.assertTrue(
            train_mjx_3d_walking_ppo._checkpoint_is_selectable_walking_3d(
                walking, None
            )
        )

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
