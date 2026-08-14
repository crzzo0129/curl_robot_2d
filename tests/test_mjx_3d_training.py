import io
import math
from contextlib import redirect_stderr
from pathlib import Path
import unittest

import numpy as np

from curl_robot_2d_mjx.environment_3d import DEFAULT_3D_CEM_CONTROLLER
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
from scripts import (
    evaluate_mjx_3d_policy,
    render_mjx_3d_policy,
    train_mjx_3d_real_geometry_nominal,
    train_mjx_3d_residual_ppo,
)


class MJX3DTrainingEntrypointTest(unittest.TestCase):
    def test_cg20_physics_profile_is_selectable(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--physics-profile", "cg20"]
        )

        self.assertEqual(args.physics_profile, "cg20")

    def test_training_entry_imports_without_jax(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args([])

        self.assertTrue(callable(train_mjx_3d_residual_ppo.main))
        self.assertEqual(args.preset, "smoke")
        self.assertEqual(args.geometry, "pupper_open60")
        self.assertEqual(args.recipe, "anchored_v1")
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.curriculum, "none")
        self.assertIsNone(args.curriculum_stage)
        self.assertEqual(args.controller, DEFAULT_3D_CEM_CONTROLLER)
        self.assertEqual(args.reference_weight, 1.0)
        self.assertEqual(args.minimum_residual_gain, 0.05)
        self.assertEqual(args.reference_action_scale, 1.0)
        self.assertIsNone(args.reference_ramp_start_scale)
        self.assertEqual(args.reference_ramp_duration_s, 0.25)
        self.assertEqual(args.reference_startup_boost, 0.0)
        self.assertEqual(args.reference_startup_boost_duration_s, 0.25)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.entropy_cost, 1e-2)
        self.assertEqual(args.selection_target_turns, 1.0)
        self.assertFalse(args.zero_residual_policy_init)
        self.assertEqual(args.initial_policy_std, 1.0)
        self.assertIsNone(args.residual_pair_differential_scale)
        self.assertIsNone(args.explicit_phase_observation)
        self.assertTrue(args.deterministic_eval)
        self.assertFalse(args.save_ppo_checkpoints)
        self.assertIsNone(args.ppo_checkpoint_dir)
        self.assertIsNone(args.restore_params)

    def test_real_geometry_contact_recipe_is_nominal_and_contact_gated(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            [
                "--geometry",
                "real",
                "--recipe",
                "real_geometry_contact_v1",
            ]
        )

        self.assertEqual(args.geometry, "real")
        self.assertEqual(args.curriculum, "none")
        self.assertEqual(args.selection_objective, "contact")
        self.assertEqual(args.selection_target_turns, 5.0)
        self.assertEqual(args.minimum_residual_gain, 0.20)
        self.assertTrue(args.zero_residual_policy_init)

    def test_real_geometry_nominal_wrapper_pins_geometry_and_reference(self) -> None:
        args = train_mjx_3d_real_geometry_nominal.parse_args(
            ["--preset", "h200", "--seed", "3"]
        )
        values = train_mjx_3d_real_geometry_nominal.training_argv(args)

        self.assertIn("real", values)
        self.assertIn("real_geometry_contact_v2", values)
        self.assertIn("12", values)
        self.assertIn(
            "results\\mjx_3d_real_geometry_contact_v2_h200_seed3",
            values,
        )

    def test_real_geometry_v2_starts_with_common_only_small_residual(self) -> None:
        wrapper_args = train_mjx_3d_real_geometry_nominal.parse_args([])
        args = train_mjx_3d_residual_ppo.parse_args(
            train_mjx_3d_real_geometry_nominal.training_argv(wrapper_args)
        )

        self.assertEqual(args.recipe, "real_geometry_contact_v2")
        self.assertEqual(args.minimum_residual_gain, 0.08)
        self.assertEqual(args.residual_pair_differential_scale, 0.0)
        self.assertEqual(args.initial_policy_std, 0.05)
        self.assertEqual(args.learning_rate, 3e-5)
        self.assertEqual(args.selection_objective, "contact")

    def test_reset_curriculum_allocates_every_stage_training_work(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--curriculum", "reset_v1"]
        )
        values = train_mjx_3d_residual_ppo.PRESETS["smoke"].copy()

        plan = train_mjx_3d_residual_ppo._curriculum_training_plan(
            args, values
        )

        self.assertEqual(
            [item["stage"].name for item in plan],
            [
                "symmetric_reset",
                "differential_005",
                "differential_010",
                "differential_025",
            ],
        )
        self.assertTrue(
            all(item["schedule"]["effective_steps"] > 0 for item in plan)
        )
        self.assertGreaterEqual(sum(item["num_evals"] for item in plan), 4)

    def test_reset_v2_allocates_every_tilt_stage_training_work(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--curriculum", "reset_v2"]
        )
        values = train_mjx_3d_residual_ppo.PRESETS["smoke"].copy()

        plan = train_mjx_3d_residual_ppo._curriculum_training_plan(
            args, values
        )

        self.assertEqual(
            [item["stage"].name for item in plan],
            [
                "tilt_v2_0000",
                "tilt_v2_0100",
                "tilt_v2_0150",
                "tilt_v2_0175",
                "tilt_v2_0200",
                "tilt_v2_0300",
            ],
        )
        self.assertTrue(
            all(item["schedule"]["effective_steps"] > 0 for item in plan)
        )
        self.assertGreaterEqual(sum(item["num_evals"] for item in plan), 6)

    def test_friction_v1_allocates_three_progressive_stages(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--curriculum", "friction_v1"]
        )
        values = train_mjx_3d_residual_ppo.PRESETS["smoke"].copy()

        plan = train_mjx_3d_residual_ppo._curriculum_training_plan(
            args, values
        )

        self.assertEqual(
            [item["stage"].name for item in plan],
            ["friction_02", "friction_05", "friction_10"],
        )
        self.assertTrue(
            all(item["schedule"]["effective_steps"] > 0 for item in plan)
        )
        self.assertGreaterEqual(sum(item["num_evals"] for item in plan), 3)

    def test_mass_v1_allocates_two_progressive_stages(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--curriculum", "mass_v1"]
        )
        values = train_mjx_3d_residual_ppo.PRESETS["smoke"].copy()

        plan = train_mjx_3d_residual_ppo._curriculum_training_plan(
            args, values
        )

        self.assertEqual(
            [item["stage"].name for item in plan], ["mass_02", "mass_05"]
        )
        self.assertTrue(
            all(item["schedule"]["effective_steps"] > 0 for item in plan)
        )
        self.assertGreaterEqual(sum(item["num_evals"] for item in plan), 2)

    def test_curriculum_stage_must_belong_to_selected_curriculum(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mjx_3d_residual_ppo.parse_args(
                [
                    "--curriculum",
                    "reset_v1",
                    "--curriculum-stage",
                    "friction",
                ]
            )

    def test_restore_params_and_orbax_checkpoint_are_mutually_exclusive(
        self,
    ) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mjx_3d_residual_ppo.parse_args(
                [
                    "--restore-params",
                    "params_best",
                    "--restore-checkpoint",
                    "ppo_checkpoint",
                ]
            )

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

    def test_phase_locked_safe_v4_claws_back_failed_progress(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--recipe", "phase_locked_safe_v4"]
        )

        reward = train_mjx_3d_residual_ppo._reward_config_from_args(args)

        self.assertTrue(args.zero_residual_policy_init)
        self.assertEqual(args.minimum_residual_gain, 0.15)
        self.assertEqual(reward.roll_progress, 8.0)
        self.assertEqual(reward.lateral_velocity, 4.0)
        self.assertEqual(reward.lateral_drift, 6.0)
        self.assertEqual(reward.failure_progress_clawback, 8.0)
        self.assertEqual(reward.termination, 40.0)
        self.assertEqual(reward.severe_extra_termination, 40.0)

    def test_phase_locked_safe_v5_uses_partial_progress_clawback(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--recipe", "phase_locked_safe_v5"]
        )

        reward = train_mjx_3d_residual_ppo._reward_config_from_args(args)

        self.assertTrue(args.zero_residual_policy_init)
        self.assertEqual(reward.roll_progress, 8.0)
        self.assertEqual(reward.failure_progress_clawback, 2.0)
        self.assertEqual(reward.lateral_drift, 6.0)
        self.assertEqual(reward.termination, 40.0)
        self.assertEqual(reward.severe_extra_termination, 40.0)

    def test_phase_locked_coupled_v6_limits_differential_exploration(
        self,
    ) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            ["--recipe", "phase_locked_coupled_v6"]
        )

        reward = train_mjx_3d_residual_ppo._reward_config_from_args(args)

        self.assertEqual(args.residual_pair_differential_scale, 0.25)
        self.assertTrue(args.explicit_phase_observation)
        self.assertEqual(args.minimum_residual_gain, 0.15)
        self.assertEqual(args.initial_policy_std, 0.20)
        self.assertEqual(reward.failure_progress_clawback, 2.0)
        self.assertEqual(reward.lateral_drift, 6.0)

    def test_reference_startup_boost_args_are_exposed(self) -> None:
        args = train_mjx_3d_residual_ppo.parse_args(
            [
                "--reference-action-scale",
                "1.05",
                "--reference-ramp-start-scale",
                "0.25",
                "--reference-ramp-duration-s",
                "0.1",
                "--reference-startup-boost",
                "0.2",
                "--reference-startup-boost-duration-s",
                "0.4",
            ]
        )

        self.assertEqual(args.reference_action_scale, 1.05)
        self.assertEqual(args.reference_ramp_start_scale, 0.25)
        self.assertEqual(args.reference_ramp_duration_s, 0.1)
        self.assertEqual(args.reference_startup_boost, 0.2)
        self.assertEqual(args.reference_startup_boost_duration_s, 0.4)

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

    def test_evaluator_exposes_curriculum_reset_distribution(self) -> None:
        args = evaluate_mjx_3d_policy.parse_args(
            [
                "params_best",
                "--out",
                "eval_curriculum_reset",
                "--reset-joint-noise-rad",
                "0.015",
                "--reset-velocity-noise",
                "0.03",
                "--reset-pair-differential-scale",
                "0.25",
                "--reset-axis-tilt-noise-rad",
                "0.03",
                "--geom-friction-scale",
                "0.90",
                "--body-mass-scale",
                "0.95",
                "--body-mass-left-scale",
                "1.05",
                "--body-mass-right-scale",
                "0.95",
            ]
        )

        self.assertEqual(args.reset_joint_noise_rad, 0.015)
        self.assertEqual(args.reset_velocity_noise, 0.03)
        self.assertEqual(args.reset_pair_differential_scale, 0.25)
        self.assertEqual(args.reset_axis_tilt_noise_rad, 0.03)
        self.assertEqual(args.geom_friction_scale, 0.90)
        self.assertEqual(args.body_mass_scale, 0.95)
        self.assertEqual(args.body_mass_left_scale, 1.05)
        self.assertEqual(args.body_mass_right_scale, 0.95)

    def test_evaluator_rejects_nonpositive_friction_scale(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_mjx_3d_policy.parse_args(
                [
                    "params_best",
                    "--out",
                    "eval_bad_friction",
                    "--geom-friction-scale",
                    "0",
                ]
            )

    def test_evaluator_rejects_nonpositive_body_mass_scale(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_mjx_3d_policy.parse_args(
                [
                    "params_best",
                    "--out",
                    "eval_bad_mass",
                    "--body-mass-left-scale",
                    "0",
                ]
            )

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

    def test_contact_selection_requires_five_turns_and_prefers_less_contact(
        self,
    ) -> None:
        base = {
            "eval/avg_episode_length": 500.0,
            "eval/episode_failed": 0.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_roll_progress_rad": 5.1 * 2.0 * math.pi,
            "eval/avg_lateral_drift_m": 0.01,
            "eval/avg_axis_tilt_rad": 0.05,
            "eval/avg_forbidden_penetration_m": 0.0001,
            "eval/avg_forbidden_contact_count": 0.01,
            "eval/avg_first_turn_forbidden_contact_count": 0.005,
        }
        cleaner = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            base,
            episode_length=500,
            target_turns=5.0,
            objective="contact",
        )
        colliding = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            {
                **base,
                "eval/avg_forbidden_contact_count": 0.03,
                "eval/avg_first_turn_forbidden_contact_count": 0.015,
            },
            episode_length=500,
            target_turns=5.0,
            objective="contact",
        )
        too_slow = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            {**base, "eval/episode_roll_progress_rad": 4.9 * 2.0 * math.pi},
            episode_length=500,
            target_turns=5.0,
            objective="contact",
        )

        self.assertGreater(cleaner["score"], colliding["score"])
        self.assertTrue(too_slow["rejected"])

    def test_checkpoint_selection_rejects_excess_failure_rate(self) -> None:
        metrics = {
            "eval/avg_episode_length": 490.0,
            "eval/episode_failed": 0.125,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_roll_progress_rad": 12.0 * math.pi,
            "eval/avg_lateral_drift_m": 0.01,
            "eval/avg_axis_tilt_rad": 0.05,
            "eval/avg_forbidden_penetration_m": 0.0,
            "eval/avg_forbidden_contact_count": 0.0,
        }
        rejected = train_mjx_3d_residual_ppo._checkpoint_selection_3d(
            metrics,
            episode_length=500,
        )
        curriculum_candidate = (
            train_mjx_3d_residual_ppo._checkpoint_selection_3d(
                metrics,
                episode_length=500,
                maximum_failure_rate=1.0,
            )
        )

        self.assertTrue(rejected["rejected"])
        self.assertFalse(curriculum_candidate["rejected"])
        self.assertFalse(
            curriculum_candidate["passes_acceptance_failure_rate"]
        )

    def test_final_eval_selection_uses_exact_final_params(self) -> None:
        stale_callback_params = object()
        final_params = object()

        resolved, source = train_mjx_3d_residual_ppo._resolve_best_params(
            {
                "step": 76_800,
                "params": stale_callback_params,
                "params_step": 51_200,
            },
            final_params,
            [{"step": 51_200}, {"step": 76_800}],
        )

        self.assertIs(resolved, final_params)
        self.assertEqual(source, "final_eval")

    def test_earlier_best_uses_matching_callback_snapshot(self) -> None:
        callback_params = object()

        resolved, source = train_mjx_3d_residual_ppo._resolve_best_params(
            {
                "step": 51_200,
                "params": callback_params,
                "params_step": 51_200,
            },
            object(),
            [{"step": 51_200}, {"step": 76_800}],
        )

        self.assertIs(resolved, callback_params)
        self.assertEqual(source, "callback_step_51200")

    def test_final_param_sources_reuse_the_post_training_rollout(self) -> None:
        self.assertTrue(
            train_mjx_3d_residual_ppo._best_and_final_share_checkpoint(
                "final_eval"
            )
        )
        self.assertTrue(
            train_mjx_3d_residual_ppo._best_and_final_share_checkpoint(
                "final_fallback"
            )
        )
        self.assertFalse(
            train_mjx_3d_residual_ppo._best_and_final_share_checkpoint(
                "callback_step_51200"
            )
        )

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

    def test_large_batch_eval_defaults_match_validated_cg20_ramp(self) -> None:
        args = evaluate_mjx_3d_policy.parse_args(
            ["params_best", "--out", "eval_best_batch"]
        )

        self.assertEqual(args.evaluation_mode, "policy")
        self.assertEqual(args.physics_profile, "cg20")
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.chunk_size, 512)
        self.assertEqual(args.progress_every, 100)
        self.assertEqual(args.episode_length, 500)
        self.assertEqual(args.minimum_residual_gain, 0.15)
        self.assertEqual(args.reset_joint_noise_rad, 0.005)
        self.assertEqual(args.reset_velocity_noise, 0.005)
        self.assertEqual(args.reference_ramp_start_scale, 0.50)
        self.assertEqual(args.reference_ramp_duration_s, 0.10)
        self.assertEqual(args.reference_startup_boost, 0.0)
        self.assertTrue(args.zero_residual_policy_init)
        self.assertEqual(args.initial_policy_std, 0.20)
        self.assertTrue(args.explicit_phase_observation)
        self.assertFalse(args.resume)
        self.assertEqual(args.diagnostic_rollouts, 0)
        self.assertTrue(args.diagnostic_reference)

    def test_reference_only_batch_eval_does_not_require_checkpoint(self) -> None:
        args = evaluate_mjx_3d_policy.parse_args(
            [
                "--evaluation-mode",
                "reference",
                "--out",
                "eval_reference_batch",
            ]
        )

        self.assertEqual(args.evaluation_mode, "reference")
        self.assertIsNone(args.checkpoint)

    def test_policy_batch_eval_still_requires_checkpoint(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_mjx_3d_policy.parse_args(
                ["--out", "eval_policy_batch"]
            )

    def test_reference_batch_eval_rejects_ignored_checkpoint(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_mjx_3d_policy.parse_args(
                [
                    "params_best",
                    "--evaluation-mode",
                    "reference",
                    "--out",
                    "eval_reference_batch",
                ]
            )

    def test_diagnostic_selection_keeps_failures_and_boundary_successes(self) -> None:
        count = 8
        arrays = {
            "steps": np.asarray([500, 320, 490, 500, 440, 500, 500, 470]),
            "failed": np.asarray([0, 1, 1, 0, 1, 0, 0, 1]),
            "seed_index": np.arange(count),
            "final_lateral_drift_m": np.asarray(
                [0.01, 0.21, -0.22, 0.03, 0.20, -0.18, 0.02, -0.21]
            ),
            "max_abs_lateral_drift_m": np.asarray(
                [0.02, 0.21, 0.22, 0.04, 0.20, 0.18, 0.03, 0.21]
            ),
            "conservative_turns": np.asarray(
                [8.7, 5.0, 8.4, 8.71, 7.2, 8.68, 8.72, 8.0]
            ),
        }

        selected = evaluate_mjx_3d_policy._select_diagnostic_rollouts(
            arrays, 6
        )

        self.assertEqual(len(selected), 6)
        selected_indices = {item["array_index"] for item in selected}
        self.assertIn(1, selected_indices)
        self.assertIn(2, selected_indices)
        self.assertTrue(selected_indices.intersection({0, 3, 5, 6}))

    def test_3d_policy_renderer_defaults_to_cg20_freejoint_model(self) -> None:
        args = render_mjx_3d_policy.parse_args(["evaluation_rollout.npz"])

        self.assertEqual(args.physics_profile, "cg20")
        self.assertEqual(args.control_dt, 0.02)
        self.assertEqual(args.fps, 20.0)
        self.assertEqual(args.width, 720)
        self.assertEqual(args.height, 540)

    def test_real_geometry_evaluator_and_renderer_are_selectable(self) -> None:
        eval_args = evaluate_mjx_3d_policy.parse_args(
            [
                "--out",
                "eval",
                "--evaluation-mode",
                "reference",
                "--geometry",
                "real",
            ]
        )
        render_args = render_mjx_3d_policy.parse_args(
            ["evaluation_rollout.npz", "--geometry", "real"]
        )

        self.assertEqual(eval_args.geometry, "real")
        self.assertEqual(render_args.geometry, "real")


if __name__ == "__main__":
    unittest.main()
