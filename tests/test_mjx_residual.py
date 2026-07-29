from pathlib import Path
import math
import unittest

import numpy as np

from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceConfig,
    DEFAULT_CEM_CONTROLLER,
    advance_oscillator,
    effective_residual_action,
    expected_budget_steps,
    load_cem_reference,
    reference_action,
)
from scripts.train_mjx_residual_ppo import (
    _distribution_summary,
    _eval_visualization_dir,
    _exact_stage_eval_schedule,
    _gate_assessment,
    _last_trained_stage_spec,
    _parse_args,
    _residual_checkpoint_rank,
    _safe_stage_checkpoint,
    _stage_plan,
    _target_eval_eligible,
)
from scripts.optimize_phase_controller import (
    controller_targets,
    knee_bias_for_foot_gap,
)


class MJXResidualTest(unittest.TestCase):
    def test_disturbance_cli_defaults_off_and_accepts_impulses(self) -> None:
        defaults = _parse_args(["--retain-cem"])
        disturbed = _parse_args(
            [
                "--retain-cem",
                "--disturbance-root-x-velocity",
                "0.2",
                "--disturbance-root-pitch-velocity",
                "0.8",
                "--disturbance-min-step",
                "100",
                "--disturbance-max-step",
                "400",
            ]
        )

        self.assertEqual(defaults.disturbance_root_x_velocity, 0.0)
        self.assertEqual(defaults.disturbance_root_pitch_velocity, 0.0)
        self.assertEqual(disturbed.disturbance_root_x_velocity, 0.2)
        self.assertEqual(disturbed.disturbance_root_pitch_velocity, 0.8)

    def test_target_gate_defaults_to_stage_gate_but_can_be_stricter(self) -> None:
        defaults = _parse_args(["--retain-cem", "--gate-min-turns", "7.0"])
        explicit = _parse_args(
            [
                "--retain-cem",
                "--gate-min-turns",
                "7.0",
                "--target-min-turns",
                "7.5",
                "--robust-eval-envs",
                "64",
            ]
        )

        self.assertEqual(defaults.target_min_turns, 7.0)
        self.assertEqual(defaults.robust_eval_envs, 128)
        self.assertEqual(explicit.target_min_turns, 7.5)
        self.assertEqual(explicit.robust_eval_envs, 64)

    def test_retained_cem_curriculums_only_residual_scale(self) -> None:
        args = _parse_args(["--retain-cem"])

        self.assertEqual(
            _stage_plan(args),
            [
                {
                    "reference_weight": 1.0,
                    "minimum_residual_gain": scale,
                }
                for scale in (0.05, 0.10, 0.20, 0.30)
            ],
        )

    def test_legacy_curriculum_still_withdraws_reference(self) -> None:
        args = _parse_args([])

        self.assertEqual(
            [stage["reference_weight"] for stage in _stage_plan(args)],
            [1.0, 0.5, 0.0],
        )

    def test_target_gate_ignores_untrained_step_zero(self) -> None:
        self.assertFalse(_target_eval_eligible(0, 1, 0, 131_072))
        self.assertTrue(_target_eval_eligible(0, 1, 131_072, 131_072))
        self.assertFalse(_target_eval_eligible(0, 2, 131_072, 131_072))

    def test_checkpoint_rank_prefers_more_turns_after_safety_gate(self) -> None:
        slower = {"turns": 7.6, "survival": 1.0}
        faster = {"turns": 8.2, "survival": 0.99}
        metrics = {
            "eval/episode_failed": 0.0,
            "eval/avg_forbidden_penetration_m": 0.0,
        }
        gate = {
            "checks": {
                "survival": True,
                "turns": False,
                "failure_rate": True,
                "finite": True,
            }
        }

        self.assertGreater(
            _residual_checkpoint_rank(faster, metrics),
            _residual_checkpoint_rank(slower, metrics),
        )
        self.assertTrue(_safe_stage_checkpoint(gate))
        gate["checks"]["failure_rate"] = False
        self.assertFalse(_safe_stage_checkpoint(gate))

    def test_failed_curriculum_evaluates_last_trained_scale(self) -> None:
        plan = [
            {"reference_weight": 1.0, "minimum_residual_gain": scale}
            for scale in (0.03, 0.05, 0.10)
        ]
        history = [{"stage_index": 0}, {"stage_index": 1}]

        self.assertEqual(
            _last_trained_stage_spec(plan, history)[
                "minimum_residual_gain"
            ],
            0.05,
        )

    def test_robust_distribution_reports_lower_tail(self) -> None:
        summary = _distribution_summary(np.arange(10.0))

        self.assertEqual(summary["min"], 0.0)
        self.assertAlmostEqual(summary["p10"], 0.9)
        self.assertEqual(summary["median"], 4.5)
        self.assertEqual(summary["max"], 9.0)

    def test_checked_in_cem_controller_loads(self) -> None:
        reference = load_cem_reference(DEFAULT_CEM_CONTROLLER)

        self.assertEqual(len(reference.coefficients), 8)
        self.assertAlmostEqual(
            reference.oscillator_rate_rad_s, 3.5193055141676064
        )
        self.assertAlmostEqual(
            reference.oscillator_coupling_per_s, 4.864379682159608
        )
        self.assertAlmostEqual(reference.with_weight(1.0).residual_gain, 0.05)
        self.assertAlmostEqual(reference.with_weight(0.5).residual_gain, 0.525)
        self.assertAlmostEqual(reference.with_weight(0.0).residual_gain, 1.0)

    def test_residual_authority_increases_as_reference_recedes(self) -> None:
        reference = load_cem_reference(
            DEFAULT_CEM_CONTROLLER, minimum_residual_gain=0.25
        )

        self.assertAlmostEqual(reference.with_weight(1.0).residual_gain, 0.25)
        self.assertAlmostEqual(reference.with_weight(0.5).residual_gain, 0.625)
        self.assertAlmostEqual(reference.with_weight(0.0).residual_gain, 1.0)

    def test_zero_reference_action_is_pure_policy(self) -> None:
        reference = load_cem_reference(
            DEFAULT_CEM_CONTROLLER, reference_weight=0.0
        )
        policy = np.asarray([0.2, -0.3, 0.4, -0.5])
        cem = np.asarray([-0.8, 0.7, -0.6, 0.5])

        effective = effective_residual_action(
            np, policy, cem, reference
        )

        np.testing.assert_allclose(effective, policy)

    def test_reference_oscillator_and_action_are_finite(self) -> None:
        reference = load_cem_reference(DEFAULT_CEM_CONTROLLER)
        oscillator_phase = advance_oscillator(
            np, 0.0, 0.0, 0.001, reference
        )
        action = reference_action(
            np,
            oscillator_phase,
            reference,
            compact_ctrl=np.asarray([0.3, 1.0, 0.3, 1.0]),
            action_scales=np.asarray([0.8, 1.2, 0.8, 1.2]),
            joint_low=np.asarray([-1.12, -0.61, -1.12, -0.61]),
            joint_high=np.asarray([2.41, 2.69, 2.41, 2.69]),
        )

        self.assertGreater(oscillator_phase, 0.0)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(np.all(np.abs(action) <= 1.0))

    def test_reference_target_matches_existing_cem_policy(self) -> None:
        reference = load_cem_reference(DEFAULT_CEM_CONTROLLER)
        compact = np.asarray(
            [0.3141592654, 1.05650322, 0.3141592654, 1.05650322]
        )
        scales = np.asarray([0.8, 1.2, 0.8, 1.2])
        oscillator_phase = 0.7
        expected = controller_targets(
            0.4,
            0.5,
            np.asarray(reference.coefficients),
            oscillator_rate=reference.oscillator_rate_rad_s,
            control_phase=oscillator_phase,
        )
        normalized = reference_action(
            np,
            oscillator_phase,
            reference,
            compact_ctrl=compact,
            action_scales=scales,
            joint_low=np.asarray([-1.12, -0.61, -1.12, -0.61]),
            joint_high=np.asarray([2.41, 2.69, 2.41, 2.69]),
        )

        np.testing.assert_allclose(compact + normalized * scales, expected)

    def test_reference_action_applies_saved_knee_bias(self) -> None:
        reference = CEMReferenceConfig(
            coefficients=(0.0,) * 8,
            oscillator_rate_rad_s=1.0,
            oscillator_coupling_per_s=0.0,
            knee_bias_rad=-0.01,
        )
        compact = np.asarray([0.3, 1.0, 0.3, 1.0])
        scales = np.ones(4)

        normalized = reference_action(
            np,
            0.0,
            reference,
            compact_ctrl=compact,
            action_scales=scales,
            joint_low=np.full(4, -2.0),
            joint_high=np.full(4, 2.0),
        )

        np.testing.assert_allclose(
            compact + normalized * scales,
            [0.3, 0.99, 0.3, 0.99],
        )

    def test_projected_reference_matches_cpu_cem_target(self) -> None:
        coefficients = np.asarray(
            [-0.1, 0.04, 0.16, 0.82, -0.03, 0.60, -1.0, 0.54]
        )
        knee_bias = knee_bias_for_foot_gap(0.002)
        reference = CEMReferenceConfig(
            coefficients=tuple(coefficients),
            oscillator_rate_rad_s=3.3,
            oscillator_coupling_per_s=4.5,
            knee_bias_rad=knee_bias,
            minimum_foot_surface_gap_m=0.002,
            foot_gap_tracking_margin_m=0.004,
        )
        compact = np.asarray(
            [0.3141592654, 1.05650322, 0.3141592654, 1.05650322]
        )
        scales = np.asarray([0.8, 1.2, 0.8, 1.2])
        phase = 0.7
        expected = controller_targets(
            0.0,
            1.0,
            coefficients,
            control_phase=phase,
            knee_bias_rad=knee_bias,
            minimum_foot_surface_gap_m=0.002,
            foot_gap_tracking_margin_m=0.004,
        )
        normalized = reference_action(
            np,
            phase,
            reference,
            compact_ctrl=compact,
            action_scales=scales,
            joint_low=np.asarray([-1.12, -0.61, -1.12, -0.61]),
            joint_high=np.asarray([2.41, 2.69, 2.41, 2.69]),
        )

        np.testing.assert_allclose(
            compact + normalized * scales, expected, atol=2.0e-5
        )

    def test_two_million_budget_and_gate_interval_are_fixed(self) -> None:
        quantum = 512 * 16 * 16
        budget = expected_budget_steps(2_000_000, quantum)
        schedule = _exact_stage_eval_schedule(
            budget, quantum, 500_000
        )

        self.assertEqual(quantum, 131_072)
        self.assertEqual(budget, 2_097_152)
        self.assertEqual(schedule["eval_interval_steps"], 524_288)
        self.assertEqual(schedule["num_evals"], 5)

    def test_curriculum_gate_uses_physical_metrics(self) -> None:
        passing = {
            "eval/avg_episode_length": 450.0,
            "eval/episode_roll_progress_rad": 4.0 * 2.0 * math.pi,
            "eval/episode_failed": 0.1,
            "eval/episode_failure_nonfinite": 0.0,
        }
        failing = {**passing, "eval/episode_failed": 0.5}

        pass_result = _gate_assessment(
            passing,
            episode_length=500,
            minimum_survival=0.8,
            minimum_turns=3.0,
            maximum_failure_rate=0.2,
        )
        fail_result = _gate_assessment(
            failing,
            episode_length=500,
            minimum_survival=0.8,
            minimum_turns=3.0,
            maximum_failure_rate=0.2,
        )

        self.assertTrue(pass_result["passed"])
        self.assertFalse(fail_result["passed"])
        self.assertFalse(fail_result["checks"]["failure_rate"])

    def test_eval_visualization_path_identifies_policy(self) -> None:
        path = _eval_visualization_dir(
            Path("results/run"), 3, 1_048_576, 0.5
        )

        self.assertEqual(
            path,
            Path("results/run")
            / "eval_visualizations"
            / "eval_003_step_001048576_ref_0p50",
        )

    def test_environment_keeps_reference_optional(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "curl_robot_2d_mjx"
            / "environment.py"
        ).read_text(encoding="utf-8")

        self.assertIn("cem_reference: CEMReferenceConfig | None = None", source)
        self.assertIn("if reference_settings is None:", source)
        self.assertIn("reference_settings.reference_weight", source)
        observation_source = source[source.index("        def _observation(") :]
        self.assertNotIn(
            "reference_settings.residual_gain", observation_source
        )
        self.assertIn(
            "return 30 if reference_settings is not None else 23", source
        )


if __name__ == "__main__":
    unittest.main()
