from pathlib import Path
import math
import unittest

import numpy as np

from curl_robot_2d_mjx.cem_reference import (
    DEFAULT_CEM_CONTROLLER,
    advance_oscillator,
    effective_residual_action,
    expected_budget_steps,
    load_cem_reference,
    reference_action,
)
from scripts.train_mjx_residual_ppo import (
    _eval_visualization_dir,
    _exact_stage_eval_schedule,
    _gate_assessment,
)
from scripts.optimize_phase_controller import controller_targets


class MJXResidualTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
