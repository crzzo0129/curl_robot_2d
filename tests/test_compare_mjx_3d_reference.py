import unittest

from scripts import compare_mjx_3d_reference as comparison


class CompareMJX3DReferenceTest(unittest.TestCase):
    def test_default_matrix_contains_all_parity_cases(self) -> None:
        args = comparison.parse_args([])

        self.assertEqual(tuple(args.cases), comparison.DEFAULT_CASE_NAMES)
        self.assertEqual(args.episode_length, 500)
        self.assertEqual(args.noise_seeds, 64)
        self.assertEqual(args.reference_action_scale, 1.0)
        self.assertIsNone(args.reference_ramp_start_scale)
        self.assertEqual(args.reference_ramp_duration_s, 0.25)
        self.assertEqual(args.reference_startup_boost, 0.0)
        self.assertEqual(args.reference_startup_boost_duration_s, 0.25)

    def test_distribution_reports_median_and_range(self) -> None:
        result = comparison._distribution([1.0, 2.0, 9.0])

        self.assertEqual(result["mean"], 4.0)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 9.0)

    def test_cpu_summary_uses_conservative_physical_turns(self) -> None:
        result = comparison._cpu_result(
            {
                "rolling_phase_turns": 7.9,
                "absolute_rotation_turns": 8.0,
                "distance_as_shell_turns": 8.2,
                "nonfinite": False,
                "physics_profile": "cg12",
                "solver": "cg",
            }
        )

        self.assertEqual(result["name"], "cpu_cg12_exact")
        self.assertEqual(result["solver"], "cg")
        self.assertEqual(result["physics_profile"], "cg12")
        self.assertEqual(result["reference_action_scale"], 1.0)
        self.assertIsNone(result["reference_ramp_start_scale"])
        self.assertEqual(result["reference_ramp_duration_s"], 0.25)
        self.assertEqual(result["reference_startup_boost"], 0.0)
        self.assertEqual(result["conservative_turns"]["mean"], 8.0)
        self.assertAlmostEqual(
            result["rotation_translation_mismatch_turns"]["mean"], -0.2
        )
        self.assertEqual(result["failure_rate"], 0.0)

    def test_mjx_matrix_isolates_solver_and_reset_noise(self) -> None:
        specs = comparison._mjx_case_specs(500)
        newton8, newton8_batch, newton8_reset = specs["mjx_newton8_exact"]
        cg_exact, cg_exact_batch, cg_exact_reset = specs["mjx_cg12_exact"]
        cg_noisy, cg_noisy_batch, cg_noisy_reset = specs["mjx_cg12_noisy"]
        cg20, cg20_batch, cg20_reset = specs["mjx_cg20_exact"]

        self.assertEqual(newton8.solver_name, "newton")
        self.assertEqual(newton8.solver_iterations, 8)
        self.assertEqual(newton8.solver_ls_iterations, 8)
        self.assertEqual(newton8_batch, 1)
        self.assertEqual(newton8_reset, "exact")
        self.assertEqual(cg_exact.solver_name, "cg")
        self.assertEqual(cg_exact.reset_velocity_noise, 0.0)
        self.assertEqual(cg_exact_batch, 1)
        self.assertEqual(cg_exact_reset, "exact")
        self.assertEqual(cg20.solver_name, "cg")
        self.assertEqual(cg20.solver_iterations, 20)
        self.assertEqual(cg20.solver_ls_iterations, 10)
        self.assertEqual(cg20_batch, 1)
        self.assertEqual(cg20_reset, "exact")
        self.assertGreater(cg_noisy.reset_joint_noise_rad, 0.0)
        self.assertIsNone(cg_noisy_batch)
        self.assertEqual(cg_noisy_reset, "noise")

    def test_mjx_matrix_applies_reference_startup_boost(self) -> None:
        specs = comparison._mjx_case_specs(
            500,
            reference_action_scale=1.05,
            reference_ramp_start_scale=0.25,
            reference_ramp_duration_s=0.1,
            reference_startup_boost=0.20,
            reference_startup_boost_duration_s=0.4,
        )

        task, _, _ = specs["mjx_cg20_exact"]
        self.assertEqual(task.reference_action_scale, 1.05)
        self.assertEqual(task.reference_ramp_start_scale, 0.25)
        self.assertEqual(task.reference_ramp_duration_s, 0.1)
        self.assertEqual(task.reference_startup_boost, 0.20)
        self.assertEqual(task.reference_startup_boost_duration_s, 0.4)


if __name__ == "__main__":
    unittest.main()
