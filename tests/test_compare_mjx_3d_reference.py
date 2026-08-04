import unittest

from scripts import compare_mjx_3d_reference as comparison


class CompareMJX3DReferenceTest(unittest.TestCase):
    def test_default_matrix_contains_all_parity_cases(self) -> None:
        args = comparison.parse_args([])

        self.assertEqual(tuple(args.cases), comparison.CASE_NAMES)
        self.assertEqual(args.episode_length, 500)
        self.assertEqual(args.noise_seeds, 64)

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
                "distance_as_shell_turns": 8.2,
                "nonfinite": False,
            }
        )

        self.assertEqual(result["conservative_turns"]["mean"], 7.9)
        self.assertAlmostEqual(result["slip_turns"]["mean"], -0.3)
        self.assertEqual(result["failure_rate"], 0.0)

    def test_mjx_matrix_isolates_solver_and_reset_noise(self) -> None:
        specs = comparison._mjx_case_specs(500)
        newton, newton_batch, newton_reset = specs["mjx_newton_exact"]
        cg_exact, cg_exact_batch, cg_exact_reset = specs["mjx_cg12_exact"]
        cg_noisy, cg_noisy_batch, cg_noisy_reset = specs["mjx_cg12_noisy"]

        self.assertEqual(newton.solver_name, "newton")
        self.assertEqual(newton.reset_joint_noise_rad, 0.0)
        self.assertEqual(newton_batch, 1)
        self.assertEqual(newton_reset, "exact")
        self.assertEqual(cg_exact.solver_name, "cg")
        self.assertEqual(cg_exact.reset_velocity_noise, 0.0)
        self.assertEqual(cg_exact_batch, 1)
        self.assertEqual(cg_exact_reset, "exact")
        self.assertGreater(cg_noisy.reset_joint_noise_rad, 0.0)
        self.assertIsNone(cg_noisy_batch)
        self.assertEqual(cg_noisy_reset, "noise")


if __name__ == "__main__":
    unittest.main()
