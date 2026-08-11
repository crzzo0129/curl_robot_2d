import unittest

from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS
from scripts import run_staged_cem_at_com


class StagedCEMGeometryTest(unittest.TestCase):
    def test_final_stage_triples_forbidden_collision_penalty(self) -> None:
        stage = run_staged_cem_at_com.STAGES[-1]

        self.assertEqual(stage.name, "03_strict_forbidden_collision")
        self.assertTrue(stage.allow_foot_contact)
        self.assertIsNone(stage.maximum_self_contact_time_s)
        self.assertEqual(stage.collision_penalty_scale, 3.0)
        self.assertEqual(stage.minimum_foot_gap_m, 0.0)
        self.assertEqual(stage.generations, 20)
        self.assertEqual(stage.population, 64)

    def test_real_geometry_keeps_default_60_mm_foot(self) -> None:
        args = run_staged_cem_at_com.parse_args(["--geometry", "real"])
        parameters = run_staged_cem_at_com._geometry_parameters(args)

        self.assertAlmostEqual(
            parameters.foot_radius,
            REAL_GEOMETRY_PARAMETERS.foot_radius,
        )
        self.assertAlmostEqual(2.0 * parameters.foot_radius, 0.060)

    def test_real_geometry_accepts_39_mm_foot_override(self) -> None:
        args = run_staged_cem_at_com.parse_args(
            ["--geometry", "real", "--foot-diameter-mm", "39"]
        )
        parameters = run_staged_cem_at_com._geometry_parameters(args)

        self.assertAlmostEqual(parameters.foot_radius, 0.0195)
        self.assertAlmostEqual(2.0 * parameters.foot_radius, 0.039)
        self.assertAlmostEqual(parameters.edge_length, 0.180)
        self.assertAlmostEqual(parameters.lower_proxy_radius, 0.025)

    def test_foot_diameter_override_must_be_positive(self) -> None:
        args = run_staged_cem_at_com.parse_args(
            ["--geometry", "real", "--foot-diameter-mm", "0"]
        )
        with self.assertRaises(SystemExit):
            run_staged_cem_at_com._geometry_parameters(args)


if __name__ == "__main__":
    unittest.main()
