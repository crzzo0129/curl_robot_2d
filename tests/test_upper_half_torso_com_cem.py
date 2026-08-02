import unittest

from scripts import sweep_upper_half_torso_com_cem as sweep


class UpperHalfTorsoCOMCEMTest(unittest.TestCase):
    def test_default_grid_has_44_valid_points(self) -> None:
        points = sweep.upper_half_points(
            sweep.DEFAULT_X_CENTER_MM,
            sweep.DEFAULT_Z_CENTER_MM,
            140.0,
        )

        self.assertEqual(len(points), 44)

    def test_grid_contains_original_root_com(self) -> None:
        points = sweep.upper_half_points(
            sweep.DEFAULT_X_CENTER_MM,
            sweep.DEFAULT_Z_CENTER_MM,
            140.0,
        )

        self.assertTrue(
            any(
                abs(x_root - 0.025) < 1.0e-12
                and abs(z_root - 0.015) < 1.0e-12
                for _, _, x_root, z_root in points
            )
        )

    def test_point_keys_ignore_insignificant_float_noise(self) -> None:
        self.assertEqual(
            sweep._point_key(0.025, 0.118228644035),
            sweep._point_key(0.02500000001, 0.1182286440346),
        )

    def test_rolling_selection_prefers_completed_two_turns(self) -> None:
        fast_failure = {
            "completed_two_turns": False,
            "conservative_rolling_turns": 1.9,
            "forbidden_contact_total_s": 0.0,
            "score": 100.0,
        }
        sustained = {
            "completed_two_turns": True,
            "conservative_rolling_turns": 2.0,
            "forbidden_contact_total_s": 0.1,
            "score": 1.0,
        }

        self.assertGreater(
            sweep._rolling_selection_key(sustained),
            sweep._rolling_selection_key(fast_failure),
        )

    def test_cold_start_flag_disables_controller_initialization(self) -> None:
        args = sweep.parse_args(["--cold-start"])

        self.assertTrue(args.cold_start)


if __name__ == "__main__":
    unittest.main()
