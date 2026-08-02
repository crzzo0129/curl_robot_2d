import unittest

import numpy as np

from curl_robot_2d_mjx.reward_3d import (
    REWARD_3D_TERM_NAMES,
    Rolling3DRewardConfig,
    conservative_rolling_potential,
    reward_terms_3d,
)


def zero_inputs():
    zero = np.asarray(0.0, dtype=np.float32)
    return {
        "conservative_progress": zero,
        "mismatch_progress": zero,
        "backward_progress": zero,
        "lateral_velocity_squared": zero,
        "lateral_drift_abs": zero,
        "axis_tilt_squared": zero,
        "action_rate": zero,
        "residual_action_cost": zero,
        "torque_cost": zero,
        "control_dt": np.asarray(0.02, dtype=np.float32),
        "forbidden_active": zero,
        "forbidden_depth": zero,
        "forbidden_max_increment": zero,
        "same_side_foot_contact_start": zero,
        "same_side_foot_contact_active": zero,
        "same_side_foot_excess": zero,
        "same_side_foot_max_increment": zero,
        "cross_side_foot_contact": zero,
        "failed": zero,
        "failure_severe": zero,
        "failure_nonfinite": zero,
        "remaining_fraction": zero,
    }


class MJX3DRewardTest(unittest.TestCase):
    def test_cumulative_potential_requires_rotation_and_translation(self) -> None:
        rotation = np.asarray([0.0, 0.2, 0.2], dtype=np.float32)
        translation = np.asarray([0.0, 0.0, 0.2], dtype=np.float32)

        potential = conservative_rolling_potential(
            np, rotation, translation
        )

        np.testing.assert_allclose(
            np.diff(potential), np.asarray([0.0, 0.2])
        )

    def test_reward_terms_are_named_and_independent(self) -> None:
        config = Rolling3DRewardConfig()
        inputs = zero_inputs()
        inputs["conservative_progress"] = np.asarray(
            0.1, dtype=np.float32
        )
        inputs["axis_tilt_squared"] = np.asarray(0.01, dtype=np.float32)

        terms = reward_terms_3d(np, config, inputs)

        self.assertEqual(tuple(terms), REWARD_3D_TERM_NAMES)
        self.assertAlmostEqual(float(terms["roll_progress"]), 0.6)
        self.assertAlmostEqual(float(terms["axis_tilt"]), -0.08)
        self.assertAlmostEqual(float(terms["collision"]), 0.0)

    def test_same_side_foot_contact_penalty_matches_design(self) -> None:
        config = Rolling3DRewardConfig(
            foot_contact_event=2.0,
            foot_contact_time=4.0,
            allowed_excess_integral=8000.0,
            maximum_allowed_excess=2000.0,
        )
        inputs = zero_inputs()
        inputs["same_side_foot_contact_start"] = np.asarray(
            1.0, dtype=np.float32
        )
        inputs["same_side_foot_contact_active"] = np.asarray(
            1.0, dtype=np.float32
        )
        inputs["same_side_foot_excess"] = np.asarray(
            0.0005, dtype=np.float32
        )
        inputs["same_side_foot_max_increment"] = np.asarray(
            0.0005, dtype=np.float32
        )

        terms = reward_terms_3d(np, config, inputs)

        self.assertAlmostEqual(float(terms["collision"]), -3.16, places=5)

    def test_cross_side_foot_contact_is_strongly_penalized(self) -> None:
        inputs = zero_inputs()
        inputs["cross_side_foot_contact"] = np.asarray(
            1.0, dtype=np.float32
        )

        terms = reward_terms_3d(np, Rolling3DRewardConfig(), inputs)

        self.assertAlmostEqual(float(terms["collision"]), -30.0)

    def test_severe_and_nonfinite_terminal_penalties(self) -> None:
        severe_inputs = zero_inputs()
        severe_inputs["failed"] = np.asarray(1.0, dtype=np.float32)
        severe_inputs["failure_severe"] = np.asarray(1.0, dtype=np.float32)
        severe_inputs["remaining_fraction"] = np.asarray(
            1.0, dtype=np.float32
        )

        severe_terms = reward_terms_3d(
            np, Rolling3DRewardConfig(), severe_inputs
        )

        self.assertEqual(float(severe_terms["termination"]), -40.0)
        self.assertEqual(float(severe_terms["early_termination"]), -40.0)

        nonfinite_inputs = zero_inputs()
        nonfinite_inputs["failed"] = np.asarray(1.0, dtype=np.float32)
        nonfinite_inputs["failure_nonfinite"] = np.asarray(
            1.0, dtype=np.float32
        )

        nonfinite_terms = reward_terms_3d(
            np, Rolling3DRewardConfig(), nonfinite_inputs
        )

        self.assertEqual(float(nonfinite_terms["termination"]), -80.0)


if __name__ == "__main__":
    unittest.main()
