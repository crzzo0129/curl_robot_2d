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
        "yaw_rate_squared": zero,
        "lateral_yaw_abs": zero,
        "axis_tilt_squared": zero,
        "action_rate": zero,
        "residual_action_cost": zero,
        "torque_cost": zero,
        "control_dt": np.asarray(0.02, dtype=np.float32),
        "forbidden_active": zero,
        "first_turn_active": zero,
        "forbidden_depth": zero,
        "forbidden_max_increment": zero,
        "same_side_foot_contact_start": zero,
        "same_side_foot_contact_active": zero,
        "same_side_foot_excess": zero,
        "same_side_foot_max_increment": zero,
        "cross_side_foot_contact": zero,
        "roll_potential_positive": zero,
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

    def test_first_turn_forbidden_contact_can_receive_extra_weight(self) -> None:
        config = Rolling3DRewardConfig(
            forbidden_contact_time=4.0,
            first_turn_forbidden_contact_multiplier=3.0,
        )
        inputs = zero_inputs()
        inputs["forbidden_active"] = np.asarray(1.0, dtype=np.float32)
        inputs["first_turn_active"] = np.asarray(1.0, dtype=np.float32)

        terms = reward_terms_3d(np, config, inputs)

        self.assertAlmostEqual(float(terms["collision"]), -0.32)

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

    def test_failure_claws_back_accumulated_progress_reward(self) -> None:
        inputs = zero_inputs()
        inputs["failed"] = np.asarray(1.0, dtype=np.float32)
        inputs["roll_potential_positive"] = np.asarray(
            2.0 * np.pi, dtype=np.float32
        )

        terms = reward_terms_3d(
            np,
            Rolling3DRewardConfig(failure_progress_clawback=8.0),
            inputs,
        )

        self.assertAlmostEqual(
            float(terms["failure_progress_clawback"]),
            -16.0 * np.pi,
            places=5,
        )

    def test_late_lateral_failure_cannot_outrank_stable_roll(self) -> None:
        config = Rolling3DRewardConfig(
            roll_progress=8.0,
            lateral_drift=6.0,
            failure_progress_clawback=4.0,
            termination=40.0,
        )
        progress_per_step = 8.6 * 2.0 * np.pi / 500.0

        stable_inputs = zero_inputs()
        stable_inputs["conservative_progress"] = np.asarray(
            progress_per_step, dtype=np.float32
        )
        stable_inputs["lateral_yaw_abs"] = np.asarray(
            0.04, dtype=np.float32
        )
        stable_step_reward = sum(
            reward_terms_3d(np, config, stable_inputs).values()
        )
        stable_return = 500.0 * float(stable_step_reward)

        failed_inputs = dict(stable_inputs)
        failed_inputs["lateral_yaw_abs"] = np.asarray(
            0.03, dtype=np.float32
        )
        failed_step_reward = sum(
            reward_terms_3d(np, config, failed_inputs).values()
        )
        failure_inputs = dict(failed_inputs)
        failure_inputs["failed"] = np.asarray(1.0, dtype=np.float32)
        failure_inputs["roll_potential_positive"] = np.asarray(
            7.7 * 2.0 * np.pi, dtype=np.float32
        )
        failure_inputs["remaining_fraction"] = np.asarray(
            0.10, dtype=np.float32
        )
        terminal_adjustment = sum(
            reward_terms_3d(np, config, failure_inputs).values()
        ) - failed_step_reward
        failed_return = (
            450.0 * float(failed_step_reward)
            + float(terminal_adjustment)
        )

        self.assertGreater(stable_return, failed_return)


if __name__ == "__main__":
    unittest.main()
