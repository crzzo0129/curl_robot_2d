import unittest

import numpy as np

from curl_robot_2d_mjx.reward_walking_3d import (
    WALKING_REWARD_TERM_NAMES_3D,
    Walking3DRewardConfig,
    reward_terms_walking_3d,
)


def zero_inputs():
    zero = np.asarray(0.0, dtype=np.float32)
    return {
        "forward_velocity_error": zero,
        "normalized_forward_velocity": zero,
        "upright_tilt": zero,
        "root_height_error": zero,
        "heading_error": zero,
        "lateral_velocity": zero,
        "lateral_drift": zero,
        "vertical_velocity": zero,
        "roll_pitch_angular_velocity_squared": zero,
        "foot_air_time_reward": zero,
        "swing_clearance_cost": zero,
        "foot_slip_velocity_squared": zero,
        "action_rate_cost": zero,
        "action_magnitude_cost": zero,
        "joint_velocity_squared": zero,
        "joint_limit_cost": zero,
        "torque_cost": zero,
        "nonfoot_contact_active": zero,
        "nonfoot_depth": zero,
        "self_contact_active": zero,
        "self_contact_depth": zero,
        "failed": zero,
        "failure_severe": zero,
        "failure_nonfinite": zero,
        "remaining_fraction": zero,
    }


class MJX3DWalkingRewardTest(unittest.TestCase):
    def test_commanded_forward_step_has_positive_task_terms(self) -> None:
        inputs = zero_inputs()
        inputs["normalized_forward_velocity"] = np.asarray(1.0)

        terms = reward_terms_walking_3d(
            np, Walking3DRewardConfig(), inputs
        )

        self.assertEqual(tuple(terms), WALKING_REWARD_TERM_NAMES_3D)
        self.assertAlmostEqual(float(terms["alive"]), 0.0)
        self.assertAlmostEqual(float(terms["velocity_tracking"]), 2.00)
        self.assertAlmostEqual(float(terms["forward_progress"]), 0.0)
        self.assertAlmostEqual(float(terms["upright"]), 0.50)

    def test_generic_motion_costs_are_independent(self) -> None:
        config = Walking3DRewardConfig()
        inputs = zero_inputs()
        inputs["root_height_error"] = np.asarray(config.height_sigma_m)
        inputs["vertical_velocity"] = np.asarray(
            config.vertical_velocity_sigma_m_s
        )
        inputs["foot_slip_velocity_squared"] = np.asarray(
            config.foot_slip_sigma_m_s**2
        )
        inputs["swing_clearance_cost"] = np.asarray(0.5)
        inputs["nonfoot_contact_active"] = np.asarray(1.0)
        inputs["nonfoot_depth"] = np.asarray(0.002)

        terms = reward_terms_walking_3d(np, config, inputs)

        self.assertAlmostEqual(float(terms["height"]), 0.0)
        self.assertAlmostEqual(float(terms["vertical_velocity"]), -0.05)
        self.assertAlmostEqual(float(terms["foot_slip"]), -0.05)
        self.assertAlmostEqual(float(terms["swing_clearance"]), -0.025)
        self.assertAlmostEqual(float(terms["collision"]), -2.70, places=6)

    def test_task_rewards_are_gated_by_upright_posture(self) -> None:
        config = Walking3DRewardConfig(forward_progress=1.0)
        inputs = zero_inputs()
        inputs["normalized_forward_velocity"] = np.asarray(1.0)
        inputs["upright_tilt"] = np.asarray(config.upright_sigma_rad)

        terms = reward_terms_walking_3d(np, config, inputs)
        upright_gate = np.exp(-1.0)

        self.assertAlmostEqual(
            float(terms["velocity_tracking"]),
            config.velocity_tracking * upright_gate,
        )
        self.assertAlmostEqual(
            float(terms["forward_progress"]), upright_gate
        )

    def test_touchdown_air_time_is_rewarded_without_phase_schedule(self) -> None:
        inputs = zero_inputs()
        inputs["foot_air_time_reward"] = np.asarray(0.5)

        terms = reward_terms_walking_3d(
            np, Walking3DRewardConfig(), inputs
        )

        self.assertAlmostEqual(float(terms["foot_air_time"]), 0.075)

    def test_severe_and_nonfinite_termination_are_strong(self) -> None:
        severe = zero_inputs()
        severe["failed"] = np.asarray(1.0)
        severe["failure_severe"] = np.asarray(1.0)
        severe["remaining_fraction"] = np.asarray(1.0)

        severe_terms = reward_terms_walking_3d(
            np, Walking3DRewardConfig(), severe
        )

        self.assertEqual(float(severe_terms["termination"]), -40.0)
        self.assertEqual(float(severe_terms["early_termination"]), -40.0)

        nonfinite = zero_inputs()
        nonfinite["failed"] = np.asarray(1.0)
        nonfinite["failure_nonfinite"] = np.asarray(1.0)
        nonfinite_terms = reward_terms_walking_3d(
            np, Walking3DRewardConfig(), nonfinite
        )

        self.assertEqual(float(nonfinite_terms["termination"]), -80.0)


if __name__ == "__main__":
    unittest.main()
