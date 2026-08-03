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
        "stance_miss_fraction": zero,
        "swing_contact_fraction": zero,
        "swing_clearance_cost": zero,
        "joint_tracking_cost": zero,
        "action_rate_cost": zero,
        "residual_action_cost": zero,
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
    def test_nominal_forward_step_has_all_positive_task_terms(self) -> None:
        inputs = zero_inputs()
        inputs["normalized_forward_velocity"] = np.asarray(1.0)

        terms = reward_terms_walking_3d(
            np, Walking3DRewardConfig(), inputs
        )

        self.assertEqual(tuple(terms), WALKING_REWARD_TERM_NAMES_3D)
        self.assertAlmostEqual(float(terms["alive"]), 0.05)
        self.assertAlmostEqual(float(terms["velocity_tracking"]), 1.40)
        self.assertAlmostEqual(float(terms["forward_progress"]), 0.40)
        self.assertAlmostEqual(float(terms["upright"]), 0.45)

    def test_height_contact_and_clearance_costs_are_independent(self) -> None:
        config = Walking3DRewardConfig()
        inputs = zero_inputs()
        inputs["root_height_error"] = np.asarray(config.height_sigma_m)
        inputs["swing_clearance_cost"] = np.asarray(0.5)
        inputs["nonfoot_contact_active"] = np.asarray(1.0)
        inputs["nonfoot_depth"] = np.asarray(0.002)

        terms = reward_terms_walking_3d(np, config, inputs)

        self.assertAlmostEqual(float(terms["height"]), -0.50)
        self.assertAlmostEqual(float(terms["swing_clearance"]), -0.06)
        self.assertAlmostEqual(float(terms["collision"]), -2.20, places=6)

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
