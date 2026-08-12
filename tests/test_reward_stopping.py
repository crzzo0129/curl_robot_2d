import math
import unittest

import numpy as np

from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig
from curl_robot_2d_mjx.environment_stopping_2d import scaled_reference_frequency
from curl_robot_2d_mjx.reward_stopping import (
    StoppingRewardConfig,
    StoppingTaskConfig,
    bounded_normalized_square,
    braking_reference_scales,
    desired_braking_speed,
    select_reachable_target_phase_xp,
    stopping_observation_features,
    stopping_reward_terms,
)
from scripts.train_mjx_stopping_ppo import _checkpoint_rank


class StoppingRewardTest(unittest.TestCase):
    def test_reachable_target_adds_a_turn_when_window_is_too_close(self):
        config = StoppingTaskConfig(braking_margin_rad=0.5)
        target, distance = select_reachable_target_phase_xp(
            np, -0.1, 4.0, config
        )
        required = 4.0**2 / (2.0 * config.maximum_deceleration_rad_s2) + 0.5
        self.assertGreaterEqual(distance, required)
        self.assertAlmostEqual(target % (2.0 * math.pi), 0.0)

    def test_desired_speed_goes_to_zero_at_target(self):
        config = StoppingTaskConfig()
        self.assertAlmostEqual(desired_braking_speed(np, 0.0, config), 0.0)
        self.assertAlmostEqual(
            desired_braking_speed(np, 10.0, config), config.nominal_roll_rate_rad_s
        )

    def test_observation_features_are_fixed_width_and_finite(self):
        features = stopping_observation_features(
            np,
            body_phase=1.0,
            target_phase=2.0,
            initial_distance=1.5,
            linear_speed=0.2,
            angular_speed=1.0,
            elapsed_s=0.5,
            config=StoppingTaskConfig(),
        )
        self.assertEqual(features.shape, (12,))
        self.assertTrue(np.isfinite(features).all())

    def test_reference_schedule_keeps_roll_then_brakes_smoothly(self):
        config = StoppingTaskConfig(reference_brake_start_progress=0.75)
        rate_early, amplitude_early = braking_reference_scales(
            np, 0.5, 1.0, config
        )
        rate_final, amplitude_final = braking_reference_scales(
            np, 0.0, 1.0, config
        )
        self.assertAlmostEqual(rate_early, 1.0)
        self.assertAlmostEqual(amplitude_early, 1.0)
        self.assertAlmostEqual(rate_final, 0.4)
        self.assertAlmostEqual(amplitude_final, 0.8)

    def test_normalized_square_is_capped_before_reward_weighting(self):
        self.assertAlmostEqual(
            bounded_normalized_square(np, 10.0, 2.0, 1.0), 1.0
        )
        self.assertAlmostEqual(
            bounded_normalized_square(np, 1.0, 2.0, 1.0), 0.25
        )

    def test_success_bonus_dominates_small_control_cost(self):
        zero = {
            "target_progress": 0.0,
            "speed_error_sq": 0.0,
            "linear_speed_sq": 0.0,
            "phase_error_sq": 0.0,
            "overshoot": 0.0,
            "action_rate_sq": 0.1,
            "residual_action_sq": 0.1,
            "torque_sq": 0.1,
            "internal_contact": 0.0,
            "torso_contact": 0.0,
            "success": 1.0,
            "failure": 0.0,
            "timeout": 0.0,
        }
        terms = stopping_reward_terms(np, StoppingRewardConfig(), zero)
        self.assertGreater(sum(terms.values()), 95.0)

    def test_frequency_scaling_matches_cpu_controller_semantics(self):
        reference = CEMReferenceConfig(
            coefficients=(0.0,) * 8,
            oscillator_rate_rad_s=4.0,
            oscillator_coupling_per_s=2.0,
        )
        scaled = scaled_reference_frequency(reference, 1.0 / math.pi)
        self.assertAlmostEqual(scaled.oscillator_rate_rad_s, 2.0)
        self.assertAlmostEqual(scaled.oscillator_coupling_per_s, 1.0)

    def test_checkpoint_rank_prioritizes_success_over_reward_proxies(self):
        successful = {
            "eval/episode_stop_success": 0.25,
            "eval/episode_failed": 0.5,
            "eval/avg_episode_length": 100.0,
            "eval/episode_phase_error_rad": 50.0,
            "eval/episode_angular_speed_rad_s": 50.0,
        }
        accurate_but_unsuccessful = {
            "eval/episode_stop_success": 0.0,
            "eval/episode_failed": 0.0,
            "eval/avg_episode_length": 100.0,
            "eval/episode_phase_error_rad": 0.0,
            "eval/episode_angular_speed_rad_s": 0.0,
        }
        self.assertGreater(
            _checkpoint_rank(successful),
            _checkpoint_rank(accurate_but_unsuccessful),
        )

    def test_early_failure_penalty_exceeds_success_bonus(self):
        reward = StoppingRewardConfig()
        task = StoppingTaskConfig()
        earliest_failure_penalty = reward.failure * (
            1.0 + task.early_failure_scale
        )
        self.assertGreater(earliest_failure_penalty, reward.success)


if __name__ == "__main__":
    unittest.main()
