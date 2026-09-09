import unittest
import numpy as np
from curl_robot_2d_mjx.transition_console import summarize_raw_evaluation


class RawEvaluationSummaryTests(unittest.TestCase):
    def test_per_episode_normalization_keeps_rewards_and_outcomes(self):
        metrics = summarize_raw_evaluation({
            'terminal_step': np.array([100, 400]),
            'foot_contact_count_per_step': np.array([400, 800]),
            'reward_reference_tracking': np.array([-10, -30]),
            'transition_success': np.array([1, 1]),
        })
        self.assertEqual(metrics['eval/episode_foot_contact_count_per_step'], 3.)
        self.assertEqual(metrics['eval/episode_reward_reference_tracking'], -20.)
        self.assertEqual(metrics['eval/episode_transition_success'], 1.)

    def test_incomplete_episode_uses_recorded_length(self):
        metrics = summarize_raw_evaluation({
            'terminal_step': np.array([0]), 'recorded_steps': np.array([500]),
            'root_z_m_per_step': np.array([75.]),
        })
        self.assertAlmostEqual(metrics['eval/episode_root_z_m_per_step'], .15)


if __name__ == '__main__':
    unittest.main()
