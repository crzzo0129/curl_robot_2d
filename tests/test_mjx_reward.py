import unittest

import numpy as np

from curl_robot_2d_mjx.reward import REWARD_TERM_NAMES, reward_terms
from curl_robot_2d_mjx.reward_config import RollingRewardConfig
from scripts.train_mjx_ppo import (
    _add_per_step_eval_metrics,
    _split_metrics,
)


def zero_inputs():
    zero = np.asarray(0.0, dtype=np.float32)
    return {
        "conservative_progress": zero,
        "mismatch": zero,
        "backward": zero,
        "action_rate": zero,
        "torque_cost": zero,
        "airborne": zero,
        "foot_distance": zero,
        "control_dt": np.asarray(0.02, dtype=np.float32),
        "forbidden_count": zero,
        "forbidden_depth": zero,
        "forbidden_max_increment": zero,
        "allowed_excess": zero,
        "allowed_max_increment": zero,
        "leg_crossing": zero,
        "failed": zero,
    }


class MJXRewardTest(unittest.TestCase):
    def test_reward_terms_are_named_and_independent(self) -> None:
        config = RollingRewardConfig()
        inputs = zero_inputs()
        inputs["conservative_progress"] = np.asarray(
            0.1, dtype=np.float32
        )
        inputs["failed"] = np.asarray(1.0, dtype=np.float32)
        terms = reward_terms(np, config, inputs)

        self.assertEqual(tuple(terms), REWARD_TERM_NAMES)
        self.assertAlmostEqual(float(terms["roll_progress"]), 0.5)
        self.assertAlmostEqual(
            float(terms["termination"]), -config.termination
        )
        self.assertAlmostEqual(float(terms["collision"]), 0.0)

    def test_eval_reward_metrics_are_split_from_other_metrics(self) -> None:
        metrics = {
            "eval/episode_reward": 10.0,
            "eval/episode_reward_termination": -5.0,
            "eval/episode_root_height_m": 20.0,
            "eval/avg_episode_length": 100.0,
        }
        _add_per_step_eval_metrics(metrics)
        rewards, ordinary = _split_metrics(metrics)

        self.assertEqual(rewards["eval/avg_reward"], 0.1)
        self.assertEqual(
            rewards["eval/avg_reward_termination"], -0.05
        )
        self.assertEqual(ordinary["eval/avg_root_height_m"], 0.2)
        self.assertNotIn("eval/episode_reward", ordinary)


if __name__ == "__main__":
    unittest.main()
