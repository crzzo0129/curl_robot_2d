import argparse
import unittest

import numpy as np

from curl_robot_2d_mjx.reward import (
    REWARD_TERM_NAMES,
    conservative_rolling_potential,
    reward_terms,
)
from curl_robot_2d_mjx.reward_config import RollingRewardConfig
from scripts.train_mjx_ppo import (
    _add_reward_arguments,
    _add_per_step_eval_metrics,
    _checkpoint_selection,
    _format_eval_report,
    _reward_config_from_args,
    _split_metrics,
    _training_step_schedule,
)


def zero_inputs():
    zero = np.asarray(0.0, dtype=np.float32)
    return {
        "conservative_progress": zero,
        "mismatch_progress": zero,
        "backward": zero,
        "action_rate": zero,
        "residual_action_cost": zero,
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
        "remaining_fraction": zero,
    }


class MJXRewardTest(unittest.TestCase):
    def test_cumulative_potential_rewards_asynchronous_rolling(self) -> None:
        phase = np.asarray([0.0, 0.1, 0.1], dtype=np.float32)
        translation = np.asarray([0.0, 0.0, 0.1], dtype=np.float32)

        potential = conservative_rolling_potential(
            np, phase, translation
        )
        progress = np.diff(potential)

        np.testing.assert_allclose(progress, np.asarray([0.0, 0.1]))
        self.assertAlmostEqual(float(np.sum(progress)), 0.1)

    def test_cumulative_mismatch_is_repaid_when_motion_catches_up(
        self,
    ) -> None:
        config = RollingRewardConfig()
        phase = np.asarray([0.0, 0.1, 0.1], dtype=np.float32)
        translation = np.asarray([0.0, 0.0, 0.1], dtype=np.float32)
        mismatch_potential = np.abs(phase - translation)
        mismatch_progress = np.diff(mismatch_potential)
        rewards = []
        for progress in mismatch_progress:
            inputs = zero_inputs()
            inputs["mismatch_progress"] = progress
            rewards.append(reward_terms(np, config, inputs)["roll_mismatch"])

        np.testing.assert_allclose(rewards, np.asarray([-0.05, 0.05]))
        self.assertAlmostEqual(float(np.sum(rewards)), 0.0)

    def test_cli_can_override_reward_without_changing_source_defaults(
        self,
    ) -> None:
        parser = argparse.ArgumentParser()
        _add_reward_arguments(parser)
        args = parser.parse_args(
            [
                "--reward-termination",
                "20",
                "--reward-early-termination-scale",
                "0.5",
                "--reward-roll-progress",
                "4",
            ]
        )

        config = _reward_config_from_args(args)

        self.assertEqual(config.termination, 20.0)
        self.assertEqual(config.early_termination_scale, 0.5)
        self.assertEqual(config.roll_progress, 4.0)
        self.assertEqual(
            config.action_rate, RollingRewardConfig().action_rate
        )

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

    def test_residual_action_cost_is_explicit_and_optional(self) -> None:
        inputs = zero_inputs()
        inputs["residual_action_cost"] = np.asarray(0.25, dtype=np.float32)

        default_term = reward_terms(
            np, RollingRewardConfig(), inputs
        )["residual_action"]
        penalized_term = reward_terms(
            np, RollingRewardConfig(residual_action=0.1), inputs
        )["residual_action"]

        self.assertEqual(float(default_term), 0.0)
        self.assertAlmostEqual(float(penalized_term), -0.025)

    def test_early_failure_penalty_decays_with_remaining_length(self) -> None:
        config = RollingRewardConfig(
            termination=10.0, early_termination_scale=1.0
        )
        inputs = zero_inputs()
        inputs["failed"] = np.asarray(1.0, dtype=np.float32)

        inputs["remaining_fraction"] = np.asarray(1.0, dtype=np.float32)
        early_terms = reward_terms(np, config, inputs)
        inputs["remaining_fraction"] = np.asarray(0.0, dtype=np.float32)
        late_terms = reward_terms(np, config, inputs)

        self.assertEqual(float(early_terms["termination"]), -10.0)
        self.assertEqual(float(early_terms["early_termination"]), -10.0)
        self.assertEqual(float(late_terms["termination"]), -10.0)
        self.assertEqual(float(late_terms["early_termination"]), 0.0)

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

    def test_checkpoint_selection_prefers_physical_progress_and_survival(
        self,
    ) -> None:
        short_backward = {
            "eval/avg_episode_length": 60.0,
            "eval/episode_failed": 1.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_roll_progress_rad": -0.5,
            "eval/avg_forbidden_penetration_m": 0.0,
        }
        longer_forward = {
            "eval/avg_episode_length": 300.0,
            "eval/episode_failed": 1.0,
            "eval/episode_failure_nonfinite": 0.0,
            "eval/episode_roll_progress_rad": 2.0,
            "eval/avg_forbidden_penetration_m": 0.0,
        }

        short_score = _checkpoint_selection(short_backward, 500)
        longer_score = _checkpoint_selection(longer_forward, 500)

        self.assertGreater(longer_score["score"], short_score["score"])
        self.assertGreater(longer_score["turns"], 0.0)
        self.assertFalse(longer_score["rejected"])

    def test_checkpoint_selection_rejects_nonfinite_policy(self) -> None:
        metrics = {
            "eval/avg_episode_length": 500.0,
            "eval/episode_failed": 0.0,
            "eval/episode_failure_nonfinite": 0.1,
            "eval/episode_roll_progress_rad": 20.0,
            "eval/avg_forbidden_penetration_m": 0.0,
        }

        selection = _checkpoint_selection(metrics, 500)

        self.assertTrue(selection["rejected"])

    def test_eval_report_contains_reward_and_failure_details(self) -> None:
        metrics = {
            "eval/episode_reward": -12.0,
            "eval/avg_episode_length": 100.0,
            "eval/episode_failed": 1.0,
            "eval/episode_timeout": 0.0,
            "eval/episode_roll_progress_rad": 1.0,
            "eval/episode_forbidden_penetration_m": 0.01,
            "training/sps": 4200.0,
            "training/kl_mean": 0.02,
            "training/policy_loss": -0.1,
            "training/v_loss": 0.03,
            "training/policy_dist_mean_std": 0.5,
        }
        for name in REWARD_TERM_NAMES:
            metrics[f"eval/episode_reward_{name}"] = -1.0
        _add_per_step_eval_metrics(metrics)
        selection = _checkpoint_selection(metrics, 500)

        report = _format_eval_report(
            2,
            6,
            655_360,
            metrics,
            episode_length=500,
            control_dt=0.02,
            selection=selection,
            selected=True,
        )

        self.assertIn("[eval 2/6]", report)
        self.assertIn("turns/episode=", report)
        self.assertIn("reward/step", report)
        self.assertIn("early=-0.0100", report)
        self.assertIn("failures", report)
        self.assertIn("ppo", report)

    def test_training_schedule_reports_brax_rollout_rounding(self) -> None:
        schedule = _training_step_schedule(
            requested_steps=50_000_000,
            num_evals=10,
            batch_size=1024,
            unroll_length=20,
            num_minibatches=32,
        )

        self.assertEqual(schedule["rollout_quantum"], 655_360)
        self.assertEqual(schedule["eval_interval_steps"], 5_898_240)
        self.assertEqual(schedule["effective_steps"], 53_084_160)


if __name__ == "__main__":
    unittest.main()
