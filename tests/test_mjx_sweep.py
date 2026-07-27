import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.sweep_mjx_ppo import (
    DEFAULT_BUDGETS,
    DEFAULT_CANDIDATES,
    _training_command,
    score_training,
)


class MJXSweepTest(unittest.TestCase):
    def test_default_sweep_keeps_physics_fixed_and_scans_training(self) -> None:
        self.assertGreaterEqual(len(DEFAULT_CANDIDATES), 6)
        self.assertTrue(
            any(
                candidate.discounting > DEFAULT_CANDIDATES[0].discounting
                for candidate in DEFAULT_CANDIDATES
            )
        )
        self.assertTrue(
            any(
                candidate.learning_rate
                < DEFAULT_CANDIDATES[0].learning_rate
                for candidate in DEFAULT_CANDIDATES
            )
        )
        self.assertEqual(
            DEFAULT_BUDGETS["4090"]["final_steps"], 20_000_000
        )
        self.assertEqual(
            DEFAULT_BUDGETS["h200"]["final_steps"], 50_000_000
        )

    def test_training_command_carries_selected_reward_override(self) -> None:
        candidate = DEFAULT_CANDIDATES[1]
        command = _training_command(
            candidate=candidate,
            preset="4090",
            physics_profile="cg12",
            steps=1000,
            num_evals=2,
            seed=3,
            episode_length=500,
            output_dir=Path("result"),
        )

        self.assertIn("--reward-termination", command)
        self.assertIn(str(candidate.reward_termination), command)
        self.assertIn("--discounting", command)
        self.assertIn("--episode-length", command)
        self.assertNotIn("--restore-checkpoint", command)

    def test_scoring_uses_physical_metrics_and_rewards_are_not_required(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            history = [
                {
                    "step": step,
                    "eval/avg_episode_length": 400.0,
                    "eval/episode_failed": 0.2,
                    "eval/episode_timeout": 0.8,
                    "eval/episode_failure_nonfinite": 0.0,
                    "eval/episode_failure_root_low": 0.1,
                    "eval/episode_failure_foot_gap": 0.1,
                    "eval/avg_roll_progress_rad": 0.04,
                    "eval/avg_forbidden_penetration_m": 0.00001,
                }
                for step in (100, 200, 300)
            ]
            (output_dir / "metrics_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )

            score = score_training(output_dir, episode_length=500)

        self.assertGreater(score["selection_score"], 0.0)
        self.assertAlmostEqual(score["survival_fraction"], 0.8)
        self.assertGreater(score["estimated_net_turns"], 2.0)

    def test_nonfinite_candidate_is_never_promoted(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            history = [
                {
                    "step": 100,
                    "eval/avg_episode_length": 500.0,
                    "eval/episode_failed": 1.0,
                    "eval/episode_failure_nonfinite": 1.0,
                    "eval/avg_roll_progress_rad": 1.0,
                    "eval/avg_forbidden_penetration_m": 0.0,
                }
            ]
            (output_dir / "metrics_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )

            score = score_training(output_dir, episode_length=500)

        self.assertTrue(score["rejected"])
        self.assertLess(score["selection_score"], -1000.0)


if __name__ == "__main__":
    unittest.main()
