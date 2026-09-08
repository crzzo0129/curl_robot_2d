from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.config_stand_to_roll import (
    STAND_TO_ROLL_CURRICULUM_STAGES,
    StandToRollConfig,
    stand_to_roll_curriculum_config,
    validate_stand_to_roll_config,
)
from curl_robot_2d_mjx.stand_to_roll_training import (
    action_center_and_scale,
    build_cem_bc_dataset,
    initialize_ppo_actor_from_bc,
    observation_normalizer,
)
from scripts.train_mjx_3d_stand_to_roll import parse_args


class StandToRollContractTest(unittest.TestCase):
    def test_curriculum_reaches_exact_full_stand_without_reward_changes(self):
        base = StandToRollConfig()
        configs = [stand_to_roll_curriculum_config(name, base)
                   for name in STAND_TO_ROLL_CURRICULUM_STAGES]
        self.assertEqual((configs[-1].reset_alpha_min, configs[-1].reset_alpha_max),
                         (1.0, 1.0))
        self.assertTrue(all(c.reward_cem_progress == base.reward_cem_progress
                            for c in configs))
        self.assertTrue(all(c.reward_compact_progress == base.reward_compact_progress
                            for c in configs))
        self.assertTrue(all(c.reward_capture_bonus == base.reward_capture_bonus
                            for c in configs))

    def test_action_range_is_not_walking_deploy_range(self):
        center, scale = action_center_and_scale(StandToRollConfig())
        np.testing.assert_allclose(center[[0, 3, 6, 9]], 0.0)
        np.testing.assert_allclose(scale[[1, 4, 7, 10]], 0.8)
        np.testing.assert_allclose(scale[[2, 5, 8, 11]], 1.2)
        stand = np.asarray((0.0, 0.9, 1.15) * 4)
        normalized_stand = (stand - center) / scale
        self.assertLessEqual(np.max(np.abs(normalized_stand)), 1.0)

    def test_invalid_reset_bounds_rejected(self):
        with self.assertRaises(ValueError):
            validate_stand_to_roll_config(
                StandToRollConfig(reset_alpha_min=0.8, reset_alpha_max=0.2)
            )

    def test_cli_requires_restore_after_compact(self):
        with self.assertRaises(SystemExit):
            parse_args(["--stage", "crouch", "--dry-run"])
        args = parse_args(["--stage", "compact", "--dry-run"])
        self.assertEqual(args.stage, "compact")


class CEMBehaviorCloningDatasetTest(unittest.TestCase):
    def test_deploy_history_is_newest_first_and_720_wide(self):
        n = 24
        qpos = np.zeros((n, 19), dtype=np.float64)
        indices = np.arange(7, 19)
        for i in range(n):
            qpos[i, indices] = i
        orientation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        angular = np.column_stack((np.arange(n), np.zeros(n), np.zeros(n)))
        center = np.zeros(12)
        scale = np.ones(12)
        targets = np.linspace(-0.5, 0.5, n)[:, None] * np.ones((1, 12))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cem.npz"
            np.savez_compressed(
                path,
                qpos=qpos,
                orientation=orientation,
                angular_velocity=angular,
                joint_target=targets,
            )
            observations, actions = build_cem_bc_dataset(
                path,
                controller_qpos_indices=indices,
                action_center=center,
                action_scale=scale,
            )
        self.assertEqual(observations.shape, (5, 720))
        self.assertEqual(actions.shape, (5, 12))
        frames = observations[0].reshape(20, 36)
        self.assertEqual(frames[0, 0], 19.0)
        self.assertEqual(frames[-1, 0], 0.0)
        self.assertEqual(frames[0, 12], 19.0)
        self.assertEqual(frames[-1, 12], 0.0)

    def test_normalizer_has_positive_floor(self):
        norm = observation_normalizer(np.zeros((4, 720), dtype=np.float32))
        self.assertTrue(np.all(norm["std"] == 1.0e-3))

    def test_bc_actor_copies_into_location_half_of_ppo_head(self):
        hidden = (4, 3, 2)
        bc = {"params": {
            "hidden_0": {"kernel": np.ones((720, 4)), "bias": np.ones(4)},
            "hidden_1": {"kernel": np.ones((4, 3)), "bias": np.ones(3)},
            "hidden_2": {"kernel": np.ones((3, 2)), "bias": np.ones(2)},
            "location": {"kernel": np.ones((2, 12)), "bias": np.arange(12)},
        }}
        ppo = {"params": {
            "hidden_0": {"kernel": np.zeros((720, 4)), "bias": np.zeros(4)},
            "hidden_1": {"kernel": np.zeros((4, 3)), "bias": np.zeros(3)},
            "hidden_2": {"kernel": np.zeros((3, 2)), "bias": np.zeros(2)},
            "hidden_3": {"kernel": np.zeros((2, 24)), "bias": np.zeros(24)},
        }}
        result = initialize_ppo_actor_from_bc(
            np, ppo, bc, hidden_layers=hidden, initial_std=0.05
        )
        np.testing.assert_array_equal(
            result["params"]["hidden_3"]["kernel"][:, :12], 1.0
        )
        np.testing.assert_array_equal(
            result["params"]["hidden_3"]["kernel"][:, 12:], 0.0
        )
        np.testing.assert_array_equal(
            result["params"]["hidden_3"]["bias"][:12], np.arange(12)
        )


if __name__ == "__main__":
    unittest.main()
