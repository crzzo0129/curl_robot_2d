from dataclasses import asdict
import unittest

import numpy as np

from curl_robot_2d_mjx.config_3d import Rolling3DConfig
from curl_robot_2d_mjx.handoff_probe_3d import FAILURES, perturbation_batch
from scripts.probe_3d_compact_coldstart import aggregate, initial_offsets, report_rows, sweep_cases


class CompactColdstartTest(unittest.TestCase):
    def test_exact_and_training_distribution(self):
        task = Rolling3DConfig()
        cases = sweep_cases(task)
        self.assertTrue(all(v == 0 for v in asdict(cases["exact"]).values()))
        noise = cases["training_noise"]
        self.assertEqual(noise.joint_position_rad, task.reset_joint_noise_rad)
        self.assertEqual(noise.joint_velocity_rad_s, task.reset_velocity_noise)
        self.assertEqual(noise.root_linear_velocity_m_s, task.reset_root_velocity_noise)
        self.assertEqual(noise.root_angular_velocity_rad_s, task.reset_root_velocity_noise)
        self.assertEqual(noise.oscillator_phase_rad, 0)
        self.assertEqual(noise.previous_action_normalized, 0)

    def test_velocity_sweep_is_paired_and_does_not_change_history_or_pose_noise(self):
        cases = sweep_cases(Rolling3DConfig())
        low = perturbation_batch("state_noise", cases["joint_002"], 31, 16)
        high = perturbation_batch("state_noise", cases["joint_010"], 31, 16)
        np.testing.assert_allclose(low["dqd"] * 5, high["dqd"], atol=1e-8)
        np.testing.assert_array_equal(low["dq"], high["dq"])
        for key in ("dv", "daxis", "dphase", "dhistory"):
            np.testing.assert_array_equal(low[key], 0)

    def test_individual_root_components_are_isolated(self):
        cases = sweep_cases(Rolling3DConfig())
        linear = perturbation_batch("state_noise", cases["linear_003"], 31, 16)
        angular = perturbation_batch("state_noise", cases["angular_010"], 31, 16)
        self.assertTrue(np.any(linear["dv"][:, :3] != 0))
        self.assertTrue(np.any(angular["dv"][:, 3:] != 0))
        np.testing.assert_array_equal(linear["dv"][:, 3:], 0)
        np.testing.assert_array_equal(angular["dv"][:, :3], 0)

    def features(self):
        first = {"qpos": np.zeros((2, 19)), "radius": np.full(2, .1275),
                 "absolute_rotation": np.zeros(2), "rolling_phase": np.zeros(2),
                 "time": np.zeros(2), "y": np.zeros(2), "failed": np.zeros(2, dtype=bool),
                 **{f"failure_{k}": np.zeros(2, dtype=bool) for k in FAILURES}}
        return first, {k: v.copy() for k, v in first.items()}

    def test_signed_x_velocity_is_fixed_and_all_other_initial_conditions_are_paired(self):
        cases = sweep_cases(Rolling3DConfig())
        neg = initial_offsets("vx_neg_003", cases["vx_neg_003"], 34, 4)
        pos = initial_offsets("vx_pos_003", cases["vx_pos_003"], 34, 4)
        np.testing.assert_allclose(neg["dv"][:, 0], -.03)
        np.testing.assert_allclose(pos["dv"][:, 0], .03)
        np.testing.assert_array_equal(neg["dv"][:, 1:], 0)
        np.testing.assert_array_equal(pos["dv"][:, 1:], 0)
        for key in ("dq", "dqd", "daxis", "dphase", "dhistory"):
            np.testing.assert_array_equal(neg[key], pos[key])

    def test_survival_and_success_are_distinct_and_require_signed_forward_turns(self):
        first, current = self.features()
        current["time"][:] = 10
        current["qpos"][:, 0] = 2 * np.pi * .1275 * 6
        current["absolute_rotation"][:] = 2 * np.pi * 6
        current["rolling_phase"][:] = (2 * np.pi * 6, -2 * np.pi * 6)
        rows = report_rows(first, current, np.zeros(2), np.zeros(2), "test", 10, .02, 5)
        self.assertTrue(rows[0]["success"])
        self.assertFalse(rows[1]["success"])
        self.assertTrue(rows[1]["failure_free"])
        summary = aggregate(rows)[0]
        self.assertEqual(summary["success_rate"], .5)
        self.assertEqual(summary["failure_free_rate"], 1)
        current["qpos"][:, 0] = 0  # Rotation alone cannot pass.
        self.assertFalse(report_rows(first, current, np.zeros(2), np.zeros(2), "test", 10, .02, 5)[0]["success"])

    def test_stopped_or_failed_episodes_cannot_pass_full_horizon(self):
        first, current = self.features()
        current["time"][:] = (9, 10)
        current["failed"][1] = True
        current["failure_lateral_drift"][1] = True
        rows = report_rows(first, current, np.zeros(2), np.zeros(2), "test", 10, .02, 0)
        self.assertFalse(any(r["failure_free"] for r in rows))
        self.assertEqual(aggregate(rows)[0]["failure_rates"]["lateral_drift"], .5)


if __name__ == "__main__":
    unittest.main()
