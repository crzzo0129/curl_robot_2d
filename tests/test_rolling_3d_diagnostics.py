from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.rolling_diagnostics_3d import (
    common_differential_action_3d,
    lateral_state_features_3d,
    save_lateral_trace,
    summarize_lateral_trace,
)
from scripts.train_mjx_3d_roll_distillation import parse_args


def sample_trace():
    shape = (3, 3)
    active = np.asarray([[True, True, True], [True, True, False], [True, False, False]])
    student = np.zeros(shape + (8,))
    student[..., 0] = 1.0
    student[..., 2] = -1.0
    student[~active] = 999.0
    student[1, 0] = 300.0  # invalid expert label: exclude from error metrics
    label_valid = active.copy()
    label_valid[1, 0] = False
    lateral_failed = np.asarray([[False, False, True], [False, True, True], [False, True, True]])
    return {
        "active": active,
        "time_s": np.broadcast_to(np.arange(3)[:, None] * 0.02, shape),
        "next_time_s": np.broadcast_to((np.arange(3)[:, None] + 1) * 0.02, shape),
        "y_m": np.zeros(shape),
        "vy_m_s": np.zeros(shape),
        "heading_rad": np.zeros(shape),
        "next_y_m": np.asarray([[0.01, 0.10, -0.21], [0.02, 0.21, 999], [0.03, 999, 999]]),
        "next_vy_m_s": np.zeros(shape),
        "next_heading_rad": np.zeros(shape),
        "student_action": student,
        "teacher_action": np.zeros_like(student),
        "teacher_label_valid": label_valid,
        "failed": lateral_failed.copy(),
        "lateral_failed": lateral_failed,
        "turns": np.broadcast_to(np.arange(3)[:, None] + 1.0, shape),
    }


class RollingDiagnosticsTest(unittest.TestCase):
    def test_common_and_differential_order_and_sign(self):
        common, differential = common_differential_action_3d(
            np, np.asarray([[3, 8, 1, 4, 7, 10, 5, 6]], dtype=float)
        )
        np.testing.assert_array_equal(common, [[2, 6, 6, 8]])
        np.testing.assert_array_equal(differential, [[1, 2, 1, 2]])

    def test_signed_world_state_and_heading_survive_pitch_rotation(self):
        heading = 0.17
        rotation_z = np.asarray([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0], [0, 0, 1],
        ])
        rotation_y = np.diag([-1.0, 1.0, -1.0])
        rotations = np.stack([rotation_z, rotation_z @ rotation_y])
        qpos = np.zeros((2, 7))
        qpos[:, 1] = [0.15, -0.15]
        qvel = np.zeros((2, 6))
        qvel[:, 1] = [0.02, -0.02]
        result = lateral_state_features_3d(np, qpos, qvel, rotations, np.asarray([0.1, -0.1]))
        np.testing.assert_allclose(result["y_m"], [0.05, -0.05])
        np.testing.assert_allclose(result["vy_m_s"], [0.02, -0.02])
        np.testing.assert_allclose(result["heading_rad"], heading)

    def test_terminal_sample_included_and_frozen_states_excluded(self):
        summary, episodes, time_rows = summarize_lateral_trace(sample_trace())
        self.assertEqual(summary["active_transitions"], 6)
        self.assertEqual(summary["valid_teacher_labels"], 5)
        self.assertEqual(summary["lateral_failure_positive_count"], 1)
        self.assertEqual(summary["lateral_failure_negative_count"], 1)
        self.assertAlmostEqual(summary["differential_error"]["rmse"], 0.5)
        self.assertAlmostEqual(summary["common_error"]["rmse"], 0.0)
        self.assertEqual(summary["differential_error"]["bias_by_channel"]["front_hip"], 1.0)
        self.assertEqual([row["active_episodes"] for row in time_rows], [3, 2, 1])
        self.assertEqual([row["steps"] for row in episodes], [3, 2, 1])
        self.assertEqual(episodes[1]["terminal_time_s"], 0.04)
        self.assertEqual(episodes[1]["final_y_m"], 0.21)
        self.assertEqual(episodes[2]["max_abs_y_m"], 0.21)
        self.assertEqual(summary["groups"]["failure_free"]["episodes"], 1)

    def test_output_reports_are_readable_and_raw_trace_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_lateral_trace(root, sample_trace())
            summary = json.loads((root / "lateral_diagnostics.json").read_text())
            self.assertEqual(summary["episodes"], 3)
            for name in ("lateral_episodes.csv", "lateral_timeseries.csv"):
                with (root / name).open(newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 3)
            with np.load(root / "lateral_trace.npz") as archive:
                self.assertEqual(archive["student_action"].shape, (3, 3, 8))
                self.assertEqual(archive["differential_error"].shape, (3, 3, 4))
                self.assertEqual(archive["active"].sum(), 6)

    def test_eval_only_requires_checkpoint_and_enables_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            student = root / "student"
            controller = root / "controller.json"
            for path in (teacher, student, controller):
                path.write_bytes(b"placeholder")
            argv = [str(teacher), "--controller", str(controller), "--out", str(root / "out"), "--eval-only"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
            args = parse_args(argv + ["--restore-student", str(student), "--eval-seed", "7"])
            self.assertTrue(args.eval_only)
            self.assertTrue(args.record_diagnostics)
            self.assertEqual(args.eval_seed, 7)
            self.assertEqual(student.read_bytes(), b"placeholder")


if __name__ == "__main__":
    unittest.main()
