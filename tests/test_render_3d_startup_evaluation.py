from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.render_3d_startup_evaluation import _episode_trace


class StartupEvaluationRenderTest(unittest.TestCase):
    def test_extracts_episode_truncates_frozen_tail_and_marks_failure(self):
        qpos = np.zeros((5, 2, 19))
        qpos[0, :, 2] = .1
        qpos[1, :, 2] = .2
        qpos[2:, :, 2] = .3
        gate = np.arange(10).reshape(5, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_best_arrays.npz"
            np.savez(path, qpos_first_episodes=qpos,
                     gate_error_first_episodes=gate,
                     teacher_active_next_first_episodes=np.zeros((5, 2)),
                     failure_axis_tilt=np.array((1., 0.)))
            first = _episode_trace(path, 0)
            self.assertEqual(first["qpos"].shape, (3, 19))
            np.testing.assert_array_equal(first["gate_error"], (0, 2, 4))
            np.testing.assert_array_equal(first["failure_axis_tilt"], (0, 0, 1))
            second = _episode_trace(path, 1)
            self.assertNotIn("failure_axis_tilt", second)


if __name__ == "__main__":
    unittest.main()
