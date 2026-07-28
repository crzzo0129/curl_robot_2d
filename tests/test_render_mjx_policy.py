from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.render_mjx_policy import _frame_indices, _load_rollout


class RenderMJXPolicyTest(unittest.TestCase):
    def test_frame_indices_preserve_duration_and_final_state(self) -> None:
        indices = _frame_indices(500, control_dt=0.02, fps=20)

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 499)
        self.assertGreater(len(indices), 190)
        self.assertLess(len(indices), 210)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_load_rollout_validates_saved_arrays(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_rollout.npz"
            np.savez_compressed(
                path,
                qpos=np.zeros((5, 7)),
                reward=np.arange(5, dtype=float),
            )

            qpos, reward = _load_rollout(path)

        self.assertEqual(qpos.shape, (5, 7))
        np.testing.assert_array_equal(reward, np.arange(5, dtype=float))


if __name__ == "__main__":
    unittest.main()
