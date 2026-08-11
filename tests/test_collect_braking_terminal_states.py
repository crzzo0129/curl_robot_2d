import unittest

import numpy as np

from scripts.collect_braking_terminal_states import select_snapshot_indices


class BrakingTerminalCollectionTest(unittest.TestCase):
    def test_selects_clean_snapshots_per_requested_bin(self):
        phase_bins = np.asarray([0, 0, 0, 8, 8, 8])
        contacts = np.zeros((6, 7))
        contacts[0, 2] = 1
        contacts[4, 3] = 1
        selected = select_snapshot_indices(phase_bins, contacts, [0, 8], 2)
        self.assertEqual(selected, [(0, 1), (0, 2), (8, 3), (8, 5)])

    def test_rejects_missing_quota(self):
        with self.assertRaises(RuntimeError):
            select_snapshot_indices(
                np.asarray([0]), np.zeros((1, 7)), [0], samples_per_bin=2
            )


if __name__ == "__main__":
    unittest.main()
