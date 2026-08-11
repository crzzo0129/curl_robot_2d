import unittest

import numpy as np

from scripts.search_deploy_trajectory import balanced_subset, deployment_midpoint


class DeploySearchTest(unittest.TestCase):
    def test_balanced_subset_skips_invalid_rows_and_caps_each_bin(self):
        bins = np.asarray([0, 0, 0, 1, 1, 1])
        valid = np.asarray([True, False, True, True, True, True])
        selected = balanced_subset(bins, valid, per_bin=2)
        np.testing.assert_array_equal(selected, [0, 2, 3, 4])

    def test_sequential_midpoints_move_only_one_leg(self):
        capture = np.asarray([0.4, 1.4, 0.7, 1.7])
        park = np.zeros(4)
        np.testing.assert_allclose(
            deployment_midpoint("front_first", capture, park), [0, 0, 0.7, 1.7]
        )
        np.testing.assert_allclose(
            deployment_midpoint("rear_first", capture, park), [0.4, 1.4, 0, 0]
        )


if __name__ == "__main__":
    unittest.main()
