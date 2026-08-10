import unittest

import numpy as np

from scripts.search_deploy_trajectory import balanced_subset


class DeploySearchTest(unittest.TestCase):
    def test_balanced_subset_skips_invalid_rows_and_caps_each_bin(self):
        bins = np.asarray([0, 0, 0, 1, 1, 1])
        valid = np.asarray([True, False, True, True, True, True])
        selected = balanced_subset(bins, valid, per_bin=2)
        np.testing.assert_array_equal(selected, [0, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
