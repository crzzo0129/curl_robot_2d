import math
import unittest

from scripts.collect_rolling_snapshots import PhaseBalancedBuffer, phase_bin_index


class RollingSnapshotCollectionTest(unittest.TestCase):
    def test_phase_bins_wrap_at_two_pi(self):
        self.assertEqual(phase_bin_index(0.0, 8), 0)
        self.assertEqual(phase_bin_index(2.0 * math.pi, 8), 0)
        self.assertEqual(phase_bin_index(-0.01, 8), 7)
        self.assertEqual(phase_bin_index(math.pi, 8), 4)

    def test_buffer_enforces_equal_per_bin_quota(self):
        buffer = PhaseBalancedBuffer(bin_count=2, samples_per_bin=2)
        self.assertTrue(buffer.add(0.1, {"value": 1}))
        self.assertTrue(buffer.add(0.2, {"value": 2}))
        self.assertFalse(buffer.add(0.3, {"value": 3}))
        self.assertTrue(buffer.add(math.pi + 0.1, {"value": 4}))
        self.assertTrue(buffer.add(math.pi + 0.2, {"value": 5}))
        self.assertTrue(buffer.complete)


if __name__ == "__main__":
    unittest.main()
