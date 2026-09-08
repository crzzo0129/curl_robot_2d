import math
import unittest
import numpy as np
from scripts.run_handcrafted_roll_to_stand import near_target, interpolate


class HandcraftedTransitionTest(unittest.TestCase):
    def test_nose_up_approaching_stand(self):
        self.assertTrue(near_target(math.radians(14), -3., 0., math.radians(15)))
        self.assertFalse(near_target(math.radians(16), -3., 0., math.radians(15)))
        self.assertFalse(near_target(math.radians(14), 3., 0., math.radians(15)))
        self.assertFalse(near_target(math.radians(-1), -3., 0., math.radians(15)))
        self.assertFalse(near_target(math.radians(5), 0., 0., math.radians(15)))

    def test_angle_wrap_does_not_trigger_upside_down(self):
        for pitch in (-179., 179., 90., -90.):
            self.assertFalse(near_target(math.radians(pitch), -3., 0., math.radians(15)))

    def test_interpolation_continuity_and_hold(self):
        start, stand = np.array([.1, 1.2]), np.array([.9, 1.15])
        np.testing.assert_array_equal(interpolate(start, stand, 0., .15), start)
        np.testing.assert_allclose(interpolate(start, stand, .075, .15), (start + stand) / 2)
        np.testing.assert_array_equal(interpolate(start, stand, .15, .15), stand)
        np.testing.assert_array_equal(interpolate(start, stand, 1., .15), stand)


if __name__ == "__main__":
    unittest.main()
