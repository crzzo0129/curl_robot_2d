import unittest

import numpy as np

from scripts.search_braking_schedule import BrakingSchedule, schedule_scales


class BrakingScheduleTest(unittest.TestCase):
    def test_schedule_starts_at_active_roll_and_reaches_terminal_values(self):
        schedule = BrakingSchedule(
            1.2, 0.5, 0.1, 0.25,
            (0.05, -0.1, 0.15, -0.2),
            (0.1, -0.2, 0.3, -0.4),
        )
        rate, amplitude, offsets = schedule_scales(schedule, 0.0, 0.6)
        self.assertAlmostEqual(rate, 0.6)
        self.assertAlmostEqual(amplitude, 1.0)
        np.testing.assert_allclose(offsets, 0.0)
        rate, amplitude, offsets = schedule_scales(schedule, 1.0, 0.6)
        self.assertAlmostEqual(rate, 0.1)
        self.assertAlmostEqual(amplitude, 0.25)
        np.testing.assert_allclose(offsets, [0.1, -0.2, 0.3, -0.4])


if __name__ == "__main__":
    unittest.main()
