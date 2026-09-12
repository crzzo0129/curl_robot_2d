"""No ROS or motors: run on the robot or cloud, not in the local simulator."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'neural_controller/scripts'))
from rolling_gamepad_mapping import rolling_command
from gamepad_sequence import Sequence


class MappingTests(unittest.TestCase):
    def test_speed_and_vy(self):
        for axis, vx in [(-1., .45), (0., .60), (1., .75)]:
            command = rolling_command([1., axis, -1., 0.])
            self.assertAlmostEqual(command[0], vx)
            self.assertEqual(command[1:], (0., 0.))

    def test_deadzone_and_yaw(self):
        for yaw in [-.10, 0., .10]:
            self.assertEqual(rolling_command([0., .05, 0., yaw]), (.6, 0., 0.))
        for yaw in [-1., -.5, -.1001, .1001, .5, 1.]:
            actual = rolling_command([0., 0., 0., yaw])[2]
            self.assertGreaterEqual(abs(actual), .02)
            self.assertLessEqual(abs(actual), .07000000001)
            self.assertGreater(actual * yaw, 0.)
        self.assertAlmostEqual(rolling_command([0., 0., 0., 1.])[2], .07)

    def test_invalid_axes(self):
        for axes in [[], [0., 0.], [0., float('nan'), 0., 0.],
                     [0., 0., 0., float('inf')], [0., 2., 0., 0.]]:
            with self.assertRaises(ValueError):
                rolling_command(axes)


class ButtonTests(unittest.TestCase):
    def sequence(self, state='rolling'):
        seq = Sequence([0, 3, 2, 10, 7], 1., expected_button_count=11, circle_index=1)
        seq.state = state
        return seq

    def buttons(self, *indices):
        b = [0] * 11
        for i in indices:
            b[i] = 1
        return b

    def test_connection_with_circle_held(self):
        seq = self.sequence()
        self.assertIsNone(seq.buttons(self.buttons(1)))
        self.assertIsNone(seq.buttons(self.buttons(1)))
        seq.buttons(self.buttons())
        self.assertEqual(seq.buttons(self.buttons(1)), 'continuous_roll')

    def test_only_request_from_startup_policy(self):
        for state in ['idle', 'walking', 'settling', 'starting', 'roll_requested',
                      'continuous_rolling', 'stand_requested', 'estop']:
            seq = self.sequence(state)
            seq.buttons(self.buttons())
            self.assertIsNone(seq.buttons(self.buttons(1)), state)

    def test_stop_priority(self):
        seq = self.sequence()
        seq.buttons(self.buttons())
        self.assertEqual(seq.buttons(self.buttons(1, 2)), 'stand')
        for state in ['rolling', 'roll_requested', 'continuous_rolling']:
            seq = self.sequence(state)
            seq.buttons(self.buttons())
            self.assertEqual(seq.buttons(self.buttons(1, 2, 10)), 'stop')

    def test_square_in_both_rolling_stages(self):
        for state in ['rolling', 'roll_requested', 'continuous_rolling']:
            seq = self.sequence(state)
            seq.buttons(self.buttons())
            self.assertEqual(seq.buttons(self.buttons(2)), 'stand')

    def test_invalid_layout_stops(self):
        seq = self.sequence('continuous_rolling')
        self.assertEqual(seq.buttons([0] * 10), 'stop')


if __name__ == '__main__':
    unittest.main()
