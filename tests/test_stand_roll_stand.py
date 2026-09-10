"""Regression checks for time/angle handoff and checkpoint provenance."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

from scripts.render_stand_roll_stand import load_assets, parse_args, pitch_from_quaternion, pitch_gate, lift_cem_targets


class HandoffTests(unittest.TestCase):
    def test_cem_preserves_abduction_calibration_and_mirrors_sagittal_targets(self):
        compact = (-.17, 0, 0, -.17, 0, 0, .17, 0, 0, .17, 0, 0)
        targets = lift_cem_targets((.2, .8, .3, .9), compact)
        self.assertEqual(targets, (-.17, .2, .8, -.17, .2, .8,
                                   .17, .3, .9, .17, .3, .9))

    def test_early_ninety_does_not_interrupt_five_seconds(self):
        for elapsed in (0., 1., 4.999):
            self.assertFalse(pitch_gate(elapsed, math.pi/2))
        self.assertTrue(pitch_gate(5., math.pi/2))

    def test_wait_for_correct_orientation_after_five_seconds(self):
        for angle in (0., -90., 180., 270., 80., 100.):
            self.assertFalse(pitch_gate(5.4, math.radians(angle)))
        self.assertTrue(pitch_gate(5.8, math.radians(90.5)))
        self.assertTrue(pitch_gate(6.7, math.radians(450.5)))

    def test_pitch_retains_full_rotation_and_quaternion_sign_invariance(self):
        for angle in (-179., -135., -90., 0., 90., 135., 179.):
            half = math.radians(-angle)/2
            q = (math.cos(half), 0., math.sin(half), 0.)
            for sign in (-1., 1.):
                self.assertAlmostEqual(math.degrees(pitch_from_quaternion(
                    tuple(sign*v for v in q))), angle)

    def test_cem_wait_cannot_be_configured_as_shorter_than_five_seconds(self):
        for value in ("4.99", "nan", "inf"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--cem-seconds", value])

    def test_all_missing_cloud_assets_are_reported_before_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(["--student-params", str(root/"student_params"),
                               "--stand-params", str(root/"params_final")])
            with self.assertRaises(FileNotFoundError) as error:
                load_assets(args)
            self.assertIn("student_params", str(error.exception))
            self.assertIn("params_final", str(error.exception))
            self.assertIn("student_source.json", str(error.exception))


if __name__ == "__main__":
    unittest.main()
