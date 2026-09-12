import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.steering_calibration import (
    calibrated_steering_amplitude, calibrated_steering_prior, load_steering_calibration,
    text_sha256,
)
from curl_robot_2d_mjx.environment_3d import rolling_axis_stability_tilt_3d, model_path_3d
from curl_robot_2d_mjx.cem_reference import load_cem_reference
from scripts.calibrate_cem_steering import TASK, CONTROLLER

ASSET = Path(__file__).resolve().parents[1] / 'assets/controllers/rollingquad_abd10_high_speed_steering_v1.json'


class SteeringCalibrationTests(unittest.TestCase):
    def test_batched_jit_lookup_and_no_double_scaling(self):
        import jax
        import jax.numpy as jp
        table = dict(speeds_m_s=[.4, .8], yaw_commands_rad_s=[-.08, 0., .08],
                     offsets=[[-.06, .002, .07], [-.02, .001, .025]])
        speed = np.array([.4, .6, .8, .2, 1.])
        yaw = np.array([.08, .04, -.08, .2, -.2])
        expected = np.array([.07, .0245, -.02, .07, -.02])
        np.testing.assert_allclose(calibrated_steering_amplitude(np, table, speed, yaw), expected)
        result = jax.jit(lambda v, w: calibrated_steering_prior(jp, table, v, w))(speed, yaw)
        np.testing.assert_allclose(result[:, 0], expected, rtol=1e-6)
        np.testing.assert_allclose(result[:, 2], -expected, rtol=1e-6)
        self.assertEqual(result.shape, (5, 8))

    def test_heading_is_not_a_fall_in_calibrated_turns(self):
        heading, elevation = .8, .1
        axis = np.array([-np.sin(heading)*np.cos(elevation),
                         np.cos(heading)*np.cos(elevation), np.sin(elevation)])
        self.assertGreater(rolling_axis_stability_tilt_3d(np, axis, .08), .5)
        self.assertAlmostEqual(float(rolling_axis_stability_tilt_3d(np, axis, .08, calibrated=True)), .1)
        self.assertGreater(rolling_axis_stability_tilt_3d(np, axis, 0., calibrated=True), .5)
        tilted = np.array([0., np.cos(.6), np.sin(.6)])
        self.assertGreater(rolling_axis_stability_tilt_3d(np, tilted, -.08, calibrated=True), .5)

    def test_asset_provenance_and_domain_guards(self):
        task = replace(TASK, forward_command_enabled=True, forward_command_min_m_s=.4111,
                       forward_command_max_m_s=.811, turn_command_enabled=True)
        reference = load_cem_reference(CONTROLLER, minimum_residual_gain=.15)
        table = load_steering_calibration(ASSET, task=task, reference=reference, model_path=model_path_3d(task.geometry))
        self.assertEqual(table['validation_status'], 'passed_cpu_holdout')
        for bad_task in (replace(task, forward_command_max_m_s=.9),
                         replace(task, turn_command_max_rad_s=.1),
                         replace(task, self_collision_enabled=True),
                         replace(task, body_mass_scale=1.1)):
            with self.assertRaises(ValueError):
                load_steering_calibration(ASSET, task=bad_task, reference=reference, model_path=model_path_3d(task.geometry))
        with self.assertRaisesRegex(ValueError, 'reference override'):
            load_steering_calibration(ASSET, task=task, reference=replace(reference, oscillator_coupling_per_s=2.), model_path=model_path_3d(task.geometry))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'
            bad = copy.deepcopy(table)
            bad['provenance']['controller_text_sha256'] = '0'*64
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_steering_calibration(path, task=task, reference=reference, model_path=model_path_3d(task.geometry))

    def test_fingerprint_survives_git_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            a.write_bytes(b'{\n"x": 1\n}\n')
            b.write_bytes(b'{\r\n"x": 1\r\n}\r\n')
            self.assertEqual(text_sha256(a), text_sha256(b))
            b.write_bytes(b'{\n"x": 2\n}\n')
            self.assertNotEqual(text_sha256(a), text_sha256(b))


if __name__ == '__main__':
    unittest.main()
