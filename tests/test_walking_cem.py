import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from curl_robot_2d.walking_cem import MODEL_PATH, GaitParameters, SineWalkingController, rollout
from scripts.walk_cem import DEFAULT_POLICY, load_policy, save_json


class WalkingCemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    def test_reset_and_hold_are_exact_stand(self):
        controller = SineWalkingController(self.model)
        data = mujoco.MjData(self.model)
        data.qpos[:] = 100
        controller.reset(data)
        np.testing.assert_array_equal(data.qpos, self.model.key_qpos[controller.key])
        np.testing.assert_array_equal(data.ctrl, self.model.key_ctrl[controller.key])
        np.testing.assert_array_equal(controller.targets(0.5, data), data.ctrl)
        self.assertLess(np.max(np.abs(controller.targets(0.500001, data) - data.ctrl)), 1e-8)

    def test_named_joint_mapping_and_limits(self):
        controller = SineWalkingController(self.model)
        # CAD qpos order deliberately differs from actuator/policy order.
        self.assertNotEqual(controller.qadr.ravel().tolist(), list(range(7, 19)))
        for t in np.linspace(0, 5, 200):
            targets = controller.targets(t)
            self.assertTrue(np.all(np.isfinite(targets)))
            self.assertTrue(np.all(targets >= controller.lower))
            self.assertTrue(np.all(targets <= controller.upper))

    def test_sine_walk_advances_without_falling(self):
        summary, trajectory = rollout(self.model, SineWalkingController(self.model), duration_s=6, record=True)
        self.assertTrue(summary["completed"], summary)
        self.assertGreater(summary["distance_x_m"], 0.25)
        self.assertLess(abs(summary["drift_y_m"]), 0.08)
        self.assertLess(summary["max_tilt_deg"], 15)
        self.assertEqual(summary["nonfoot_contact_fraction"], 0)
        np.testing.assert_array_equal(trajectory["qpos"][0], self.model.key_qpos[self.model.key("stand").id])
        self.assertEqual(len(trajectory["time"]), len(trajectory["ctrl"]))
        self.assertTrue(np.all(np.diff(trajectory["time"]) > 0))

    def test_invalid_parameters_and_duration(self):
        with self.assertRaises(ValueError):
            GaitParameters.from_vector(np.full(12, np.nan))
        with self.assertRaises(ValueError):
            SineWalkingController(self.model, ramp_s=0)
        with self.assertRaises(ValueError):
            rollout(self.model, SineWalkingController(self.model), duration_s=0.1)

    def test_saved_cem_controller_long_rollout(self):
        parameters, metadata = load_policy(DEFAULT_POLICY)
        controller = SineWalkingController(self.model, parameters,
            hold_s=metadata["hold_s"], ramp_s=metadata["ramp_s"],
            heading_feedback=metadata["heading_feedback"])
        summary, _ = rollout(self.model, controller, duration_s=30)
        self.assertTrue(summary["completed"], summary)
        self.assertGreater(summary["distance_x_m"], 6)
        self.assertLess(abs(summary["drift_y_m"]), 0.2)
        self.assertLess(summary["max_tilt_deg"], 12)
        self.assertEqual(summary["nonfoot_contact_fraction"], 0)
        self.assertTrue(all(v > 0 for v in summary["foot_swing_samples"].values()))

    def test_json_roundtrip(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            save_json(path, dict(format="rollingquad-sine-cem-v1", parameters=asdict(GaitParameters())))
            parameters, _ = load_policy(path)
            np.testing.assert_array_equal(parameters.vector(), GaitParameters().vector())
            path.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(path)


if __name__ == "__main__":
    unittest.main()
