"""CPU-only checks against neural_controller's raw 36x20 sensor/action ABI."""

from dataclasses import replace
import json
import unittest

import numpy as np

from curl_robot_2d_mjx.config_transition_3d import (
    Transition3DConfig, validate_transition_config_3d,
)
from curl_robot_2d_mjx.deployment_transition_3d import (
    initial_transition_history_3d, transition_controller_frame_3d,
    push_transition_frame_3d, transition_controller_metadata_3d,
)
from curl_robot_2d_mjx.environment_transition_3d import TRANSITION_MODEL_PATH_3D
from curl_robot_2d_mjx.transition_initialization_3d import transition_target_ctrl_3d
from scripts.export_transition_rtneural import convert_transition
from scripts.export_rtneural import _activation, _run_layers


class TransitionObservationTests(unittest.TestCase):
    def test_single_frame_exact_fields(self):
        frame = transition_controller_frame_3d(
            np, angular_velocity_body=np.array([.1, .2, .3]),
            projected_gravity=np.array([.4, .5, -.6]),
            joint_position_offset=np.arange(12.) + 10,
            last_action=np.arange(12.) + 30,
        )
        self.assertEqual(frame.shape, (36,))
        np.testing.assert_array_equal(frame[:6], [.1, .2, .3, .4, .5, -.6])
        np.testing.assert_array_equal(frame[6:12], [0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(frame[12:24], np.arange(12.) + 10)
        np.testing.assert_array_equal(frame[24:], np.arange(12.) + 30)

    def test_cold_history_matches_cpp_activation(self):
        frame = np.zeros(36, dtype=np.float32)
        frame[5], frame[11] = -1, 1
        actual = initial_transition_history_3d(np)
        np.testing.assert_array_equal(actual, np.tile(frame, 20))
        self.assertEqual(actual.dtype, np.float32)

    def test_history_matches_cpp_rotate_then_store_output(self):
        # Independently reproduce the C++ mutable buffer: update first 24,
        # clip, infer, rotate right 36, store RAW output in first frame 24:36.
        cpp = initial_transition_history_3d(np)
        history = cpp.copy()
        previous_action = np.zeros(12, dtype=np.float32)
        rng = np.random.default_rng(3)
        for tick in range(25):
            sensors = rng.normal(size=24).astype(np.float32)
            sensors[6:12] = [0, 0, 0, 0, 0, 1]
            cpp[:24] = sensors
            cpp = np.clip(cpp, -100, 100)
            frame = transition_controller_frame_3d(
                np, angular_velocity_body=sensors[:3], projected_gravity=sensors[3:6],
                joint_position_offset=sensors[12:24], last_action=previous_action)
            history = push_transition_frame_3d(np, history, frame)
            np.testing.assert_array_equal(history, cpp, err_msg=f"tick {tick}")
            # Include outputs > 1: history stores raw, not targets or clipped a.
            previous_action = np.full(12, tick * .25, dtype=np.float32)
            cpp = np.roll(cpp, 36)
            cpp[24:36] = previous_action

    def test_clip_is_observation_limit_not_action_limit(self):
        frame = np.zeros(36)
        frame[0], frame[12], frame[24] = 120, -120, 2.5
        result = push_transition_frame_3d(np, initial_transition_history_3d(np), frame)
        np.testing.assert_array_equal(result[[0, 12, 24]], [100, -100, 2.5])

    def test_invalid_observation_limit_rejected(self):
        for limit in (0, -1, float("nan"), float("inf")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                validate_transition_config_3d(replace(Transition3DConfig(), observation_limit=limit))


class TransitionDeploymentTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        self.config = Transition3DConfig()
        self.metadata = transition_controller_metadata_3d(self.model, self.config)

    def test_metadata_matches_hardware_names_pose_and_rate(self):
        meta = self.metadata
        self.assertEqual(meta["observation_history"] * meta["single_observation_size"], 720)
        self.assertAlmostEqual(meta["policy_frequency_hz"], 52)
        self.assertEqual(meta["joint_names"], [f"leg_{leg}_{suffix}"
                         for leg in ("front_l", "front_r", "back_l", "back_r")
                         for suffix in (2, 1, 3)])
        np.testing.assert_allclose(meta["default_joint_pos"], [0, .9, 1.15] * 4)
        self.assertIsInstance(meta["kp"], float)
        self.assertIsInstance(meta["kd"], float)
        np.testing.assert_allclose(meta["kp"], 5)
        np.testing.assert_allclose(meta["kd"], .1)
        self.assertEqual(meta["activation"], "elu")
        self.assertEqual(meta["actor_output"], "tanh_location")
        json.dumps(meta, allow_nan=False)

    def test_nonuniform_gains_rejected_by_current_cpp_contract(self):
        self.model.actuator_gainprm[0, 0] = 6.0
        with self.assertRaisesRegex(ValueError, "uniform scalar kp/kd"):
            transition_controller_metadata_3d(self.model, self.config)

    def test_action_is_cpp_fixed_scale_affine_then_joint_clip(self):
        meta = self.metadata
        nominal = np.array(meta["default_joint_pos"])
        low, high = np.array(meta["joint_lower_limits"]), np.array(meta["joint_upper_limits"])
        scale = np.array(meta["action_scale"])
        for value in (-2.5, -.75, -.1, 0, .1, .75, 2.5):
            action = np.full(12, value)
            expected = np.clip(nominal + scale * action, low, high)
            actual = transition_target_ctrl_3d(np, action, nominal, low, high)
            np.testing.assert_allclose(actual, expected)

    def _checkpoint(self):
        rng = np.random.default_rng(19)
        kernels = [rng.normal(0, .05, size=(720, 16)).astype(np.float32),
                   rng.normal(0, .1, size=(16, 24)).astype(np.float32)]
        biases = [rng.normal(0, .1, size=k.shape[1]).astype(np.float32) for k in kernels]
        actor = {"params": {f"hidden_{i}": {"kernel": k, "bias": b}
                             for i, (k, b) in enumerate(zip(kernels, biases))}}
        normalizer = {
            "mean": {"state": rng.normal(size=720).astype(np.float32),
                     "privileged_state": np.full(86, 777)},
            "std": {"state": rng.uniform(.3, 2, size=720).astype(np.float32),
                    "privileged_state": np.full(86, 999)},
        }
        return (normalizer, actor, {"never_export_critic": np.ones(86)}), kernels, biases

    def test_export_matches_normalized_actor_and_excludes_critic(self):
        checkpoint, kernels, biases = self._checkpoint()
        document = convert_transition(checkpoint, self.metadata)
        observations = np.random.default_rng(2).normal(size=(32, 720)).astype(np.float32)
        norm = checkpoint[0]
        expected = (observations - norm["mean"]["state"]) / norm["std"]["state"]
        expected = _activation("elu", expected @ kernels[0] + biases[0])
        expected = np.tanh((expected @ kernels[1] + biases[1])[:, :12])
        actual = _run_layers(observations, document["layers"])
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(document["in_shape"], [1, 720])
        self.assertEqual(document["out_shape"], [1, 12])
        # Corrupt all privileged data: export must remain bit-for-bit equal.
        norm["mean"]["privileged_state"][:] = -123
        norm["std"]["privileged_state"][:] = 0
        changed = convert_transition((norm, checkpoint[1], None), self.metadata)
        self.assertEqual(document, changed)
        json.dumps(document, allow_nan=False)

    def test_export_rejects_old_actor_and_bad_metadata(self):
        checkpoint, _, _ = self._checkpoint()
        for override in ({"contract_version": "old"}, {"activation": "swish"},
                         {"observation_history": 1}, {"actor_output": "normal"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                convert_transition(checkpoint, {**self.metadata, **override})
        checkpoint[1]["params"]["hidden_0"]["kernel"] = np.zeros((66, 16))
        with self.assertRaisesRegex(ValueError, "720-input"):
            convert_transition(checkpoint, self.metadata)


if __name__ == "__main__":
    unittest.main()
