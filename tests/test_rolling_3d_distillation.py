from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from curl_robot_2d_mjx.deployment_rolling_3d import (
    CONTROLLER_JOINT_NAMES_3D,
    HARDWARE_CONTROLLER_JOINT_NAMES_3D,
    HARDWARE_IMU_PUBLISH_FREQUENCY_HZ_3D,
    HARDWARE_POLICY_FREQUENCY_HZ_3D,
    ROLLING_DEPLOY_OBSERVATION_HISTORY_3D,
    ROLLING_DEPLOY_OBSERVATION_SIZE_3D,
    ROLLING_DEPLOY_SINGLE_OBSERVATION_SIZE_3D,
    controller_action_to_effective_action_3d,
    effective_action_to_controller_action_3d,
    initial_rolling_deploy_history_3d,
    push_rolling_deploy_frame_3d,
    rolling_deploy_frame_3d,
    ROLLING_CONTROLLER_ACTION_MASK_3D,
)
from scripts import train_mjx_3d_roll_distillation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad.xml"
)


class Rolling3DDistillationContractTest(unittest.TestCase):
    def test_hardware_policy_and_imu_rates_are_not_confused(self):
        self.assertEqual(HARDWARE_POLICY_FREQUENCY_HZ_3D, 52.0)
        self.assertEqual(HARDWARE_IMU_PUBLISH_FREQUENCY_HZ_3D, 260.0)

    def test_controller_joint_order_maps_hardware_hip_abduction_knee(self):
        self.assertEqual(
            CONTROLLER_JOINT_NAMES_3D[:3],
            (
                "front_left_hip_abduction",
                "front_left_hip",
                "front_left_knee",
            ),
        )
        self.assertEqual(
            HARDWARE_CONTROLLER_JOINT_NAMES_3D[:3],
            ("leg_front_l_2", "leg_front_l_1", "leg_front_l_3"),
        )

    def test_effective_action_round_trip_and_locked_abduction(self):
        effective = np.arange(8, dtype=np.float32)

        controller = effective_action_to_controller_action_3d(
            np, effective
        )

        self.assertEqual(controller.shape, (12,))
        np.testing.assert_array_equal(controller[[0, 3, 6, 9]], 0.0)
        np.testing.assert_array_equal(
            controller_action_to_effective_action_3d(np, controller),
            effective,
        )
        np.testing.assert_array_equal(
            np.asarray(ROLLING_CONTROLLER_ACTION_MASK_3D)[[0, 3, 6, 9]],
            0.0,
        )

    def test_deploy_frame_matches_36_value_cpp_layout(self):
        frame = rolling_deploy_frame_3d(
            np,
            angular_velocity_body=np.asarray((1.0, 2.0, 3.0)),
            projected_gravity=np.asarray((4.0, 5.0, 6.0)),
            command=np.asarray((7.0, 8.0, 9.0)),
            desired_world_z=np.asarray((10.0, 11.0, 12.0)),
            joint_position_offset=np.arange(12, dtype=np.float64) + 20.0,
            last_action=np.arange(12, dtype=np.float64) + 40.0,
        )

        self.assertEqual(
            frame.shape, (ROLLING_DEPLOY_SINGLE_OBSERVATION_SIZE_3D,)
        )
        np.testing.assert_array_equal(frame[:12], np.arange(1.0, 13.0))
        np.testing.assert_array_equal(frame[12:24], np.arange(20.0, 32.0))
        np.testing.assert_array_equal(frame[24:36], np.arange(40.0, 52.0))

    def test_batched_deploy_frame_broadcasts_constant_channels(self):
        batch_size = 32
        frame = rolling_deploy_frame_3d(
            np,
            angular_velocity_body=np.zeros((batch_size, 3)),
            projected_gravity=np.zeros((batch_size, 3)),
            joint_position_offset=np.zeros((batch_size, 12)),
            last_action=np.zeros((batch_size, 12)),
        )

        self.assertEqual(frame.shape, (batch_size, 36))
        np.testing.assert_array_equal(frame[:, 6:9], 0.0)
        np.testing.assert_array_equal(
            frame[:, 9:12],
            np.broadcast_to((0.0, 0.0, 1.0), (batch_size, 3)),
        )

    def test_history_is_newest_first_and_matches_controller_startup(self):
        history = initial_rolling_deploy_history_3d(np)
        self.assertEqual(
            history.shape, (ROLLING_DEPLOY_OBSERVATION_SIZE_3D,)
        )
        reshaped = history.reshape(
            ROLLING_DEPLOY_OBSERVATION_HISTORY_3D,
            ROLLING_DEPLOY_SINGLE_OBSERVATION_SIZE_3D,
        )
        np.testing.assert_array_equal(reshaped[:, 5], -1.0)
        np.testing.assert_array_equal(reshaped[:, 11], 1.0)
        frame = np.arange(36, dtype=np.float32)

        pushed = push_rolling_deploy_frame_3d(np, history, frame)

        np.testing.assert_array_equal(pushed[:36], frame)
        np.testing.assert_array_equal(pushed[36:72], history[:36])

    def test_student_metadata_uses_compact_pose_and_12_motor_contract(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

        config = train_mjx_3d_roll_distillation.student_controller_config(
            model
        )

        self.assertEqual(config["observation_history"], 20)
        self.assertEqual(len(config["action_scale"]), 12)
        np.testing.assert_array_equal(
            np.asarray(config["action_scale"])[[0, 3, 6, 9]], 0.0
        )
        self.assertEqual(len(config["default_joint_pos"]), 12)
        np.testing.assert_allclose(
            np.asarray(config["default_joint_pos"])[[0, 3, 6, 9]], 0.0
        )
        np.testing.assert_allclose(
            np.asarray(config["default_joint_pos"])[[1, 4, 7, 10]],
            0.1108283051,
        )
        np.testing.assert_allclose(
            np.asarray(config["default_joint_pos"])[[2, 5, 8, 11]],
            0.9092586986,
        )
        self.assertEqual(config["kp"], 5.0)
        self.assertEqual(config["kd"], 0.1)

    def test_smoke_cli_keeps_50hz_simulation_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "params_best"
            teacher.write_bytes(b"placeholder")
            controller = root / "controller.json"
            controller.write_text("{}", encoding="utf-8")

            args = train_mjx_3d_roll_distillation.parse_args(
                [
                    str(teacher),
                    "--controller",
                    str(controller),
                    "--out",
                    str(root / "output"),
                    "--preset",
                    "smoke",
                ]
            )
            task = train_mjx_3d_roll_distillation._task(
                episode_length=args.episode_length
            )

        self.assertEqual(args.envs, 32)
        self.assertEqual(task.physics_timestep, 0.001)
        self.assertEqual(task.action_repeat, 20)
        self.assertAlmostEqual(task.control_timestep, 0.02)
        self.assertTrue(task.explicit_phase_observation)
        self.assertFalse(task.direct_effective_action)

        direct_task = train_mjx_3d_roll_distillation._task(
            episode_length=args.episode_length,
            direct_effective_action=True,
        )
        self.assertFalse(direct_task.explicit_phase_observation)
        self.assertTrue(direct_task.direct_effective_action)
        self.assertIsNone(direct_task.residual_pair_differential_scale)


if __name__ == "__main__":
    unittest.main()
