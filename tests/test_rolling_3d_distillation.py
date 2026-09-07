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
from scripts.export_rtneural import convert as convert_rtneural
from curl_robot_2d_mjx.randomization_3d import (
    RollingStudentDeployDomainRandomization,
    validate_student_deploy_domain_randomization_3d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad.xml"
)


class Rolling3DDistillationContractTest(unittest.TestCase):
    def test_deploy_dr_curriculum_contracts_toward_nominal(self):
        full = RollingStudentDeployDomainRandomization()
        quarter = full.scaled(0.25)

        self.assertEqual(quarter.sliding_friction, (0.9, 1.1))
        self.assertEqual(quarter.motor_kp_scale, (0.9625, 1.0375))
        self.assertEqual(quarter.motor_kd_scale, (0.95, 1.05))
        self.assertEqual(
            quarter.action_latency_probabilities,
            (0.9, 0.075, 0.025),
        )
        self.assertAlmostEqual(
            quarter.control_deadline_miss_probability, 0.0125
        )
        self.assertAlmostEqual(quarter.motor_zero_bias_rad, 0.005)
        self.assertAlmostEqual(quarter.encoder_fixed_bias_rad, 0.0025)
        validate_student_deploy_domain_randomization_3d(quarter)

    def test_dagger_teacher_probability_reaches_both_endpoints(self):
        probability = (
            train_mjx_3d_roll_distillation.dagger_teacher_probability
        )
        self.assertEqual(probability(0, 10_000, 0.25, 0.0), 0.25)
        self.assertEqual(probability(9_999, 10_000, 0.25, 0.0), 0.0)
        self.assertEqual(probability(0, 1, 0.25, 0.0), 0.25)

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
        # rollingquad.xml bakes front -15 deg / rear +15 deg abduction into the
        # compact keyframe; the four abduction action scales remain locked at
        # zero so the deployable student holds this offset without commanding it.
        np.testing.assert_allclose(
            np.asarray(config["default_joint_pos"])[[0, 3, 6, 9]],
            np.asarray((-0.2617993878, -0.2617993878, 0.2617993878, 0.2617993878)),
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
        self.assertEqual(args.dagger_steps, 16)
        self.assertEqual(task.physics_timestep, 0.001)
        self.assertEqual(task.action_repeat, 20)
        self.assertAlmostEqual(task.control_timestep, 0.02)
        self.assertTrue(task.explicit_phase_observation)
        self.assertFalse(task.direct_effective_action)

        direct_task = train_mjx_3d_roll_distillation._task(
            episode_length=args.episode_length,
            direct_effective_action=True,
        )
        self.assertTrue(direct_task.explicit_phase_observation)
        self.assertTrue(direct_task.direct_effective_action)
        self.assertIsNone(direct_task.residual_pair_differential_scale)

    def test_primitive_geometry_selects_matching_reference_and_lateral_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "params_best"
            teacher.write_bytes(b"placeholder")
            args = train_mjx_3d_roll_distillation.parse_args(
                [
                    str(teacher),
                    "--geometry",
                    "rollingquad_2_primitive",
                    "--lateral-drift-diagnostic-only",
                    "--out",
                    str(root / "output"),
                ]
            )
            task = train_mjx_3d_roll_distillation._task(
                episode_length=args.episode_length,
                geometry=args.geometry,
                lateral_drift_diagnostic_only=(
                    args.lateral_drift_diagnostic_only
                ),
            )

        self.assertEqual(args.geometry, "rollingquad_2_primitive")
        self.assertIn("rollingquad_primitive_stiff_cem", str(args.controller))
        self.assertEqual(task.geometry, "rollingquad_2_primitive")
        self.assertFalse(task.lateral_drift_termination)

    def test_restore_student_must_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "params_best"
            teacher.write_bytes(b"placeholder")
            controller = root / "controller.json"
            controller.write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit):
                train_mjx_3d_roll_distillation.parse_args(
                    [
                        str(teacher),
                        "--restore-student",
                        str(root / "missing_student"),
                        "--controller",
                        str(controller),
                        "--out",
                        str(root / "output"),
                    ]
                )

    def test_deploy_dr_requires_and_reuses_existing_student(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "params_best"
            teacher.write_bytes(b"placeholder")
            student = root / "student_params"
            student.write_bytes(b"placeholder")
            controller = root / "controller.json"
            controller.write_text("{}", encoding="utf-8")
            base = [
                str(teacher),
                "--controller",
                str(controller),
                "--out",
                str(root / "output"),
                "--deploy-dr",
            ]

            with self.assertRaises(SystemExit):
                train_mjx_3d_roll_distillation.parse_args(base)
            args = train_mjx_3d_roll_distillation.parse_args(
                base
                + [
                    "--restore-student",
                    str(student),
                    "--deploy-dr-strength",
                    "0.25",
                ]
            )

        self.assertTrue(args.deploy_dr)
        self.assertEqual(args.deploy_dr_strength, 0.25)


def _dense(rng, inputs, outputs):
    return {
        "kernel": rng.normal(size=(inputs, outputs)).astype(np.float32),
        "bias": rng.normal(size=(outputs,)).astype(np.float32),
    }


class Rolling3DVelocityEstimationContractTest(unittest.TestCase):
    def test_privileged_velocity_slice_matches_mirror_contract(self):
        self.assertEqual(
            train_mjx_3d_roll_distillation.PRIVILEGED_BASE_VELOCITY_INDICES_3D,
            (9, 10, 11),
        )
        observation = np.arange(65, dtype=np.float32)
        velocity = train_mjx_3d_roll_distillation.privileged_base_velocity_3d(
            observation
        )

        self.assertEqual(velocity.shape, (3,))
        np.testing.assert_array_equal(velocity, observation[9:12])
        batched = train_mjx_3d_roll_distillation.privileged_base_velocity_3d(
            observation[np.newaxis, :]
        )
        np.testing.assert_array_equal(batched, observation[np.newaxis, 9:12])

    def test_action_only_params_strips_velocity_head(self):
        rng = np.random.default_rng(11)
        params = {
            "params": {
                "hidden_0": _dense(rng, 5, 4),
                "location": _dense(rng, 4, 12),
                "velocity_estimator": _dense(rng, 4, 3),
            }
        }

        stripped = train_mjx_3d_roll_distillation.action_only_student_params(
            params
        )

        self.assertNotIn("velocity_estimator", stripped["params"])
        self.assertIn("location", stripped["params"])
        self.assertIn("hidden_0", stripped["params"])
        # The original dict is left untouched.
        self.assertIn("velocity_estimator", params["params"])

    def test_velocity_head_would_break_export_chain_unless_stripped(self):
        rng = np.random.default_rng(13)
        action_scale = [1.0] * 12
        mean = np.zeros(5, dtype=np.float32)
        std = np.ones(5, dtype=np.float32)

        with_head = {
            "params": {
                "hidden_0": _dense(rng, 5, 4),
                "location": _dense(rng, 4, 12),
                "velocity_estimator": _dense(rng, 4, 3),
            }
        }
        with self.assertRaises(ValueError):
            convert_rtneural(
                ({"mean": mean, "std": std}, with_head, {}),
                {"action_scale": action_scale},
                activation="elu",
            )

        stripped = train_mjx_3d_roll_distillation.action_only_student_params(
            with_head
        )
        document = convert_rtneural(
            ({"mean": mean, "std": std}, stripped, {}),
            {"action_scale": action_scale},
            activation="elu",
        )
        self.assertEqual(document["in_shape"], [1, 5])
        self.assertEqual(document["out_shape"], [1, 12])

    def test_velocity_loss_weight_cli_defaults_and_overrides(self):
        # parse_args only checks that the teacher/controller paths exist; the
        # real XML model stands in for both so the test needs no temp directory.
        out = MODEL_PATH.parent / "unused_velocity_cli_output"
        base = [
            str(MODEL_PATH),
            "--controller",
            str(MODEL_PATH),
            "--out",
            str(out),
        ]
        default_args = train_mjx_3d_roll_distillation.parse_args(base)
        self.assertAlmostEqual(default_args.velocity_loss_weight, 0.2)

        weighted = train_mjx_3d_roll_distillation.parse_args(
            base + ["--velocity-loss-weight", "0.0"]
        )
        self.assertEqual(weighted.velocity_loss_weight, 0.0)

        with self.assertRaises(SystemExit):
            train_mjx_3d_roll_distillation.parse_args(
                base + ["--velocity-loss-weight", "-1.0"]
            )

    def test_terrain_cli_and_task_config(self):
        out = MODEL_PATH.parent / "unused_terrain_cli_output"
        base = [
            str(MODEL_PATH),
            "--controller",
            str(MODEL_PATH),
            "--out",
            str(out),
        ]
        args = train_mjx_3d_roll_distillation.parse_args(base)
        self.assertFalse(args.terrain_enabled)
        self.assertAlmostEqual(args.terrain_slope_probability, 0.30)
        self.assertAlmostEqual(args.terrain_max_angle_deg, 2.0)

        terrain_args = train_mjx_3d_roll_distillation.parse_args(
            base + ["--terrain-enabled", "--terrain-max-angle-deg", "4.0"]
        )
        self.assertTrue(terrain_args.terrain_enabled)
        self.assertAlmostEqual(terrain_args.terrain_max_angle_deg, 4.0)

        task = train_mjx_3d_roll_distillation._task(
            episode_length=500, terrain_enabled=True
        )
        self.assertTrue(task.terrain_enabled)
        self.assertEqual(task.terrain_slope_angle_deg, 0.0)

        with self.assertRaises(SystemExit):
            train_mjx_3d_roll_distillation.parse_args(
                base + ["--terrain-enabled", "--terrain-slope-probability", "0.0"]
            )


if __name__ == "__main__":
    unittest.main()
