import math
import unittest

import numpy as np

from curl_robot_2d_mjx.stop_evaluation import (
    ParkPoseStaticGate,
    ParkPoseStaticMetrics,
    StopEpisodeMetrics,
    StopEvaluationGate,
    park_pose_failure_reasons,
    park_pose_succeeded,
    stop_episode_succeeded,
    stop_failure_reasons,
)
from curl_robot_2d_mjx.stop_task import (
    ParkPose,
    StopMode,
    StopReferenceSchedule,
    StopState,
    StopTaskConfig,
    StopTransitionInput,
    advance_stop_state,
    blend_joint_reference,
    forward_phase_delta,
    reference_schedule,
    required_braking_phase_distance,
    select_reachable_target_phase_unwrapped,
    select_target_phase_unwrapped,
    stop_observation_features,
    stop_succeeded,
    wrap_to_pi,
)
from curl_robot_2d_mjx.deploy_trajectory import deploy_trajectory_sample


class StopTaskTest(unittest.TestCase):
    def setUp(self):
        self.pose = ParkPose(
            joint_targets_rad=(0.2, 0.4),
            foot_down_phase_rad=0.1,
            required_grounded_feet=2,
        )
        self.config = StopTaskConfig(
            maximum_brake_duration_s=2.0,
            deploy_duration_s=1.0,
            required_hold_duration_s=0.3,
            brake_phase_margin_rad=0.01,
        )

    def input(self, **overrides):
        values = {
            "time_s": 0.0,
            "body_phase_unwrapped_rad": 0.0,
            "stop_command": False,
            "linear_speed_m_s": 0.0,
            "angular_speed_rad_s": 0.0,
            "joint_pose_rms_error_rad": 0.0,
            "root_pitch_error_rad": 0.0,
            "grounded_feet": 2,
        }
        values.update(overrides)
        return StopTransitionInput(**values)

    def test_phase_wrap_and_forward_target_cross_boundary(self):
        self.assertAlmostEqual(wrap_to_pi(3.0 * math.pi), -math.pi)
        distance = forward_phase_delta(2.0 * math.pi - 0.05, 0.10, 1.0)
        self.assertAlmostEqual(distance, 0.15)
        target, target_distance = select_target_phase_unwrapped(
            2.0 * math.pi - 0.05, 0.10, 1.0
        )
        self.assertAlmostEqual(target, 2.0 * math.pi + 0.10)
        self.assertAlmostEqual(target_distance, 0.15)
        self.assertAlmostEqual(forward_phase_delta(0.05, -0.10, -1.0), 0.15)

    def test_reachable_phase_skips_an_unsafe_near_window(self):
        required = required_braking_phase_distance(4.0, 2.0, 0.2)
        self.assertAlmostEqual(required, 4.2)
        target, distance = select_reachable_target_phase_unwrapped(
            0.0, 0.1, required, 1.0
        )
        self.assertAlmostEqual(target, 2.0 * math.pi + 0.1)
        self.assertAlmostEqual(distance, 2.0 * math.pi + 0.1)

    def test_state_machine_reaches_hold_and_requires_continuous_settling(self):
        state = StopState()
        state = advance_stop_state(
            state,
            self.input(
                time_s=1.0,
                body_phase_unwrapped_rad=2.0 * math.pi - 0.05,
                stop_command=True,
            ),
            self.pose,
            self.config,
            0.1,
        )
        self.assertEqual(state.mode, StopMode.BRAKE_ALIGN)
        self.assertAlmostEqual(state.target_phase_unwrapped_rad, 2.0 * math.pi + 0.1)

        state = advance_stop_state(
            state,
            self.input(
                time_s=1.1,
                body_phase_unwrapped_rad=state.target_phase_unwrapped_rad,
            ),
            self.pose,
            self.config,
            0.1,
        )
        self.assertEqual(state.mode, StopMode.PARK_DEPLOY)
        state = advance_stop_state(
            state,
            self.input(
                time_s=2.1,
                body_phase_unwrapped_rad=state.target_phase_unwrapped_rad,
            ),
            self.pose,
            self.config,
            0.1,
        )
        self.assertEqual(state.mode, StopMode.HOLD)

        for index in range(3):
            state = advance_stop_state(
                state,
                self.input(time_s=2.2 + 0.1 * index),
                self.pose,
                self.config,
                0.1,
            )
        self.assertTrue(stop_succeeded(state, self.config))
        state = advance_stop_state(
            state,
            self.input(time_s=2.5, linear_speed_m_s=0.2),
            self.pose,
            self.config,
            0.1,
        )
        self.assertEqual(state.settled_duration_s, 0.0)

    def test_schedule_and_three_reference_blend(self):
        state = StopState(
            mode=StopMode.BRAKE_ALIGN,
            stop_command_time_s=0.0,
            mode_start_time_s=0.0,
            target_phase_unwrapped_rad=1.0,
            initial_phase_distance_rad=1.0,
        )
        schedule = reference_schedule(
            state,
            time_s=1.0,
            body_phase_unwrapped_rad=0.5,
            config=self.config,
        )
        self.assertAlmostEqual(schedule.rolling_reference_scale, 0.5)
        blended = blend_joint_reference(
            np,
            np.array([1.0, 3.0]),
            np.array([0.0, 1.0]),
            np.array([-1.0, -1.0]),
            schedule,
        )
        np.testing.assert_allclose(blended, [0.5, 2.0])

        deploy_schedule = StopReferenceSchedule(
            mode=StopMode.PARK_DEPLOY,
            rolling_reference_scale=0.0,
            parking_reference_blend=0.5,
            target_phase_error_rad=0.0,
            time_since_stop_s=2.0,
            time_in_mode_s=0.5,
        )
        blended = blend_joint_reference(
            np,
            np.array([1.0, 3.0]),
            np.array([0.0, 1.0]),
            np.array([-1.0, -1.0]),
            deploy_schedule,
        )
        np.testing.assert_allclose(blended, [-0.5, 0.0])

    def test_observation_features_have_stable_width(self):
        state = StopState(mode=StopMode.HOLD)
        schedule = StopReferenceSchedule(
            mode=StopMode.HOLD,
            rolling_reference_scale=0.0,
            parking_reference_blend=1.0,
            target_phase_error_rad=0.2,
            time_since_stop_s=2.0,
            time_in_mode_s=1.0,
        )
        features = stop_observation_features(
            np, state, schedule, np.float32(0.3), True
        )
        self.assertEqual(features.shape, (13,))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(features[:4], [0.0, 0.0, 0.0, 1.0])

    def test_deploy_quintic_matches_captured_boundary_state(self):
        q0 = np.array([0.2, -0.4])
        v0 = np.array([0.3, -0.1])
        q1 = np.array([1.0, 0.5])
        start = deploy_trajectory_sample(
            np, q0, v0, q1, elapsed_s=0.0, duration_s=1.0
        )
        end = deploy_trajectory_sample(
            np, q0, v0, q1, elapsed_s=1.0, duration_s=1.0
        )
        np.testing.assert_allclose(start.position, q0)
        np.testing.assert_allclose(start.velocity, v0)
        np.testing.assert_allclose(start.acceleration, 0.0, atol=1e-12)
        np.testing.assert_allclose(end.position, q1, atol=1e-12)
        np.testing.assert_allclose(end.velocity, 0.0, atol=1e-12)
        np.testing.assert_allclose(end.acceleration, 0.0, atol=1e-12)
        self.assertTrue(end.finished)


class StopEvaluationTest(unittest.TestCase):
    def good_episode(self, **overrides):
        values = {
            "survived": True,
            "numerical_failure": False,
            "stop_time_s": 2.0,
            "extra_distance_m": 0.5,
            "final_linear_speed_m_s": 0.01,
            "final_angular_speed_rad_s": 0.02,
            "final_phase_error_rad": 0.02,
            "final_pose_rms_error_rad": 0.02,
            "settled_duration_s": 2.1,
            "grounded_feet": 2,
            "torso_contact_total_s": 0.0,
            "hold_internal_contact_total_s": 0.0,
            "leg_crossing": False,
            "lateral_drift_m": 0.01,
            "maximum_torque_nm": 4.0,
        }
        values.update(overrides)
        return StopEpisodeMetrics(**values)

    def test_stop_gate_reports_specific_failures(self):
        gate = StopEvaluationGate()
        self.assertTrue(stop_episode_succeeded(self.good_episode(), gate))
        failed = self.good_episode(
            grounded_feet=1,
            torso_contact_total_s=0.1,
            hold_internal_contact_total_s=0.2,
        )
        self.assertEqual(
            stop_failure_reasons(failed, gate),
            ("insufficient_support", "torso_contact", "internal_contact"),
        )

    def test_static_park_gate(self):
        gate = ParkPoseStaticGate()
        values = {
            "survived": True,
            "numerical_failure": False,
            "duration_s": 3.0,
            "final_linear_speed_m_s": 0.01,
            "final_angular_speed_rad_s": 0.02,
            "final_torso_tilt_rad": 0.01,
            "maximum_torso_tilt_rad": 0.02,
            "final_joint_pose_rms_error_rad": 0.02,
            "grounded_feet": 4,
            "internal_contact_total_s": 0.0,
            "torso_ground_contact_total_s": 0.0,
            "torso_internal_contact_total_s": 0.0,
            "lateral_drift_m": 0.0,
            "minimum_root_height_m": 0.3,
            "maximum_torque_nm": 1.0,
        }
        good = ParkPoseStaticMetrics(**values)
        self.assertTrue(park_pose_succeeded(good, gate))
        failed = ParkPoseStaticMetrics(**{**values, "grounded_feet": 2})
        self.assertEqual(
            park_pose_failure_reasons(failed, gate),
            ("insufficient_support",),
        )


if __name__ == "__main__":
    unittest.main()
