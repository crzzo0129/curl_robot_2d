from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model_3d import FOOT_SITE_NAMES_3D
from curl_robot_2d.parameters import (
    PUPPER_ORIGINAL_SHELL_60_PARAMETERS,
    REAL_GEOMETRY_PARAMETERS,
)
from curl_robot_2d_mjx.config_walking_3d import (
    Walking3DConfig,
    validate_walking_3d_config,
    walking_geometry_config_3d,
    walking_physics_profile_3d,
)
from curl_robot_2d_mjx.environment_walking_3d import (
    EXPECTED_WALKING_JOINT_AXES_3D,
    WALKING_ACTION_SIZE_3D,
    WALKING_MODEL_PATH_3D,
    WALKING_JOINT_NAMES_3D,
    WALKING_OBSERVATION_SIZE_3D,
    command_overspeed_3d,
    freejoint_body_velocity_3d,
    heading_frame_planar_velocity_3d,
    normalized_command_progress_3d,
    swing_clearance_reward_3d,
    torso_local_points_3d,
    validate_walking_morphology_3d,
)
from curl_robot_2d_mjx.environment_3d import model_path_3d
from curl_robot_2d_mjx.environment_3d import apply_physics_options_3d
from scripts import mjx_3d_walking_smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MJX3DWalkingContractTest(unittest.TestCase):
    def test_task_uses_eth_style_12_dof_proprioceptive_contract(self) -> None:
        config = Walking3DConfig()

        self.assertEqual(
            WALKING_MODEL_PATH_3D,
            PROJECT_ROOT
            / "assets"
            / "curl_robot_3d_pupper_r127p5_open60_width120.xml",
        )
        self.assertEqual(config.reset_keyframe_name, "stand")
        self.assertEqual(len(WALKING_JOINT_NAMES_3D), WALKING_ACTION_SIZE_3D)
        self.assertEqual(WALKING_ACTION_SIZE_3D, 12)
        self.assertEqual(WALKING_OBSERVATION_SIZE_3D, 48)
        self.assertEqual(config.geometry, "pupper_open60")
        self.assertAlmostEqual(config.desired_speed_m_s, 0.20)
        self.assertAlmostEqual(config.diagnostic_lateral_drift_m, 1.50)
        self.assertEqual(
            config.action_scales,
            (0.10, 0.40, 0.55) * 4,
        )
        self.assertEqual(config.observation_scale_linear_velocity, 2.0)
        self.assertEqual(config.observation_scale_angular_velocity, 0.25)
        self.assertEqual(config.observation_scale_command_linear_velocity, 2.0)
        self.assertEqual(config.observation_scale_command_yaw_rate, 0.25)
        self.assertEqual(config.observation_scale_joint_velocity, 0.05)

    def test_freejoint_angular_velocity_is_already_body_local(self) -> None:
        rotation = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        qvel = np.asarray((0.0, 1.0, 0.0, 0.1, 0.2, 0.3))

        linear, angular = freejoint_body_velocity_3d(np, rotation, qvel)

        np.testing.assert_allclose(linear, (1.0, 0.0, 0.0), atol=1.0e-8)
        np.testing.assert_allclose(angular, (0.1, 0.2, 0.3), atol=1.0e-8)

    def test_planar_reward_velocity_ignores_pitch_and_vertical_motion(self) -> None:
        pitch = np.deg2rad(45.0)
        rotation = np.asarray(
            (
                (np.cos(pitch), 0.0, np.sin(pitch)),
                (0.0, 1.0, 0.0),
                (-np.sin(pitch), 0.0, np.cos(pitch)),
            )
        )

        body_linear, _ = freejoint_body_velocity_3d(
            np, rotation, np.asarray((0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
        )
        reward_planar = heading_frame_planar_velocity_3d(
            np, rotation, np.asarray((0.0, 0.0, -1.0))
        )

        self.assertGreater(float(body_linear[0]), 0.5)
        np.testing.assert_allclose(reward_planar, (0.0, 0.0), atol=1.0e-8)

    def test_planar_reward_velocity_uses_heading_not_world_axes(self) -> None:
        rotation = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )

        reward_planar = heading_frame_planar_velocity_3d(
            np, rotation, np.asarray((0.0, 0.1, 0.7))
        )

        np.testing.assert_allclose(reward_planar, (0.1, 0.0), atol=1.0e-8)

    def test_command_progress_is_target_relative_and_saturating(self) -> None:
        command = np.asarray((0.1, 0.0, 0.0))

        self.assertAlmostEqual(
            float(
                normalized_command_progress_3d(
                    np, np.asarray((0.0, 0.0)), command
                )
            ),
            0.0,
        )
        self.assertAlmostEqual(
            float(
                normalized_command_progress_3d(
                    np, np.asarray((0.05, 0.0)), command
                )
            ),
            0.5,
        )
        for forward_speed in (0.1, 0.4):
            with self.subTest(forward_speed=forward_speed):
                self.assertAlmostEqual(
                    float(
                        normalized_command_progress_3d(
                            np,
                            np.asarray((forward_speed, 0.0)),
                            command,
                        )
                    ),
                    1.0,
                )
        self.assertAlmostEqual(
            float(
                normalized_command_progress_3d(
                    np, np.asarray((-0.1, 0.0)), command
                )
            ),
            0.0,
        )

    def test_swing_clearance_requires_relative_horizontal_foot_motion(self) -> None:
        contact = np.asarray((False, True, True, True))
        height = np.asarray((0.015, 0.0, 0.0, 0.0))
        rigid_foot_velocity = np.zeros((4, 2))

        rigid_reward = swing_clearance_reward_3d(
            np,
            contact,
            height,
            rigid_foot_velocity,
            clearance_m=0.015,
            swing_speed_m_s=0.10,
        )
        swinging_foot_velocity = rigid_foot_velocity.copy()
        swinging_foot_velocity[0, 0] = 0.10
        swing_reward = swing_clearance_reward_3d(
            np,
            contact,
            height,
            swinging_foot_velocity,
            clearance_m=0.015,
            swing_speed_m_s=0.10,
        )

        self.assertAlmostEqual(float(rigid_reward), 0.0)
        self.assertAlmostEqual(float(swing_reward), 0.25)

    def test_torso_local_feet_ignore_rigid_translation_and_rotation(self) -> None:
        theta = np.deg2rad(35.0)
        rotation = np.asarray(
            (
                (np.cos(theta), -np.sin(theta), 0.0),
                (np.sin(theta), np.cos(theta), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        local_points = np.asarray(
            ((0.1, 0.05, -0.2), (-0.1, -0.05, -0.2))
        )
        torso_position = np.asarray((1.2, -0.4, 0.3))
        world_points = local_points @ rotation.T + torso_position

        recovered = torso_local_points_3d(
            np, rotation, torso_position, world_points
        )

        np.testing.assert_allclose(recovered, local_points, atol=1.0e-8)

    def test_command_overspeed_has_a_margin(self) -> None:
        command = np.asarray((0.1, 0.0, 0.0))

        self.assertAlmostEqual(
            float(
                command_overspeed_3d(
                    np, np.asarray((0.15, 0.0)), command, 0.05
                )
            ),
            0.0,
        )
        self.assertAlmostEqual(
            float(
                command_overspeed_3d(
                    np, np.asarray((0.4, 0.0)), command, 0.05
                )
            ),
            0.25,
        )

    def test_model_matches_mirrored_planar_leg_morphology(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(WALKING_MODEL_PATH_3D))

        validate_walking_morphology_3d(model)

        for name, expected_axis in EXPECTED_WALKING_JOINT_AXES_3D.items():
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            np.testing.assert_allclose(model.jnt_axis[joint_id], expected_axis)
            self.assertAlmostEqual(np.linalg.norm(model.jnt_axis[joint_id]), 1.0)

    @unittest.skip("legacy 8-DoF model is outside the 12-DoF walking task")
    def test_real_geometry_model_matches_walking_morphology(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(model_path_3d("real")))

        validate_walking_morphology_3d(model, REAL_GEOMETRY_PARAMETERS)
        config = walking_geometry_config_3d(Walking3DConfig(geometry="real"))
        self.assertEqual(config.geometry, "real")
        self.assertAlmostEqual(
            config.foot_radius_m, REAL_GEOMETRY_PARAMETERS.foot_radius
        )
        self.assertAlmostEqual(
            config.nominal_root_height_m,
            REAL_GEOMETRY_PARAMETERS.stand_3d_root_height,
        )

        apply_physics_options_3d(model, config)
        self.assertEqual(model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)

    def test_real_geometry_stand_is_finite_for_one_second(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(model_path_3d("real")))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)
        torso_id = model.body("torso").id
        maximum_tilt = 0.0

        for _ in range(round(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            body_z = data.xmat[torso_id].reshape(3, 3)[:, 2]
            maximum_tilt = max(
                maximum_tilt,
                float(np.arccos(np.clip(body_z[2], -1.0, 1.0))),
            )

        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(float(data.qpos[2]), 0.30)
        self.assertLess(maximum_tilt, 0.08)

    def test_stand_keyframe_places_all_four_feet_on_floor(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(WALKING_MODEL_PATH_3D))
        data = mujoco.MjData(model)

        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)

        for name in FOOT_SITE_NAMES_3D:
            with self.subTest(site=name):
                foot_z = data.site_xpos[model.site(name).id, 2]
                self.assertAlmostEqual(
                    foot_z,
                    PUPPER_ORIGINAL_SHELL_60_PARAMETERS.foot_radius,
                    delta=1.0e-4,
                )
        self.assertAlmostEqual(
            float(data.qpos[2]),
            PUPPER_ORIGINAL_SHELL_60_PARAMETERS.stand_3d_root_height,
        )

    def test_zero_action_center_has_stable_one_second_start(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(WALKING_MODEL_PATH_3D))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)
        initial_x = float(data.qpos[0])
        initial_y = float(data.qpos[1])
        torso_id = model.body("torso").id
        maximum_tilt = 0.0

        for _ in range(round(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            body_z = data.xmat[torso_id].reshape(3, 3)[:, 2]
            maximum_tilt = max(
                maximum_tilt,
                float(np.arccos(np.clip(body_z[2], -1.0, 1.0))),
            )

        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(float(data.qpos[2]), 0.14)
        self.assertLess(maximum_tilt, 0.08)
        self.assertLess(abs(float(data.qpos[0]) - initial_x), 0.04)
        self.assertAlmostEqual(float(data.qpos[1]) - initial_y, 0.0, places=8)

    def test_config_validation_and_fast_profile(self) -> None:
        validate_walking_3d_config(Walking3DConfig())
        cg12 = walking_physics_profile_3d("cg12")

        self.assertAlmostEqual(cg12.control_timestep, 0.02)
        self.assertEqual(cg12.solver_name, "cg")
        self.assertEqual(cg12.solver_iterations, 12)
        for values in (
            {"action_scales": (1.0,)},
            {"desired_speed_m_s": 0.0},
            {"reset_root_xy_velocity_noise_m_s": -0.01},
            {"terminate_root_z_min": 0.5},
            {"soft_joint_limit_fraction": 0.0},
            {"observation_scale_joint_velocity": 0.0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_walking_3d_config(Walking3DConfig(**values))

    def test_smoke_entry_imports_without_jax(self) -> None:
        args = mjx_3d_walking_smoke.parse_args([])

        self.assertTrue(callable(mjx_3d_walking_smoke.main))
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.steps, 8)
        self.assertAlmostEqual(args.desired_speed, 0.20)
        self.assertAlmostEqual(args.action_scale_abduction, 0.10)
        self.assertAlmostEqual(args.action_scale_hip, 0.40)
        self.assertAlmostEqual(args.action_scale_knee, 0.55)

    def test_environment_has_no_gait_reference_path(self) -> None:
        source = (
            PROJECT_ROOT / "curl_robot_2d_mjx" / "environment_walking_3d.py"
        ).read_text(encoding="utf-8")
        for token in (
            "walking_reference_3d",
            "oscillator_phase",
            "reference_target",
            "stance_miss",
        ):
            self.assertNotIn(token, source)
        for token in (
            "self.nominal_ctrl",
            "self.sys = mjx.put_model",
            "mjx.forward(self.sys",
            "mjx.step(self.sys",
            "foot_air_time",
            "foot_slip_velocity_squared",
            "joint_limit_cost",
            "jax.lax.cond",
            "transition_finite",
            '"time_out": timeout_bool.astype(jp.float32)',
        ):
            self.assertIn(token, source)

        self.assertIn('"lateral_drift_exceeded"', source)
        self.assertNotIn("failure_lateral_drift", source)
        self.assertNotIn("| lateral_drift_exceeded", source)
        self.assertNotIn("startup_action_ramp", source)
        self.assertNotIn("smoothstep_ramp", source)
        self.assertIn("+ policy_action * self.action_scales", source)


if __name__ == "__main__":
    unittest.main()
