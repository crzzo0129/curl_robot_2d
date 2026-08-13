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
        self.assertEqual(
            config.action_scales,
            (0.10, 0.40, 0.55) * 4,
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
            {"terminate_root_z_min": 0.5},
            {"soft_joint_limit_fraction": 0.0},
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
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
