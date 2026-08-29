from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d_mjx.config_walking_3d import (
    ROLLINGQUAD_2_FOOT_RADIUS_M,
    ROLLINGQUAD_2_STAND_ROOT_HEIGHT_M,
    WALKING_GEOMETRY_NAMES_3D,
    Walking3DConfig,
    walking_geometry_config_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    ROLLINGQUAD_2_MODEL_PATH_3D,
    model_path_3d,
)
from curl_robot_2d_mjx.environment_walking_3d import (
    WALKING_JOINT_NAMES_3D,
    validate_walking_morphology_3d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "curl_robot_3d_pupper_r127p5_open60_width120.xml"
)


class Rollingquad2IntegrationTest(unittest.TestCase):
    def test_geometry_is_available_to_walking_training(self) -> None:
        self.assertIn("rollingquad_2", WALKING_GEOMETRY_NAMES_3D)
        self.assertEqual(model_path_3d("rollingquad_2"), ROLLINGQUAD_2_MODEL_PATH_3D)
        self.assertTrue(ROLLINGQUAD_2_MODEL_PATH_3D.exists())

        config = walking_geometry_config_3d(
            Walking3DConfig(geometry="rollingquad_2")
        )
        self.assertAlmostEqual(
            config.nominal_root_height_m, ROLLINGQUAD_2_STAND_ROOT_HEIGHT_M
        )
        self.assertAlmostEqual(config.foot_radius_m, ROLLINGQUAD_2_FOOT_RADIUS_M)

    def test_model_matches_policy_names_axes_and_actuator_order(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROLLINGQUAD_2_MODEL_PATH_3D))

        self.assertEqual((model.nq, model.nv, model.nu), (19, 18, 12))
        validate_walking_morphology_3d(model, geometry_name="rollingquad_2")
        actuator_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(model.nu)
        )
        self.assertEqual(
            actuator_names,
            tuple(f"{name}_servo" for name in WALKING_JOINT_NAMES_3D),
        )

    def test_keyframes_match_reference_by_semantic_joint_name(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROLLINGQUAD_2_MODEL_PATH_3D))
        reference = mujoco.MjModel.from_xml_path(str(REFERENCE_MODEL_PATH))

        for key_name in ("open", "stand_previous", "park", "compact"):
            key_id = model.key(key_name).id
            reference_key_id = reference.key(key_name).id
            for joint_name in WALKING_JOINT_NAMES_3D:
                qpos_index = model.jnt_qposadr[model.joint(joint_name).id]
                reference_qpos_index = reference.jnt_qposadr[
                    reference.joint(joint_name).id
                ]
                self.assertAlmostEqual(
                    model.key_qpos[key_id, qpos_index],
                    reference.key_qpos[reference_key_id, reference_qpos_index],
                    places=6,
                    msg=f"{key_name}: {joint_name}",
                )
                actuator_name = f"{joint_name}_servo"
                self.assertAlmostEqual(
                    model.key_ctrl[key_id, model.actuator(actuator_name).id],
                    reference.key_ctrl[
                        reference_key_id, reference.actuator(actuator_name).id
                    ],
                    places=6,
                    msg=f"{key_name}: {actuator_name}",
                )

    def test_crouched_walking_stand_has_one_mm_cad_clearance(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROLLINGQUAD_2_MODEL_PATH_3D))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)

        expected = {
            "hip_abduction": 0.0,
            "hip": 0.90,
            "knee": 1.15,
        }
        for joint_name in WALKING_JOINT_NAMES_3D:
            joint_kind = next(
                kind for kind in expected if joint_name.endswith(kind)
            )
            qpos_index = model.jnt_qposadr[model.joint(joint_name).id]
            self.assertAlmostEqual(
                data.qpos[qpos_index], expected[joint_kind], places=6
            )

        self.assertAlmostEqual(
            data.qpos[2], ROLLINGQUAD_2_STAND_ROOT_HEIGHT_M, places=9
        )
        mujoco.mj_forward(model, data)
        floor_id = model.geom("floor").id
        distances = []
        for leg in ("front_left", "front_right", "rear_left", "rear_right"):
            fromto = np.zeros(6)
            distances.append(mujoco.mj_geomDistance(
                model, data, floor_id, model.geom(f"{leg}_foot_proxy").id,
                0.01, fromto,
            ))
        self.assertAlmostEqual(min(distances), 0.001, places=6)

    def test_servo_limits_and_short_cpu_dynamics_are_finite(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROLLINGQUAD_2_MODEL_PATH_3D))
        np.testing.assert_allclose(
            model.actuator_forcerange,
            np.tile((-3.0, 3.0), (model.nu, 1)),
        )
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], 5.0)
        np.testing.assert_allclose(
            model.actuator_biasprm[:, 1:3],
            np.tile((-5.0, -0.1), (model.nu, 1)),
        )

        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        for _ in range(round(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        self.assertTrue(np.isfinite(data.actuator_force).all())


if __name__ == "__main__":
    unittest.main()
