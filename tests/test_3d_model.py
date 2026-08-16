from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model_3d import (
    FOOT_SITE_NAMES_3D,
    JOINT_NAMES_3D,
    build_mjcf_3d,
)
from curl_robot_2d.parameters import FIXED_PARAMETERS, REAL_GEOMETRY_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"
REAL_GEOMETRY_MODEL_PATH = (
    PROJECT_ROOT / "assets" / "curl_robot_3d_real_geometry.xml"
)


class CurlRobot3DModelTest(unittest.TestCase):
    def test_real_geometry_3d_candidate_contract(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(REAL_GEOMETRY_MODEL_PATH))
        p = REAL_GEOMETRY_PARAMETERS

        self.assertEqual(
            REAL_GEOMETRY_MODEL_PATH.read_text(encoding="utf-8"),
            build_mjcf_3d(p, detailed_structure=True),
        )

        self.assertEqual(model.nq, 15)
        self.assertEqual(model.nu, 8)
        self.assertEqual(model.nkey, 4)
        self.assertEqual(
            {model.key(index).name for index in range(model.nkey)},
            {"open", "stand", "park", "compact"},
        )
        self.assertEqual(model.npair, 0)
        torso_id = model.geom("torso_box_proxy").id
        np.testing.assert_allclose(model.geom_size[torso_id], [0.06, 0.06, 0.06])
        self.assertAlmostEqual(model.geom_pos[torso_id, 2], -0.04)
        geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(model.ngeom)
        ]
        self.assertEqual(sum(name.endswith("_motor") for name in geom_names), 8)
        self.assertEqual(sum("_motor_collision_" in name for name in geom_names), 32)
        self.assertEqual(sum("_shell_" in name for name in geom_names), 60)
        self.assertEqual(sum(name.endswith("_foot_proxy") for name in geom_names), 4)
        for side in ("left", "right"):
            expected_y = 0.06 if side == "left" else -0.06
            for end in ("front", "rear"):
                body_id = model.body(f"{end}_{side}_thigh").id
                self.assertAlmostEqual(model.body_pos[body_id, 1], expected_y)
                foot_id = model.geom(f"{end}_{side}_foot_proxy").id
                self.assertAlmostEqual(model.geom_size[foot_id, 0], 0.03)
                for joint in ("hip", "knee"):
                    motor_id = model.geom(f"{end}_{side}_{joint}_motor").id
                    self.assertAlmostEqual(model.geom_size[motor_id, 0], 0.027)
                    self.assertAlmostEqual(model.geom_size[motor_id, 1], 0.012)
                    self.assertEqual(
                        model.geom_type[motor_id],
                        mujoco.mjtGeom.mjGEOM_CYLINDER,
                    )
                    self.assertEqual(model.geom_contype[motor_id], 0)
                    self.assertEqual(model.geom_conaffinity[motor_id], 0)
                    for proxy_index in range(4):
                        proxy_id = model.geom(
                            f"{end}_{side}_{joint}_motor_collision_{proxy_index:02d}"
                        ).id
                        self.assertEqual(
                            model.geom_type[proxy_id],
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                        )
                        self.assertAlmostEqual(model.geom_size[proxy_id, 0], 0.012)
                        self.assertAlmostEqual(model.geom_size[proxy_id, 1], 0.015)
                        self.assertEqual(model.geom_contype[proxy_id], 2)
                        self.assertEqual(model.geom_conaffinity[proxy_id], 7)

    def test_checked_in_3d_model_matches_generator(self) -> None:
        self.assertEqual(MODEL_PATH.read_text(encoding="utf-8"), build_mjcf_3d())

    def test_generator_can_make_shell_visual_only(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            build_mjcf_3d(shell_collisions_enabled=False)
        )
        shell_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if "_shell_" in (model.geom(geom_id).name or "")
        ]

        self.assertGreater(len(shell_ids), 0)
        for geom_id in shell_ids:
            self.assertEqual(int(model.geom_contype[geom_id]), 0)
            self.assertEqual(int(model.geom_conaffinity[geom_id]), 0)

    def test_3d_model_contract(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

        self.assertEqual(model.nq, 15)
        self.assertEqual(model.nv, 14)
        self.assertEqual(model.nu, 8)
        self.assertEqual(model.nkey, 3)
        self.assertEqual(model.npair, 2)
        self.assertEqual(len(JOINT_NAMES_3D), 8)
        self.assertAlmostEqual(float(model.body_mass.sum()), FIXED_PARAMETERS.total_mass)
        self.assertEqual(
            sum(
                "_shell_" in (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    or ""
                )
                for geom_id in range(model.ngeom)
            ),
            10 * FIXED_PARAMETERS.shell_segments_per_edge,
        )

    def test_compact_keyframe_has_two_side_rails_and_touching_feet(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
        mujoco.mj_forward(model, data)

        sites = {
            name: data.site_xpos[model.site(name).id].copy()
            for name in FOOT_SITE_NAMES_3D
        }
        self.assertAlmostEqual(
            sites["front_left_foot_site"][1],
            FIXED_PARAMETERS.side_rail_half_width,
        )
        self.assertAlmostEqual(
            sites["front_right_foot_site"][1],
            -FIXED_PARAMETERS.side_rail_half_width,
        )
        for name, position in sites.items():
            with self.subTest(site=name):
                self.assertAlmostEqual(position[2], FIXED_PARAMETERS.foot_radius)

        for side in ("left", "right"):
            distance = np.linalg.norm(
                sites[f"front_{side}_foot_site"] - sites[f"rear_{side}_foot_site"]
            )
            self.assertAlmostEqual(distance, FIXED_PARAMETERS.compact_foot_center_distance)

        contacts = {
            frozenset(
                (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                )
            )
            for contact in data.contact
        }
        self.assertIn(frozenset(("front_left_foot_proxy", "rear_left_foot_proxy")), contacts)
        self.assertIn(frozenset(("front_right_foot_proxy", "rear_right_foot_proxy")), contacts)

    def test_3d_actuators_are_single_motor_scale(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

        for name in JOINT_NAMES_3D:
            actuator_id = model.actuator(f"{name}_servo").id
            self.assertAlmostEqual(model.actuator_forcerange[actuator_id, 0], -3.0)
            self.assertAlmostEqual(model.actuator_forcerange[actuator_id, 1], 3.0)
            self.assertAlmostEqual(model.actuator_gainprm[actuator_id, 0], 5.0)
            self.assertAlmostEqual(model.actuator_biasprm[actuator_id, 2], -0.1)


if __name__ == "__main__":
    unittest.main()
