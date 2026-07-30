from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model_3d import (
    FOOT_SITE_NAMES_3D,
    JOINT_NAMES_3D,
    build_mjcf_3d,
)
from curl_robot_2d.parameters import FIXED_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"


class CurlRobot3DModelTest(unittest.TestCase):
    def test_checked_in_3d_model_matches_generator(self) -> None:
        self.assertEqual(MODEL_PATH.read_text(encoding="utf-8"), build_mjcf_3d())

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
