"""Native MuJoCo static checks only: no JAX imports, stepping or rollouts."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.deploy_collision import audit_foot_sphere_model, foot_sphere_only_xml
from scripts.deploy_terrain import RoughTerrainConfig, inject_heightfield, reference_terrain_data


LEGS = ("front_left", "front_right", "rear_left", "rear_right")
SOURCE = (Path(__file__).resolve().parents[1] / "assets" /
          "rollingquad_description_2" / "mjcf" / "rollingquad.xml")


class FootSphereCollisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ET.fromstring(SOURCE.read_text(encoding="utf-8"))
        root.find("compiler").set("meshdir", SOURCE.parent.as_posix())
        cls.source_xml = ET.tostring(root, encoding="unicode")
        cls.xml = foot_sphere_only_xml(cls.source_xml, LEGS, 0.0195)
        cls.source = mujoco.MjModel.from_xml_string(cls.source_xml)
        cls.spheres = mujoco.MjModel.from_xml_string(cls.xml)

    def test_compiled_contract_and_inertias(self):
        report = audit_foot_sphere_model(self.spheres, LEGS, 0.0195)
        self.assertEqual(report["ground_pairs"], 4)
        self.assertEqual((self.spheres.nq, self.spheres.nv, self.spheres.nu), (19, 18, 12))
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                      "body_pos", "body_quat", "jnt_axis", "jnt_range", "key_qpos",
                      "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                      "actuator_ctrlrange", "actuator_trnid"):
            np.testing.assert_array_equal(getattr(self.source, field), getattr(self.spheres, field))

    def test_native_contacts_and_stand_bottom(self):
        source_data = mujoco.MjData(self.source)
        sphere_data = mujoco.MjData(self.spheres)
        for model, data in ((self.source, source_data), (self.spheres, sphere_data)):
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            mujoco.mj_resetDataKeyframe(model, data, kid)
            mujoco.mj_forward(model, data)
        for leg in LEGS:
            sid = mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_SITE, leg + "_foot_site")
            gid = mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_GEOM, leg + "_foot_proxy")
            mid = self.source.geom_dataid[gid]
            start = self.source.mesh_vertadr[mid]
            verts = self.source.mesh_vert[start:start + self.source.mesh_vertnum[mid]]
            world = verts @ source_data.geom_xmat[gid].reshape(3, 3).T + source_data.geom_xpos[gid]
            sphere_bottom = sphere_data.site_xpos[sid, 2] - 0.0195
            self.assertLess(abs(sphere_bottom - world[:, 2].min()), 0.0003)
        # Lower the pose enough for contact and run collision detection only.
        sphere_data.qpos[2] -= 0.002
        mujoco.mj_forward(self.spheres, sphere_data)
        self.assertEqual(sphere_data.ncon, 4)
        floor = mujoco.mj_name2id(self.spheres, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        expected = {mujoco.mj_name2id(self.spheres, mujoco.mjtObj.mjOBJ_GEOM,
                                    leg + "_foot_collision") for leg in LEGS}
        actual = set()
        for contact in sphere_data.contact:
            self.assertIn(floor, (contact.geom1, contact.geom2))
            actual.add(contact.geom2 if contact.geom1 == floor else contact.geom1)
        self.assertEqual(actual, expected)

    def test_explicit_pairs_cannot_reenable_cad(self):
        root = ET.fromstring(self.source_xml)
        contact = ET.SubElement(root, "contact")
        ET.SubElement(contact, "pair", geom1="floor", geom2="torso_mesh")
        ET.SubElement(contact, "pair", geom1="front_left_foot_proxy", geom2="rear_right_foot_proxy")
        model = mujoco.MjModel.from_xml_string(foot_sphere_only_xml(
            ET.tostring(root, encoding="unicode"), LEGS, 0.0195))
        audit_foot_sphere_model(model, LEGS, 0.0195)

    def test_terrain_uses_same_four_spheres(self):
        config = RoughTerrainConfig()
        model = mujoco.MjModel.from_xml_string(inject_heightfield(self.xml, config))
        model.hfield_data[:] = reference_terrain_data(config)
        self.assertEqual(model.nhfield, 1)
        audit_foot_sphere_model(model, LEGS, 0.0195)


if __name__ == "__main__":
    if "jax" in sys.modules:
        raise RuntimeError("static collision checks must not import JAX")
    unittest.main()
