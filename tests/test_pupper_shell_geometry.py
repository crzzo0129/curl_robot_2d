from dataclasses import replace
import math
import unittest

from curl_robot_2d.pupper_shell_geometry import (
    PupperShellDesign,
    solve_compact_geometry,
)
from curl_robot_2d.pupper_shell_model import build_pupper_shell_mjcf
from curl_robot_2d.model import build_mjcf
from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_PARAMETERS
import mujoco


class PupperShellGeometryTest(unittest.TestCase):
    def test_127p5mm_geometry_satisfies_original_pupper_lengths(self):
        design = PupperShellDesign()
        solution = solve_compact_geometry(design)
        hip = (design.hip_half_distance, 0.0)
        knee = (solution.knee_x, solution.knee_below_hip)
        foot = (solution.foot_x, solution.foot_below_hip)
        center = (0.0, solution.shell_center_below_hip)

        self.assertAlmostEqual(math.dist(hip, knee), design.upper_leg_length)
        self.assertAlmostEqual(math.dist(knee, foot), design.lower_leg_length)
        self.assertAlmostEqual(
            math.dist(center, knee),
            design.shell_outer_radius - design.motor_envelope_radius,
        )
        self.assertAlmostEqual(
            math.dist(center, foot),
            design.shell_outer_radius - design.foot_radius,
        )
        self.assertAlmostEqual(2.0 * solution.foot_x, 0.043)
        self.assertAlmostEqual(design.foot_surface_gap, 0.004)
        self.assertAlmostEqual(solution.shell_center_below_hip, 0.0395170689)
        self.assertAlmostEqual(solution.hip_motor_radial_clearance, 0.0105491981)

    def test_integer_search_range_is_feasible(self):
        for radius in (0.122, 0.1275, 0.155):
            solution = solve_compact_geometry(
                replace(PupperShellDesign(), shell_outer_radius=radius)
            )
            self.assertGreaterEqual(solution.shell_center_below_hip, 0.0)
            self.assertGreaterEqual(solution.hip_motor_radial_clearance, 0.0)

    def test_rejects_touching_feet(self):
        with self.assertRaises(ValueError):
            PupperShellDesign(compact_foot_center_distance=0.039)

    def test_shell_centerline_keeps_requested_outer_radius(self):
        design = PupperShellDesign(shell_outer_radius=0.1275)
        self.assertAlmostEqual(
            design.shell_centerline_radius + design.shell_capsule_radius,
            0.1275,
        )

    def test_generated_model_has_original_dimensions_and_full_shell(self):
        design = PupperShellDesign()
        model = mujoco.MjModel.from_xml_string(build_pupper_shell_mjcf(design))
        self.assertEqual(model.nq, 7)
        self.assertEqual(
            sum(
                "_shell_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                for i in range(model.ngeom)
            ),
            design.shell_segments,
        )
        self.assertAlmostEqual(model.geom_size[model.geom("front_foot_proxy").id, 0], 0.0195)
        self.assertAlmostEqual(model.geom_size[model.geom("front_hip_motor").id, 0], 0.032)

    def test_compact_keyframe_has_43mm_foot_center_distance(self):
        design = PupperShellDesign()
        model = mujoco.MjModel.from_xml_string(build_pupper_shell_mjcf(design))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
        mujoco.mj_forward(model, data)
        front = data.geom_xpos[model.geom("front_foot_proxy").id]
        rear = data.geom_xpos[model.geom("rear_foot_proxy").id]
        self.assertAlmostEqual(abs(front[0] - rear[0]), 0.043, places=8)
        shell_bottom = min(
            data.geom_xpos[i, 2] - model.geom_size[i, 0]
            for i in range(model.ngeom)
            if "_shell_" in (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            )
        )
        self.assertGreaterEqual(shell_bottom, -0.002)

    def test_existing_model_pipeline_supports_pupper_geometry(self):
        parameters = PUPPER_ORIGINAL_SHELL_PARAMETERS
        model = mujoco.MjModel.from_xml_string(
            build_mjcf(
                parameters,
                detailed_structure=True,
                include_motor_collisions=True,
                disable_shell_shell_collision=True,
            )
        )
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
        mujoco.mj_forward(model, data)
        front = data.geom_xpos[model.geom("front_foot_proxy").id]
        rear = data.geom_xpos[model.geom("rear_foot_proxy").id]
        self.assertAlmostEqual(abs(front[0] - rear[0]), 0.043, places=8)
        shell_count = sum(
                "_shell_" in (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                    or ""
                )
                for i in range(model.ngeom)
            )
        self.assertEqual(shell_count, parameters.shell_segments_full_circle - 12)
        self.assertAlmostEqual(
            model.geom_size[model.geom("front_hip_motor").id, 0],
            0.032,
        )


if __name__ == "__main__":
    unittest.main()
