from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model import build_mjcf
from curl_robot_2d.parameters import FIXED_PARAMETERS
from curl_robot_2d.planar_geometry import (
    proper_segments_intersect,
    segment_distance,
)
from scripts.analyze_roll_phase import analyze_rigid_phase
from scripts.optimize_phase_controller import controller_targets, rollout_controller
from scripts.run_release_baseline import run_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"


class ModelContractTest(unittest.TestCase):
    def test_planar_crossing_geometry_excludes_shared_endpoint(self) -> None:
        self.assertTrue(
            proper_segments_intersect(
                np.array([0.0, 0.0]),
                np.array([1.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 0.0]),
            )
        )
        self.assertFalse(
            proper_segments_intersect(
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([1.0, 1.0]),
            )
        )
        self.assertAlmostEqual(
            segment_distance(
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 0.5]),
                np.array([1.0, 0.5]),
            ),
            0.5,
        )

    def test_checked_in_model_matches_generator(self) -> None:
        self.assertEqual(MODEL_PATH.read_text(encoding="utf-8"), build_mjcf())

    def test_model_contract(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

        self.assertEqual(model.nq, 7)
        self.assertEqual(model.nv, 7)
        self.assertEqual(model.nu, 4)
        self.assertEqual(model.nkey, 3)
        self.assertEqual(model.neq, 4)
        self.assertFalse(np.asarray(model.eq_active0).any())
        self.assertEqual(
            sum(
                "_shell_" in mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                for geom_id in range(model.ngeom)
            ),
            5 * FIXED_PARAMETERS.shell_segments_per_edge,
        )
        self.assertAlmostEqual(
            float(model.body_mass.sum()), FIXED_PARAMETERS.total_mass
        )

    def test_five_centerline_edges_have_one_length(self) -> None:
        p = FIXED_PARAMETERS
        self.assertAlmostEqual(p.torso_length, p.edge_length)
        self.assertAlmostEqual(2.0 * p.hip_half_span, p.edge_length)
        self.assertAlmostEqual(p.upper_length, p.edge_length)
        self.assertAlmostEqual(p.lower_length, p.edge_length)

    def test_compact_keyframe_touches_finite_size_feet(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
        front_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
        )
        rear_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
        )
        front_hip = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "front_thigh"
        )
        front_knee = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "front_shank"
        )
        rear_hip = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "rear_thigh"
        )
        rear_knee = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "rear_shank"
        )

        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        foot_center_distance = np.linalg.norm(
            data.site_xpos[front_site] - data.site_xpos[rear_site]
        )
        self.assertAlmostEqual(
            foot_center_distance,
            FIXED_PARAMETERS.compact_foot_center_distance,
        )
        self.assertAlmostEqual(
            foot_center_distance - 2.0 * FIXED_PARAMETERS.foot_radius,
            FIXED_PARAMETERS.compact_foot_surface_gap,
        )
        self.assertAlmostEqual(
            data.site_xpos[front_site][2], FIXED_PARAMETERS.foot_radius
        )
        self.assertAlmostEqual(
            data.site_xpos[rear_site][2], FIXED_PARAMETERS.foot_radius
        )
        edge_lengths = np.asarray(
            [
                np.linalg.norm(data.xpos[front_hip] - data.xpos[rear_hip]),
                np.linalg.norm(data.xpos[front_knee] - data.xpos[front_hip]),
                np.linalg.norm(
                    data.site_xpos[front_site] - data.xpos[front_knee]
                ),
                np.linalg.norm(
                    data.site_xpos[rear_site] - data.xpos[rear_knee]
                ),
                np.linalg.norm(data.xpos[rear_knee] - data.xpos[rear_hip]),
            ]
        )
        np.testing.assert_allclose(
            edge_lengths, FIXED_PARAMETERS.edge_length, atol=1e-9
        )

    def test_shell_design_gap_when_thigh_and_shank_are_collinear(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "open")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        def shell_segments(prefix: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
            segments = []
            for geom_id in range(model.ngeom):
                name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                if not (name or "").startswith(f"{prefix}_shell_"):
                    continue
                rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
                axis = rotation[:, 2][[0, 2]]
                center = np.asarray(data.geom_xpos[geom_id])[[0, 2]]
                half_length = float(model.geom_size[geom_id, 1])
                segments.append(
                    (
                        center - half_length * axis,
                        center + half_length * axis,
                        float(model.geom_size[geom_id, 0]),
                    )
                )
            return segments

        for thigh_name, shank_name in (
            ("front_thigh", "front_shank"),
            ("rear_thigh", "rear_shank"),
        ):
            clearances = []
            for thigh_start, thigh_end, thigh_radius in shell_segments(
                thigh_name
            ):
                for shank_start, shank_end, shank_radius in shell_segments(
                    shank_name
                ):
                    clearances.append(
                        segment_distance(
                            thigh_start,
                            thigh_end,
                            shank_start,
                            shank_end,
                        )
                        - thigh_radius
                        - shank_radius
                    )
            self.assertAlmostEqual(
                min(clearances), FIXED_PARAMETERS.shell_design_gap, places=9
            )

    def test_shells_clear_full_safe_joint_ranges(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "open"
        )
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        open_qpos = data.qpos.copy()

        def shell_segments(
            prefix: str,
        ) -> list[tuple[np.ndarray, np.ndarray, float]]:
            segments = []
            for geom_id in range(model.ngeom):
                name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                if not (name or "").startswith(f"{prefix}_shell_"):
                    continue
                rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
                axis = rotation[:, 2][[0, 2]]
                center = np.asarray(data.geom_xpos[geom_id])[[0, 2]]
                half_length = float(model.geom_size[geom_id, 1])
                segments.append(
                    (
                        center - half_length * axis,
                        center + half_length * axis,
                        float(model.geom_size[geom_id, 0]),
                    )
                )
            return segments

        cases = (
            (
                "front_hip",
                FIXED_PARAMETERS.hip.safe_range,
                "torso",
                "front_thigh",
            ),
            (
                "front_knee",
                FIXED_PARAMETERS.knee.safe_range,
                "front_thigh",
                "front_shank",
            ),
        )
        for joint_name, angle_range, first_prefix, second_prefix in cases:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos_address = model.jnt_qposadr[joint_id]
            minimum_clearance = np.inf
            for angle in np.linspace(*angle_range, 101):
                data.qpos[:] = open_qpos
                data.qpos[qpos_address] = angle
                mujoco.mj_forward(model, data)
                for first_start, first_end, first_radius in shell_segments(
                    first_prefix
                ):
                    for (
                        second_start,
                        second_end,
                        second_radius,
                    ) in shell_segments(second_prefix):
                        minimum_clearance = min(
                            minimum_clearance,
                            segment_distance(
                                first_start,
                                first_end,
                                second_start,
                                second_end,
                            )
                            - first_radius
                            - second_radius,
                        )
            self.assertGreaterEqual(minimum_clearance, 0.002)

    def test_walking_keyframe_has_clear_shells_and_swing_foot(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "walk"
        )
        front_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
        )
        rear_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
        )

        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        self.assertAlmostEqual(
            float(data.site_xpos[front_site, 2]),
            FIXED_PARAMETERS.foot_radius,
        )
        self.assertGreater(
            float(data.site_xpos[rear_site, 2]),
            FIXED_PARAMETERS.foot_radius + 0.01,
        )
        self.assertEqual(data.ncon, 0)

    def test_collision_classes_and_compact_foot_contact(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        for name in (
            "torso_proxy",
            "front_thigh_proxy",
            "front_shank_proxy",
            "front_foot_proxy",
            "rear_thigh_proxy",
            "rear_shank_proxy",
            "rear_foot_proxy",
        ):
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            self.assertEqual(model.geom_contype[geom_id], 2)
            self.assertEqual(model.geom_conaffinity[geom_id], 7)
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if "_shell_" in (name or ""):
                self.assertEqual(model.geom_contype[geom_id], 4)
                self.assertEqual(model.geom_conaffinity[geom_id], 7)

        contacts = {
            frozenset(
                (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                    ),
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                    ),
                )
            )
            for contact in data.contact
        }
        self.assertIn(
            frozenset(("front_foot_proxy", "rear_foot_proxy")), contacts
        )
        self.assertEqual(model.npair, 1)
        self.assertAlmostEqual(
            float(model.opt.timestep), FIXED_PARAMETERS.timestep
        )

        for joint_name, expected_range in (
            ("front_hip", FIXED_PARAMETERS.hip.shell_compatible_range),
            ("rear_hip", FIXED_PARAMETERS.hip.shell_compatible_range),
            ("front_knee", FIXED_PARAMETERS.knee.shell_compatible_range),
            ("rear_knee", FIXED_PARAMETERS.knee.shell_compatible_range),
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            np.testing.assert_allclose(
                model.jnt_range[joint_id], expected_range, atol=1e-9
            )

    def test_keyframes_respect_joint_limits(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)

        for key_id in range(model.nkey):
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            for joint_id in range(model.njnt):
                if not model.jnt_limited[joint_id]:
                    continue
                qpos_address = model.jnt_qposadr[joint_id]
                low, high = model.jnt_range[joint_id]
                self.assertLessEqual(low, data.qpos[qpos_address])
                self.assertLessEqual(data.qpos[qpos_address], high)

    def test_rigid_phase_analysis_preserves_compact_geometry(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        analysis = analyze_rigid_phase(model, samples=361)
        p = FIXED_PARAMETERS

        np.testing.assert_allclose(
            analysis.column("circle_center_z_m"),
            p.shell_contact_radius,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            analysis.column("circle_center_x_m"),
            p.shell_contact_radius * analysis.column("phase_rad"),
            atol=1e-10,
        )
        eccentricity = analysis.column("com_offset_radius_m")
        np.testing.assert_allclose(eccentricity, eccentricity[0], atol=1e-10)

        expected_potential_range = (
            2.0
            * p.total_mass
            * 9.81
            * float(analysis.summary["com_eccentricity_m"])
        )
        self.assertAlmostEqual(
            float(analysis.summary["potential_peak_to_peak_J"]),
            expected_potential_range,
            places=5,
        )
        expected_peak_torque = (
            p.total_mass
            * 9.81
            * float(analysis.summary["com_eccentricity_m"])
        )
        self.assertAlmostEqual(
            float(analysis.summary["gravity_torque_max_Nm"]),
            expected_peak_torque,
            places=4,
        )

    def test_rigid_release_uses_runtime_locks(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        result = run_release(model, joint_mode="rigid", duration=0.05)

        self.assertEqual(result.summary["joint_mode"], "rigid")
        self.assertTrue(result.summary["actuation_disabled"])
        self.assertLess(float(result.summary["max_joint_error_rad"]), 1e-3)
        self.assertGreater(result.rows.shape[0], 2)

    def test_servo_release_uses_actuators_without_rigid_locks(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        result = run_release(model, joint_mode="servo", duration=0.05)

        self.assertEqual(result.summary["joint_mode"], "servo")
        self.assertFalse(result.summary["actuation_disabled"])
        self.assertGreater(
            max(
                float(result.summary[f"{joint_name}_max_torque_Nm"])
                for joint_name, _ in (
                    ("front_hip", FIXED_PARAMETERS.compact_hip_angle),
                    ("front_knee", FIXED_PARAMETERS.compact_knee_angle),
                    ("rear_hip", FIXED_PARAMETERS.compact_hip_angle),
                    ("rear_knee", FIXED_PARAMETERS.compact_knee_angle),
                )
            ),
            0.0,
        )

    def test_phase_controller_starts_from_compact_and_rolls_out(self) -> None:
        targets = controller_targets(
            0.75, 0.0, np.full(8, 0.25, dtype=float)
        )
        np.testing.assert_allclose(
            targets,
            [
                FIXED_PARAMETERS.compact_hip_angle,
                FIXED_PARAMETERS.compact_knee_angle,
                FIXED_PARAMETERS.compact_hip_angle,
                FIXED_PARAMETERS.compact_knee_angle,
            ],
        )

        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        rollout = rollout_controller(
            model, np.zeros(8), duration=0.02, detailed=True
        )
        self.assertIsNotNone(rollout.rows)
        self.assertGreater(rollout.rows.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
