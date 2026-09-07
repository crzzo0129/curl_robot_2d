from __future__ import annotations

import math
import unittest
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d_mjx.terrain_3d import (
    SlopeTerrainConfig,
    column_centers_x,
    flat_hfield_data_3d,
    hfield_data_3d,
    hfield_data_flat_column_major,
    inject_hfield_into_mjcf,
    slope_terrain_config_from_task,
    terrain_height,
    terrain_height_array,
    terrain_hfield_candidates_3d,
    terrain_surface_offset,
    terrain_surface_z_at,
    validate_slope_terrain_config,
)
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, validate_3d_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MESH_MODEL = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad.xml"
)


class SlopeTerrainProfileTest(unittest.TestCase):
    def setUp(self):
        self.config = SlopeTerrainConfig(
            slope_angle_deg=2.0,
            slope_start_distance_m=3.0,
            transition_length_m=0.5,
            slope_length_m=2.0,
        )

    def test_flat_config_yields_zero_height(self):
        flat = SlopeTerrainConfig(slope_angle_deg=0.0)
        x = np.linspace(-5.0, 5.0, 101)
        np.testing.assert_array_equal(
            terrain_height_array(x, flat), np.zeros_like(x)
        )

    def test_uphill_profile_reaches_total_rise_and_is_smooth(self):
        rise = math.tan(math.radians(2.0))
        config = self.config

        self.assertAlmostEqual(terrain_height(0.0, config), 0.0, places=9)
        self.assertAlmostEqual(
            terrain_height(config.slope_start_distance_m, config), 0.0, places=9
        )
        # Mid entry transition is half of a half: rise * L / 2 accumulated over
        # the full transition, so at the midpoint of the entry the height is
        # rise * L * smoothstep_integral(0.5).
        entry_mid = config.slope_start_distance_m + 0.5 * config.transition_length_m
        expected_entry_mid = rise * config.transition_length_m * (
            0.5**3 - 0.5 * 0.5**4
        )
        self.assertAlmostEqual(
            terrain_height(entry_mid, config), expected_entry_mid, places=9
        )
        # Top plateau.
        top = config.slope_start_distance_m + config.transition_length_m + config.slope_length_m + config.transition_length_m
        self.assertAlmostEqual(
            terrain_height(top, config), config.total_rise, places=9
        )
        self.assertAlmostEqual(config.total_rise, rise * (0.5 + 2.0), places=9)

    def test_downhill_profile_is_negative_and_symmetric(self):
        up = SlopeTerrainConfig(slope_angle_deg=3.0)
        down = SlopeTerrainConfig(slope_angle_deg=-3.0)
        x = np.linspace(-2.0, 12.0, 200)
        np.testing.assert_allclose(
            terrain_height_array(x, down),
            -terrain_height_array(x, up),
            atol=1e-12,
        )

    def test_profile_is_monotonic_for_uphill(self):
        x = np.linspace(-2.0, 12.0, 500)
        heights = terrain_height_array(x, self.config)
        self.assertTrue(np.all(np.diff(heights) >= -1e-12))


class SlopeTerrainHfieldTest(unittest.TestCase):
    def setUp(self):
        self.config = SlopeTerrainConfig(
            slope_angle_deg=2.0,
            ncol=64,
            nrow=4,
        )

    def test_hfield_data_shape_and_constant_along_y(self):
        data = hfield_data_3d(self.config)
        self.assertEqual(data.shape, (self.config.nrow, self.config.ncol))
        for row in range(self.config.nrow):
            np.testing.assert_array_equal(data[row], data[0])

    def test_hfield_data_surface_matches_profile(self):
        data = hfield_data_3d(self.config)
        centers = column_centers_x(self.config)
        surface = self.config.hfield_base_z + self.config.hfield_scale_z * data[0]
        expected = terrain_surface_z_at(self.config, centers)
        np.testing.assert_allclose(surface, expected, atol=1e-9)
        # Data must stay nonnegative for the hfield to keep colliding.
        self.assertTrue(np.all(data >= -1e-9))

    def test_surface_offset_keeps_start_at_ground_for_uphill(self):
        uphill = SlopeTerrainConfig(slope_angle_deg=4.0)
        self.assertAlmostEqual(
            terrain_surface_offset(uphill), uphill.hfield_base_z, places=9
        )
        self.assertAlmostEqual(
            terrain_surface_z_at(uphill, 0.0), uphill.hfield_base_z, places=9
        )

    def test_surface_offset_lifts_downhill_start(self):
        downhill = SlopeTerrainConfig(
            slope_angle_deg=-4.0,
            transition_length_m=0.5,
            slope_length_m=2.0,
        )
        self.assertAlmostEqual(
            terrain_surface_offset(downhill),
            downhill.hfield_base_z - downhill.total_rise,
            places=9,
        )
        # Start is on the flat high plateau.
        self.assertAlmostEqual(
            terrain_surface_z_at(downhill, 0.0),
            terrain_surface_offset(downhill),
            places=9,
        )

    def test_flat_hfield_data_is_constant(self):
        data = flat_hfield_data_3d(self.config)
        self.assertEqual(data.shape, (self.config.nrow, self.config.ncol))
        self.assertTrue(np.all(data == data[0, 0]))

    def test_column_major_flatten_layout(self):
        data_2d = np.arange(self.config.nrow * self.config.ncol, dtype=np.float64).reshape(
            self.config.nrow, self.config.ncol
        )
        flat = hfield_data_flat_column_major(self.config, data_2d)
        self.assertEqual(flat.shape, (self.config.nrow * self.config.ncol,))
        # MuJoCo order: column (x) is the outer index.
        self.assertEqual(flat[0], data_2d[0, 0])
        self.assertEqual(flat[1], data_2d[1, 0])
        self.assertEqual(flat[self.config.nrow], data_2d[0, 1])

    def test_task_derived_config_uses_flat_fields(self):
        task = Rolling3DConfig(
            terrain_enabled=True,
            terrain_slope_angle_deg=-2.0,
            terrain_slope_start_distance_m=4.0,
        )
        validate_3d_config(task)
        config = slope_terrain_config_from_task(task)
        self.assertEqual(config.slope_angle_deg, -2.0)
        self.assertEqual(config.slope_start_distance_m, 4.0)


class SlopeTerrainMjcfBakeTest(unittest.TestCase):
    def test_bake_compiles_and_keeps_self_collision_contract(self):
        from curl_robot_2d_mjx.environment_3d import (
            validate_rollingquad_self_collision_contract_3d,
        )

        # Bake next to the source so the relative ../meshes references resolve.
        output = MESH_MODEL.with_name("_terrain_test_bake.xml")
        config = SlopeTerrainConfig(slope_angle_deg=2.0)
        try:
            inject_hfield_into_mjcf(MESH_MODEL, output, config, force=True)
            model = mujoco.MjModel.from_xml_path(str(output))
            floor_id = model.geom("floor").id
            self.assertEqual(model.geom_type[floor_id], mujoco.mjtGeom.mjGEOM_HFIELD)
            self.assertEqual(model.nhfield, 1)
            self.assertEqual(model.hfield_nrow[0], config.nrow)
            self.assertEqual(model.hfield_ncol[0], config.ncol)
            # The floor's contype/conaffinity must remain (1, 0).
            self.assertEqual(int(model.geom_contype[floor_id]), 1)
            self.assertEqual(int(model.geom_conaffinity[floor_id]), 0)
            validate_rollingquad_self_collision_contract_3d(model, "rollingquad_2")
        finally:
            if output.exists():
                output.unlink()

    def test_bake_rejects_existing_output_without_force(self):
        output = MESH_MODEL.with_name("_terrain_test_existing.xml")
        output.write_text("<mujoco/>", encoding="utf-8")
        try:
            with self.assertRaises(FileExistsError):
                inject_hfield_into_mjcf(MESH_MODEL, output, SlopeTerrainConfig())
        finally:
            if output.exists():
                output.unlink()


class SlopeTerrainCommittedModelTest(unittest.TestCase):
    def test_primitive_abd10_terrain_model_validates_and_writes_hfield(self):
        from curl_robot_2d_mjx.environment_3d import (
            terrain_model_path_3d,
            validate_rollingquad_self_collision_contract_3d,
        )

        path = terrain_model_path_3d("rollingquad_2_primitive_abd10")
        model = mujoco.MjModel.from_xml_path(str(path))
        self.assertEqual(model.nhfield, 1)
        self.assertEqual(model.geom_type[model.geom("floor").id], mujoco.mjtGeom.mjGEOM_HFIELD)
        validate_rollingquad_self_collision_contract_3d(
            model, "rollingquad_2_primitive_abd10"
        )

        config = SlopeTerrainConfig(slope_angle_deg=2.0, ncol=model.hfield_ncol[0], nrow=model.hfield_nrow[0])
        data_2d = hfield_data_3d(config)
        data_flat = hfield_data_flat_column_major(config, data_2d)
        model.hfield_data[:] = data_flat
        # Round-trip through the model preserves the column-major flat array
        # (the model stores hfield data as float32).
        np.testing.assert_allclose(model.hfield_data, data_flat, rtol=1e-5, atol=1e-6)
        # The logical row 0 (all x columns) matches the intended surface.
        surface = config.hfield_base_z + config.hfield_scale_z * data_2d[0]
        np.testing.assert_allclose(
            surface,
            terrain_surface_z_at(config, column_centers_x(config)),
            atol=1e-9,
        )


class SlopeCurriculumTest(unittest.TestCase):
    def test_slope_v1_stage_enables_terrain_and_keeps_flat_base(self):
        from curl_robot_2d_mjx.curriculum_3d import curriculum_stages_3d

        stages = curriculum_stages_3d("slope_v1")
        self.assertEqual(len(stages), 1)
        stage = stages[0]
        self.assertEqual(stage.name, "slope_02")
        self.assertTrue(stage.terrain_enabled)
        self.assertAlmostEqual(stage.terrain_slope_probability, 0.30)
        self.assertAlmostEqual(stage.terrain_max_angle_deg, 2.0)

        task = stage.task_config(Rolling3DConfig())
        self.assertTrue(task.terrain_enabled)
        self.assertEqual(task.terrain_slope_angle_deg, 0.0)
        validate_slope_terrain_config(slope_terrain_config_from_task(task))

    def test_slope_v2_expands_slope_magnitude(self):
        from curl_robot_2d_mjx.curriculum_3d import curriculum_stages_3d

        stages = curriculum_stages_3d("slope_v2")
        self.assertEqual([s.name for s in stages], [
            "slope_02", "slope_04", "slope_06", "slope_08", "slope_10",
        ])
        self.assertEqual(
            [s.terrain_max_angle_deg for s in stages], [2.0, 4.0, 6.0, 8.0, 10.0]
        )
        self.assertTrue(all(s.terrain_enabled for s in stages))
        self.assertTrue(all(s.terrain_slope_probability == 0.30 for s in stages))

    def test_terrain_candidates_flat_uphill_downhill(self):
        config = SlopeTerrainConfig(ncol=16, nrow=2)
        labels, data = terrain_hfield_candidates_3d(config, 2.0)

        self.assertEqual(labels, ("flat", "uphill", "downhill"))
        self.assertEqual(data.shape, (3, 2 * 16))
        self.assertTrue(np.all(data[0] >= -1e-9))
        # Uphill and downhill surface profiles are not identical.
        self.assertFalse(np.allclose(data[1], data[2]))


if __name__ == "__main__":
    unittest.main()
